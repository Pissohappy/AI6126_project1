# train.py
import argparse
import json
import os
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import (
    SegPaths,
    FaceParsingDataset,
    ComposeJoint,
    JointResize,
    JointRandomHorizontalFlip,
    JointRandomAffine,
    JointColorJitter,
    ToTensorNormalize,
)
from src.model import MiniUNet, SRResNetSeg, count_params
from src.metrics import f1_score_multiclass


def load_split(split_path: str):
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index >= 0:
            valid = target != self.ignore_index

        target_one_hot = torch.zeros_like(probs)
        target_clamped = target.clamp(min=0, max=num_classes - 1)
        target_one_hot.scatter_(1, target_clamped.unsqueeze(1), 1)

        valid = valid.unsqueeze(1)
        probs = probs * valid
        target_one_hot = target_one_hot * valid

        dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dims)
        union = probs.sum(dims) + target_one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter, higher gamma means more focus on hard examples
        alpha: class weights (optional)
        ignore_index: index to ignore in loss calculation
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """
        Args:
            logits: (B, C, H, W) raw predictions
            target: (B, H, W) ground truth labels
        """
        ce_loss = F.cross_entropy(
            logits, target,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        # p_t = exp(-ce_loss) is the probability of the correct class
        pt = torch.exp(-ce_loss)

        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        focal_loss = focal_term * ce_loss

        # Mask out ignored indices
        if self.ignore_index >= 0:
            valid_mask = target != self.ignore_index
            focal_loss = focal_loss[valid_mask]

        return focal_loss.mean()


def compute_sample_weights(
    file_list: List[str],
    masks_dir: str,
    rare_classes: List[int] = None,
    rare_weight: float = 5.0
) -> List[float]:
    """Compute sample weights for balanced sampling.

    Samples containing rare classes get higher weights to oversample them.

    Args:
        file_list: list of image filenames
        masks_dir: directory containing mask files
        rare_classes: list of rare class ids to oversample
        rare_weight: weight multiplier for samples containing rare classes

    Returns:
        List of sample weights
    """
    if rare_classes is None:
        # Default rare classes based on data analysis
        rare_classes = [3, 14, 15, 16]  # glasses, earring, neck, necklace

    weights = []
    for fname in file_list:
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(masks_dir, stem + ".png")

        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            # Check if any rare class exists in this mask
            has_rare = any(c in mask for c in rare_classes)
            w = rare_weight if has_rare else 1.0
        else:
            w = 1.0
        weights.append(w)

    return weights


def _maybe_init_wandb(args, model, n_params: int):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except Exception as e:
        raise RuntimeError("wandb 未安装，请先 pip install wandb") from e

    # 设置 wandb 日志保存目录，避免与 wandb 包名冲突
    os.makedirs("wandb_logs", exist_ok=True)
    os.environ["WANDB_DIR"] = "wandb_logs"

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        tags=tags,
        mode=args.wandb_mode,  # "online" | "offline" | "disabled"
        config=vars(args),
    )

    # 记录一些静态信息
    wandb.config.update({"model_params": n_params}, allow_val_change=True)

    if args.wandb_watch != "none":
        # log="gradients" 或 "all"
        wandb.watch(model, log=args.wandb_watch, log_freq=max(1, args.log_freq))

    return run


@torch.no_grad()
def _log_val_images_wandb(run, imgs, masks, logits, num_classes: int, max_items: int = 4, step: Optional[int] = None):
    """
    记录一些验证集可视化到 wandb：
    - 输入图（归一化后可能看起来怪，这里尽量反归一化到 0~1）
    - GT mask（类别 id）
    - Pred mask（argmax）
    """
    if run is None:
        return
    import wandb

    # imgs: (B,3,H,W), masks: (B,H,W), logits:(B,C,H,W)
    b = min(imgs.size(0), max_items)
    preds = torch.argmax(logits, dim=1)[:b].detach().cpu()
    gts = masks[:b].detach().cpu()
    ims = imgs[:b].detach().cpu()

    # 尝试把输入图拉到 0~1（不假设具体 mean/std，只做 min-max）
    ims_vis = []
    for i in range(b):
        x = ims[i]
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        ims_vis.append(x)

    rows = []
    for i in range(b):
        # wandb.Image 支持 mask 显示；class_labels 可以让 UI 显示类别名（这里用数字占位）
        class_labels = {k: str(k) for k in range(num_classes)}
        rows.append(
            wandb.Image(
                ims_vis[i].permute(1, 2, 0).numpy(),
                masks={
                    "ground_truth": {"mask_data": gts[i].numpy(), "class_labels": class_labels},
                    "prediction": {"mask_data": preds[i].numpy(), "class_labels": class_labels},
                },
            )
        )

    wandb.log({"val/examples": rows}, step=step)


@torch.no_grad()
def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = -1):
    """
    pred/target: (B,H,W) int64
    return: (C,C) where rows=gt, cols=pred
    """
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)

    if ignore_index >= 0:
        m = target != ignore_index
        pred = pred[m]
        target = target[m]

    # clamp to valid range just in case
    pred = pred.clamp(0, num_classes - 1)
    target = target.clamp(0, num_classes - 1)

    k = target * num_classes + pred
    binc = torch.bincount(k, minlength=num_classes * num_classes)
    cm = binc.view(num_classes, num_classes)
    return cm


@torch.no_grad()
def per_class_f1_from_cm(cm: torch.Tensor, eps: float = 1e-12):
    """
    cm: (C,C) rows=gt, cols=pred
    returns: precision(C), recall(C), f1(C)
    """
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def log_confusion_matrix_wandb(cm: torch.Tensor, class_names, step: int, title: str = "val/confusion_matrix"):
    """
    用 heatmap 方式把 confusion matrix 画成图丢给 wandb。
    """
    import wandb
    import matplotlib.pyplot as plt

    cm = cm.detach().cpu().to(torch.float32)
    cm_sum = cm.sum(dim=1, keepdim=True).clamp_min(1.0)
    cm_norm = cm / cm_sum  # 按 GT 行归一化：每行和为 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm.numpy())  # 不显式指定颜色；用默认 colormap
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    # colorbar（不指定颜色风格）
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    wandb.log({title: wandb.Image(fig)}, step=step)
    plt.close(fig)


def build_experiment_config(args):
    """Build a consistent experiment config from preset + CLI args."""
    presets = {
        "baseline": {
            "model_arch": "miniunet",
            "model": {
                "base_ch": 32,
                "backbone": "plain",
                "use_dsconv": False,
                "use_attention": False,
                "use_context": False,
                "use_aux_head": False,
                "sr_num_blocks": 8,
            },
            "training": {
                "use_dice": False,
                "use_balanced_sampling": False,
                "use_focal": False,
            },
        },
        "full_strong": {
            "model_arch": "miniunet",
            "model": {
                "base_ch": 32,
                "backbone": "plain",
                "use_dsconv": True,
                "use_attention": True,
                "use_context": True,
                "use_aux_head": True,
                "sr_num_blocks": 8,
            },
            "training": {
                "use_dice": True,
                "use_balanced_sampling": True,
                "use_focal": False,
            },
        },
        "baseline4": {
            "model_arch": "miniunet",
            "model": {
                "base_ch": 32,
                "backbone": "residual",
                "use_dsconv": True,
                "use_attention": True,
                "use_context": True,
                "use_aux_head": True,
                "sr_num_blocks": 8,
            },
            "training": {
                "use_dice": True,
                "use_balanced_sampling": True,
                "use_focal": True,
            },
        },
        "srresnet_baseline": {
            "model_arch": "srresnet",
            "model": {
                "base_ch": 32,
                "backbone": "plain",
                "use_dsconv": False,
                "use_attention": False,
                "use_context": False,
                "use_aux_head": False,
                "sr_num_blocks": 8,
            },
            "training": {
                "use_dice": False,
                "use_balanced_sampling": False,
                "use_focal": False,
            },
        },
        "custom": {
            "model_arch": args.model_arch,
            "model": {
                "base_ch": 32,
                "backbone": args.backbone,
                "use_dsconv": args.use_dsconv,
                "use_attention": args.use_attention,
                "use_context": args.use_context,
                "use_aux_head": args.use_aux_head,
                "sr_num_blocks": args.sr_num_blocks,
            },
            "training": {
                "use_dice": args.use_dice,
                "use_balanced_sampling": args.use_balanced_sampling,
                "use_focal": args.use_focal,
            },
        },
    }

    if args.exp_preset not in presets:
        raise ValueError(f"Unknown exp preset: {args.exp_preset}")
    return presets[args.exp_preset]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="splits/train_split.json")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--use_hflip", action="store_true")
    parser.add_argument("--use_affine", action="store_true")
    parser.add_argument("--use_color_jitter", action="store_true")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "onecycle"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--class_weights", type=str, default="", help="Comma-separated class weights")
    parser.add_argument("--use_dice", action="store_true")
    parser.add_argument("--dice_weight", type=float, default=0.4)
    parser.add_argument("--use_dsconv", action="store_true")
    parser.add_argument("--use_aux_head", action="store_true")
    parser.add_argument("--aux_weight", type=float, default=0.3)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--use_context", action="store_true")
    parser.add_argument("--backbone", type=str, default="plain", choices=["plain", "residual"],
                        help="Encoder-decoder block type. residual keeps params under the fairness limit.")
    parser.add_argument("--model_arch", type=str, default="miniunet", choices=["miniunet", "srresnet"])
    parser.add_argument("--sr_num_blocks", type=int, default=8, help="Number of SRResNet residual blocks.")
    parser.add_argument("--exp_preset", type=str, default="custom",
                        choices=["baseline", "full_strong", "baseline4", "srresnet_baseline", "custom"])

    # ---- Focal Loss ----
    parser.add_argument("--use_focal", action="store_true", help="Use Focal Loss instead of CrossEntropy")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for CrossEntropy (ignored when using focal loss)")

    # ---- Balanced Sampling ----
    parser.add_argument("--use_balanced_sampling", action="store_true",
                        help="Use WeightedRandomSampler to oversample images with rare classes")
    parser.add_argument("--rare_classes", type=str, default="3,14,15,16",
                        help="Comma-separated rare class ids to oversample (default: glasses,earring,neck,necklace)")
    parser.add_argument("--rare_weight", type=float, default=5.0,
                        help="Weight multiplier for samples containing rare classes")

    # ---- wandb ----
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="face-parsing")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity/team, optional")
    parser.add_argument("--wandb_name", type=str, default="", help="Run name, optional")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated tags")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_watch", type=str, default="none", choices=["none", "gradients", "all"])
    parser.add_argument("--log_images", action="store_true", help="Log a few val images each epoch")
    parser.add_argument("--log_freq", type=int, default=50, help="wandb.watch log frequency (steps)")
    parser.add_argument("--wandb_artifact", action="store_true", help="Upload best checkpoint as artifact")

    # ---- wandb logging granularity ----
    parser.add_argument("--train_log_every", type=int, default=50, help="log train metrics every N steps")
    parser.add_argument("--val_log_cm", action="store_true", help="log confusion matrix on val")
    parser.add_argument("--val_log_cm_max_items", type=int, default=200000,
                        help="(optional) max pixels if using y_true/y_pred list; not used in heatmap mode")

    args = parser.parse_args()
    exp_cfg = build_experiment_config(args)
    model_cfg = exp_cfg["model"]
    train_cfg = exp_cfg["training"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ----- paths -----
    train_images = os.path.join(args.data_root, "train", "train", "images")
    train_masks = os.path.join(args.data_root, "train", "train", "masks")

    split = load_split(args.split)
    train_list: List[str] = split["train"]
    val_list: List[str] = split["val"]

    # ----- transforms -----
    train_tfms = [JointResize((args.size, args.size))]
    if args.use_hflip:
        train_tfms.append(JointRandomHorizontalFlip(p=0.5))
    if args.use_affine:
        train_tfms.append(JointRandomAffine(p=0.5))
    if args.use_color_jitter:
        train_tfms.append(JointColorJitter(p=0.5, brightness=0.12, contrast=0.12, saturation=0.12))
    train_tfms.append(ToTensorNormalize())
    train_joint = ComposeJoint(train_tfms)

    val_joint = ComposeJoint([
        JointResize((args.size, args.size)),
        ToTensorNormalize(),
    ])

    train_ds = FaceParsingDataset(
        paths=SegPaths(images_dir=train_images, masks_dir=train_masks),
        file_list=train_list,
        joint_transform=train_joint,
        return_name=False
    )
    val_ds = FaceParsingDataset(
        paths=SegPaths(images_dir=train_images, masks_dir=train_masks),
        file_list=val_list,
        joint_transform=val_joint,
        return_name=False
    )

    # ----- Balanced Sampling -----
    sampler = None
    shuffle = True
    if train_cfg["use_balanced_sampling"]:
        rare_classes = [int(x.strip()) for x in args.rare_classes.split(",") if x.strip()]
        print(f"Using balanced sampling with rare classes: {rare_classes}, weight: {args.rare_weight}")
        sample_weights = compute_sample_weights(
            train_list, train_masks, rare_classes=rare_classes, rare_weight=args.rare_weight
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_list),
            replacement=True
        )
        shuffle = False  # When using sampler, shuffle must be False
        print(f"  Total samples: {len(train_list)}, "
              f"rare samples: {sum(1 for w in sample_weights if w > 1.0)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ----- model -----
    model_arch = exp_cfg["model_arch"]
    if model_arch == "miniunet":
        model = MiniUNet(
            num_classes=args.num_classes,
            base_ch=model_cfg["base_ch"],
            backbone=model_cfg["backbone"],
            use_dsconv=model_cfg["use_dsconv"],
            use_attention=model_cfg["use_attention"],
            use_context=model_cfg["use_context"],
            use_aux_head=model_cfg["use_aux_head"],
        ).to(device)
    else:
        model = SRResNetSeg(
            num_classes=args.num_classes,
            base_ch=model_cfg["base_ch"],
            num_blocks=model_cfg["sr_num_blocks"],
        ).to(device)

    n_params = count_params(model)
    print(f"Model params: {n_params:,}")
    if n_params >= 1_821_085:
        raise RuntimeError(f"Model params {n_params:,} exceed limit 1,821,085. Abort for fairness.")

    # ----- wandb init -----
    run = _maybe_init_wandb(args, model, n_params)

    # ----- loss/optim -----
    class_weights = None
    if args.class_weights:
        values = [float(x.strip()) for x in args.class_weights.split(",") if x.strip()]
        if len(values) != args.num_classes:
            raise ValueError(f"Expected {args.num_classes} class weights, got {len(values)}")
        class_weights = torch.tensor(values, dtype=torch.float32, device=device)

    # Choose loss function
    if train_cfg["use_focal"]:
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weights)
        print(f"Using Focal Loss with gamma={args.focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"Using CrossEntropy Loss (label_smoothing={args.label_smoothing})")

    dice_criterion = DiceLoss()

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.1,
        )

    best_f1 = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            if model_arch == "miniunet" and model_cfg["use_aux_head"]:
                logits, aux = out
            else:
                logits = out
                aux = None

            loss = criterion(logits, masks)
            if train_cfg["use_dice"]:
                loss = (1 - args.dice_weight) * loss + args.dice_weight * dice_criterion(logits, masks)

            if aux is not None:
                aux_loss = criterion(aux, masks)
                if train_cfg["use_dice"]:
                    aux_loss = (1 - args.dice_weight) * aux_loss + args.dice_weight * dice_criterion(aux, masks)
                loss = loss + args.aux_weight * aux_loss

            loss.backward()
            optimizer.step()
            if scheduler is not None and args.scheduler == "onecycle":
                scheduler.step()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            global_step += 1

            # ---- wandb step logging ----
            if run is not None and (global_step % max(1, args.train_log_every) == 0):
                import wandb
                current_lr_step = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "train/loss_step": float(loss.item()),
                        "lr": float(current_lr_step),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        train_loss = running_loss / len(train_ds)

        # ---- val ----
        model.eval()
        f1_sum = 0.0
        n_batches = 0

        cm_total = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64, device="cpu")

        first_batch = None

        for imgs, masks in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            out = model(imgs)
            logits = out[0] if (model_arch == "miniunet" and model_cfg["use_aux_head"]) else out

            if first_batch is None:
                first_batch = (imgs.detach(), masks.detach(), logits.detach())

            f1 = f1_score_multiclass(logits, masks, num_classes=args.num_classes, ignore_background=True)
            f1_sum += float(f1)
            n_batches += 1

            # confusion matrix (per-pixel)
            pred = torch.argmax(logits, dim=1)  # (B,H,W)
            cm = compute_confusion_matrix(
                pred=pred.detach().cpu(),
                target=masks.detach().cpu(),
                num_classes=args.num_classes,
                ignore_index=-1,
            )
            cm_total += cm

        val_f1 = f1_sum / max(1, n_batches)

        if scheduler is not None and args.scheduler != "onecycle":
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:03d} | lr={current_lr:.6g} | train_loss={train_loss:.4f} | val_f1={val_f1:.4f}")

        # ---- wandb log (per-epoch) ----
        if run is not None:
            import wandb

            # 统一用 global_step 作为 wandb step（训练/验证曲线会对齐）
            wandb.log(
                {
                    "epoch": epoch,
                    "lr": float(current_lr),
                    "train/loss_epoch": float(train_loss),
                    "val/f1": float(val_f1),
                },
                step=global_step,
            )

            # 记录 per-class 指标 + confusion matrix（可选）
            if args.val_log_cm:
                # class_names：如果你有真实类别名，换成你的；没有就用 0..C-1
                class_names = [str(i) for i in range(args.num_classes)]

                prec, rec, f1c = per_class_f1_from_cm(cm_total)

                f1_no_bg = f1c[1:].mean().item() if args.num_classes > 1 else f1c.mean().item()

                log_dict = {"val/f1_no_bg_from_cm": float(f1_no_bg)}
                for i in range(args.num_classes):
                    log_dict[f"val/f1_class/{i}"] = float(f1c[i].item())
                    log_dict[f"val/precision_class/{i}"] = float(prec[i].item())
                    log_dict[f"val/recall_class/{i}"] = float(rec[i].item())

                wandb.log(log_dict, step=global_step)

                # confusion matrix heatmap
                log_confusion_matrix_wandb(cm_total, class_names, step=global_step, title="val/confusion_matrix")

        # ---- save best ----
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(args.ckpt_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "best_f1": best_f1,
                "num_classes": args.num_classes,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved best checkpoint: {ckpt_path} (best_f1={best_f1:.4f})")

            # 上传 artifact（可选）
            if run is not None and args.wandb_artifact:
                import wandb
                art = wandb.Artifact(name="best-checkpoint", type="model")
                art.add_file(ckpt_path)
                wandb.log_artifact(art)

    print(f"Done. Best val_f1={best_f1:.4f}")
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
