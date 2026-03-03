# train.py
import argparse
import json
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
from src.model import MiniUNet, count_params
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


def _maybe_init_wandb(args, model, n_params: int):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except Exception as e:
        raise RuntimeError("wandb 未安装，请先 pip install wandb") from e

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
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--class_weights", type=str, default="", help="Comma-separated class weights")
    parser.add_argument("--use_dice", action="store_true")
    parser.add_argument("--dice_weight", type=float, default=0.4)
    parser.add_argument("--use_dsconv", action="store_true")
    parser.add_argument("--use_aux_head", action="store_true")
    parser.add_argument("--aux_weight", type=float, default=0.3)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--use_context", action="store_true")

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
    parser.add_argument("--val_log_cm_max_items", type=int, default=200000, help="(optional) max pixels if using y_true/y_pred list; not used in heatmap mode")

    args = parser.parse_args()

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ----- model -----
    model = MiniUNet(
        num_classes=args.num_classes,
        base_ch=32,
        use_dsconv=args.use_dsconv,
        use_attention=args.use_attention,
        use_context=args.use_context,
        use_aux_head=args.use_aux_head,
    ).to(device)

    n_params = count_params(model)
    print(f"Model params: {n_params:,}")
    if n_params >= 1_821_085:
        print("WARNING: parameter count exceeds limit!")

    # ----- wandb init -----
    run = _maybe_init_wandb(args, model, n_params)

    # ----- loss/optim -----
    class_weights = None
    if args.class_weights:
        values = [float(x.strip()) for x in args.class_weights.split(",") if x.strip()]
        if len(values) != args.num_classes:
            raise ValueError(f"Expected {args.num_classes} class weights, got {len(values)}")
        class_weights = torch.tensor(values, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    dice_criterion = DiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
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
            if args.use_aux_head:
                logits, aux = out
            else:
                logits = out
                aux = None

            loss = criterion(logits, masks)
            if args.use_dice:
                loss = (1 - args.dice_weight) * loss + args.dice_weight * dice_criterion(logits, masks)

            if aux is not None:
                aux_loss = criterion(aux, masks)
                if args.use_dice:
                    aux_loss = (1 - args.dice_weight) * aux_loss + args.dice_weight * dice_criterion(aux, masks)
                loss = loss + args.aux_weight * aux_loss

            loss.backward()
            optimizer.step()

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

        # # ---- val ----
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
            logits = out[0] if args.use_aux_head else out

            if first_batch is None:
                first_batch = (imgs.detach(), masks.detach(), logits.detach())

            # 你原来的 overall f1（macro/whatever 由你的实现决定）
            f1 = f1_score_multiclass(logits, masks, num_classes=args.num_classes, ignore_background=True)
            f1_sum += float(f1)
            n_batches += 1

            # confusion matrix (per-pixel)
            pred = torch.argmax(logits, dim=1)  # (B,H,W)
            cm = compute_confusion_matrix(
                pred=pred.detach().cpu(),
                target=masks.detach().cpu(),
                num_classes=args.num_classes,
                ignore_index=-1,  # 如果你的 dataset 有 ignore_index，改成对应值
            )
            cm_total += cm

        val_f1 = f1_sum / max(1, n_batches)
        # model.eval()
        # f1_sum = 0.0
        # n_batches = 0

        # # 为了 log_images，缓存一小批
        # first_batch = None

        # for imgs, masks in val_loader:
        #     imgs = imgs.to(device, non_blocking=True)
        #     masks = masks.to(device, non_blocking=True)

        #     out = model(imgs)
        #     logits = out[0] if args.use_aux_head else out

        #     if first_batch is None:
        #         first_batch = (imgs.detach(), masks.detach(), logits.detach())

        #     f1 = f1_score_multiclass(logits, masks, num_classes=args.num_classes, ignore_background=True)
        #     f1_sum += float(f1)
        #     n_batches += 1

        # val_f1 = f1_sum / max(1, n_batches)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:03d} | lr={current_lr:.6g} | train_loss={train_loss:.4f} | val_f1={val_f1:.4f}")

        # ---- wandb log (per-epoch) ----
        # if run is not None:
        #     import wandb
        #     wandb.log(
        #         {
        #             "epoch": epoch,
        #             "lr": current_lr,
        #             "train/loss": train_loss,
        #             "val/f1": val_f1,
        #         },
        #         step=epoch,
        #     )
        #     if args.log_images and first_batch is not None:
        #         imgs_b, masks_b, logits_b = first_batch
        #         _log_val_images_wandb(run, imgs_b, masks_b, logits_b, num_classes=args.num_classes, max_items=4, step=epoch)
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

                # 如果你希望像 overall f1 一样忽略 background=0，这里也可以排除 0
                # 你也可以不排除，看你的定义
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