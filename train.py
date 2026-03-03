# train.py
import argparse
import json
import os
from typing import List

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="splits/train_split.json")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=20)
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ----- paths -----
    train_images = os.path.join(args.data_root, "train", "train", "images")
    train_masks  = os.path.join(args.data_root, "train", "train", "masks")

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

    # ----- loss/optim -----
    # criterion = nn.CrossEntropyLoss()

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

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        # ---- val ----
        model.eval()
        f1_sum = 0.0
        n_batches = 0
        for imgs, masks in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            out = model(imgs)
            logits = out[0] if args.use_aux_head else out
            f1 = f1_score_multiclass(logits, masks, num_classes=args.num_classes, ignore_background=True)
            f1_sum += f1
            n_batches += 1

        val_f1 = f1_sum / max(1, n_batches)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:03d} | lr={current_lr:.6g} | train_loss={train_loss:.4f} | val_f1={val_f1:.4f}")

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

    print(f"Done. Best val_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()