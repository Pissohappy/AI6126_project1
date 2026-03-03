# train.py
import argparse
import json
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SegPaths, FaceParsingDataset, ComposeJoint, JointResize, ToTensorNormalize
from src.model import MiniUNet, count_params
from src.metrics import f1_score_multiclass


def load_split(split_path: str):
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ----- paths -----
    train_images = os.path.join(args.data_root, "train", "train", "images")
    train_masks  = os.path.join(args.data_root, "train", "train", "masks")

    split = load_split(args.split)
    train_list: List[str] = split["train"]
    val_list: List[str] = split["val"]

    # ----- transforms (no augmentation yet) -----
    joint = ComposeJoint([
        JointResize((512, 512)),
        ToTensorNormalize(),
    ])

    train_ds = FaceParsingDataset(
        paths=SegPaths(images_dir=train_images, masks_dir=train_masks),
        file_list=train_list,
        joint_transform=joint,
        return_name=False
    )
    val_ds = FaceParsingDataset(
        paths=SegPaths(images_dir=train_images, masks_dir=train_masks),
        file_list=val_list,
        joint_transform=joint,
        return_name=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ----- model -----
    model = MiniUNet(num_classes=args.num_classes, base_ch=32).to(device)
    n_params = count_params(model)
    print(f"Model params: {n_params:,}")
    if n_params >= 1_821_085:
        print("WARNING: parameter count exceeds limit!")

    # ----- loss/optim -----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, masks)
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
            logits = model(imgs)
            f1 = f1_score_multiclass(logits, masks, num_classes=args.num_classes, ignore_background=True)
            f1_sum += f1
            n_batches += 1

        val_f1 = f1_sum / max(1, n_batches)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_f1={val_f1:.4f}")

        # ---- save best ----
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(args.ckpt_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_f1": best_f1,
                "num_classes": args.num_classes,
            }, ckpt_path)
            print(f"  Saved best checkpoint: {ckpt_path} (best_f1={best_f1:.4f})")

    print(f"Done. Best val_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()