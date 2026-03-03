
# debug_masks.py
import os
import json
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    SegPaths,
    FaceParsingDataset,
    ComposeJoint,
    JointResize,
    ToTensorNormalize,
)


def load_split(split_path: str):
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    data_root = "data"
    split_path = "splits/train_split.json"
    size = 512
    batch_size = 2
    num_workers = 0  # debug 时建议设 0

    train_images = os.path.join(data_root, "train", "train", "images")
    train_masks = os.path.join(data_root, "train", "train", "masks")

    split = load_split(split_path)
    train_list = split["train"]

    joint = ComposeJoint([
        JointResize((size, size)),
        ToTensorNormalize(),
    ])

    dataset = FaceParsingDataset(
        paths=SegPaths(images_dir=train_images, masks_dir=train_masks),
        file_list=train_list,
        joint_transform=joint,
        return_name=False
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("Checking a few batches...\n")

    for i, (imgs, masks) in enumerate(loader):
        print(f"Batch {i}")
        print("  dtype:", masks.dtype)
        print("  shape:", masks.shape)

        unique_vals = torch.unique(masks)
        print("  unique values:", unique_vals)
        print("  min:", masks.min().item())
        print("  max:", masks.max().item())

        if 255 in unique_vals:
            print("  ⚠ Found value 255 → likely ignore_index = 255")
        if -1 in unique_vals:
            print("  ⚠ Found value -1 → likely ignore_index = -1")

        print("-" * 50)

        if i == 4:  # 只看前 5 个 batch
            break


if __name__ == "__main__":
    main()