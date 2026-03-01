import argparse
import os
import random

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask(path: str) -> np.ndarray:
    m = Image.open(path)
    return np.array(m)


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Simple overlay: convert mask to a color map for visualization.
    Works best when mask is single-channel class ids.
    """
    out = img.copy()

    if mask.ndim == 3:
        # RGB mask: overlay directly (just to visualize)
        m_rgb = mask
    else:
        # class-id mask -> colorize
        # Use matplotlib colormap; avoid hardcoding colors
        cmap = plt.get_cmap("tab20")
        m_norm = (mask.astype(np.float32) / (mask.max() + 1e-6))
        m_rgb = (cmap(m_norm)[..., :3] * 255).astype(np.uint8)

    out = (out * (1 - alpha) + m_rgb * alpha).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    imgs = sorted([f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not imgs:
        raise RuntimeError("No images found.")

    random.seed(args.seed)
    sample = random.sample(imgs, k=min(args.n, len(imgs)))

    # 1) Print mask stats for a few samples
    print("=== Mask sanity check ===")
    global_uniques = set()

    for f in sample:
        stem = os.path.splitext(f)[0]
        img_path = os.path.join(args.images_dir, f)
        mask_path = os.path.join(args.masks_dir, stem + ".png")

        img = load_rgb(img_path)
        m = load_mask(mask_path)

        print(f"\nFile: {f}")
        print(f"  image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")
        print(f"  mask  shape: {m.shape}, dtype: {m.dtype}")

        if m.ndim == 2:
            u = np.unique(m)
            print(f"  mask unique count: {len(u)}")
            print(f"  mask unique (first 30): {u[:30]}")
            for x in u.tolist():
                global_uniques.add(int(x))
        else:
            # RGB mask case
            u = np.unique(m.reshape(-1, m.shape[-1]), axis=0)
            print(f"  mask appears RGB/3ch, unique colors (sample up to 10): {u[:10]} (total colors: {len(u)})")

    if global_uniques:
        g = np.array(sorted(list(global_uniques)))
        print("\n=== Global (sampled) unique class ids ===")
        print(f"count={len(g)}, min={g.min()}, max={g.max()}")
        print(g[:60])

    # 2) Visualization
    cols = 3
    rows = len(sample)
    plt.figure(figsize=(cols * 5, rows * 4))

    for i, f in enumerate(sample):
        stem = os.path.splitext(f)[0]
        img_path = os.path.join(args.images_dir, f)
        mask_path = os.path.join(args.masks_dir, stem + ".png")

        img = load_rgb(img_path)
        m = load_mask(mask_path)
        ov = overlay(img, m)

        plt.subplot(rows, cols, i * cols + 1)
        plt.imshow(img)
        plt.title(f"Image: {f}")
        plt.axis("off")

        plt.subplot(rows, cols, i * cols + 2)
        if m.ndim == 2:
            plt.imshow(m)
            plt.title("Mask (class ids)")
        else:
            plt.imshow(m)
            plt.title("Mask (RGB?)")
        plt.axis("off")

        plt.subplot(rows, cols, i * cols + 3)
        plt.imshow(ov)
        plt.title("Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()