import argparse
import json
import os
import random
from typing import List

def list_images(images_dir: str) -> List[str]:
    files = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg"))]
    files.sort()
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="split/train_split.json")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    files = list_images(args.images_dir)
    if len(files) == 0:
        raise RuntimeError(f"No images found")
    
    random.seed(args.seed)
    random.shuffle(files)
    
    n_val = int(len(files) * args.val_ratio)
    val_files = sorted(files[:n_val])
    train_files = sorted(files[n_val:])
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "train" : train_files,
            "val" : val_files 
        }, f, indent=2)
        
    print(f"Total: {len(files)} | Train: {len(train_files)} | Val: {len(val_files)}")
    print(f"Saved to: {args.out}")

if __name__ == "__main__":
    main()
    