import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

@dataclass
class SegPaths:
    image_dir: str
    masks_dir: Optional[str] = None # none的时候就是val集合

def load_image_rgb(path: str) -> Image.Image:
    """Return RGB PIL image."""
    # img格式都是‘RGB’，如果存在异常也要转成RGB
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    return img

def load_mask(path: str) -> Image.Image:
    """Return mask as PIL image"""
    # 一共有19种masks的类别 0-18
    # 注意masks的图片格式是‘P’，不是RGB
    return Image.open(path)

class FaceParsingDataset(Dataset):
    def __init__(
        self,
        paths: SegPaths,
        file_list: Optional[List[str]] = None,
        joint_transform = None,
        return_name: bool = False,
    ):
        super().__init__()
        self.paths = paths
        self.joint_transform = joint_transform
        self.return_name = return_name
        
        if file_list is None:
            self.files = sorted([
                f for f in os.listdir(paths.image_dir)
                if f.lower().endswith((".jpg", ".png"))
            ])
        else:
            self.files = file_list
        
        if len(self.files) == 0:
            raise RuntimeError(f"No image found in {paths.image_dir}")

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index: int):
        fname = self.files[index]
        img_path = os.path.join(self.paths.image_dir, fname)
        img = load_image_rgb(img_path)
        
        mask = None
        if self.paths.masks_dir is not None:
            stem = os.path.splitext(fname)[0]
            mask_path = os.path.join(self.paths.masks_dir, stem + ".png")
            mask = load_mask(mask_path)
        
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        
        if self.return_name:
            return img, mask, fname
        
        return img , mask 