import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image
from PIL import ImageEnhance

import torch
from torch.utils.data import Dataset

@dataclass
class SegPaths:
    images_dir: str
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


# ---------- Step 3: joint transform (minimal, safe) ----------
class JointResize:
    """Resize image with bilinear, mask with nearest."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (H, W)

    def __call__(self, img: Image.Image, mask: Optional[Image.Image]):
        img = img.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((self.size[1], self.size[0]), resample=Image.NEAREST)
        return img, mask


class JointRandomHorizontalFlip:
    """Flip image and mask together."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, mask: Optional[Image.Image]):
        if np.random.rand() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class JointRandomAffine:
    """Apply the same small affine transform to image and mask."""
    def __init__(
        self,
        p: float = 0.5,
        max_rotate: float = 8.0,
        max_translate_ratio: float = 0.04,
        min_scale: float = 0.95,
        max_scale: float = 1.05,
    ):
        self.p = p
        self.max_rotate = max_rotate
        self.max_translate_ratio = max_translate_ratio
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image, mask: Optional[Image.Image]):
        if np.random.rand() >= self.p:
            return img, mask

        w, h = img.size
        angle = float(np.random.uniform(-self.max_rotate, self.max_rotate))
        scale = float(np.random.uniform(self.min_scale, self.max_scale))
        tx = float(np.random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * w)
        ty = float(np.random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * h)

        # PIL.Image.rotate does not support `scale` across Pillow versions.
        # Use generic affine matrix to keep compatibility.
        rad = np.deg2rad(angle)
        cos = float(np.cos(rad)) / scale
        sin = float(np.sin(rad)) / scale

        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5

        # output(x, y) -> input(a*x + b*y + c, d*x + e*y + f)
        a = cos
        b = sin
        d = -sin
        e = cos
        c = cx - a * cx - b * cy - tx
        f = cy - d * cx - e * cy - ty
        coeffs = (a, b, c, d, e, f)

        img = img.transform(
            size=(w, h),
            method=Image.AFFINE,
            data=coeffs,
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        if mask is not None:
            mask = mask.transform(
                size=(w, h),
                method=Image.AFFINE,
                data=coeffs,
                resample=Image.NEAREST,
                fillcolor=0,
            )

        # img = img.rotate(
        #     angle,
        #     resample=Image.BILINEAR,
        #     translate=(tx, ty),
        #     scale=scale,
        #     fillcolor=(0, 0, 0),
        # )
        # if mask is not None:
        #     mask = mask.rotate(
        #         angle,
        #         resample=Image.NEAREST,
        #         translate=(tx, ty),
        #         scale=scale,
        #         fillcolor=0,
        #     )
        return img, mask


class JointColorJitter:
    """Color jitter only for image, mask unchanged."""
    def __init__(
        self,
        p: float = 0.5,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.15,
    ):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def _factor(self, strength: float) -> float:
        return float(np.random.uniform(max(0.0, 1 - strength), 1 + strength))

    def __call__(self, img: Image.Image, mask: Optional[Image.Image]):
        if np.random.rand() >= self.p:
            return img, mask

        if self.brightness > 0:
            img = ImageEnhance.Brightness(img).enhance(self._factor(self.brightness))
        if self.contrast > 0:
            img = ImageEnhance.Contrast(img).enhance(self._factor(self.contrast))
        if self.saturation > 0:
            img = ImageEnhance.Color(img).enhance(self._factor(self.saturation))
        return img, mask


class ToTensorNormalize:
    """
    Convert:
      - image: PIL RGB -> float tensor (C,H,W) in [0,1]
      - mask : PIL L   -> long tensor (H,W) with class ids
    """
    def __call__(self, img: Image.Image, mask: Optional[Image.Image]):
        img_np = np.array(img).astype(np.float32) / 255.0  # (H,W,3)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # (3,H,W)

        if mask is None:
            return img_t, None

        mask_np = np.array(mask)  # (H,W), uint8 class ids
        mask_t = torch.from_numpy(mask_np).long()
        return img_t, mask_t


class ComposeJoint:
    """Compose joint transforms that take (img, mask) and return (img, mask)."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


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
                f for f in os.listdir(paths.images_dir)
                if f.lower().endswith((".jpg", ".png"))
            ])
        else:
            self.files = file_list
        
        if len(self.files) == 0:
            raise RuntimeError(f"No image found in {paths.images_dir}")

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index: int):
        fname = self.files[index]
        img_path = os.path.join(self.paths.images_dir, fname)
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