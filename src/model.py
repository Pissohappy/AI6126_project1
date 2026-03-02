# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        """
        in_ch: channels of the feature being upsampled (from deeper layer)
        skip_ch: channels from skip connection
        out_ch: output channels after fusion conv
        """
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class MiniUNet(nn.Module):
    """
    Lightweight U-Net for 512x512 segmentation, from scratch.
    """
    def __init__(self, num_classes: int = 19, base_ch: int = 32):
        super().__init__()
        c1 = base_ch          # 32
        c2 = base_ch * 2      # 64
        c3 = base_ch * 4      # 128
        c4 = base_ch * 6      # 192 (a bit smaller than 256 to keep params down)

        self.stem = DoubleConv(3, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        self.up2 = Up(c4, c3, c3)   # 192 + 128 -> 128
        self.up1 = Up(c3, c2, c2)   # 128 + 64  -> 64
        self.up0 = Up(c2, c1, c1)   # 64  + 32  -> 32

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)     # 32
        x1 = self.down1(x0)   # 64
        x2 = self.down2(x1)   # 128
        x3 = self.down3(x2)   # 192

        y2 = self.up2(x3, x2) # 128
        y1 = self.up1(y2, x1) # 64
        y0 = self.up0(y1, x0) # 32

        logits = self.head(y0)  # (B, num_classes, H, W)
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)