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

class DSConvBNReLU(nn.Module):
    """Depthwise-separable conv for efficiency."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch, bias=False),  # depthwise
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_dsconv: bool = False):
        super().__init__()
        conv_layer = DSConvBNReLU if use_dsconv else ConvBNReLU
        self.net = nn.Sequential(
            # ConvBNReLU(in_ch, out_ch),
            # ConvBNReLU(out_ch, out_ch),
            conv_layer(in_ch, out_ch),
            conv_layer(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_dsconv: bool = False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, use_dsconv=use_dsconv)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_dsconv: bool = False):
        """
        in_ch: channels of the feature being upsampled (from deeper layer)
        skip_ch: channels from skip connection
        out_ch: output channels after fusion conv
        """
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, use_dsconv=use_dsconv)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class PPMLite(nn.Module):
    """Tiny pyramid pooling context module."""
    def __init__(self, channels: int):
        super().__init__()
        inter = max(16, channels // 4)
        self.scales = (1, 2, 4)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(channels, inter, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter),
                nn.ReLU(inplace=True),
            )
            for s in self.scales
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + inter * len(self.scales), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [x]
        for p in self.proj:
            y = p(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(y)
        return self.fuse(torch.cat(feats, dim=1))


class MiniUNet(nn.Module):
    """
    Lightweight U-Net for 512x512 segmentation, from scratch.
    """
    def __init__(
            self,
            num_classes: int = 19,
            base_ch: int = 32,
            use_dsconv: bool = False,
            use_attention: bool = False,
            use_context: bool = False,
            use_aux_head: bool = False,
        ):
        super().__init__()
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 6

        self.use_aux_head = use_aux_head

        self.stem = DoubleConv(3, c1, use_dsconv=False)
        self.down1 = Down(c1, c2, use_dsconv=use_dsconv)
        self.down2 = Down(c2, c3, use_dsconv=use_dsconv)
        self.down3 = Down(c3, c4, use_dsconv=use_dsconv)


        self.context = PPMLite(c4) if use_context else nn.Identity()
        self.attn_deep = SEAttention(c4) if use_attention else nn.Identity()

        self.up2 = Up(c4, c3, c3, use_dsconv=use_dsconv)   # 192 + 128 -> 128
        self.up1 = Up(c3, c2, c2, use_dsconv=use_dsconv)   # 128 + 64  -> 64
        self.up0 = Up(c2, c1, c1, use_dsconv=use_dsconv)   # 64  + 32  -> 32

        self.attn_shallow = SEAttention(c2) if use_attention else nn.Identity()
        
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

        self.aux_head = nn.Conv2d(c2, num_classes, kernel_size=1) if use_aux_head else None

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x3 = self.context(x3)
        x3 = self.attn_deep(x3)

        y2 = self.up2(x3, x2)
        y1 = self.up1(y2, x1)
        y1 = self.attn_shallow(y1)
        y0 = self.up0(y1, x0)

        logits = self.head(y0)
        if not self.use_aux_head:
            return logits
        
        aux = self.aux_head(y1)
        aux = F.interpolate(aux, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        return logits, aux


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)