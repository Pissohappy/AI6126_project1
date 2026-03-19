import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
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
            nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_dsconv: bool = False):
        super().__init__()
        conv_layer = DSConvBNReLU if use_dsconv else ConvBNReLU
        self.net = nn.Sequential(conv_layer(in_ch, out_ch), conv_layer(out_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_dsconv: bool = False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, use_dsconv=use_dsconv)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_dsconv: bool = False):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, use_dsconv=use_dsconv)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


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
        return x * self.fc(self.pool(x))


class PPMLite(nn.Module):
    """Tiny pyramid pooling context module."""

    def __init__(self, channels: int):
        super().__init__()
        inter = max(16, channels // 4)
        self.scales = (1, 2, 4)
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(channels, inter, kernel_size=1, bias=False),
                    nn.BatchNorm2d(inter),
                    nn.ReLU(inplace=True),
                )
                for s in self.scales
            ]
        )
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
            feats.append(F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(feats, dim=1))


class MiniUNet(nn.Module):
    """Lightweight U-Net for segmentation, from scratch."""

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

        self.up2 = Up(c4, c3, c3, use_dsconv=use_dsconv)
        self.up1 = Up(c3, c2, c2, use_dsconv=use_dsconv)
        self.up0 = Up(c2, c1, c1, use_dsconv=use_dsconv)

        self.attn_shallow = SEAttention(c2) if use_attention else nn.Identity()
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(c2, num_classes, kernel_size=1) if use_aux_head else None

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x3 = self.attn_deep(self.context(x3))

        y2 = self.up2(x3, x2)
        y1 = self.attn_shallow(self.up1(y2, x1))
        y0 = self.up0(y1, x0)

        logits = self.head(y0)
        if not self.use_aux_head:
            return logits

        aux = self.aux_head(y1)
        aux = F.interpolate(aux, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        return logits, aux


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int = 4):
        super().__init__()
        assert stride in (1, 2)
        hidden = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, hidden, k=1, p=0),
            ConvBNReLU(hidden, hidden, k=3, s=stride, p=1, groups=hidden),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class TinyUNetMobileNetV2(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 0.5):
        super().__init__()
        c1 = max(16, int(24 * width_mult))
        c2 = max(24, int(48 * width_mult))
        c3 = max(32, int(96 * width_mult))
        c4 = max(48, int(160 * width_mult))

        self.stem = ConvBNReLU(3, c1, k=3, s=2, p=1)
        self.enc1 = nn.Sequential(InvertedResidual(c1, c2, stride=2), InvertedResidual(c2, c2, stride=1))
        self.enc2 = nn.Sequential(InvertedResidual(c2, c3, stride=2), InvertedResidual(c3, c3, stride=1))
        self.enc3 = nn.Sequential(InvertedResidual(c3, c4, stride=2), InvertedResidual(c4, c4, stride=1))

        self.up2 = Up(c4, c3, c3, use_dsconv=True)
        self.up1 = Up(c3, c2, c2, use_dsconv=True)
        self.up0 = Up(c2, c1, c1, use_dsconv=True)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        in_size = x.shape[-2:]
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        y2 = self.up2(x3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        logits = self.head(y0)
        return F.interpolate(logits, size=in_size, mode="bilinear", align_corners=False)


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ShuffleV2Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        assert stride in (1, 2)
        branch_ch = out_ch // 2
        self.stride = stride

        if stride == 2:
            self.branch1 = nn.Sequential(
                ConvBNReLU(in_ch, in_ch, k=3, s=2, p=1, groups=in_ch),
                ConvBNReLU(in_ch, branch_ch, k=1, p=0),
            )
            in_branch2 = in_ch
        else:
            self.branch1 = nn.Identity()
            in_branch2 = in_ch // 2

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_branch2, branch_ch, k=1, p=0),
            ConvBNReLU(branch_ch, branch_ch, k=3, s=stride, p=1, groups=branch_ch),
            ConvBNReLU(branch_ch, branch_ch, k=1, p=0),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, groups=2)


class TinyUNetShuffleNetV2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        c1, c2, c3, c4 = 24, 48, 96, 192
        self.stem = ConvBNReLU(3, c1, k=3, s=2, p=1)
        self.enc1 = nn.Sequential(ShuffleV2Block(c1, c2, stride=2), ShuffleV2Block(c2, c2, stride=1))
        self.enc2 = nn.Sequential(ShuffleV2Block(c2, c3, stride=2), ShuffleV2Block(c3, c3, stride=1))
        self.enc3 = nn.Sequential(ShuffleV2Block(c3, c4, stride=2), ShuffleV2Block(c4, c4, stride=1))

        self.up2 = Up(c4, c3, c3, use_dsconv=True)
        self.up1 = Up(c3, c2, c2, use_dsconv=True)
        self.up0 = Up(c2, c1, c1, use_dsconv=True)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        in_size = x.shape[-2:]
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        y2 = self.up2(x3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        logits = self.head(y0)
        return F.interpolate(logits, size=in_size, mode="bilinear", align_corners=False)


class BiSeNetV2Tiny(nn.Module):
    """Compact BiSeNetV2-style network with reduced semantic channels."""

    def __init__(self, num_classes: int, semantic_scale: float = 0.75):
        super().__init__()
        sem_c = max(16, int(32 * semantic_scale))
        detail_c = 24

        # Detail branch (keeps stronger spatial cues)
        self.detail = nn.Sequential(
            ConvBNReLU(3, detail_c, k=3, s=2, p=1),
            DSConvBNReLU(detail_c, detail_c),
            ConvBNReLU(detail_c, detail_c * 2, k=3, s=2, p=1),
            DSConvBNReLU(detail_c * 2, detail_c * 2),
            ConvBNReLU(detail_c * 2, detail_c * 2, k=3, s=2, p=1),
        )

        # Semantic branch (aggressively compact)
        self.semantic = nn.Sequential(
            ConvBNReLU(3, sem_c, k=3, s=2, p=1),
            InvertedResidual(sem_c, sem_c, stride=2, expand_ratio=2),
            InvertedResidual(sem_c, sem_c * 2, stride=2, expand_ratio=2),
            InvertedResidual(sem_c * 2, sem_c * 2, stride=1, expand_ratio=2),
            PPMLite(sem_c * 2),
        )

        fuse_in = detail_c * 2 + sem_c * 2
        self.fuse = nn.Sequential(
            ConvBNReLU(fuse_in, 64, k=1, p=0),
            DSConvBNReLU(64, 64),
        )
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        in_size = x.shape[-2:]
        d = self.detail(x)
        s = self.semantic(x)
        s = F.interpolate(s, size=d.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([d, s], dim=1))
        logits = self.head(y)
        return F.interpolate(logits, size=in_size, mode="bilinear", align_corners=False)


def build_model(arch: str, num_classes: int, **kwargs) -> nn.Module:
    pretrained = kwargs.pop("pretrained", False)
    if pretrained:
        raise ValueError("All supported models are train-from-scratch only; set pretrained=False.")

    arch = arch.lower()
    if arch == "mini_unet":
        return MiniUNet(num_classes=num_classes, **kwargs)

    if arch == "unet_mobilenetv2_tiny":
        width_mult = float(kwargs.pop("width_mult", 0.5))
        if width_mult not in (0.5, 0.75):
            raise ValueError("unet_mobilenetv2_tiny supports width_mult in {0.5, 0.75}.")
        return TinyUNetMobileNetV2(num_classes=num_classes, width_mult=width_mult)

    if arch == "unet_shufflenetv2_tiny":
        return TinyUNetShuffleNetV2(num_classes=num_classes)

    if arch == "bisenetv2_tiny":
        semantic_scale = float(kwargs.pop("semantic_scale", 0.75))
        return BiSeNetV2Tiny(num_classes=num_classes, semantic_scale=semantic_scale)

    raise ValueError(f"Unsupported architecture: {arch}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
