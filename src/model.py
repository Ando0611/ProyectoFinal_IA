"""U-Net multi-tarea implementado desde cero en PyTorch.

Arquitectura:

    Encoder (4 niveles)
        DoubleConv -> MaxPool
    Bottleneck
        DoubleConv profundo
    Decoder (4 niveles)
        UpSample + skip concat -> DoubleConv
    Cabeza de segmentación
        Conv 1x1 -> [B, NUM_SEG_CLASSES, H, W]
    Cabeza de clasificación (sobre bottleneck)
        GlobalAvgPool -> MLP -> [B, NUM_ATTRS]  (logits, sin sigmoid)

Diseñado para ser entrenable desde inicialización aleatoria
(He init) en una M4 con 16 GB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_ATTRS, NUM_SEG_CLASSES


class DoubleConv(nn.Module):
    """Bloque típico de U-Net: dos Conv3x3 + BN + ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Reducción espacial 2x: MaxPool + DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Aumento espacial 2x con bilinear upsample + concat con skip + DoubleConv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Por si la diferencia espacial no es exacta (paddings), recortamos
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetMultiTask(nn.Module):
    """U-Net + cabeza global de clasificación de atributos."""

    def __init__(
        self,
        in_channels=3,
        num_seg_classes=NUM_SEG_CLASSES,
        num_attrs=NUM_ATTRS,
        base_channels=32,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        # Decoder
        self.up1 = Up(c5, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)

        # Cabeza de segmentación: 1x1 a NUM_SEG_CLASSES
        self.seg_head = nn.Conv2d(c1, num_seg_classes, kernel_size=1)

        # Cabeza de clasificación global (sobre el bottleneck)
        self.attr_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_attrs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)             # bottleneck

        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)

        seg_logits = self.seg_head(d)               # [B, NUM_SEG_CLASSES, H, W]
        attr_logits = self.attr_head(x5)            # [B, NUM_ATTRS]

        return seg_logits, attr_logits


def get_device():
    """Selecciona el mejor device disponible: cuda > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
