# ──────────────────────────────────────────────────────────────────────────────
# custom_layers.py  (guárdalo en el mismo directorio que tu script de entrenamiento)
# ──────────────────────────────────────────────────────────────────────────────
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from ultralytics.nn.modules import Conv  # acceso al registro interno
from ultralytics.nn.modules import AConv, SPPF  # AConv = conv normal de YOLO

# ------------------------------------------------------------------ #
# Global Attention Mechanism (GAM)  ─ versión resumida del paper
# ------------------------------------------------------------------ #
class GAM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Attention canal
        attn_c = torch.sigmoid(self.mlp(self.avg(x)))
        x = x * attn_c
        # Attention espacial
        attn_s = torch.sigmoid(self.spatial(x))
        return x * attn_s

# ------------------------------------------------------------------ #
# SimSPPF  (igual que SPPF pero con ReLU en vez de SiLU)             #
# ------------------------------------------------------------------ #
class SimSPPF(SPPF):
    def __init__(self, c1, c2, k=5):
        super().__init__(c1, c2, k)
        # Reemplazamos la activación por ReLU en self.cv1 y self.cv2
        self.cv1.act = nn.ReLU(inplace=True)
        self.cv2.act = nn.ReLU(inplace=True)

# ------------------------------------------------------------------ #
# AKConv  (envoltorio simplificado usando deform-conv de tu proyecto)#
# ------------------------------------------------------------------ #
from your_utils.deform_conv import DeformConv2d  # ya la tenías importada

class AKConv(nn.Module):
    """Alterable Kernel Convolution (versión ligera)
       Se comporta como Conv 1×1 + DeformConv2d 3×3."""
    def __init__(self, c1, c2, k=3, stride=1, bias=False):
        super().__init__()
        self.pw   = Conv(c1, c2, k=1, act=False)  # point-wise
        self.dcn  = DeformConv2d(c2, c2, kernel_size=k, stride=stride,
                                 padding=k//2, bias=bias)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.pw(x)
        x = self.dcn(x)
        return self.act(x)

# ------------------------------------------------------------------ #
# Registro para que Ultralytics pueda leerlos desde YAML
# ------------------------------------------------------------------ #
import ultralytics.nn.modules as _ultra_mod
_ultra_mod.GAM     = GAM
_ultra_mod.SimSPPF = SimSPPF
_ultra_mod.AKConv  = AKConv
