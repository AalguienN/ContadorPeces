from ultralytics import YOLO
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Module definitions ---
class ResidualBlock(nn.Module):
    """
    A basic Residual Block: Conv -> BN -> ReLU -> Conv -> BN + identity skip
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class DSAM(nn.Module):
    """
    Dynamic Selection Aggregation Module: aggregates multi-scale features via learned weights
    """
    def __init__(self, in_channels_list, reduction=16):
        super().__init__()
        total_c = sum(in_channels_list)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_c, total_c // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_c // reduction, len(in_channels_list), 1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, features):
        # features already share the same spatial size
        concat  = torch.cat(features, dim=1)
        weights = self.fc(concat)
        return sum(weights[:, i:i+1] * f for i, f in enumerate(features))


class CAFS(nn.Module):
    """
    Context-Aware Feature Selection: spatial masks for context-driven feature emphasis
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels // 2, 1, bias=False)
        self.bn   = nn.BatchNorm2d(channels // 2)
        self.attn = nn.Sequential(
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f    = F.relu(self.bn(self.conv(x)))
        mask = self.attn(f)
        return x * mask


# --- Load YOLOv8n and apply custom modules ---
model = YOLO("yolov8n.pt")  # pretrained nano model

# 1) Replace first 7 Conv2d in the backbone with ResidualBlocks
for idx, layer in enumerate(model.model.model[:7]):
    if isinstance(layer, nn.Conv2d):
        model.model.model[idx] = ResidualBlock(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            stride=layer.stride[0]
        )

# 2) Locate the Detect head and derive neck channels from its cv2 ModuleList
detect_idx   = next(
    i for i, l in enumerate(model.model.model)
    if l.__class__.__name__ == "Detect"
)
detect_layer = model.model.model[detect_idx]

# For each branch in cv2, grab the in_channels of its first Conv2d
in_channels_list = [
    seq[0].conv.in_channels
    for seq in detect_layer.cv2
]

# Insert DSAM right after the neck (position 13)
model.model.model.insert(13, DSAM(in_channels_list=in_channels_list))

# 3) Insert CAFS just before the Detect head (using the first channel)
head_out_ch = in_channels_list[0]  # e.g. 64 for yolov8n
# since DSAM was inserted at 13, Detect shifted by +1
model.model.model.insert(detect_idx + 1, CAFS(head_out_ch))


# --- Fine-tuning setup ---
# Freeze backbone layers (first 7)
for layer in model.model.model[:7]:
    for p in layer.parameters():
        p.requires_grad = False

results_stage1 = model.train(
    data="config.yaml",
    epochs=30,
    batch=16,
    lr0=1e-4,
    freeze=7
)

# --- Stage 2: unfreeze all, lower lr, fine-tune end-to-end ---
for layer in model.model.model:
    for p in layer.parameters():
        p.requires_grad = True

results_stage2 = model.train(
    data="config.yaml",
    epochs=15,
    batch=16,
    lr0=1e-5,
    freeze=0
)
