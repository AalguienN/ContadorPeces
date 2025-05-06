import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck

class CSP1(nn.Module):
    def __init__(self, channels, n=1, shortcut=True):
        super().__init__()
        self.cv1 = Conv(channels, channels, 1, 1)
        self.cv2 = Conv(channels*2, channels, 1, 1)
        self.m   = nn.Sequential(*[Bottleneck(channels, shortcut=shortcut) for _ in range(n)])
    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = x
        return self.cv2(torch.cat((y1, y2), dim=1))
