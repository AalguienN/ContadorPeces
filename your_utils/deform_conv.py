import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d as TorchDeformConv2d

class DeformConv2d(nn.Module):
    """
    Deformable Convolution 2D wrapper: predicts offsets and applies
    torchvision.ops.DeformConv2d.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True):
        super().__init__()
        # Offset prediction convolution
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        # Deformable convolution
        self.deform_conv = TorchDeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        offsets = self.offset_conv(x)
        return self.deform_conv(x, offsets)
