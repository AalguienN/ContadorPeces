import torch
import torch.nn as nn

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Reference: Woo et al., ECCV 2018.
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size,
                                      padding=(kernel_size-1)//2,
                                      bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        x = x * self.sigmoid_channel(avg_out + max_out)
        # Spatial attention
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sigmoid_spatial(
            self.conv_spatial(torch.cat([avg_sp, max_sp], dim=1))
        )
        return x
    
class DSAM(nn.Module):
    """
    Dynamic Spatial Attention Module (DSAM) from AquaYOLO JMSE paper:
    - Aligns adjacent-level features via upsampling + 1x1 conv + BN + ReLU
    - Fuses by element-wise summation with the target-level feature
    """
    def __init__(self, channels):
        super(DSAM, self).__init__()
        # FAU: upsample + conv1x1 to match channel count + BN + ReLU
        self.align = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_low, x_high):
        """
        x_low: lower-level feature (target resolution)
        x_high: higher-level feature (to be aligned)
        returns: fused feature at x_low resolution
        """
        # Align higher-level feature to low-level
        aligned = self.align(x_high)
        # Fuse with low-level feature
        out = x_low + aligned
        return out


class CAFS(nn.Module):
    """
    Context-Aware Feature Selection (CAFS) from AquaYOLO JMSE paper:
    - Concatenates two adjacent features
    - Learns spatial weights via softmax to select between them
    - Computes fused feature + attention-based enhancement
    """
    def __init__(self, channels):
        super(CAFS, self).__init__()
        # Extraction: reduce concat of two features back to 'channels'
        self.extract = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # Produce two spatial weight maps (Wa, Wb)
        self.weight_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        # Fusion enhancement
        self.fuse_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_a, x_b):
        """
        x_a, x_b: two aligned feature maps with same shape [B, C, H, W]
        returns: enhanced fused feature map
        """
        # Concatenate along channel dim
        x_cat = torch.cat([x_a, x_b], dim=1)
        x = self.extract(x_cat)
        # Compute spatial weights and normalize
        w = self.weight_conv(x)  # [B, 2, H, W]
        w = self.softmax(w)
        wa, wb = w[:, 0:1, ...], w[:, 1:2, ...]
        # Select features
        sel_a = x_a * wa
        sel_b = x_b * wb
        fused = sel_a + sel_b
        # Enhancement via learned fusion
        wf = self.sigmoid(self.fuse_conv(fused))
        out = fused + fused * wf
        return out
