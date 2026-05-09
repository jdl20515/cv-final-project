import torch
import torch.nn as nn


class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network (Shi et al. 2016).

    Lightweight single-image super-resolution using sub-pixel convolution.
    Processes each frame independently — no temporal awareness.

    Args:
        scale: upscaling factor (default 2)
        n_channels: number of input/output channels (default 3 for RGB)
        n_features: number of intermediate feature channels (default 64)
    """

    def __init__(self, scale=2, n_channels=3, n_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, n_features, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features // 2, n_channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] low-resolution input
        Returns:
            [B, C, H*scale, W*scale] super-resolved output
        """
        return self.net(x)
