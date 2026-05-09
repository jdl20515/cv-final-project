import torch
import torch.nn as nn


class MultiFrameSR(nn.Module):
    """Multi-frame sliding window super-resolution.

    Takes 3 consecutive LR frames concatenated along the channel dimension,
    and outputs the super-resolved center frame. Early conv layers jointly
    process all 3 frames to capture temporal information.

    Args:
        scale: upscaling factor (default 2)
        n_channels: channels per frame (default 3 for RGB)
        n_frames: number of input frames (default 3)
        n_features: number of intermediate feature channels (default 64)
    """

    def __init__(self, scale=2, n_channels=3, n_frames=3, n_features=64):
        super().__init__()
        self.n_frames = n_frames
        in_channels = n_channels * n_frames  # 9 for 3 RGB frames

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, n_features, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features // 2, n_channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, frames):
        """
        Args:
            frames: [B, n_frames*C, H, W] channel-concatenated LR frames
                    OR [B, n_frames, C, H, W] sequence of LR frames
        Returns:
            [B, C, H*scale, W*scale] super-resolved center frame
        """
        if frames.dim() == 5:
            B, N, C, H, W = frames.shape
            frames = frames.reshape(B, N * C, H, W)
        return self.net(frames)
