import torch.nn as nn

from .baseline import ESPCN


class TemporalESPCN(nn.Module):
    """ESPCN trained with temporal consistency loss.

    Same architecture as the baseline ESPCN, but designed to be trained
    on frame pairs/sequences with an additional temporal consistency loss term.
    The training loop handles the temporal loss — this module just wraps ESPCN.

    Args:
        scale: upscaling factor (default 2)
        n_channels: number of input/output channels (default 3)
        n_features: number of intermediate feature channels (default 64)
    """

    def __init__(self, scale=2, n_channels=3, n_features=64):
        super().__init__()
        self.net = ESPCN(scale, n_channels, n_features)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] single LR frame
        Returns:
            [B, C, H*scale, W*scale] super-resolved frame
        """
        return self.net(x)

    def forward_sequence(self, lr_frames):
        """Process a sequence of LR frames independently.

        Args:
            lr_frames: [B, N, C, H, W] sequence of LR frames
        Returns:
            [B, N, C, H*scale, W*scale] sequence of SR frames
        """
        B, N, C, H, W = lr_frames.shape
        # Reshape to process all frames at once
        x = lr_frames.reshape(B * N, C, H, W)
        sr = self.net(x)
        _, C_out, H_out, W_out = sr.shape
        return sr.reshape(B, N, C_out, H_out, W_out)
