import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(img, flow):
    """Warp an image using optical flow.

    Args:
        img: [B, C, H, W] image to warp
        flow: [B, 2, H, W] optical flow (dx, dy)
    Returns:
        [B, C, H, W] warped image
    """
    B, _, H, W = flow.shape
    # Create sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=flow.dtype),
        torch.arange(W, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Add flow to grid and normalize to [-1, 1]
    x = grid_x + flow[:, 0]
    y = grid_y + flow[:, 1]
    x = 2.0 * x / (W - 1) - 1.0
    y = 2.0 * y / (H - 1) - 1.0

    grid = torch.stack([x, y], dim=-1)  # [B, H, W, 2]
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=True)


class SmallFlowNet(nn.Module):
    """Lightweight optical flow estimation network.

    Simple encoder-decoder that estimates flow between two frames.
    """

    def __init__(self, n_channels=3):
        super().__init__()
        # Encoder: takes concatenated pair of frames
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels * 2, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, padding=1),  # 2 channels for (dx, dy)
        )

    def forward(self, frame1, frame2):
        """Estimate flow from frame1 to frame2.

        Args:
            frame1: [B, C, H, W]
            frame2: [B, C, H, W]
        Returns:
            flow: [B, 2, H, W]
        """
        return self.encoder(torch.cat([frame1, frame2], dim=1))


class FlowWarpSR(nn.Module):
    """Flow-based multi-frame super-resolution.

    Estimates optical flow to warp neighboring frames to align with the
    center frame, then fuses aligned frames before upscaling.

    Args:
        scale: upscaling factor (default 2)
        n_channels: input/output channels (default 3)
        n_neighbors: number of neighbors on each side (default 1, total 3 frames)
        n_features: feature channels (default 64)
    """

    def __init__(self, scale=2, n_channels=3, n_neighbors=1, n_features=64):
        super().__init__()
        self.n_neighbors = n_neighbors
        n_frames = 2 * n_neighbors + 1  # e.g., 3 for n_neighbors=1

        # Flow estimation network
        self.flow_net = SmallFlowNet(n_channels)

        # Fusion network: takes all aligned frames concatenated
        self.fusion = nn.Sequential(
            nn.Conv2d(n_channels * n_frames, n_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Upscale
        self.upscale = nn.Sequential(
            nn.Conv2d(n_features, n_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features // 2, n_channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, frames):
        """
        Args:
            frames: [B, N, C, H, W] sequence of LR frames (N = 2*n_neighbors + 1)
                    OR [B, N*C, H, W] channel-concatenated
        Returns:
            sr: [B, C, H*scale, W*scale] super-resolved center frame
        """
        if frames.dim() == 4:
            B, NC, H, W = frames.shape
            C = 3
            N = NC // C
            frames = frames.reshape(B, N, C, H, W)

        B, N, C, H, W = frames.shape
        center_idx = N // 2
        center = frames[:, center_idx]  # [B, C, H, W]

        # Warp all frames to align with center
        aligned = []
        for i in range(N):
            if i == center_idx:
                aligned.append(center)
            else:
                flow = self.flow_net(center, frames[:, i])
                warped = warp(frames[:, i], flow)
                aligned.append(warped)

        # Fuse aligned frames
        fused = torch.cat(aligned, dim=1)  # [B, N*C, H, W]
        features = self.fusion(fused)
        sr = self.upscale(features)
        return sr
