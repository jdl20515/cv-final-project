import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatial-temporal processing."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        # Gates: input, forget, output, cell candidate
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, state):
        """
        Args:
            x: [B, C_in, H, W]
            state: tuple (h, c) each [B, C_hidden, H, W], or None
        Returns:
            (h_new, c_new)
        """
        if state is None:
            B, _, H, W = x.shape
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class RecurrentSR(nn.Module):
    """Recurrent super-resolution with ConvLSTM.

    Processes frames sequentially, maintaining hidden state between frames
    to capture temporal dependencies.

    Args:
        scale: upscaling factor (default 2)
        n_channels: input/output channels (default 3)
        n_features: feature channels (default 64)
        hidden_channels: ConvLSTM hidden channels (default 32)
    """

    def __init__(self, scale=2, n_channels=3, n_features=64, hidden_channels=32):
        super().__init__()
        self.scale = scale

        # Feature extraction from LR frame
        self.feature_extract = nn.Sequential(
            nn.Conv2d(n_channels, n_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ConvLSTM for temporal modeling
        self.convlstm = ConvLSTMCell(n_features, hidden_channels)

        # Reconstruction from hidden state
        self.reconstruct = nn.Sequential(
            nn.Conv2d(hidden_channels, n_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x, state=None):
        """Process a single frame with recurrent state.

        Args:
            x: [B, C, H, W] single LR frame
            state: optional (h, c) from previous frame
        Returns:
            sr: [B, C, H*scale, W*scale] super-resolved frame
            state: (h, c) hidden state for next frame
        """
        features = self.feature_extract(x)
        state = self.convlstm(features, state)
        h, _ = state
        sr = self.reconstruct(h)
        return sr, state

    def forward_sequence(self, lr_frames):
        """Process a full sequence of LR frames.

        Args:
            lr_frames: [B, N, C, H, W] sequence of LR frames
        Returns:
            sr_frames: [B, N, C, H*scale, W*scale] sequence of SR frames
        """
        B, N, C, H, W = lr_frames.shape
        sr_list = []
        state = None
        for t in range(N):
            sr, state = self.forward(lr_frames[:, t], state)
            sr_list.append(sr)
        return torch.stack(sr_list, dim=1)
