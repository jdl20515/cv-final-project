import torch
import torch.nn as nn
import torchvision.models as models


class PixelLoss(nn.Module):
    """Pixel-wise reconstruction loss (L1 or MSE)."""

    def __init__(self, loss_type="l1"):
        super().__init__()
        if loss_type == "l1":
            self.loss = nn.L1Loss()
        elif loss_type == "mse":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown pixel loss type: {loss_type}")

    def forward(self, sr, hr):
        return self.loss(sr, hr)


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss using conv3_3 features."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # conv3_3 is at index 15 in vgg16 features
        self.feature_extractor = nn.Sequential(*list(vgg.features[:16]))
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, sr, hr):
        sr_features = self.feature_extractor(self._normalize(sr))
        hr_features = self.feature_extractor(self._normalize(hr))
        return nn.functional.l1_loss(sr_features, hr_features)


class TemporalConsistencyLoss(nn.Module):
    """Penalizes temporal inconsistency between consecutive SR frames.

    Loss = ||  (SR_t - SR_{t-1}) - (HR_t - HR_{t-1})  ||_1
    """

    def forward(self, sr_frames, hr_frames):
        """
        Args:
            sr_frames: [B, N, C, H, W] or list of [B, C, H, W] tensors
            hr_frames: [B, N, C, H, W] or list of [B, C, H, W] tensors
        """
        if isinstance(sr_frames, (list, tuple)):
            sr_frames = torch.stack(sr_frames, dim=1)
        if isinstance(hr_frames, (list, tuple)):
            hr_frames = torch.stack(hr_frames, dim=1)

        sr_diff = sr_frames[:, 1:] - sr_frames[:, :-1]  # [B, N-1, C, H, W]
        hr_diff = hr_frames[:, 1:] - hr_frames[:, :-1]
        return nn.functional.l1_loss(sr_diff, hr_diff)


class CombinedLoss(nn.Module):
    """Weighted combination of losses.

    Args:
        pixel_weight: weight for pixel loss
        perceptual_weight: weight for perceptual loss (0 to disable)
        temporal_weight: weight for temporal consistency loss (0 to disable)
        pixel_type: 'l1' or 'mse'
    """

    def __init__(self, pixel_weight=1.0, perceptual_weight=0.0,
                 temporal_weight=0.0, pixel_type="l1"):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight

        self.pixel_loss = PixelLoss(pixel_type)
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None
        self.temporal_loss = TemporalConsistencyLoss() if temporal_weight > 0 else None

    def forward(self, sr, hr, sr_frames=None, hr_frames=None):
        """
        Args:
            sr: [B, C, H, W] single frame SR output
            hr: [B, C, H, W] single frame HR target
            sr_frames: optional sequence for temporal loss
            hr_frames: optional sequence for temporal loss
        """
        loss = self.pixel_weight * self.pixel_loss(sr, hr)

        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            loss = loss + self.perceptual_weight * self.perceptual_loss(sr, hr)

        if (self.temporal_loss is not None and self.temporal_weight > 0
                and sr_frames is not None and hr_frames is not None):
            loss = loss + self.temporal_weight * self.temporal_loss(sr_frames, hr_frames)

        return loss
