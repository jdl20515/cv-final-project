import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class VideoSRDataset(Dataset):
    """Vimeo-90K septuplet dataset for video super-resolution.

    Each sample is a sequence of consecutive frames. Returns downscaled LR
    frames and original HR frames.

    Args:
        root: Path to vimeo_septuplet directory
        split: 'train' or 'test'
        n_frames: Number of consecutive frames to return (default 7)
        scale: Downscale factor (default 2)
        patch_size: Random crop size for HR patches during training (default 64)
    """

    def __init__(self, root, split="train", n_frames=7, scale=2, patch_size=64):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.n_frames = n_frames
        self.scale = scale
        self.patch_size = patch_size
        self.sequences_dir = self.root / "sequences"

        # Load sequence list
        list_file = f"sep_{'trainlist' if split == 'train' else 'testlist'}.txt"
        list_path = self.root / list_file
        if not list_path.exists():
            raise FileNotFoundError(
                f"Split list not found: {list_path}. "
                "Run scripts/download_data.sh first."
            )

        with open(list_path, "r") as f:
            self.sequences = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.sequences)

    def _load_frames(self, seq_path):
        """Load all 7 frames from a sequence directory."""
        frames = []
        for i in range(1, 8):  # im1.png through im7.png
            img_path = seq_path / f"im{i}.png"
            img = Image.open(img_path).convert("RGB")
            frames.append(img)
        return frames

    def _random_crop(self, frames, patch_size):
        """Apply the same random crop to all frames."""
        w, h = frames[0].size
        if w < patch_size or h < patch_size:
            # If image is smaller than patch, resize up
            scale = max(patch_size / w, patch_size / h) + 0.01
            new_w, new_h = int(w * scale), int(h * scale)
            frames = [f.resize((new_w, new_h), Image.BICUBIC) for f in frames]
            w, h = new_w, new_h

        left = random.randint(0, w - patch_size)
        top = random.randint(0, h - patch_size)
        return [f.crop((left, top, left + patch_size, top + patch_size)) for f in frames]

    def _augment(self, frames):
        """Random horizontal flip and temporal reverse."""
        if random.random() < 0.5:
            frames = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in frames]
        if random.random() < 0.5:
            frames = frames[::-1]
        return frames

    def _downsample(self, frame_tensor):
        """Downsample a [C,H,W] tensor by the scale factor using bicubic."""
        _, h, w = frame_tensor.shape
        new_h, new_w = h // self.scale, w // self.scale
        lr = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        return lr.clamp(0, 1)

    def __getitem__(self, idx):
        seq_name = self.sequences[idx]
        seq_path = self.sequences_dir / seq_name

        frames = self._load_frames(seq_path)

        # Select n_frames consecutive frames from the 7 available
        if self.n_frames < 7:
            start = random.randint(0, 7 - self.n_frames)
            frames = frames[start:start + self.n_frames]

        # Crop and augment (training only)
        if self.split == "train":
            frames = self._random_crop(frames, self.patch_size)
            frames = self._augment(frames)

        to_tensor = transforms.ToTensor()  # Converts to [C,H,W] float 0-1
        hr_frames = torch.stack([to_tensor(f) for f in frames])  # [N,C,H,W]
        lr_frames = torch.stack([self._downsample(hf) for hf in hr_frames])  # [N,C,H/s,W/s]

        return {"lr_frames": lr_frames, "hr_frames": hr_frames}


def get_dataloader(root, split, n_frames=7, scale=2, patch_size=64,
                   batch_size=16, num_workers=4):
    """Create a DataLoader for the VideoSR dataset."""
    dataset = VideoSRDataset(root, split, n_frames, scale, patch_size)
    shuffle = split == "train"
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
