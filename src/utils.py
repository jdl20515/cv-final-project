import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def get_device():
    """Get the best available device (MPS for Apple Silicon, else CUDA, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, psnr, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "psnr": psnr,
    }, path)


def load_checkpoint(model, path, optimizer=None, device=None):
    """Load model checkpoint. Returns (epoch, psnr)."""
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("psnr", 0.0)


def tensor_to_numpy(tensor):
    """Convert [C,H,W] float tensor (0-1) to [H,W,C] uint8 numpy array."""
    img = tensor.detach().cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255).astype(np.uint8)


def frames_to_video(frames, output_path, fps=24):
    """Stitch list of [C,H,W] tensors into an mp4 video.

    Args:
        frames: list of [C,H,W] float tensors (0-1 range)
        output_path: path for the output .mp4 file
        fps: frames per second
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imgs = [tensor_to_numpy(f) for f in frames]
    h, w = imgs[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for img in imgs:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()


def visualize_comparison(lr_frame, sr_frame, hr_frame, save_path=None):
    """Side-by-side comparison of LR (upscaled), SR, and HR frames.

    Args:
        lr_frame, sr_frame, hr_frame: [C,H,W] float tensors
        save_path: optional path to save the figure
    """
    lr_np = tensor_to_numpy(lr_frame)
    sr_np = tensor_to_numpy(sr_frame)
    hr_np = tensor_to_numpy(hr_frame)

    # Upscale LR for visual comparison
    h, w = hr_np.shape[:2]
    lr_up = cv2.resize(lr_np, (w, h), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, [lr_up, sr_np, hr_np],
                               ["LR (bicubic)", "SR (model)", "HR (ground truth)"]):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
