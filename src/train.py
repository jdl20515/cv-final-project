"""Training script for video super-resolution models.

Usage:
    python src/train.py --config configs/baseline.yaml
"""

import argparse
import os
import sys
import time

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataloader
from src.losses import CombinedLoss
from src.models import build_model
from src.utils import count_parameters, get_device, save_checkpoint, load_checkpoint


def compute_psnr(sr, hr):
    """Compute PSNR between SR and HR tensors."""
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)


def train_one_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch. Returns average loss and PSNR."""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    n_batches = 0
    model_name = config["model"]["name"]

    for batch in tqdm(dataloader, desc="Train", leave=False):
        lr_frames = batch["lr_frames"].to(device)  # [B, N, C, H, W]
        hr_frames = batch["hr_frames"].to(device)

        optimizer.zero_grad()

        if model_name == "baseline":
            # Process center frame only
            center = lr_frames.shape[1] // 2
            lr = lr_frames[:, center]
            hr = hr_frames[:, center]
            sr = model(lr)
            loss = criterion(sr, hr)

        elif model_name == "temporal_loss":
            # Process all frames, compute temporal loss
            sr_all = model.forward_sequence(lr_frames)  # [B, N, C, H, W]
            center = lr_frames.shape[1] // 2
            sr = sr_all[:, center]
            hr = hr_frames[:, center]
            loss = criterion(sr, hr, sr_frames=sr_all, hr_frames=hr_frames)

        elif model_name == "multiframe":
            # Use 3 consecutive frames centered around the middle
            N = lr_frames.shape[1]
            center = N // 2
            start = max(0, center - 1)
            end = min(N, center + 2)
            lr_window = lr_frames[:, start:end]  # [B, 3, C, H, W]
            hr = hr_frames[:, center]
            sr = model(lr_window)
            loss = criterion(sr, hr)

        elif model_name == "recurrent":
            # Process full sequence
            sr_all = model.forward_sequence(lr_frames)  # [B, N, C, H, W]
            # Average loss over all frames
            B, N, C, H, W = sr_all.shape
            sr_flat = sr_all.reshape(B * N, C, H, W)
            hr_flat = hr_frames.reshape(B * N, C, H, W)
            loss = criterion(sr_flat, hr_flat, sr_frames=sr_all, hr_frames=hr_frames)
            sr = sr_all[:, N // 2]
            hr = hr_frames[:, N // 2]

        elif model_name == "flow_warp":
            # Use 3 consecutive frames
            N = lr_frames.shape[1]
            center = N // 2
            start = max(0, center - 1)
            end = min(N, center + 2)
            lr_window = lr_frames[:, start:end]
            hr = hr_frames[:, center]
            sr = model(lr_window)
            loss = criterion(sr, hr)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += compute_psnr(sr.detach(), hr).item()
        n_batches += 1

    return total_loss / n_batches, total_psnr / n_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    """Validate. Returns average loss and PSNR."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n_batches = 0
    model_name = config["model"]["name"]

    for batch in tqdm(dataloader, desc="Val", leave=False):
        lr_frames = batch["lr_frames"].to(device)
        hr_frames = batch["hr_frames"].to(device)

        if model_name == "baseline":
            center = lr_frames.shape[1] // 2
            lr = lr_frames[:, center]
            hr = hr_frames[:, center]
            sr = model(lr)
            loss = criterion(sr, hr)

        elif model_name == "temporal_loss":
            sr_all = model.forward_sequence(lr_frames)
            center = lr_frames.shape[1] // 2
            sr = sr_all[:, center]
            hr = hr_frames[:, center]
            loss = criterion(sr, hr, sr_frames=sr_all, hr_frames=hr_frames)

        elif model_name == "multiframe":
            N = lr_frames.shape[1]
            center = N // 2
            start = max(0, center - 1)
            end = min(N, center + 2)
            lr_window = lr_frames[:, start:end]
            hr = hr_frames[:, center]
            sr = model(lr_window)
            loss = criterion(sr, hr)

        elif model_name == "recurrent":
            sr_all = model.forward_sequence(lr_frames)
            B, N, C, H, W = sr_all.shape
            sr_flat = sr_all.reshape(B * N, C, H, W)
            hr_flat = hr_frames.reshape(B * N, C, H, W)
            loss = criterion(sr_flat, hr_flat, sr_frames=sr_all, hr_frames=hr_frames)
            sr = sr_all[:, N // 2]
            hr = hr_frames[:, N // 2]

        elif model_name == "flow_warp":
            N = lr_frames.shape[1]
            center = N // 2
            start = max(0, center - 1)
            end = min(N, center + 2)
            lr_window = lr_frames[:, start:end]
            hr = hr_frames[:, center]
            sr = model(lr_window)
            loss = criterion(sr, hr)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        total_loss += loss.item()
        total_psnr += compute_psnr(sr, hr).item()
        n_batches += 1

    return total_loss / n_batches, total_psnr / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train video SR model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    # Build model
    model = build_model(config).to(device)
    n_params = count_parameters(model)
    print(f"Model: {config['model']['name']} ({n_params:,} parameters)")

    # Data
    data_cfg = config["data"]
    train_loader = get_dataloader(
        root=data_cfg["root"],
        split="train",
        n_frames=data_cfg.get("n_frames", 7),
        scale=data_cfg.get("scale", 2),
        patch_size=data_cfg.get("patch_size", 64),
        batch_size=config["training"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
    )
    val_loader = get_dataloader(
        root=data_cfg["root"],
        split="test",
        n_frames=data_cfg.get("n_frames", 7),
        scale=data_cfg.get("scale", 2),
        patch_size=data_cfg.get("patch_size", 64),
        batch_size=config["training"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
    )

    # Loss
    loss_cfg = config.get("loss", {})
    criterion = CombinedLoss(
        pixel_weight=loss_cfg.get("pixel_weight", 1.0),
        perceptual_weight=loss_cfg.get("perceptual_weight", 0.0),
        temporal_weight=loss_cfg.get("temporal_weight", 0.0),
        pixel_type=loss_cfg.get("pixel_type", "l1"),
    ).to(device)

    # Optimizer
    train_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0),
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg.get("lr_step", 30),
        gamma=train_cfg.get("lr_gamma", 0.5),
    )

    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(model, args.resume, optimizer, device)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    # TensorBoard
    log_dir = os.path.join("results", "logs", config["model"]["name"])
    writer = SummaryWriter(log_dir)

    # Checkpoint dir
    ckpt_dir = os.path.join("checkpoints", config["model"]["name"])
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    epochs = train_cfg.get("epochs", 100)
    print(f"Training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        val_loss, val_psnr = validate(
            model, val_loader, criterion, device, config
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} PSNR: {train_psnr:.2f} | "
            f"Val Loss: {val_loss:.4f} PSNR: {val_psnr:.2f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("PSNR/train", train_psnr, epoch)
        writer.add_scalar("PSNR/val", val_psnr, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Save best checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                model, optimizer, epoch, best_psnr,
                os.path.join(ckpt_dir, "best.pt"),
            )
            print(f"  -> New best PSNR: {best_psnr:.2f}")

        # Save periodic checkpoint
        if (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            save_checkpoint(
                model, optimizer, epoch, val_psnr,
                os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"),
            )

    writer.close()
    print(f"Training complete. Best val PSNR: {best_psnr:.2f}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
