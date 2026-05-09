"""Evaluation script for video super-resolution models.

Usage:
    python src/evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/best.pt
"""

import argparse
import json
import os
import sys

import torch
import yaml
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataloader
from src.models import build_model
from src.utils import (
    get_device,
    load_checkpoint,
    tensor_to_numpy,
    frames_to_video,
    visualize_comparison,
)


def compute_psnr(sr, hr):
    """PSNR between two [C,H,W] tensors."""
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(sr, hr):
    """SSIM between two [C,H,W] tensors."""
    sr_np = tensor_to_numpy(sr)
    hr_np = tensor_to_numpy(hr)
    return ssim(hr_np, sr_np, channel_axis=2, data_range=255)


def compute_temporal_consistency(sr_frames, hr_frames):
    """Mean temporal consistency error over a sequence.

    Args:
        sr_frames: [N, C, H, W] tensor
        hr_frames: [N, C, H, W] tensor
    Returns:
        float: mean || (SR_t - SR_{t-1}) - (HR_t - HR_{t-1}) ||_1
    """
    if sr_frames.shape[0] < 2:
        return 0.0
    sr_diff = sr_frames[1:] - sr_frames[:-1]
    hr_diff = hr_frames[1:] - hr_frames[:-1]
    return torch.mean(torch.abs(sr_diff - hr_diff)).item()


def run_model_on_sequence(model, lr_frames, config, device):
    """Run model on a sequence of LR frames.

    Args:
        model: the SR model
        lr_frames: [N, C, H, W] sequence of LR frames
        config: config dict
        device: torch device
    Returns:
        sr_frames: [N, C, H, W] sequence of SR frames
    """
    model_name = config["model"]["name"]

    if model_name in ("baseline", "temporal_loss"):
        # Process each frame independently
        sr_list = []
        for t in range(lr_frames.shape[0]):
            lr = lr_frames[t:t+1].to(device)
            if model_name == "baseline":
                sr = model(lr)
            else:
                sr = model(lr)
            sr_list.append(sr.squeeze(0).cpu())
        return torch.stack(sr_list)

    elif model_name == "multiframe":
        sr_list = []
        N = lr_frames.shape[0]
        for t in range(N):
            start = max(0, t - 1)
            end = min(N, t + 2)
            window = lr_frames[start:end]  # [<=3, C, H, W]
            # Pad if at boundaries
            if window.shape[0] < 3:
                if start == 0:
                    window = torch.cat([window[:1], window], dim=0)
                if end == N and window.shape[0] < 3:
                    window = torch.cat([window, window[-1:]], dim=0)
            window = window.unsqueeze(0).to(device)  # [1, 3, C, H, W]
            sr = model(window)
            sr_list.append(sr.squeeze(0).cpu())
        return torch.stack(sr_list)

    elif model_name == "recurrent":
        lr_seq = lr_frames.unsqueeze(0).to(device)  # [1, N, C, H, W]
        sr_seq = model.forward_sequence(lr_seq)
        return sr_seq.squeeze(0).cpu()

    elif model_name == "flow_warp":
        sr_list = []
        N = lr_frames.shape[0]
        for t in range(N):
            start = max(0, t - 1)
            end = min(N, t + 2)
            window = lr_frames[start:end]
            if window.shape[0] < 3:
                if start == 0:
                    window = torch.cat([window[:1], window], dim=0)
                if end == N and window.shape[0] < 3:
                    window = torch.cat([window, window[-1:]], dim=0)
            window = window.unsqueeze(0).to(device)
            sr = model(window)
            sr_list.append(sr.squeeze(0).cpu())
        return torch.stack(sr_list)

    else:
        raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def evaluate(model, dataloader, config, device, output_dir, max_videos=5):
    """Full evaluation: PSNR, SSIM, temporal consistency, and visualizations."""
    model.eval()

    all_psnr = []
    all_ssim = []
    all_tc = []
    n_videos = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        lr_frames = batch["lr_frames"]  # [B, N, C, H, W]
        hr_frames = batch["hr_frames"]

        for b in range(lr_frames.shape[0]):
            lr_seq = lr_frames[b]  # [N, C, H, W]
            hr_seq = hr_frames[b]

            sr_seq = run_model_on_sequence(model, lr_seq, config, device)

            # Per-frame metrics
            for t in range(sr_seq.shape[0]):
                psnr_val = compute_psnr(sr_seq[t], hr_seq[t])
                ssim_val = compute_ssim(sr_seq[t], hr_seq[t])
                all_psnr.append(psnr_val)
                all_ssim.append(ssim_val)

            # Temporal consistency
            tc_val = compute_temporal_consistency(sr_seq, hr_seq)
            all_tc.append(tc_val)

            # Save videos and comparisons for first few sequences
            if n_videos < max_videos:
                vid_dir = os.path.join(output_dir, f"seq_{n_videos:03d}")
                os.makedirs(vid_dir, exist_ok=True)

                # Save comparison video
                frames_to_video(
                    list(sr_seq), os.path.join(vid_dir, "sr.mp4")
                )
                frames_to_video(
                    list(hr_seq), os.path.join(vid_dir, "hr.mp4")
                )

                # Save frame comparison for center frame
                center = sr_seq.shape[0] // 2
                visualize_comparison(
                    lr_seq[center], sr_seq[center], hr_seq[center],
                    save_path=os.path.join(vid_dir, "comparison.png"),
                )

            n_videos += 1

    # Aggregate metrics
    results = {
        "model": config["model"]["name"],
        "psnr_mean": float(np.mean(all_psnr)),
        "psnr_std": float(np.std(all_psnr)),
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
        "temporal_consistency": float(np.mean(all_tc)),
        "n_frames_evaluated": len(all_psnr),
        "n_sequences_evaluated": len(all_tc),
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"Results for {config['model']['name']}:")
    print(f"  PSNR: {results['psnr_mean']:.2f} +/- {results['psnr_std']:.2f} dB")
    print(f"  SSIM: {results['ssim_mean']:.4f} +/- {results['ssim_std']:.4f}")
    print(f"  Temporal Consistency: {results['temporal_consistency']:.4f}")
    print(f"  Evaluated {results['n_sequences_evaluated']} sequences "
          f"({results['n_frames_evaluated']} frames)")
    print(f"{'='*50}\n")

    # Save results
    results_path = os.path.join(output_dir, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate video SR model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    model = build_model(config).to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    data_cfg = config["data"]
    val_loader = get_dataloader(
        root=data_cfg["root"],
        split="test",
        n_frames=data_cfg.get("n_frames", 7),
        scale=data_cfg.get("scale", 2),
        patch_size=data_cfg.get("patch_size", 128),  # larger patches for eval
        batch_size=1,  # process one sequence at a time
        num_workers=data_cfg.get("num_workers", 4),
    )

    output_dir = args.output_dir or os.path.join("results", config["model"]["name"])
    os.makedirs(output_dir, exist_ok=True)

    evaluate(model, val_loader, config, device, output_dir, args.max_videos)


if __name__ == "__main__":
    main()
