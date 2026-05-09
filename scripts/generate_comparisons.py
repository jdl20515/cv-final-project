"""Generate side-by-side comparison images across all models.

Creates a grid showing LR (bicubic) | Baseline | Temporal | MultiFrame | Recurrent | FlowWarp | HR
for several test sequences.
"""

import os
import sys
import json

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import VideoSRDataset
from src.models import build_model
from src.utils import get_device, load_checkpoint, tensor_to_numpy


MODELS = [
    ("baseline", "configs/baseline.yaml", "checkpoints/baseline/best.pt", {}),
    ("temporal_loss", "configs/temporal_loss.yaml", "checkpoints/temporal_loss/best.pt", {}),
    ("multiframe", "configs/multiframe.yaml", "checkpoints/multiframe/best.pt", {}),
    ("recurrent", "configs/recurrent.yaml", "checkpoints/recurrent/best.pt", {}),
    ("flow_warp", "configs/flow_warp.yaml", "checkpoints/flow_warp/best.pt", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}),
]


def run_single_frame(model, lr_frames, config, device):
    """Get SR output for the center frame."""
    model_name = config["model"]["name"]
    N = lr_frames.shape[0]
    center = N // 2

    if model_name in ("baseline", "temporal_loss"):
        lr = lr_frames[center:center+1].to(device)
        sr = model(lr)
        return sr.squeeze(0).cpu()

    elif model_name == "multiframe":
        start = max(0, center - 1)
        end = min(N, center + 2)
        window = lr_frames[start:end]
        if window.shape[0] < 3:
            if start == 0:
                window = torch.cat([window[:1], window], dim=0)
            if window.shape[0] < 3:
                window = torch.cat([window, window[-1:]], dim=0)
        window = window.unsqueeze(0).to(device)
        sr = model(window)
        return sr.squeeze(0).cpu()

    elif model_name == "recurrent":
        lr_seq = lr_frames.unsqueeze(0).to(device)
        sr_seq = model.forward_sequence(lr_seq)
        return sr_seq[0, center].cpu()

    elif model_name == "flow_warp":
        start = max(0, center - 1)
        end = min(N, center + 2)
        window = lr_frames[start:end]
        if window.shape[0] < 3:
            if start == 0:
                window = torch.cat([window[:1], window], dim=0)
            if window.shape[0] < 3:
                window = torch.cat([window, window[-1:]], dim=0)
        window = window.unsqueeze(0).to(device)
        sr = model(window)
        return sr.squeeze(0).cpu()


@torch.no_grad()
def main():
    device = get_device()
    os.makedirs("results", exist_ok=True)

    # Load test dataset
    dataset = VideoSRDataset("data/vimeo_septuplet", split="test", n_frames=7, scale=2, patch_size=128)

    # Pick a few diverse sequences
    indices = [0, 10, 25, 40]

    # Load all models
    loaded_models = {}
    for name, cfg_path, ckpt_path, env_vars in MODELS:
        for k, v in env_vars.items():
            os.environ[k] = v
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        model = build_model(config).to(device)
        load_checkpoint(model, ckpt_path, device=device)
        model.eval()
        loaded_models[name] = (model, config)
        print(f"Loaded {name}")

    # Load metrics for annotations
    metrics = {}
    for name, _, _, _ in MODELS:
        mp = f"results/{name}/metrics.json"
        if os.path.exists(mp):
            with open(mp) as f:
                metrics[name] = json.load(f)

    # === Figure 1: Full comparison grid ===
    n_seqs = len(indices)
    n_cols = 7  # LR, baseline, temporal, multiframe, recurrent, flow_warp, HR
    col_labels = ["LR (Bicubic)", "Baseline", "Temporal Loss", "Multi-Frame", "Recurrent", "Flow Warp", "HR (Ground Truth)"]

    fig, axes = plt.subplots(n_seqs, n_cols, figsize=(n_cols * 3, n_seqs * 3))

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        lr_frames = sample["lr_frames"]
        hr_frames = sample["hr_frames"]
        center = lr_frames.shape[0] // 2

        lr_center = lr_frames[center]
        hr_center = hr_frames[center]

        # Bicubic upscale of LR
        lr_np = tensor_to_numpy(lr_center)
        h, w = tensor_to_numpy(hr_center).shape[:2]
        lr_up = cv2.resize(lr_np, (w, h), interpolation=cv2.INTER_CUBIC)

        images = [lr_up]

        for name, _, _, _ in MODELS:
            model, config = loaded_models[name]
            sr = run_single_frame(model, lr_frames, config, device)
            images.append(tensor_to_numpy(sr))

        images.append(tensor_to_numpy(hr_center))

        for col, img in enumerate(images):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_labels[col], fontsize=10, fontweight="bold")

    plt.suptitle("Video Super-Resolution: Model Comparison (Center Frame, 2x Upscaling)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/model_comparison_grid.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/model_comparison_grid.png")

    # === Figure 2: Zoomed crop comparison ===
    sample = dataset[indices[0]]
    lr_frames = sample["lr_frames"]
    hr_frames = sample["hr_frames"]
    center = lr_frames.shape[0] // 2
    hr_center = hr_frames[center]

    # Pick a crop region (center 64x64 of the HR image)
    _, H, W = hr_center.shape
    cy, cx = H // 2, W // 2
    cs = 48  # crop half-size
    y1, y2 = cy - cs, cy + cs
    x1, x2 = cx - cs, cx + cs

    lr_np = tensor_to_numpy(lr_frames[center])
    lr_up = cv2.resize(lr_np, (W, H), interpolation=cv2.INTER_CUBIC)

    all_crops = [lr_up[y1:y2, x1:x2]]
    crop_labels = ["LR (Bicubic)"]

    for name, _, _, _ in MODELS:
        model, config = loaded_models[name]
        sr = run_single_frame(model, lr_frames, config, device)
        sr_np = tensor_to_numpy(sr)
        all_crops.append(sr_np[y1:y2, x1:x2])
        crop_labels.append(name.replace("_", " ").title())

    all_crops.append(tensor_to_numpy(hr_center)[y1:y2, x1:x2])
    crop_labels.append("HR (GT)")

    fig, axes = plt.subplots(1, len(all_crops), figsize=(len(all_crops) * 3, 3))
    for i, (crop, label) in enumerate(zip(all_crops, crop_labels)):
        axes[i].imshow(crop)
        axes[i].set_title(label, fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Zoomed Crop Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/zoomed_crop_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/zoomed_crop_comparison.png")

    # === Figure 3: Temporal profile (xt-slice) ===
    sample = dataset[indices[0]]
    lr_frames = sample["lr_frames"]
    hr_frames = sample["hr_frames"]
    _, H_hr, W_hr = hr_frames[0].shape
    slice_x = W_hr // 2  # vertical slice at center column

    profiles = {}
    # HR profile
    hr_slice = torch.stack([hr_frames[t, :, :, slice_x] for t in range(hr_frames.shape[0])])  # [N, C, H]
    profiles["HR (GT)"] = tensor_to_numpy(hr_slice.permute(1, 0, 2).reshape(3, -1, hr_slice.shape[2]))
    # Actually let's just stack vertically: each row = time, columns = spatial
    hr_profile = np.stack([tensor_to_numpy(hr_frames[t])[:, slice_x, :] for t in range(hr_frames.shape[0])])  # [N, 3] wait no

    # Simpler approach: for each model, collect the center column across all frames
    def make_temporal_profile(frames_list):
        """frames_list: list of [C,H,W] tensors. Returns [N, H, 3] image."""
        cols = []
        for f in frames_list:
            img = tensor_to_numpy(f)  # [H, W, 3]
            cols.append(img[:, slice_x, :])  # [H, 3]
        return np.stack(cols, axis=0)  # [N, H, 3]

    fig, axes = plt.subplots(1, 7, figsize=(21, 4))
    profile_labels = ["LR (Bicubic)", "Baseline", "Temporal Loss", "Multi-Frame", "Recurrent", "Flow Warp", "HR (GT)"]

    # LR bicubic
    lr_profile_frames = []
    for t in range(lr_frames.shape[0]):
        lr_np = tensor_to_numpy(lr_frames[t])
        lr_up = cv2.resize(lr_np, (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)
        lr_profile_frames.append(torch.from_numpy(lr_up).permute(2, 0, 1).float() / 255.0)

    all_profiles = [make_temporal_profile(lr_profile_frames)]

    for name, _, _, _ in MODELS:
        model, config = loaded_models[name]
        sr_frames = []
        if config["model"]["name"] == "recurrent":
            lr_seq = lr_frames.unsqueeze(0).to(device)
            sr_seq = model.forward_sequence(lr_seq).squeeze(0).cpu()
            sr_frames = [sr_seq[t] for t in range(sr_seq.shape[0])]
        else:
            for t in range(lr_frames.shape[0]):
                sr = run_single_frame(model, lr_frames, config, device)
                sr_frames.append(sr)
        all_profiles.append(make_temporal_profile(sr_frames))

    all_profiles.append(make_temporal_profile([hr_frames[t] for t in range(hr_frames.shape[0])]))

    for i, (prof, label) in enumerate(zip(all_profiles, profile_labels)):
        axes[i].imshow(prof, aspect="auto")
        axes[i].set_title(label, fontsize=9)
        axes[i].set_ylabel("Time" if i == 0 else "")
        axes[i].set_xlabel("Spatial (y)")

    plt.suptitle("Temporal Profile (vertical slice through time)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/temporal_profiles.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/temporal_profiles.png")

    # === Figure 4: Metrics bar charts ===
    available = [name for name, _, _, _ in MODELS if name in metrics]
    psnr_vals = [metrics[m]["psnr_mean"] for m in available]
    ssim_vals = [metrics[m]["ssim_mean"] for m in available]
    tc_vals = [metrics[m]["temporal_consistency"] for m in available]
    labels = [m.replace("_", " ").title() for m in available]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    bars = axes[0].bar(labels, psnr_vals, color=colors[:len(available)])
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR (higher = better)")
    axes[0].tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, psnr_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    bars = axes[1].bar(labels, ssim_vals, color=colors[:len(available)])
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM (higher = better)")
    axes[1].tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, ssim_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    bars = axes[2].bar(labels, tc_vals, color=colors[:len(available)])
    axes[2].set_ylabel("Temporal Error")
    axes[2].set_title("Temporal Consistency (lower = better)")
    axes[2].tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, tc_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Quantitative Comparison Across All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/metrics_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/metrics_comparison.png")

    print("\nDone! All figures saved to results/")


if __name__ == "__main__":
    main()
