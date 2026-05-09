"""Regenerate metrics_comparison.png with honest y-axis scales.

Reads results/<model>/metrics.json and replots the bar charts with tighter
y-limits so the magnitude of differences is visible.
"""

import json
import os

import matplotlib.pyplot as plt

MODELS = ["baseline", "temporal_loss", "multiframe", "recurrent", "flow_warp"]
LABELS = ["Baseline", "Temporal Loss", "Multi-Frame", "Recurrent", "Flow Warp"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

metrics = {}
for m in MODELS:
    with open(f"results/{m}/metrics.json") as f:
        metrics[m] = json.load(f)

psnr_vals = [metrics[m]["psnr_mean"] for m in MODELS]
ssim_vals = [metrics[m]["ssim_mean"] for m in MODELS]
tc_vals = [metrics[m]["temporal_consistency"] for m in MODELS]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

bars = axes[0].bar(LABELS, psnr_vals, color=COLORS)
axes[0].set_ylabel("PSNR (dB)")
axes[0].set_title("PSNR (higher = better)")
axes[0].set_ylim(30.5, 32.7)
axes[0].tick_params(axis="x", rotation=25)
for bar, val in zip(bars, psnr_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

bars = axes[1].bar(LABELS, ssim_vals, color=COLORS)
axes[1].set_ylabel("SSIM")
axes[1].set_title("SSIM (higher = better)")
axes[1].set_ylim(0.72, 0.76)
axes[1].tick_params(axis="x", rotation=25)
for bar, val in zip(bars, ssim_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

bars = axes[2].bar(LABELS, tc_vals, color=COLORS)
axes[2].set_ylabel("Temporal Error")
axes[2].set_title("Temporal Consistency (lower = better)")
axes[2].set_ylim(0, max(tc_vals) * 1.15)
axes[2].tick_params(axis="x", rotation=25)
for bar, val in zip(bars, tc_vals):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Quantitative Comparison Across All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved results/metrics_comparison.png")
