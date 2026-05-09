"""Generate a synthetic video dataset in Vimeo-90K septuplet format.

Creates sequences of moving geometric shapes with smooth motion,
structured identically to Vimeo-90K so the same dataloader works.

Usage:
    python scripts/generate_synthetic.py --n-train 200 --n-test 50
"""

import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw


def make_sequence(width=256, height=256, n_frames=7, n_shapes=3, seed=None):
    """Generate a 7-frame sequence of moving colored shapes on textured background."""
    rng = np.random.RandomState(seed)

    # Textured background (gradient + noise)
    bg = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        grad = np.linspace(rng.uniform(0.1, 0.4), rng.uniform(0.4, 0.8), width)
        bg[:, :, c] = grad[np.newaxis, :]
    bg += rng.uniform(-0.05, 0.05, bg.shape).astype(np.float32)
    bg = np.clip(bg, 0, 1)

    # Define shapes with positions and velocities
    shapes = []
    for _ in range(n_shapes):
        shape_type = rng.choice(["circle", "rect", "triangle"])
        color = tuple(rng.randint(50, 255, 3).tolist())
        size = rng.randint(20, 60)
        x = rng.uniform(size, width - size)
        y = rng.uniform(size, height - size)
        vx = rng.uniform(-4, 4)
        vy = rng.uniform(-4, 4)
        shapes.append({
            "type": shape_type, "color": color, "size": size,
            "x": x, "y": y, "vx": vx, "vy": vy,
        })

    frames = []
    for t in range(n_frames):
        img = Image.fromarray((bg * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        for s in shapes:
            cx, cy, sz = s["x"], s["y"], s["size"]
            if s["type"] == "circle":
                draw.ellipse([cx - sz, cy - sz, cx + sz, cy + sz], fill=s["color"])
            elif s["type"] == "rect":
                draw.rectangle([cx - sz, cy - sz, cx + sz, cy + sz], fill=s["color"])
            elif s["type"] == "triangle":
                pts = [
                    (cx, cy - sz),
                    (cx - sz, cy + sz),
                    (cx + sz, cy + sz),
                ]
                draw.polygon(pts, fill=s["color"])

            # Update position with bounce
            s["x"] += s["vx"]
            s["y"] += s["vy"]
            if s["x"] < s["size"] or s["x"] > width - s["size"]:
                s["vx"] *= -1
            if s["y"] < s["size"] or s["y"] > height - s["size"]:
                s["vy"] *= -1

        frames.append(img)

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/vimeo_septuplet")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    seq_dir = os.path.join(args.output, "sequences")
    os.makedirs(seq_dir, exist_ok=True)

    train_list = []
    test_list = []
    total = args.n_train + args.n_test

    for i in range(total):
        # Use folder structure like Vimeo-90K: 00001/0001
        group = f"{(i // 100) + 1:05d}"
        seq = f"{(i % 100) + 1:04d}"
        seq_path = os.path.join(seq_dir, group, seq)
        os.makedirs(seq_path, exist_ok=True)

        frames = make_sequence(args.width, args.height, seed=i)
        for j, frame in enumerate(frames):
            frame.save(os.path.join(seq_path, f"im{j+1}.png"))

        entry = f"{group}/{seq}"
        if i < args.n_train:
            train_list.append(entry)
        else:
            test_list.append(entry)

        if (i + 1) % 50 == 0:
            print(f"Generated {i+1}/{total} sequences")

    # Write split files
    with open(os.path.join(args.output, "sep_trainlist.txt"), "w") as f:
        f.write("\n".join(train_list) + "\n")
    with open(os.path.join(args.output, "sep_testlist.txt"), "w") as f:
        f.write("\n".join(test_list) + "\n")

    print(f"\nDone! {args.n_train} train + {args.n_test} test sequences")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
