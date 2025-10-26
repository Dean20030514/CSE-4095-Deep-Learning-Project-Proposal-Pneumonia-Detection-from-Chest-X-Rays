#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate dataset overview report (Markdown + images) for CXR Pneumonia.

Inputs:
- experiments/dataset_report.json (produced by scripts/inspect_dataset.py)
- data root (to sample example images)

Outputs (under --out):
- images/class_counts.png
- images/channel_counts.png
- images/top_sizes.png
- images/samples_train.png (if available)
- images/samples_test.png (if available)
- images/samples_val.png (if available)
- data_overview.md (Markdown report embedding the above images)

Usage:
    python scripts/generate_data_report.py \
        --root data/raw \
        --report experiments/dataset_report.json \
        --out reports
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABELS = ["NORMAL", "PNEUMONIA"]


def load_report(report_path: Path) -> Dict:
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    return img_dir


def save_bar_chart(data: Dict[str, int], title: str, out_path: Path, order: List[str] | None = None):
    if order is not None:
        keys = [k for k in order if k in data]
    else:
        keys = list(data.keys())
    vals = [data[k] for k in keys]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(keys, vals, color=["#4C78A8", "#F58518", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6"])  # auto palette
    plt.title(title)
    plt.ylabel("Count")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_top_sizes(size_dist: Dict[str, int], k: int, out_path: Path):
    items = sorted(size_dist.items(), key=lambda x: (-x[1], x[0]))[:k]
    if not items:
        return
    labels, counts = zip(*items)
    plt.figure(figsize=(8, 4.8))
    y_pos = range(len(labels))
    plt.barh(y_pos, counts, color="#4C78A8")
    plt.yticks(y_pos, labels)
    plt.xlabel("Count")
    plt.title(f"Top {k} Image Sizes")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def find_samples(root: Path, split: str, label: str, k: int = 8) -> List[Path]:
    """Find up to k sample image paths for given split and label across nested structures.

    It searches recursively for directories named like the split (train/test/val)
    that contain a child directory named as the label (NORMAL/PNEUMONIA), and then
    collects a few images from there.
    """
    paths: List[Path] = []
    # First try common direct locations for speed
    common = [
        root / "chest_xray" / split / label,
        root / split / label,
    ]
    for base in common:
        if base.exists():
            for fp in base.rglob("*"):
                if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTS and not fp.name.startswith("."):
                    paths.append(fp)
                    if len(paths) >= k:
                        return paths

    # Fallback: generic recursive search matching .../<split>/<label>/...
    for split_dir in root.rglob("*"):
        if not split_dir.is_dir():
            continue
        if split_dir.name.lower() != split:
            continue
        candidate = split_dir / label
        if not candidate.is_dir():
            continue
        for fp in candidate.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTS and not fp.name.startswith("."):
                paths.append(fp)
                if len(paths) >= k:
                    return paths
    return paths


def plot_sample_grid(paths: List[Path], title: str, out_path: Path, cols: int = 4):
    if not paths:
        return False
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, fp in zip(axes, paths):
        try:
            with Image.open(fp) as img:
                # show grayscale as gray, rgb as rgb
                if img.mode != "RGB":
                    img_disp = img.convert("L")
                    ax.imshow(img_disp, cmap="gray")
                else:
                    ax.imshow(img)
            ax.set_title(fp.parent.name, fontsize=9)
            ax.axis("off")
        except (UnidentifiedImageError, OSError, ValueError):
            ax.axis("off")
            ax.set_title("(unreadable)", fontsize=9)
    # hide unused axes
    for ax in axes[len(paths):]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate dataset overview report")
    parser.add_argument("--root", type=str, default="data/raw")
    parser.add_argument("--report", type=str, default="experiments/dataset_report.json")
    parser.add_argument("--out", type=str, default="reports")
    parser.add_argument("--top-k-sizes", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root)
    report_path = Path(args.report)
    out_dir = Path(args.out)

    img_dir = ensure_dirs(out_dir)

    if not report_path.exists():
        raise SystemExit(f"Report JSON not found: {report_path}. Run scripts/inspect_dataset.py first.")

    rep = load_report(report_path)

    # charts
    class_counts_path = img_dir / "class_counts.png"
    class_counts = rep.get("classes", {})
    # ensure order NORMAL, PNEUMONIA if present
    save_bar_chart(class_counts, "Class Counts", class_counts_path, order=LABELS)

    channel_counts_path = img_dir / "channel_counts.png"
    channel_counts = rep.get("channel_summary", {})
    channel_order = ["grayscale", "rgb", "rgba", "other"]
    save_bar_chart(channel_counts, "Channel Summary", channel_counts_path, order=channel_order)

    top_sizes_path = img_dir / "top_sizes.png"
    size_dist = rep.get("size_distribution", {})
    save_top_sizes(size_dist, args.top_k_sizes, top_sizes_path)

    # sample grids per split
    split_names = ["train", "test", "val"]
    sample_images_paths = {}
    for split in split_names:
        split_samples: List[Path] = []
        for label in LABELS:
            paths = find_samples(root, split, label, k=args.samples_per_class)
            split_samples.extend(paths)
        if split_samples:
            out_img = img_dir / f"samples_{split}.png"
            if plot_sample_grid(split_samples, f"{split.title()} Samples", out_img):
                sample_images_paths[split] = out_img

    # write markdown
    md_path = out_dir / "data_overview.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Data Overview\n\n")
        f.write(f"- Total images: {rep.get('total_images', 0)}\n")
        if class_counts:
            f.write("- Class counts: " + ", ".join(f"{k}: {v}" for k, v in sorted(class_counts.items())) + "\n")
        size_sum = rep.get("size_summary", {})
        if size_sum:
            f.write(
                f"- Width range: {size_sum.get('min_width')} - {size_sum.get('max_width')}, "
                f"Height range: {size_sum.get('min_height')} - {size_sum.get('max_height')}\n\n"
            )

        # class chart
        f.write("## Class Distribution\n\n")
        f.write(f"![Class Counts](images/{class_counts_path.name})\n\n")

        # channel chart
        f.write("## Channel Summary\n\n")
        f.write(f"![Channel Summary](images/{channel_counts_path.name})\n\n")

        # top sizes
        if size_dist:
            f.write("## Top Image Sizes\n\n")
            f.write(f"![Top Sizes](images/{top_sizes_path.name})\n\n")

        # samples
        if sample_images_paths:
            f.write("## Sample Visualizations\n\n")
            for split in split_names:
                if split in sample_images_paths:
                    rel = sample_images_paths[split].name
                    f.write(f"### {split.title()}\n\n")
                    f.write(f"![{split} samples](images/{rel})\n\n")

    print(f"Report generated: {md_path.resolve()}")


if __name__ == "__main__":
    main()
