#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect CXR Pneumonia dataset under a given root directory.

Outputs:
- Class sample counts (by label inferred from parent folder: NORMAL / PNEUMONIA)
- Image size (width x height) distribution and summary (min/max/top)
- Channel/mode breakdown (grayscale/RGB/other + raw Pillow modes)
- Corrupted/unreadable file count and list (limited print)

Usage:
    python scripts/inspect_dataset.py --root data/raw --max-errors 50 --save experiments/dataset_report.json

Notes:
- The script scans recursively under --root and ignores __MACOSX and dotfiles.
- Labels are inferred from the folder names containing the file (case-insensitive),
  matching exactly 'NORMAL' or 'PNEUMONIA'. If neither is found in the path parts,
  label is set to 'UNKNOWN'.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABELS = {"NORMAL", "PNEUMONIA"}
GRAYSCALE_MODES = {"L", "1", "I", "F"}
RGB_MODES = {"RGB"}
RGBA_MODES = {"RGBA"}


@dataclass
class ImageRecord:
    path: str
    label: str
    width: int
    height: int
    mode: str


@dataclass
class Report:
    total_images: int
    classes: Dict[str, int]
    splits: Dict[str, Dict[str, int]]  # split -> {label: count}
    size_distribution: Dict[str, int]   # "WxH" -> count
    size_summary: Dict[str, object]
    mode_counts: Dict[str, int]         # Pillow mode -> count
    channel_summary: Dict[str, int]     # grayscale/rgb/rgba/other -> count
    corrupted_count: int
    corrupted_examples: List[str]


def infer_label(path: Path) -> str:
    for part in path.parts[::-1]:  # search from leaf upwards
        up = part.upper()
        if up in LABELS:
            return up
    return "UNKNOWN"


def infer_split(path: Path) -> str:
    for part in path.parts[::-1]:
        low = part.lower()
        if low in {"train", "test", "val", "valid", "validation"}:
            # normalize valid/validation to val
            return "val" if low in {"valid", "validation"} else low
    return "unspecified"


def mode_category(mode: str) -> str:
    if mode in GRAYSCALE_MODES:
        return "grayscale"
    if mode in RGB_MODES:
        return "rgb"
    if mode in RGBA_MODES:
        return "rgba"
    return "other"


def scan_images(root: Path, max_errors: int = 50) -> Tuple[List[ImageRecord], List[Path]]:
    records: List[ImageRecord] = []
    corrupted: List[Path] = []

    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.name.startswith("."):
            continue
        if any(part == "__MACOSX" for part in fp.parts):
            continue
        if fp.suffix.lower() not in SUPPORTED_EXTS:
            continue
        try:
            # First verify to catch truncated files quickly
            with Image.open(fp) as img:
                img.verify()
            # Re-open to access size/mode after verify()
            with Image.open(fp) as img2:
                width, height = img2.size
                mode = img2.mode
            label = infer_label(fp.parent)
            records.append(ImageRecord(path=str(fp), label=label, width=width, height=height, mode=mode))
        except (UnidentifiedImageError, OSError, ValueError):
            corrupted.append(fp)
            if len(corrupted) <= max_errors:
                # keep scanning; we'll report at the end
                pass
            continue
    return records, corrupted


def build_report(records: List[ImageRecord], corrupted: List[Path]) -> Report:
    total = len(records)
    class_counts: Dict[str, int] = Counter(r.label for r in records)

    # split breakdown per class
    split_counts: Dict[str, Dict[str, int]] = defaultdict(Counter)
    for r in records:
        sp = infer_split(Path(r.path))
        split_counts[sp][r.label] += 1

    # size distribution
    size_counts: Dict[str, int] = Counter(f"{r.width}x{r.height}" for r in records)

    # mode counts
    mode_counts: Dict[str, int] = Counter(r.mode for r in records)

    channel_counts: Dict[str, int] = Counter(mode_category(r.mode) for r in records)

    # size summary
    if records:
        widths = [r.width for r in records]
        heights = [r.height for r in records]
        # top frequent sizes
        top_sizes = size_counts.most_common(10)
        size_summary = {
            "min_width": int(min(widths)),
            "max_width": int(max(widths)),
            "min_height": int(min(heights)),
            "max_height": int(max(heights)),
            "top_sizes": [{"size": s, "count": c} for s, c in top_sizes],
        }
    else:
        size_summary = {
            "min_width": None,
            "max_width": None,
            "min_height": None,
            "max_height": None,
            "top_sizes": [],
        }

    return Report(
        total_images=total,
        classes=dict(class_counts),
        splits={k: dict(v) for k, v in split_counts.items()},
        size_distribution=dict(size_counts),
        size_summary=size_summary,
        mode_counts=dict(mode_counts),
        channel_summary=dict(channel_counts),
        corrupted_count=len(corrupted),
        corrupted_examples=[str(p) for p in corrupted[:20]],
    )


def print_report(rep: Report) -> None:
    print("\n==== Dataset Report ====")
    print(f"Total images: {rep.total_images}")

    print("\nClass counts:")
    for k, v in sorted(rep.classes.items()):
        print(f"  {k}: {v}")

    if rep.splits:
        print("\nSplit x Class counts:")
        for sp, d in rep.splits.items():
            d_sorted = dict(sorted(d.items()))
            print(f"  {sp}: {d_sorted}")

    print("\nImage size summary:")
    s = rep.size_summary
    print(f"  Width range: {s['min_width']} - {s['max_width']}")
    print(f"  Height range: {s['min_height']} - {s['max_height']}")
    if s["top_sizes"]:
        print("  Most frequent sizes:")
        for item in s["top_sizes"]:
            print(f"    {item['size']}: {item['count']}")

    print("\nChannel/mode summary:")
    if rep.channel_summary:
        print(f"  Grayscale: {rep.channel_summary.get('grayscale', 0)}")
        print(f"  RGB: {rep.channel_summary.get('rgb', 0)}")
        print(f"  RGBA: {rep.channel_summary.get('rgba', 0)}")
        print(f"  Other: {rep.channel_summary.get('other', 0)}")
    if rep.mode_counts:
        print("  Raw Pillow modes:")
        for k, v in sorted(rep.mode_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {k}: {v}")

    print("\nCorrupted/unreadable files:")
    print(f"  Count: {rep.corrupted_count}")
    if rep.corrupted_examples:
        print("  Examples (up to 20):")
        for p in rep.corrupted_examples:
            print(f"    {p}")
    print("===================================\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect dataset under a root directory")
    parser.add_argument("--root", type=str, default="data/raw", help="Root directory to scan (recursive)")
    parser.add_argument("--max-errors", type=int, default=200, help="Max corrupted files to keep as examples")
    parser.add_argument("--save", type=str, default="experiments/dataset_report.json", help="Optional path to save JSON report")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[Error] Root not found: {root}", file=sys.stderr)
        return 2

    records, corrupted = scan_images(root, max_errors=args.max_errors)
    rep = build_report(records, corrupted)

    # print
    print_report(rep)

    # save
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(rep), f, ensure_ascii=False, indent=2)
    print(f"Report saved: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
