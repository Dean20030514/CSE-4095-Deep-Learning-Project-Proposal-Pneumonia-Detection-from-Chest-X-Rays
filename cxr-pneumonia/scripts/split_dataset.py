#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Group-aware, stratified dataset split for CXR Pneumonia.

Goal:
- Ensure images from the same patient (or near-duplicates via pHash) NEVER span across train/val/test.
- Keep class distribution stable via stratified sampling.
- Split ratios default to 70%/15%/15%.

Heuristics for patient grouping:
1) Try extracting patient ID from filename and parent directories via regexes:
   - person(\d+)
   - patient(\d+)
   - IM[-_]?(\d+)
   - p(\d+)
   - any long number sequence (>=4) as fallback
2) If no ID found, fallback to perceptual hash (pHash) grouping using the `imagehash` package.
   - By default, exact pHash groups are used (threshold=0). Use --phash-threshold>0 for more aggressive merging (may be slow).

Outputs:
- CSV manifest with columns: path,label,group,split  at data/processed/splits.csv (by default)
- Optional materialization of directory tree under data/processed/{split}/{label}/ via copy or hardlink

Usage:
    python scripts/split_dataset.py \
        --root data/raw \
        --out-csv data/processed/splits.csv \
        --train 0.7 --val 0.15 --test 0.15 \
        --strategy auto \
        [--copy | --hardlink]

Notes:
- Only NORMAL and PNEUMONIA labels are used. Others are skipped.
- "Stratified" is performed at group level based on the group's majority label.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedKFold

try:
    import imagehash  # type: ignore
    HAS_IMAGEHASH = True
except Exception:
    HAS_IMAGEHASH = False

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABELS = {"NORMAL", "PNEUMONIA"}

# Regex patterns to extract patient IDs from filenames or parent directories
PATTERNS = [
    re.compile(r"person(\d+)", re.IGNORECASE),
    re.compile(r"patient(\d+)", re.IGNORECASE),
    re.compile(r"IM[-_]?(\d+)", re.IGNORECASE),
    re.compile(r"p(\d+)", re.IGNORECASE),
    re.compile(r"(\d{4,})"),  # fallback long number sequence
]


@dataclass
class Item:
    path: Path
    label: str
    fname: str
    group_hint: Optional[str]  # from regex
    phash: Optional[str]       # hex string of perceptual hash


def infer_label(path: Path) -> Optional[str]:
    for part in path.parts[::-1]:
        up = part.upper()
        if up in LABELS:
            return up
    return None


def extract_patient_id_from_parts(path: Path) -> Optional[str]:
    parts = list(path.parts)
    for part in [path.name, path.stem] + parts[::-1]:  # check filename, stem, and dirs
        for pat in PATTERNS:
            m = pat.search(part)
            if m:
                return m.group(1)
    return None


def compute_phash(fp: Path) -> Optional[str]:
    try:
        with Image.open(fp) as img:
            img = img.convert("L")
            if HAS_IMAGEHASH:
                ph = imagehash.phash(img)
                return ph.__str__()
            else:
                # Lightweight fallback: average hash via numpy if imagehash not available
                img_small = img.resize((8, 8))
                arr = np.asarray(img_small, dtype=np.float32)
                avg = arr.mean()
                bits = (arr > avg).astype(np.uint8).flatten()
                # convert to hex string
                by = np.packbits(bits)
                return by.tobytes().hex()
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def iter_images(root: Path) -> Iterable[Path]:
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.name.startswith("."):
            continue
        if any(part == "__MACOSX" for part in fp.parts):
            continue
        if fp.suffix.lower() not in SUPPORTED_EXTS:
            continue
        yield fp


def scan_items(root: Path, use_phash: bool, phash_limit: Optional[int] = None) -> List[Item]:
    items: List[Item] = []
    for i, fp in enumerate(iter_images(root)):
        label = infer_label(fp.parent)
        if label is None:
            continue
        group_hint = extract_patient_id_from_parts(fp)
        phash = None
        if use_phash and group_hint is None:
            phash = compute_phash(fp)
        items.append(Item(path=fp, label=label, fname=fp.name, group_hint=group_hint, phash=phash))
        if phash_limit is not None and i + 1 >= phash_limit:
            break
    return items


def build_groups(items: List[Item], strategy: str = "auto", phash_threshold: int = 0) -> Tuple[Dict[str, List[int]], List[str]]:
    """Return (group_to_indices, group_label) and list group_id per item index."""
    # First pass: assign group ids
    group_id_of: List[str] = [""] * len(items)

    # Map from key to group id
    key_to_gid: Dict[str, str] = {}

    def new_gid(prefix: str, key: str) -> str:
        return f"{prefix}:{key}"

    # assign by patient hint when available
    for idx, it in enumerate(items):
        if it.group_hint:
            key = it.label + "|" + it.group_hint  # label-scoped patient to be safe
            gid = key_to_gid.get(key)
            if gid is None:
                gid = new_gid("pid", key)
                key_to_gid[key] = gid
            group_id_of[idx] = gid

    # remaining: assign by pHash if available/desired
    for idx, it in enumerate(items):
        if group_id_of[idx]:
            continue
        if strategy in ("auto", "phash") and it.phash:
            key = it.label + "|" + it.phash
            gid = key_to_gid.get(key)
            if gid is None:
                gid = new_gid("ph", key)
                key_to_gid[key] = gid
            group_id_of[idx] = gid

    # fallback: file stem grouping (very weak)
    for idx, it in enumerate(items):
        if group_id_of[idx]:
            continue
        stem = re.sub(r"\D+", "", Path(it.fname).stem)  # digits in filename
        if stem:
            key = it.label + "|stem:" + stem
        else:
            key = it.label + "|md5:" + hashlib.md5(it.fname.encode("utf-8")).hexdigest()[:8]
        gid = key_to_gid.get(key)
        if gid is None:
            gid = new_gid("fn", key)
            key_to_gid[key] = gid
        group_id_of[idx] = gid

    # collect group -> indices
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, gid in enumerate(group_id_of):
        groups[gid].append(i)

    return groups, group_id_of


def majority_label(labels: List[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def stratified_group_split(items: List[Item], groups: Dict[str, List[int]], ratios=(0.7, 0.15, 0.15), seed: int = 42) -> Dict[str, List[int]]:
    """Return mapping split -> list of item indices.

    Approach: operate at group level. Build unique group list and majority labels.
    Use StratifiedKFold(n_splits=10) to obtain folds of groups; then aggregate
    folds into 70/30; then split the 30% into val/test (approx 15/15) with StratifiedKFold(n_splits=2).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    train_r, val_r, test_r = ratios

    rng = np.random.RandomState(seed)

    group_ids = list(groups.keys())
    # assign each group a majority label
    y_group = []
    group_sizes = []
    for gid in group_ids:
        idxs = groups[gid]
        y_group.append(majority_label([items[i].label for i in idxs]))
        group_sizes.append(len(idxs))

    # 10-fold stratified on groups
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # collect folds as lists of group indices
    folds: List[List[int]] = []
    for _, test_idx in skf.split(np.zeros(len(group_ids)), np.array(y_group)):
        folds.append(list(test_idx))

    # Greedy pick folds for train to approach desired ratio by sample counts
    target_train = train_r * sum(group_sizes)
    fold_sizes = [sum(group_sizes[i] for i in fold) for fold in folds]

    order = list(range(len(folds)))
    rng.shuffle(order)

    picked = []
    total = 0
    for j in order:
        if total + fold_sizes[j] <= target_train or not picked:
            picked.append(j)
            total += fold_sizes[j]
        # stop if close enough (within one fold)
        if total / sum(group_sizes) >= train_r * 0.98:
            break

    train_group_idx = list(itertools.chain.from_iterable(folds[j] for j in picked))
    remaining = sorted(set(range(len(group_ids))) - set(train_group_idx))

    # split remaining into val/test stratified by groups
    rem_labels = [y_group[i] for i in remaining]
    rem_sizes = [group_sizes[i] for i in remaining]

    # if too few groups remain, simple split by size
    if len(remaining) < 2:
        val_groups = remaining
        test_groups = []
    else:
        skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + 1)
        # create dummy X
        splits = list(skf2.split(np.zeros(len(remaining)), np.array(rem_labels)))
        _, test_idx = splits[0]
        test_set = set(test_idx)
        test_groups = [remaining[i] for i in test_set]
        val_groups = [remaining[i] for i in range(len(remaining)) if i not in test_set]

    # map group indices to item indices
    def groups_to_items(g_indices: List[int]) -> List[int]:
        out = []
        for gi in g_indices:
            gid = group_ids[gi]
            out.extend(groups[gid])
        return out

    split_map = {
        "train": groups_to_items(train_group_idx),
        "val": groups_to_items(val_groups),
        "test": groups_to_items(test_groups),
    }
    return split_map


def materialize(split_map: Dict[str, List[int]], items: List[Item], out_root: Path, mode: Optional[str] = None):
    if mode is None:
        return
    for split, idxs in split_map.items():
        for i in idxs:
            it = items[i]
            dst = out_root / split / it.label / it.path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if mode == "copy":
                if not dst.exists():
                    # copy bytes
                    with it.path.open("rb") as src, dst.open("wb") as f:
                        f.write(src.read())
            elif mode == "hardlink":
                if not dst.exists():
                    try:
                        os.link(it.path, dst)
                    except Exception:
                        # fallback to copy if hardlink fails
                        with it.path.open("rb") as src, dst.open("wb") as f:
                            f.write(src.read())


def summarize(split_map: Dict[str, List[int]], items: List[Item]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split, idxs in split_map.items():
        lab = Counter(items[i].label for i in idxs)
        out[split] = dict(lab)
    return out


def main():
    parser = argparse.ArgumentParser(description="Group-aware stratified split for CXR dataset")
    parser.add_argument("--root", type=str, default="data/raw")
    parser.add_argument("--out-csv", type=str, default="data/processed/splits.csv")
    parser.add_argument("--out-dir", type=str, default="data/processed/splits_materialized")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--strategy", type=str, choices=["auto", "patient", "phash"], default="auto")
    parser.add_argument("--phash-threshold", type=int, default=0, help="Hamming distance threshold (0 means exact)")
    parser.add_argument("--materialize", type=str, choices=["copy", "hardlink", "none"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = (args.train, args.val, args.test)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit("Train/Val/Test ratios must sum to 1.0")

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv)
    out_dir = Path(args.out_dir)

    # Determine whether to use phash
    use_phash = args.strategy in ("auto", "phash")
    if use_phash and not HAS_IMAGEHASH:
        print("[Info] imagehash not installed; using lightweight aHash fallback.")

    print(f"Scanning images under: {root}")
    items = scan_items(root, use_phash=use_phash)
    print(f"Found {len(items)} labeled images.")

    groups, item_gids = build_groups(items, strategy=args.strategy, phash_threshold=args.phash_threshold)
    print(f"Built {len(groups)} groups.")

    split_map = stratified_group_split(items, groups, ratios=ratios, seed=args.seed)

    # Write CSV manifest
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "split"])  # header
        for split, idxs in split_map.items():
            for i in idxs:
                w.writerow([str(items[i].path), items[i].label, item_gids[i], split])
    print(f"Manifest written: {out_csv.resolve()}")

    # Optional materialization
    if args.materialize != "none":
        materialize(split_map, items, out_dir, mode=args.materialize)
        print(f"Materialized under: {out_dir.resolve()} (mode={args.materialize})")

    # Summary
    summary = summarize(split_map, items)
    print("Summary (counts per split):")
    for split in ["train", "val", "test"]:
        cnts = summary.get(split, {})
        total = sum(cnts.values())
        print(f"  {split}: total={total}, " + ", ".join(f"{k}={v}" for k, v in sorted(cnts.items())))


if __name__ == "__main__":
    main()
