from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import yaml

from datasets import CXRDataset
from transforms import get_transforms
from models import create_model
from losses import get_loss
from utils import set_seed, get_device, ensure_dir, AverageMeter, compute_metrics, save_checkpoint, timestamp


def _extract_labels(ds) -> List[int]:
    """Get labels list from CXRDataset or Subset[CXRDataset] without loading images."""
    try:
        from torch.utils.data import Subset
        if isinstance(ds, Subset):
            base = ds.dataset
            idxs = ds.indices
            if hasattr(base, "samples"):
                return [int(base.samples[i].label) for i in idxs]
            # fallback by indexing (may be slow)
            return [int(base[i][1]) for i in idxs]
        else:
            if hasattr(ds, "samples"):
                return [int(s.label) for s in ds.samples]
            return [int(ds[i][1]) for i in range(len(ds))]
    except Exception:
        # extremely defensive fallback
        return [int(ds[i][1]) for i in range(len(ds))]


def _compute_class_weights(labels: List[int]) -> Tuple[Dict[int, float], float]:
    """Compute class weights using N_total / (K * N_class_i) and pos_weight for BCE.

    Returns:
      (class_weight_map, pos_weight)
    """
    num_classes = 2
    n_total = len(labels)
    n_pos = sum(1 for x in labels if x == 1)
    n_neg = n_total - n_pos
    # avoid zero division
    n_pos = max(1, n_pos)
    n_neg = max(1, n_neg)
    w_neg = n_total / (num_classes * n_neg)
    w_pos = n_total / (num_classes * n_pos)
    class_weight = {0: float(w_neg), 1: float(w_pos)}
    pos_weight = float(n_neg) / float(n_pos)
    return class_weight, pos_weight


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, Dict[str, float], float]:
    image_root = cfg["data"].get("image_root")
    train_csv = cfg["data"].get("train_csv")
    val_csv = cfg["data"].get("val_csv")
    image_size = int(cfg["data"].get("image_size", 224))
    batch_size = int(cfg["train"].get("batch_size", 32))
    num_workers = int(cfg["train"].get("num_workers", 4))
    aug = cfg["train"].get("aug", "light")

    train_tfms, val_tfms = get_transforms(image_size=image_size, aug=aug)

    if train_csv:
        train_ds = CXRDataset(csv_path=train_csv, transform=train_tfms)
        if val_csv:
            val_ds = CXRDataset(csv_path=val_csv, transform=val_tfms)
        else:
            # split 90/10
            n_total = len(train_ds)
            n_val = max(1, int(0.1 * n_total))
            n_train = n_total - n_val
            train_ds, val_ds = random_split(train_ds, [n_train, n_val])
            # random_split wraps dataset; patch transforms for val loader if needed
            if hasattr(val_ds.dataset, "transform"):
                val_ds.dataset.transform = val_tfms
    else:
        # folder mode (image_root required)
        assert image_root, "image_root must be specified when train_csv is None"
        full_ds = CXRDataset(image_root=image_root, transform=train_tfms)
        n_total = len(full_ds)
        n_val = max(1, int(0.1 * n_total))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        if hasattr(val_ds.dataset, "transform"):
            val_ds.dataset.transform = val_tfms

    # compute class weights from TRAIN set only
    labels = _extract_labels(train_ds)
    class_weight_map, pos_weight = _compute_class_weights(labels)

    # sampler: weighted to upsample rare class
    sampler_cfg = cfg["train"].get("sampler", "none")
    if str(sampler_cfg).lower() == "weighted":
        sample_weights = [(class_weight_map[int(l)]) for l in labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, class_weight_map, pos_weight


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, scaler, device: torch.device) -> float:
    model.train()
    meter = AverageMeter()
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=True):
            logits = model(images).squeeze(-1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), n=images.size(0))
    return meter.avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    meter = AverageMeter()
    all_probs = []
    all_labels = []
    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        logits = model(images).squeeze(-1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())
        meter.update(loss.item(), n=images.size(0))
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    acc, auroc, precision, recall, f1 = compute_metrics(y_true, y_prob)
    return meter.avg, {"acc": acc, "auroc": auroc, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CXR Pneumonia classifier")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("configs").joinpath("default.yaml")), help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device()

    train_loader, val_loader, class_weight_map, pos_weight = build_dataloaders(cfg)

    model = create_model(
        name=cfg["model"].get("name", "resnet18"),
        num_classes=1,
        pretrained=bool(cfg["model"].get("pretrained", True)),
    ).to(device)

    # build loss with optional class weighting
    loss_name = cfg["loss"].get("name", "bce").lower()
    loss_params = dict(cfg["loss"].get("params", {}))
    # auto weight setting controlled by train.class_weight
    cw_cfg = str(cfg["train"].get("class_weight", "none")).lower()
    if cw_cfg == "auto":
        if loss_name in {"bce", "bcelogits", "bcewithlogits"}:
            # BCEWithLogitsLoss: use pos_weight tensor
            loss_params["pos_weight"] = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        elif loss_name in {"focal", "focalloss"}:
            # FocalLoss: pass class weights [w_neg, w_pos]
            loss_params["weight"] = torch.tensor([class_weight_map[0], class_weight_map[1]], dtype=torch.float32, device=device)
            # and set default gamma=2 if not provided
            loss_params.setdefault("gamma", 2.0)
    criterion = get_loss(loss_name, **loss_params)
    optimizer = AdamW(model.parameters(), lr=float(cfg["train"].get("lr", 3e-4)), weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    epochs = int(cfg["train"].get("epochs", 10))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) if cfg["train"].get("scheduler", "cosine") == "cosine" else None

    out_dir = ensure_dir(cfg.get("output_dir", os.path.join("experiments", f"run-{timestamp()}")))

    best_auroc = -1.0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"acc={metrics['acc']:.4f} auroc={metrics['auroc']:.4f} f1={metrics['f1']:.4f}")

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            save_checkpoint({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "config": cfg,
            }, out_dir)

    print(f"Training complete. Best AUROC={best_auroc:.4f}. Checkpoints saved in: {out_dir}")


if __name__ == "__main__":
    main()
