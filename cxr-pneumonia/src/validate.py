from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
import yaml

from datasets import CXRDataset
from transforms import get_transforms
from models import create_model, load_checkpoint
from losses import get_loss
from utils import get_device, compute_metrics
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    y_probs = []
    y_trues = []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)
        logits = model(images).squeeze(-1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        losses.append(float(loss))
        y_probs.append(probs)
        y_trues.append(labels.cpu().numpy())
    y_prob = np.concatenate(y_probs)
    y_true = np.concatenate(y_trues)
    acc, auroc, precision, recall, f1 = compute_metrics(y_true, y_prob)
    return float(np.mean(losses)), {"acc": acc, "auroc": auroc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Validate CXR model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--csv", type=str, help="CSV file for evaluation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    _, val_tfms = get_transforms(image_size=cfg["data"].get("image_size", 224), aug="none")

    csv_path = args.csv or cfg["data"].get("val_csv") or cfg["data"].get("test_csv")
    assert csv_path, "Provide --csv or set data.val_csv/test_csv in config"

    ds = CXRDataset(csv_path=csv_path, transform=val_tfms)
    loader = DataLoader(ds, batch_size=cfg["train"].get("batch_size", 32), shuffle=False, num_workers=cfg["train"].get("num_workers", 4))

    model = create_model(name=cfg["model"].get("name", "resnet18"), num_classes=1, pretrained=False).to(device)
    model = load_checkpoint(args.weights, model, device)

    criterion = get_loss(cfg["loss"].get("name", "bce"), **cfg["loss"].get("params", {}))
    val_loss, metrics = evaluate(model, loader, criterion, device)

    print({"val_loss": val_loss, **metrics})


if __name__ == "__main__":
    main()
