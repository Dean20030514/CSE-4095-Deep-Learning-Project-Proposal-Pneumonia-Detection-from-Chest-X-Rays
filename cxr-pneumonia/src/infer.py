from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from PIL import Image
import yaml

from transforms import get_transforms
from models import create_model, load_checkpoint
from utils import get_device


def list_images(path: str) -> List[str]:
    if os.path.isdir(path):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        imgs: List[str] = []
        for root, _dirs, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    imgs.append(os.path.join(root, f))
        return sorted(imgs)
    else:
        return [path]


@torch.no_grad()
def predict_images(model, image_paths: List[str], image_size: int, device: torch.device) -> List[float]:
    _, val_tfms = get_transforms(image_size=image_size, aug="none")
    probs: List[float] = []
    for p in image_paths:
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = val_tfms(img).unsqueeze(0).to(device)
        logits = model(x).squeeze(-1)
        prob = torch.sigmoid(logits).item()
        probs.append(prob)
    return probs


def main():
    parser = argparse.ArgumentParser(description="Inference for CXR Pneumonia")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image path or directory")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    model = create_model(name=cfg["model"].get("name", "resnet18"), num_classes=1, pretrained=False).to(device)
    model = load_checkpoint(args.weights, model, device)
    model.eval()

    image_paths = list_images(args.input)
    probs = predict_images(model, image_paths, cfg["data"].get("image_size", 224), device)

    for p, prob in zip(image_paths, probs):
        print(f"{p},pneumonia_prob={prob:.4f}")

    if args.output:
        import pandas as pd
        pd.DataFrame({"path": image_paths, "pneumonia_prob": probs}).to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
