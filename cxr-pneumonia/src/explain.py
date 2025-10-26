from __future__ import annotations

import argparse
import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
import yaml

from models import create_model, load_checkpoint
from transforms import get_transforms
from utils import get_device

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception as e:
    GradCAM = None  # type: ignore
    show_cam_on_image = None  # type: ignore


def get_target_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
    # For ResNet/EfficientNet typical last conv block
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Unable to find a suitable target layer for Grad-CAM")


def generate_gradcam(image_path: str, model: torch.nn.Module, image_size: int = 224, out_path: Optional[str] = None) -> str:
    assert GradCAM is not None, "pytorch-grad-cam is required. Please install it."
    device = next(model.parameters()).device
    model.eval()

    _, val_tfms = get_transforms(image_size=image_size, aug="none")
    img_pil = Image.open(image_path)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    rgb_img = np.array(img_pil).astype(np.float32) / 255.0
    input_tensor = val_tfms(img_pil).unsqueeze(0).to(device)

    target_layer = get_target_layer(model, "auto")
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == "cuda"))
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    if out_path is None:
        base, ext = os.path.splitext(image_path)
        out_path = base + "_gradcam.png"
    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap for a CXR image")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    model = create_model(name=cfg["model"].get("name", "resnet18"), num_classes=1, pretrained=False).to(device)
    model = load_checkpoint(args.weights, model, device)

    out_path = generate_gradcam(args.image, model, cfg["data"].get("image_size", 224), args.output)
    print(f"Grad-CAM saved to: {out_path}")


if __name__ == "__main__":
    main()
