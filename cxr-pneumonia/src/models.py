from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tvm


def _init_linear(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0, 0.01)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def create_model(name: str = "resnet18", num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    name = name.lower()

    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        _init_linear(m.fc)
        return m
    elif name == "resnet34":
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        _init_linear(m.fc)
        return m
    elif name in {"efficientnet_b0", "efficientnet-b0", "effnet_b0"}:
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        _init_linear(m.classifier[-1])
        return m
    else:
        raise ValueError(f"Unsupported model name: {name}")


def load_checkpoint(weights_path: str, model: nn.Module, device: Optional[torch.device] = None) -> nn.Module:
    ckpt = torch.load(weights_path, map_location=device or "cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    return model
