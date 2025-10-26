from typing import Tuple

import torch
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GaussianNoise:
    """Additive Gaussian noise on a tensor image (expects [0,1] range).

    Applied after ToTensor and before Normalize. Intensity is mild by default for medical images.
    """

    def __init__(self, std: float = 0.01):
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        noise = torch.randn_like(x) * self.std
        x = x + noise
        return x.clamp(0.0, 1.0)


def get_transforms(image_size: int = 224, aug: str = "light", resize_size: int | None = None) -> Tuple[T.Compose, T.Compose]:
    """
    Build torchvision transforms for train/val.

    aug: "none" | "light" | "medium"
    """
    if aug not in {"none", "light", "medium", "medical"}:
        aug = "light"

    train_aug = []
    if aug == "none":
        train_aug.extend([
            T.Resize((image_size, image_size)),
        ])
    elif aug == "light":
        train_aug.extend([
            T.Resize(int(image_size * 1.15)),
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
        ])
    elif aug == "medium":
        train_aug.extend([
            T.Resize(int(image_size * 1.2)),
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=7),
        ])
    else:  # medical: conservative aug for chest X-rays
        # Defaults per request: Resize(512) -> RandomResizedCrop(448) -> HFlip -> Rotate(±7°)
        # -> Brightness/Contrast (light) -> GaussianNoise (light) -> Normalize
        target_size = image_size if image_size is not None else 448
        resize_sz = resize_size if resize_size is not None else 512
        train_aug.extend([
            T.Resize(resize_sz),
            T.RandomResizedCrop(target_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=7),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    # Build common tail. For medical aug we insert mild Gaussian noise before Normalize.
    if aug == "medical":
        common = [T.ToTensor(), GaussianNoise(std=0.01), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    else:
        common = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    train_tfms = T.Compose(train_aug + common)
    if aug == "medical":
        # Val/Test: Resize(512) -> CenterCrop(448) -> Normalize
        target_size = image_size if image_size is not None else 448
        resize_sz = resize_size if resize_size is not None else 512
        val_tfms = T.Compose([
            T.Resize(resize_sz),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        val_tfms = T.Compose([
            T.Resize((image_size, image_size)),
            *common,
        ])

    return train_tfms, val_tfms
