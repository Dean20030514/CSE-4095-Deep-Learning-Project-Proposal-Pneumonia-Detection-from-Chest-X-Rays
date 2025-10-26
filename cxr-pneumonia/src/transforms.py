from typing import Tuple

import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224, aug: str = "light") -> Tuple[T.Compose, T.Compose]:
    """
    Build torchvision transforms for train/val.

    aug: "none" | "light" | "medium"
    """
    if aug not in {"none", "light", "medium"}:
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
    else:  # medium
        train_aug.extend([
            T.Resize(int(image_size * 1.2)),
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=7),
        ])

    common = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    train_tfms = T.Compose(train_aug + common)
    val_tfms = T.Compose([
        T.Resize((image_size, image_size)),
        *common,
    ])

    return train_tfms, val_tfms
