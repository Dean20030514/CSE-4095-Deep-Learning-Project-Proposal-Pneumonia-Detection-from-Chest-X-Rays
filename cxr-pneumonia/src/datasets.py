from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Sample:
    path: str
    label: int
    id: Optional[str] = None


def _scan_folder(image_root: str) -> List[Sample]:
    """
    Expect a folder with two subfolders: 'normal' and 'pneumonia' (case-insensitive).
    """
    classes = {"normal": 0, "pneumonia": 1}
    samples: List[Sample] = []
    for cls_name, label in classes.items():
        cls_dir = os.path.join(image_root, cls_name)
        if not os.path.isdir(cls_dir):
            # try capitalized variants
            for alt in [cls_name.capitalize(), cls_name.upper()]:
                alt_dir = os.path.join(image_root, alt)
                if os.path.isdir(alt_dir):
                    cls_dir = alt_dir
                    break
        if not os.path.isdir(cls_dir):
            continue
        for root, _dirs, files in os.walk(cls_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTS:
                    samples.append(Sample(path=os.path.join(root, f), label=label))
    return samples


class CXRDataset(Dataset):
    """
    A dataset for chest X-ray binary classification (Pneumonia vs Normal).

    Supports:
    1) CSV file with columns: path,label[,id]
    2) Image folder with class subfolders: normal/ and pneumonia/
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        image_root: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        assert (csv_path is not None) or (image_root is not None), "Provide csv_path or image_root"
        self.transform = transform

        if csv_path is not None:
            df = pd.read_csv(csv_path)
            if not {"path", "label"}.issubset(df.columns):
                raise ValueError("CSV must contain columns: path,label[,(optional) id]")
            self.samples: List[Sample] = [
                Sample(path=row["path"], label=int(row["label"]), id=str(row.get("id")) if "id" in df.columns else None)
                for _, row in df.iterrows()
            ]
        else:
            self.samples = _scan_folder(image_root)  # type: ignore[arg-type]

        if len(self.samples) == 0:
            raise RuntimeError("No images found. Check CSV paths or image_root structure.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = Image.open(s.path)
        # Ensure RGB 3-channel for ImageNet-pretrained backbones
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, s.label
