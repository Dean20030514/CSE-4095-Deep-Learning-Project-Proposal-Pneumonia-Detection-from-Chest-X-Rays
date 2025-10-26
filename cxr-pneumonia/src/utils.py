from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.cnt = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n

    @property
    def avg(self) -> float:
        return float(self.sum) / max(1, self.cnt)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    auroc: float
    precision: float
    recall: float
    f1: float


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return acc, auroc, precision, recall, f1


def save_checkpoint(state: Dict, out_dir: str, filename: str = "best.ckpt") -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    return path
