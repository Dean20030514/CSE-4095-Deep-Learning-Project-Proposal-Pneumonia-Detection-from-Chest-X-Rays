from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with logits input.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, 1) or (N,), targets: (N,) in {0,1}
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        loss = bce * ((1 - pt) ** self.gamma)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss(name: str = "bce", **kwargs) -> nn.Module:
    name = name.lower()
    if name in {"bce", "bcelogits", "bcewithlogits"}:
        return nn.BCEWithLogitsLoss(**kwargs)
    if name in {"focal", "focalloss"}:
        return FocalLoss(**kwargs)
    raise ValueError(f"Unknown loss: {name}")
