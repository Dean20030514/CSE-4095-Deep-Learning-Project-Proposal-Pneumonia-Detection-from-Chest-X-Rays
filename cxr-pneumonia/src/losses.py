from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with logits input.

    Args:
        alpha: class balancing factor in [0,1]; weight for positive class is alpha, negative is 1-alpha.
        gamma: focusing parameter.
        weight: Optional class weights tensor/list [w_neg, w_pos] applied after focal term.
        reduction: 'none' | 'mean' | 'sum'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight: torch.Tensor | list | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, 1) or (N,), targets: (N,) in {0,1}
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        loss = bce * ((1 - pt) ** self.gamma)
        # alpha weighting
        alpha_t = torch.where(targets == 1, torch.as_tensor(self.alpha, device=logits.device), torch.as_tensor(1 - self.alpha, device=logits.device))
        loss = alpha_t * loss
        # class weights [w_neg, w_pos]
        if self.weight is not None:
            w_neg = self.weight[0].to(logits.device)
            w_pos = self.weight[1].to(logits.device)
            w_t = torch.where(targets == 1, w_pos, w_neg)
            loss = loss * w_t
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
