import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for imbalanced classification (accepts logits).

    FL(x, y) = alpha_t * (1 - p_t)^fl_gamma * BCE(x, y)

    where p_t = sigmoid(x) if y=1, else 1-sigmoid(x).
    alpha_t = alpha if y=1, else 1-alpha.

    Args:
        alpha: positive-class weight in [0, 1]. Default 0.75 (favours minority).
        fl_gamma: focusing exponent. Default 2.0.
    """

    def __init__(self, alpha: float = 0.75, fl_gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.fl_gamma = fl_gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t) ** self.fl_gamma * bce).mean()
