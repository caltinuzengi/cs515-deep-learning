"""PGD adversarial attack implementation.

Implements the Projected Gradient Descent (PGD) attack from Madry et al.
for both L∞ and L2 norm constraints.

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial
    Attacks", ICLR 2018.  https://openreview.net/forum?id=rJzIBfZAb

Usage::

    from attacks import PGDAttack

    attacker = PGDAttack(model, norm="linf", eps=4/255, steps=20)
    x_adv = attacker.perturb(x, y)   # x: normalized tensor (B,C,H,W)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PGDAttack:
    """Projected Gradient Descent adversarial attack.

    Supports L∞ and L2 norm constraints.  Step size defaults to the Madry
    formula ``2.5 * eps / steps``, which is well-calibrated for PGD-20.

    The attack operates entirely in the *normalized* input space — the same
    space that the model expects.  The perturbation ``delta`` is constrained
    to ``[-eps, eps]`` (L∞) or ``‖delta‖₂ ≤ eps`` (L2), and the adversarial
    example is not clamped to ``[0, 1]`` because the input may lie outside
    that range after normalization.  This is the standard approach when the
    model is wrapped with dataset-specific normalization.

    Args:
        model: The target model to attack.  Must be callable with a batch of
            tensors and return logits.
        norm: Threat model — ``"linf"`` or ``"l2"``.
        eps: Perturbation budget in the *original* (un-normalized) pixel scale
            (e.g. ``4/255`` for L∞ or ``0.25`` for L2).
        steps: Number of PGD iterations.
        step_size: Per-step perturbation size.  If ``≤ 0``, defaults to the
            Madry formula ``2.5 * eps / steps``.
        random_start: If ``True``, initialise delta with a uniform random
            perturbation inside the constraint set.
    """

    def __init__(
        self,
        model: nn.Module,
        norm: str,
        eps: float,
        steps: int = 20,
        step_size: float = -1.0,
        random_start: bool = True,
    ) -> None:
        if norm not in ("linf", "l2"):
            raise ValueError(f"norm must be 'linf' or 'l2', got '{norm}'.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}.")

        self.model = model
        self.norm = norm
        self.eps = eps
        self.steps = steps
        self.step_size = step_size if step_size > 0 else 2.5 * eps / steps
        self.random_start = random_start

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples for a batch of inputs.

        The model is kept in ``eval()`` mode throughout.  Gradients are only
        computed with respect to the input, not the model parameters.

        Args:
            x: Clean input batch of shape ``(B, C, H, W)``, already
                normalized to the model's expected input range.
            y: Ground-truth labels of shape ``(B,)``.

        Returns:
            Adversarial examples of the same shape as ``x``.
        """
        self.model.eval()

        # Initialise perturbation
        delta = torch.zeros_like(x)
        if self.random_start:
            if self.norm == "linf":
                delta.uniform_(-self.eps, self.eps)
            else:  # l2
                delta = torch.randn_like(x)
                delta = self._project_l2(delta, self.eps)
        delta = delta.to(x.device)
        delta.requires_grad_(True)

        criterion = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            logits = self.model(x + delta)
            loss = criterion(logits, y)
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.detach()

                if self.norm == "linf":
                    delta.data = delta.data + self.step_size * grad.sign()
                    delta.data = self._project_linf(delta.data, self.eps)
                else:  # l2
                    # Normalise gradient to unit L2 norm per sample
                    grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)
                    grad_norm = grad_norm.view(-1, 1, 1, 1).clamp(min=1e-12)
                    delta.data = delta.data + self.step_size * (grad / grad_norm)
                    delta.data = self._project_l2(delta.data, self.eps)

            delta.grad.zero_()

        return (x + delta).detach()

    def _project_linf(self, delta: torch.Tensor, eps: float) -> torch.Tensor:
        """Project *delta* onto the L∞ ball of radius *eps*.

        Args:
            delta: Perturbation tensor of any shape.
            eps: L∞ radius.

        Returns:
            Clamped perturbation tensor with the same shape.
        """
        return delta.clamp(-eps, eps)

    def _project_l2(self, delta: torch.Tensor, eps: float) -> torch.Tensor:
        """Project *delta* onto the L2 ball of radius *eps* per sample.

        Each sample in the batch is projected independently.

        Args:
            delta: Perturbation tensor of shape ``(B, ...)``.
            eps: L2 radius.

        Returns:
            Projected perturbation tensor with the same shape.
        """
        batch_size = delta.size(0)
        norms = delta.view(batch_size, -1).norm(p=2, dim=1)
        # Only project samples that exceed the radius
        factor = (eps / norms.clamp(min=eps)).view(-1, *([1] * (delta.dim() - 1)))
        return delta * factor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"norm={self.norm}, "
            f"eps={self.eps:.6f}, "
            f"steps={self.steps}, "
            f"step_size={self.step_size:.6f}, "
            f"random_start={self.random_start})"
        )
