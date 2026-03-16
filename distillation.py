"""Knowledge distillation loss functions.

Implements:
    1. Standard KD loss (Hinton et al., 2015)
    2. Teacher-probability soft labels (difficulty-based)
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Compute the standard knowledge distillation loss.

    Combines a soft KL-divergence term (teacher's dark knowledge) with
    a hard cross-entropy term (ground-truth labels).

    .. math::
        L = \\alpha \\cdot T^2 \\cdot D_{KL}(\\sigma(z_s/T) \\| \\sigma(z_t/T))
            + (1 - \\alpha) \\cdot CE(z_s, y)

    Args:
        student_logits: Raw logits from the student, shape ``(B, C)``.
        teacher_logits: Raw logits from the teacher, shape ``(B, C)``.
        labels: Ground-truth class indices, shape ``(B,)``.
        temperature: Softmax temperature (higher = softer distributions).
        alpha: Weight for the soft loss (``1 - alpha`` for the hard loss).

    Returns:
        Scalar loss tensor.
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)

    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


def teacher_prob_soft_labels(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build soft labels using the teacher's confidence on the true class.

    For each sample the true-class probability comes from the teacher's
    softmax output; the remaining probability mass is spread equally
    across the other classes.  This assigns a *difficulty score* to each
    example — low teacher confidence means a harder example.

    Args:
        teacher_logits: Raw logits from the teacher, shape ``(B, C)``.
        labels: Ground-truth class indices, shape ``(B,)``.
        num_classes: Total number of classes (C).

    Returns:
        Soft label tensor of shape ``(B, C)`` that sums to 1 per row.
    """
    teacher_probs = F.softmax(teacher_logits, dim=1)  # (B, C)
    p_true = teacher_probs.gather(1, labels.unsqueeze(1))  # (B, 1)

    # Equal share for non-true classes
    soft = (1.0 - p_true) / (num_classes - 1)  # (B, 1)
    soft = soft.expand_as(teacher_probs).clone()

    # Assign teacher's true-class probability
    soft.scatter_(1, labels.unsqueeze(1), p_true)

    return soft


def teacher_prob_loss(
    student_logits: torch.Tensor,
    soft_labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy between student's log-softmax and teacher-derived soft labels.

    Args:
        student_logits: Raw logits from the student, shape ``(B, C)``.
        soft_labels: Soft target distribution, shape ``(B, C)``.

    Returns:
        Scalar loss tensor.
    """
    log_probs = F.log_softmax(student_logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()
