"""Multi-Layer Perceptron (MLP) model for classification tasks."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected feed-forward network with configurable depth and width.

    Architecture::
        Flatten → [Linear → (BN) → Activation → Dropout] x N → Linear(output)

    Args:
        input_size: Dimensionality of the flattened input (e.g. 784 for MNIST).
        hidden_sizes: Width of each hidden layer.
        num_classes: Number of output classes.
        dropout: Dropout probability applied after each hidden block.
        use_batchnorm: If ``True``, insert ``BatchNorm1d`` before activation.
        activation: Activation function name (``"relu"`` or ``"gelu"``).
    """
    def __init__(self, input_size: int, hidden_sizes: list[int],
                 num_classes: int, dropout: float = 0.3,
                 use_batchnorm: bool = True, activation: str = "relu") -> None:
        super().__init__()

        # Doing this before forward we aim to flatten the input
        # (B, 1, 28, 28) → (B, 784)
        layers = [nn.Flatten()]
        in_dim = input_size
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h) if use_batchnorm else nn.Identity(),
                act_fn(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        Args:
            x: Input tensor of shape ``(B, C, H, W)`` or ``(B, D)``.
        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        return self.net(x)