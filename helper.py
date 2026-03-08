"""Generate visualisations of the MLP architecture.

Produces two PNG files under ``results/``:
1. **Computational graph** with *torchviz* (``mlp_computational_graph.png``).
2. **Architecture (block) diagram** with *matplotlib* (``mlp_architecture.png``).
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchviz import make_dot

from models.MLP import MLP
from parameters import get_params

LAYER_COLORS: dict[str, str] = {
    "Flatten":     "#6C8EBF",
    "Linear":      "#D6A4E0",
    "BatchNorm1d": "#82C4A0",
    "ReLU":        "#F4A261",
    "GELU":        "#F4A261",
    "Dropout":     "#E76F51",
    "Identity":    "#CCCCCC",
}
DEFAULT_COLOR = "#AAAAAA"


def _layer_label(layer: torch.nn.Module) -> str:
    """Return a human-readable label for *layer*."""
    name = layer.__class__.__name__
    if isinstance(layer, torch.nn.Linear):
        return f"Linear({layer.in_features} → {layer.out_features})"
    if isinstance(layer, torch.nn.BatchNorm1d):
        return f"BatchNorm1d({layer.num_features})"
    if isinstance(layer, torch.nn.Dropout):
        return f"Dropout(p={layer.p})"
    return name


def draw_architecture(model: MLP, save_path: str) -> None:
    """Draw a vertical block diagram of every layer in *model.net*."""
    layers = list(model.net)
    n = len(layers)

    box_h = 0.6
    gap = 0.15
    total_h = n * box_h + (n - 1) * gap
    fig_w, fig_h = 6, max(4, total_h + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Title
    ax.text(fig_w / 2, fig_h - 0.4, "MLP Architecture",
            ha="center", va="top", fontsize=14, fontweight="bold")

    top = fig_h - 1.0  # starting y for the first box
    box_w = 4.5
    x0 = (fig_w - box_w) / 2

    for i, layer in enumerate(layers):
        y = top - i * (box_h + gap)
        name = layer.__class__.__name__
        color = LAYER_COLORS.get(name, DEFAULT_COLOR)

        rect = mpatches.FancyBboxPatch(
            (x0, y - box_h), box_w, box_h,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#333333", linewidth=1.2, alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(x0 + box_w / 2, y - box_h / 2, _layer_label(layer),
                ha="center", va="center", fontsize=10, color="white",
                fontweight="bold")

        # Arrow between consecutive boxes
        if i < n - 1:
            ax.annotate("", xy=(fig_w / 2, y - box_h - gap),
                        xytext=(fig_w / 2, y - box_h),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__" or True:    
    params = get_params()

    model = MLP(
        input_size=params["input_size"],
        hidden_sizes=params["hidden_sizes"],
        num_classes=params["num_classes"],
        activation=params["activation"],
        dropout=params["dropout"],
        use_batchnorm=params["use_batchnorm"],
    )
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)

    os.makedirs("results", exist_ok=True)

    # 1. Computational graph (torchviz)
    make_dot(y, params=dict(model.named_parameters())).render(
        "results/mlp_computational_graph", format="png",
    )
    print("Computational graph  → results/mlp_computational_graph.png")

    # 2. Architecture block diagram (matplotlib)
    draw_architecture(model, "results/mlp_architecture.png")
    print("Architecture diagram → results/mlp_architecture.png")