"""Command-line argument parsing and parameter management.

Uses ``argparse`` for CLI control and ``dataclasses`` for structured
parameter grouping.  The public :func:`get_params` function returns a flat
dictionary consumed by the rest of the codebase.

Example::

    python main.py --activation gelu --dropout 0.6 --hidden_sizes 512 256
    python main.py --lr_scheduler plateau --l1_lambda 1e-4 --epochs 15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataclasses — structured documentation for each parameter group
# ---------------------------------------------------------------------------

@dataclass
class DataParams:
    """Dataset and data-loading settings."""

    dataset: str
    data_dir: str
    num_workers: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass
class ModelParams:
    """MLP architecture settings."""

    model: str
    input_size: int
    hidden_sizes: list[int]
    num_classes: int
    dropout: float
    activation: str
    use_batchnorm: bool


@dataclass
class TrainParams:
    """Training loop hyper-parameters."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    l1_lambda: float
    lr_scheduler: str
    optimizer: str = "adam"
    early_stop_patience: int = 5
    log_interval: int = 100


@dataclass
class MiscParams:
    """Run-time and I/O settings."""

    seed: int
    device: str
    save_path: str
    mode: str
    results_dir: str
    exp_name: str = "default"


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def get_params() -> dict:
    """Parse command-line arguments and return a flat configuration dict.

    Parsed values are grouped into dataclass instances for documentation,
    then merged into a single dictionary so that callers can use simple
    key look-ups (e.g. ``params["learning_rate"]``).

    Returns:
        dict: Flat mapping of every parameter name to its value.
    """

    parser = argparse.ArgumentParser(description="CS515-Deep Learning on MNIST")

    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset",   choices=["mnist", "cifar10"],      default="mnist")
    parser.add_argument("--model",     choices=["mlp"], default="mlp")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--batch_size",type=int,   default=64)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--activation", type=str, choices=["relu", "gelu"], default="relu")
    parser.add_argument("--use_batchnorm",
                        type=lambda v: v.lower() in ("true", "1", "yes"),
                        default=True)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--l1_lambda", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, choices=["step", "cosine", "plateau"], default="step")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default="adam",
                        help="Optimizer: adam | adamw | sgd (momentum=0.9).")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name for organizing results.")

    args = parser.parse_args()

    # Dataset-dependent settings
    if args.dataset == "mnist":
        input_size = 784          # 1 × 28 × 28
        mean, std  = (0.1307,), (0.3081,)
    else:                         # cifar10
        input_size = 3072         # 3 × 32 × 32
        mean       = (0.4914, 0.4822, 0.4465)
        std        = (0.2023, 0.1994, 0.2010)

    # Populate dataclasses (serves as living documentation)
    data = DataParams(
        dataset=args.dataset, data_dir="./data",
        num_workers=2, mean=mean, std=std,
    )
    model = ModelParams(
        model=args.model, 
        input_size=input_size,
        hidden_sizes=args.hidden_sizes, 
        num_classes=10,
        dropout=args.dropout, 
        activation=args.activation,
        use_batchnorm=args.use_batchnorm,
    )
    train = TrainParams(
        epochs=args.epochs, 
        batch_size=args.batch_size,
        learning_rate=args.lr, 
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda, 
        lr_scheduler=args.lr_scheduler,
        optimizer=args.optimizer,
        early_stop_patience=args.early_stop_patience, 
        log_interval=100,
    )
    misc = MiscParams(
        seed=42, 
        device=args.device,
        save_path=args.save_path, 
        mode=args.mode,
        results_dir=args.results_dir,
        exp_name=args.exp_name,
    )

    # Merge all dataclass fields into a single flat dict
    merged: dict = {}
    for dc in (data, model, train, misc):
        for key, value in dc.__dict__.items():
            merged[key] = value

    return merged