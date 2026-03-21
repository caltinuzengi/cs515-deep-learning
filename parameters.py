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
import dataclasses
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
    val_ratio: float
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass
class ModelParams:
    """Deep Learning Model architecture settings."""

    model: str
    input_size: int
    hidden_sizes: list[int]
    num_classes: int
    dropout: float
    activation: str
    use_batchnorm: bool
    vgg_depth: str = "16"
    resnet_layers: list[int] = field(default_factory=lambda: [2, 2, 2, 2])  # Default to ResNet-18

    # def __post_init__(self):
    #     # Number of blocks per layer, e.g. [2,2,2,2] for ResNet-18
    #     if self.resnet_layers is None:
    #         self.resnet_layers = [2, 2, 2, 2]


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

@dataclass
class TransferParams:
    """Transfer learning settings."""
    
    transfer_strategy: str = "none"     # none | resize | modify_conv
    pretrained: bool = False            # Whether to use pretrained weights
    freeze_layers: bool = False         # Whether to freeze early layers in transfer learning

    # To avoid from silent bug when freeze_layers=True but transfer_strategy=modify_conv, we can add a post-init check
    def __post_init__(self):
        if self.transfer_strategy == "modify_conv" and self.freeze_layers:
            raise ValueError("Cannot freeze layers when using 'modify_conv' transfer strategy, as the architecture is changed.")

@dataclass
class DistillationParams:
    """Knowledge distillation settings."""

    training_mode: str = "standard"     # standard | distillation | teacher_prob
    teacher_path: str = ""              # Path to the teacher model for distillation
    teacher_model: str = "resnet"       # Architecture of the teacher model for distillation
    kd_temperature: float = 3.0         # Temperature for knowledge distillation
    kd_alpha: float = 0.7               # Alpha for balancing distillation loss and standard loss
    label_smoothing: float = 0.0        # Label smoothing factor (0.0 = no smoothing)

    def __post_init__(self):
        if self.training_mode in ("distillation", "teacher_prob") and not self.teacher_path:
            raise ValueError(f"teacher_path must be provided for {self.training_mode} training mode.")
        if not (0.0 <= self.kd_alpha <= 1.0):
            raise ValueError(f"kd_alpha must be in the range [0.0, 1.0]. Got: {self.kd_alpha}")
        if self.kd_temperature <= 0.0:
            raise ValueError(f"kd_temperature must be greater than 0.0. Got: {self.kd_temperature}")



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

    parser = argparse.ArgumentParser(description="CS515-Deep Learning on MNIST and CIFAR-10")

    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset",   choices=["mnist", "cifar10"],      default="mnist")
    parser.add_argument("--model",     choices=["mlp", "cnn", "resnet", "vgg16", "mobilenet"], default="mlp")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--batch_size",type=int,   default=64)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation split ratio taken from the official train split.")
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

    parser.add_argument("--transfer_strategy", type=str, choices=["none", "resize", "modify_conv"], default="none")
    parser.add_argument("--pretrained", type=lambda v: v.lower() in ("true", "1", "yes"), default=False,
                        help="Whether to use pretrained weights for transfer learning.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor (0.0 = no smoothing).")
    parser.add_argument("--training_mode", type=str, choices=["standard", "distillation", "teacher_prob"], default="standard",
                        help="Training mode: standard | distillation | teacher_prob.")
    parser.add_argument("--teacher_path", type=str, default="",
                        help="Path to the teacher model for distillation.")
    parser.add_argument("--teacher_model", type=str, choices=["mlp", "cnn", "resnet", "vgg16", "mobilenet"], default="resnet",
                        help="Architecture of the teacher model for distillation.")
    parser.add_argument("--kd_temperature", type=float, default=3.0, help="Temperature for knowledge distillation.")
    parser.add_argument("--kd_alpha", type=float, default=0.7, help="Alpha for balancing distillation loss and standard loss.")
    
    parser.add_argument("--vgg_depth", type=str, choices=["11", "13", "16", "19"], default="16",
                        help="Depth of VGG model: 11 | 13 | 16 | 19.")
    parser.add_argument("--freeze_layers", type=lambda v: v.lower() in ("true", "1", "yes"), default=False,
                        help="Freeze early layers in transfer learning.")
    parser.add_argument("--resnet_layers", type=int, nargs="+", default=[2, 2, 2, 2],
                        help="Number of blocks per ResNet layer, e.g. 2 2 2 2 for ResNet-18.")

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
        num_workers=2, val_ratio=args.val_ratio, mean=mean, std=std,
    )
    model = ModelParams(
        model=args.model, 
        input_size=input_size,
        hidden_sizes=args.hidden_sizes, 
        num_classes=10,
        dropout=args.dropout, 
        activation=args.activation,
        use_batchnorm=args.use_batchnorm,
        vgg_depth=args.vgg_depth,
        resnet_layers=args.resnet_layers,
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

    transfer = TransferParams(
        transfer_strategy=args.transfer_strategy,
        pretrained=args.pretrained,
        freeze_layers=args.freeze_layers,
    )

    distillation = DistillationParams(
        training_mode=args.training_mode,
        teacher_path=args.teacher_path,
        teacher_model=args.teacher_model,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,
        label_smoothing=args.label_smoothing,
    )

    # Merge all dataclass fields into a single flat dict
    merged: dict = {}
    for dc in (data, model, train, misc, transfer, distillation):
        merged.update(dataclasses.asdict(dc))

    return merged