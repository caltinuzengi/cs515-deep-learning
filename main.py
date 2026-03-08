"""The main function for training and evaluating the MLP Classifier."""
import csv
import os

import random
import numpy as np
import torch

from parameters import get_params
from models.MLP import MLP
from train import run_training
from test  import run_test


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params):
    """Define and build the model specified *params*."""
    model_name = params["model"]
    dataset    = params["dataset"]
    nc         = params["num_classes"]

    if model_name == "mlp":
        return MLP(
            input_size   = params["input_size"],
            hidden_sizes = params["hidden_sizes"],
            num_classes  = nc,
            dropout      = params["dropout"],
            activation   = params["activation"],
            use_batchnorm= params["use_batchnorm"],
        )

    raise ValueError(f"Unknown model: {model_name}")


def main():
    """Parse arguments, build model, train and evaluate."""
    params = get_params()

    exp_dir = os.path.join(params["results_dir"], params["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)
    params["results_dir"] = exp_dir
    params["save_path"] = os.path.join(exp_dir, "best_model.pth")

    set_seed(params["seed"])
    print(f"Seed set to: {params['seed']}")
    print(f"Dataset: {params['dataset']}  |  Model: {params['model']}")

    device = torch.device(
        params["device"] if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(params).to(device)
    print(model)

    best_val_acc = 0.0
    if params["mode"] in ("train", "both"):
        history = run_training(model, params, device)
        best_val_acc = max(history["val_acc"])

    test_acc = 0.0
    if params["mode"] in ("test", "both"):
        test_acc = run_test(model, params, device)

    log_to_csv(params, best_val_acc, test_acc)


# This function makes it easier for us to track ablation results.
def log_to_csv(params: dict, best_val_acc: float, test_acc: float) -> None:
    """Append one row of experiment results to ablation_results.csv.
    This function makes it easier for us to track ablation results.
    """
    csv_path = os.path.join(params["results_dir"], "..", "ablation_results.csv")
    file_exists = os.path.exists(csv_path)

    row = {
        "exp_name": params["exp_name"],
        "hidden_sizes": str(params["hidden_sizes"]),
        "activation": params["activation"],
        "use_batchnorm": params["use_batchnorm"],
        "dropout": params["dropout"],
        "optimizer": params["optimizer"],
        "lr_scheduler": params["lr_scheduler"],
        "weight_decay": params["weight_decay"],
        "l1_lambda": params["l1_lambda"],
        "epochs": params["epochs"],
        "best_val_acc": f"{best_val_acc:.4f}",
        "test_acc": f"{test_acc:.4f}",
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Results appended to {csv_path}")


if __name__ == "__main__":
    main()