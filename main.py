"""The main function for training and evaluating models."""
import csv
import os

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

from parameters import get_params
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.ResNet import ResNet, BasicBlock
from models.VGG import VGG
from models.mobilenet import MobileNetV2
from train import run_training, run_kd_training, run_teacher_prob_training
from test  import run_test


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_pretrained_resnet(params: dict) -> nn.Module:
    """Build a pretrained ResNet-18 adapted for CIFAR-10.

    Two strategies:
        - ``resize``:  Keep original conv1 (7×7), freeze early layers optionally.
        - ``modify_conv``: Replace conv1 with 3×3, remove maxpool, fine-tune all.
    """
    nc = params["num_classes"]
    strategy = params["transfer_strategy"]
    freeze = params["freeze_layers"]

    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

    if strategy == "modify_conv":
        # Replace 7×7 conv with 3×3 for 32×32 input (weights can't transfer)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    # Replace final FC for num_classes
    model.fc = nn.Linear(model.fc.in_features, nc)

    # Freeze everything except fc (and conv1 if modified)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def _build_pretrained_vgg(params: dict) -> nn.Module:
    """Build a pretrained VGG-16 adapted for CIFAR-10.

    Two strategies:
        - ``resize``:  Keep original features, freeze early layers optionally.
        - ``modify_conv``: Replace first conv for 32×32 input, fine-tune all.
    """
    nc = params["num_classes"]
    strategy = params["transfer_strategy"]
    freeze = params["freeze_layers"]

    model = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)

    if strategy == "modify_conv":
        # Replace first conv for native 32×32 input
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    # Replace final classifier layer for num_classes
    model.classifier[6] = nn.Linear(4096, nc)

    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def build_model(params: dict) -> nn.Module:
    """Build the model specified by *params*."""
    model_name        = params["model"]
    dataset           = params["dataset"]
    nc                = params["num_classes"]
    pretrained        = params["pretrained"]

    #---------------------------------------
    # MLP
    #---------------------------------------
    if model_name == "mlp":
        return MLP(
            input_size   = params["input_size"],
            hidden_sizes = params["hidden_sizes"],
            num_classes  = nc,
            dropout      = params["dropout"],
            activation   = params["activation"],
            use_batchnorm= params["use_batchnorm"],
        )

    #----------------------------------------  
    # CNN
    #----------------------------------------
    if model_name == "cnn":
        if dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        return SimpleCNN(num_classes=nc)

    #----------------------------------------  
    # ResNet
    #----------------------------------------
    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10.")
        if pretrained:
            return _build_pretrained_resnet(params)
        return ResNet(BasicBlock, params["resnet_layers"], num_classes=nc)

    #----------------------------------------  
    # VGG16
    #----------------------------------------
    if model_name == "vgg16":
        if dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use cifar10.")
        if pretrained:
            return _build_pretrained_vgg(params)
        return VGG(depth=params["vgg_depth"], num_class=nc)

    #----------------------------------------  
    # MobileNetV2
    #----------------------------------------
    if model_name == "mobilenet":
        if dataset == "mnist":
            raise ValueError("MobileNetV2 is designed for 3-channel images; use cifar10.")
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")


def _build_teacher(params: dict, device: torch.device) -> nn.Module:
    """Load a pre-trained teacher model from a checkpoint for KD."""
    teacher_params = dict(params)  
    teacher_params["model"] = params["teacher_model"]
    teacher_params["pretrained"] = False          
    teacher_params["transfer_strategy"] = "none"

    teacher = build_model(teacher_params).to(device)
    state = torch.load(params["teacher_path"], map_location=device)
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher ({params['teacher_model']}) loaded from {params['teacher_path']}")
    return teacher


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

    # device = torch.device(
    #     params["device"] if torch.cuda.is_available() else
    #     "mps" if torch.backends.mps.is_available() else
    #     "cpu"
    # )
    if params["device"] != "auto":
        device = torch.device(params["device"])
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    model = build_model(params).to(device)
    print(model)

    # ── Training dispatch ──
    training_mode = params.get("training_mode", "standard")
    best_val_acc = 0.0

    if params["mode"] in ("train", "both"):
        if training_mode == "distillation":
            teacher = _build_teacher(params, device)
            history = run_kd_training(model, teacher, params, device)
        elif training_mode == "teacher_prob":
            teacher = _build_teacher(params, device)
            history = run_teacher_prob_training(model, teacher, params, device)
        else:
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
        "model": params["model"],
        "training_mode": params.get("training_mode", "standard"),
        "pretrained": params.get("pretrained", False),
        "transfer_strategy": params.get("transfer_strategy", "none"),
        "label_smoothing": params.get("label_smoothing", 0.0),
        "kd_temperature": params.get("kd_temperature", ""),
        "kd_alpha": params.get("kd_alpha", ""),
        "optimizer": params["optimizer"],
        "lr_scheduler": params["lr_scheduler"],
        "learning_rate": params["learning_rate"],
        "weight_decay": params["weight_decay"],
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