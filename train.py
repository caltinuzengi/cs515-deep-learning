"""Training utilities for the MLP classification pipeline.

Provides data loading, single-epoch training with optional L1
regularisation, validation, LR scheduling, early stopping, and
plotly-based training curve visualisation.
"""

import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(params: dict, train: bool = True) -> transforms.Compose:
    """Build a torchvision transform pipeline for the chosen dataset.

    Args:
        params: Configuration dictionary (needs ``dataset``, ``mean``, ``std``).
        train: If ``True``, include data-augmentation transforms.

    Returns:
        A composed transform pipeline.
    """
    mean, std = params["mean"], params["std"]

    if params["dataset"] == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


def get_loaders(params: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.

    Args:
        params: Configuration dictionary.

    Returns:
        A ``(train_loader, val_loader)`` tuple.
    """
    train_tf = get_transforms(params, train=True)
    val_tf   = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        train_ds = datasets.MNIST(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(params["data_dir"], train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True,  num_workers=params["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"],
                              shuffle=False, num_workers=params["num_workers"])
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
    l1_lambda: float = 0.0,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: Network to train.
        loader: Training data loader.
        optimizer: Optimiser instance.
        criterion: Loss function.
        device: Torch device.
        log_interval: Print stats every *n* batches.
        l1_lambda: L1 regularisation coefficient (0 = disabled).

    Returns:
        ``(average_loss, accuracy)`` over the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        
        # L1 regularization
        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a validation / test set.

    Args:
        model: Network to evaluate.
        loader: Validation data loader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        ``(average_loss, accuracy)`` over the dataset.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def plot_training_history_png(history: dict, results_dir: str) -> None:
    """Save training curves as a PNG image using matplotlib.

    Creates a figure with two subplots (loss and accuracy) and saves
    it to ``results_dir/training_history.png``.

    Args:
        history: Dictionary with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``.
        results_dir: Directory where the PNG file is saved.
    """
    os.makedirs(results_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # -- Loss --
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "o-", label="Val Loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training History")
    ax1.grid(True, alpha=0.3)

    # -- Accuracy --
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "o-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(results_dir, "training_history.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Training plots saved to {path}")


def plot_training_history(history: dict, results_dir: str) -> None:
    """Save interactive plotly charts of the training history.

    Generates a single HTML file with two subplots (loss and accuracy)
    showing both training and validation curves.

    Args:
        history: Dictionary with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``, and ``lr``.
        results_dir: Directory where the HTML file is saved.
    """
    os.makedirs(results_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Loss", "Accuracy"),
        vertical_spacing=0.12,
    )

    # -- Loss subplot --
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
                             name="Train Loss", mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                             name="Val Loss", mode="lines+markers"), row=1, col=1)

    # -- Accuracy subplot --
    fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"],
                             name="Train Acc", mode="lines+markers"), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"],
                             name="Val Acc", mode="lines+markers"), row=2, col=1)

    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_layout(title="Training History", height=700, template="plotly_white")

    path = os.path.join(results_dir, "training_history.html")
    fig.write_html(path)
    print(f"  Training plots saved to {path}")


def run_training(model: nn.Module, params: dict, device: torch.device) -> dict:
    """Full training loop with early stopping, LR scheduling, and plotting.

    Args:
        model: Network to train.
        params: Configuration dictionary.
        device: Torch device.

    Returns:
        History dictionary with per-epoch metrics.
    """
    history = {"train_loss": [], "train_acc": [], 
                "val_loss": [], "val_acc": [],
                "lr": []}

    train_loader, val_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()

    # -- Configurable optimizer --
    if params["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=params["learning_rate"],
                                      weight_decay=params["weight_decay"])
    elif params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=params["learning_rate"],
                                    momentum=0.9,
                                    weight_decay=params["weight_decay"])
    else:  # adam (default)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=params["learning_rate"],
                                     weight_decay=params["weight_decay"])
    
    if params["lr_scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif params["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
    elif params["lr_scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    else:
        scheduler = None

    best_acc     = 0.0
    best_weights = None

    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(1, params["epochs"] + 1):
        print(f"\nEpoch {epoch}/{params['epochs']}")
        tr_loss, tr_acc = train_one_epoch(model = model, 
                                            loader = train_loader, optimizer = optimizer,
                                            criterion = criterion, device = device, 
                                            log_interval = params["log_interval"], l1_lambda = params["l1_lambda"])
        val_loss, val_acc = validate(model = model, loader = val_loader, criterion = criterion, device = device)
        # scheduler.step()
        if params["lr_scheduler"] == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params["save_path"])
            print(f"  Saved best model (val_acc={best_acc:.4f})")
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= params["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")

    plot_training_history_png(history, params["results_dir"])
    # plot_training_history(history, params["results_dir"])  # interactive HTML
    return history