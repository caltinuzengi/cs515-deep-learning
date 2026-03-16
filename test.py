"""Test evaluation, confusion matrix, and t-SNE visualisation utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import get_transforms


def get_penultimate_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the layer whose output will be used for t-SNE.

    Args:
        model: The network.
        model_name: One of ``mlp``, ``cnn``, ``resnet``, ``vgg16``, ``mobilenet``.

    Returns:
        A reference to the penultimate layer/module.
    """
    if model_name == "mlp":
        return list(model.net.children())[-2]

    if model_name == "cnn":
        # SimpleCNN / MNIST_CNN — hook on fc1 (last hidden before output)
        return model.fc1

    if model_name == "resnet":
        # Both scratch ResNet and pretrained ResNet-18 have avgpool
        return model.avgpool

    if model_name == "vgg16":
        # classifier = [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear]
        # Hook on classifier[4] = second ReLU (4096-dim features)
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            return model.classifier[4]
        return model.features  # fallback for scratch VGG

    if model_name == "mobilenet":
        # MobileNetV2 (scratch) — hook on bn2 (before avg_pool2d in forward)
        return model.bn2

    raise ValueError(f"Cannot determine penultimate layer for model: {model_name}")


def plot_confusion_matrix(all_labels: torch.Tensor, all_preds: torch.Tensor,
                          num_classes: int, results_dir: str) -> None:
    """Save a confusion matrix heatmap as PNG.

    Args:
        all_labels: Ground-truth labels.
        all_preds: Predicted labels.
        num_classes: Number of classes.
        results_dir: Directory where the PNG is saved.
    """
    os.makedirs(results_dir, exist_ok=True)
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(), labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=range(num_classes))

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path = os.path.join(results_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {path}")


def plot_tsne(features: np.ndarray, labels: np.ndarray,
              num_classes: int, results_dir: str) -> None:
    """Run t-SNE on hidden features and save a 2-D scatter plot as PNG.

    Args:
        features: Feature matrix of shape ``(N, D)``.
        labels: Ground-truth labels of shape ``(N,)``.
        num_classes: Number of classes (used for colour map).
        results_dir: Directory where the PNG is saved.
    """
    os.makedirs(results_dir, exist_ok=True)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=labels, cmap="tab10", s=5, alpha=0.6,
    )
    ax.legend(*scatter.legend_elements(), title="Classes", loc="best", fontsize=8)
    ax.set_title("t-SNE of Penultimate Layer Features")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    path = os.path.join(results_dir, "tsne.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  t-SNE plot saved to {path}")

@torch.no_grad()
def run_test(model: nn.Module, params: dict, device: torch.device) -> float:
    """Evaluate a trained model and generate confusion matrix + t-SNE plots.

    Args:
        model: Trained network.
        params: Configuration dictionary.
        device: Torch device.

    Returns:
        Overall test accuracy.
    """
    tf = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    else:  # cifar10
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)

    loader = DataLoader(test_ds, batch_size=params["batch_size"],
                        shuffle=False, num_workers=params["num_workers"])

    model.load_state_dict(torch.load(params["save_path"], map_location=device))
    model.eval()

    # This is for confusion matrix and t-SNE
    # Since for that we need all the predictions and labels

    # --- Hook to capture penultimate layer features for t-SNE ---
    feature_store: list[torch.Tensor] = []

    def _hook_fn(_module, _input, output):
        # avgpool and similar layers may return (B, C, 1, 1) — flatten to (B, C)
        feat = output.cpu()
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        feature_store.append(feat)

    penultimate = get_penultimate_layer(model, params["model"])
    hook_handle = penultimate.register_forward_hook(_hook_fn)

    # --- Collect predictions and labels ---
    all_preds = []
    all_labels = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        # Using cpu to store the predictions and labels 
        # Detach from GPU and move to CPU before stroing, since
        # numpy and list operations require cpu tensors
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    hook_handle.remove()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_features = torch.cat(feature_store).numpy()

    # --- Print accuracy ---
    correct = all_preds.eq(all_labels).sum().item()
    n = len(all_labels)

    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {correct/n:.4f}  ({correct}/{n})\n")
    for i in range(params["num_classes"]):
        mask = (all_labels == i)
        class_acc = all_preds[mask].eq(i).sum().item() / mask.sum().item()
        print(f"  Class {i}: {class_acc:.4f}  ({all_preds[mask].eq(i).sum().item()}/{mask.sum().item()})")

    # --- Plots ---
    print("Generating confusion matrix...")
    plot_confusion_matrix(all_labels, all_preds, params["num_classes"], params["results_dir"])
    print("Generating t-SNE plot...")
    plot_tsne(all_features, all_labels.numpy(), params["num_classes"], params["results_dir"])

    return correct / n