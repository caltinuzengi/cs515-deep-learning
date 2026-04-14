"""Test evaluation, confusion matrix, and t-SNE visualisation utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from attacks import PGDAttack
from gradcam import GradCAM, get_target_layer
from train import get_transforms, load_cifar10c, CIFAR10C_CORRUPTIONS


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
                          num_classes: int, results_dir: str, exp_name: str) -> None:
    """Save a confusion matrix heatmap as PNG.

    Args:
        all_labels: Ground-truth labels.
        all_preds: Predicted labels.
        num_classes: Number of classes.
        results_dir: Directory where the PNG is saved.
        exp_name: Experiment name used for the file name.
    """
    os.makedirs(results_dir, exist_ok=True)
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(), labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=range(num_classes))

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path = os.path.join(results_dir, f"{exp_name}_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {path}")


def plot_tsne(features: np.ndarray, labels: np.ndarray,
              num_classes: int, results_dir: str, exp_name: str) -> None:
    """Run t-SNE on hidden features and save a 2-D scatter plot as PNG.

    Args:
        features: Feature matrix of shape ``(N, D)``.
        labels: Ground-truth labels of shape ``(N,)``.
        num_classes: Number of classes (used for colour map).
        results_dir: Directory where the PNG is saved.
        exp_name: Experiment name used for the file name.
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
    path = os.path.join(results_dir, f"{exp_name}_tsne.png")
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
    plot_confusion_matrix(all_labels, all_preds, params["num_classes"], params["results_dir"], params["exp_name"])
    print("Generating t-SNE plot...")
    plot_tsne(all_features, all_labels.numpy(), params["num_classes"], params["results_dir"], params["exp_name"])

    return correct / n


# ---------------------------------------------------------------------------
# HW2 — helper utilities
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _load_weights(model: nn.Module, save_path: str, device: torch.device) -> None:
    """Load model weights from *save_path* into *model* in-place.

    Handles both raw state-dict files and checkpoints that wrap the state
    dict under a ``model_state_dict`` key.

    Args:
        model: Target model.
        save_path: Path to the ``.pth`` checkpoint file.
        device: Device to map tensors to while loading.
    """
    state = torch.load(save_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Denormalise a CIFAR-10 image tensor to a uint8 (H, W, 3) array.

    Assumes CIFAR-10 normalisation: mean=(0.4914,0.4822,0.4465),
    std=(0.2023,0.1994,0.2010).

    Args:
        tensor: Single image tensor of shape (C, H, W).

    Returns:
        uint8 NumPy array of shape (H, W, 3).
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# HW2 Phase 5 — evaluation functions
# ---------------------------------------------------------------------------


def run_corruption_eval(
    model: nn.Module,
    params: dict,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate robustness on CIFAR-10-C across all corruptions and severities.

    Iterates over all 19 corruption types × 5 severity levels.  For each
    pair, computes top-1 accuracy on 10 000 examples.  Results are written
    to a CSV file and a summary is printed to stdout.

    Args:
        model: Trained model (weights will be loaded from ``params['save_path']``).
        params: Configuration dictionary including ``cifar10c_dir``,
            ``batch_size``, ``num_workers``, ``results_dir``, ``exp_name``.
        device: Torch device.

    Returns:
        Mapping from corruption name to mean accuracy across severities.
    """
    _load_weights(model, params["save_path"], device)

    csv_path = os.path.join(params["results_dir"], f"{params['exp_name']}_corruption_eval.csv")
    rows: list[dict] = []

    print("\n=== Corruption Evaluation ===")
    corruption_means: dict[str, float] = {}

    with torch.no_grad():
        for corruption in CIFAR10C_CORRUPTIONS:
            accs: list[float] = []
            for severity in range(1, 6):
                loader = load_cifar10c(params, corruption, severity)
                correct = total = 0
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = model(imgs).argmax(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)
                acc = correct / total
                accs.append(acc)
                rows.append({"corruption": corruption, "severity": severity, "accuracy": f"{acc:.4f}"})

            mean_acc = float(np.mean(accs))
            corruption_means[corruption] = mean_acc
            print(f"  {corruption:<25s}  mean_acc={mean_acc:.4f}  per_severity={[f'{a:.3f}' for a in accs]}")

    overall_mean = float(np.mean(list(corruption_means.values())))
    print(f"\n  Overall mean accuracy across all corruptions: {overall_mean:.4f}")

    os.makedirs(params["results_dir"], exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["corruption", "severity", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results saved to {csv_path}")

    return corruption_means


def run_adversarial_eval(
    model: nn.Module,
    params: dict,
    attack: PGDAttack,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate clean and adversarial accuracy on a subset of the test set.

    Uses the first ``attack_n_samples`` images from the CIFAR-10 test split.
    Adversarial examples are generated with the supplied *attack* instance.

    Args:
        model: Trained model (weights loaded from ``params['save_path']``).
        params: Configuration dict.  Uses ``attack_n_samples``,
            ``batch_size``, ``results_dir``, ``exp_name``.
        attack: Pre-configured ``PGDAttack`` instance.
        device: Torch device.

    Returns:
        Tuple ``(clean_acc, adv_acc)``.
    """
    _load_weights(model, params["save_path"], device)

    n_samples = params["attack_n_samples"]
    tf = get_transforms(params, train=False)
    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    else:
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)

    subset = Subset(test_ds, range(min(n_samples, len(test_ds))))
    loader = DataLoader(subset, batch_size=params["batch_size"], shuffle=False,
                        num_workers=params["num_workers"])

    clean_correct = adv_correct = total = 0

    print(f"\n=== Adversarial Evaluation ({params['attack_norm'].upper()}, ε={params['attack_eps']:.5f}, "
          f"steps={params['attack_steps']}, n={n_samples}) ===")

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Clean accuracy (no_grad)
        with torch.no_grad():
            clean_preds = model(imgs).argmax(1)
        clean_correct += clean_preds.eq(labels).sum().item()

        # Adversarial accuracy (attack manages its own grad context)
        x_adv = attack.perturb(imgs, labels)
        with torch.no_grad():
            adv_preds = model(x_adv).argmax(1)
        adv_correct += adv_preds.eq(labels).sum().item()

        total += labels.size(0)

    clean_acc = clean_correct / total
    adv_acc   = adv_correct   / total
    print(f"  Clean accuracy : {clean_acc:.4f}  ({clean_correct}/{total})")
    print(f"  Adv   accuracy : {adv_acc:.4f}  ({adv_correct}/{total})")
    print(f"  Accuracy drop  : {clean_acc - adv_acc:.4f}")

    out_path = os.path.join(params["results_dir"],
                            f"{params['exp_name']}_adversarial_eval.txt")
    os.makedirs(params["results_dir"], exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"norm={params['attack_norm']}  eps={params['attack_eps']:.6f}  "
                f"steps={params['attack_steps']}  n_samples={n_samples}\n")
        f.write(f"clean_acc={clean_acc:.4f}\n")
        f.write(f"adv_acc={adv_acc:.4f}\n")
        f.write(f"accuracy_drop={clean_acc - adv_acc:.4f}\n")
    print(f"  Results saved to {out_path}")

    return clean_acc, adv_acc


def run_gradcam_viz(
    model: nn.Module,
    params: dict,
    attack: PGDAttack,
    device: torch.device,
    n_samples: int = 4,
) -> None:
    """Visualise Grad-CAM maps for clean vs adversarial examples.

    Searches for examples that are correctly classified on clean input but
    misclassified under the PGD attack.  For each such example, a
    four-panel figure is saved:
    ``[clean image | clean GradCAM | adversarial image | adversarial GradCAM]``.

    The clean GradCAM uses the true class as the target; the adversarial
    GradCAM uses the (wrong) predicted class to highlight what the network
    is responding to.

    Args:
        model: Trained model (weights loaded from ``params['save_path']``).
        params: Configuration dict.  Uses ``model``, ``results_dir``,
            ``exp_name``, ``attack_n_samples``, ``batch_size``.
        attack: Pre-configured ``PGDAttack`` instance.
        device: Torch device.
        n_samples: Number of qualifying examples to visualise.
    """
    _load_weights(model, params["save_path"], device)

    target_layer = get_target_layer(model, params["model"])
    gc = GradCAM(model, target_layer)

    tf = get_transforms(params, train=False)
    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    else:
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)

    max_search = min(params["attack_n_samples"], len(test_ds))
    subset = Subset(test_ds, range(max_search))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    class_names = CIFAR10_CLASSES if params["dataset"] == "cifar10" else [str(i) for i in range(10)]

    collected: list[tuple] = []  # (clean_img, cam_clean, adv_img, cam_adv, true_cls, adv_cls)

    for imgs, labels in loader:
        if len(collected) >= n_samples:
            break

        imgs, labels = imgs.to(device), labels.to(device)
        true_cls = labels.item()

        # Check clean prediction
        with torch.no_grad():
            clean_pred = model(imgs).argmax(1).item()
        if clean_pred != true_cls:
            continue  # skip misclassified clean images

        # Generate adversarial example
        x_adv = attack.perturb(imgs, labels)
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(1).item()
        if adv_pred == true_cls:
            continue  # attack failed — not interesting

        # Grad-CAM (must be outside no_grad)
        cam_clean = gc(imgs, class_idx=true_cls)
        cam_adv   = gc(x_adv, class_idx=adv_pred)

        clean_img_np = _denorm(imgs.squeeze(0))
        adv_img_np   = _denorm(x_adv.squeeze(0).detach())

        collected.append((
            clean_img_np, GradCAM.overlay(clean_img_np, cam_clean),
            adv_img_np,   GradCAM.overlay(adv_img_np,   cam_adv),
            class_names[true_cls], class_names[adv_pred],
        ))

    gc.remove_hooks()

    if not collected:
        print("  No qualifying examples found for GradCAM visualisation.")
        return

    n_found = len(collected)
    fig, axes = plt.subplots(n_found, 4, figsize=(14, 3.5 * n_found))
    if n_found == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D indexing

    col_titles = ["Clean image", "Clean Grad-CAM", "Adv image", "Adv Grad-CAM"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, (cimg, ccam, aimg, acam, true_name, adv_name) in enumerate(collected):
        for col, img in enumerate([cimg, ccam, aimg, acam]):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"true: {true_name}\npred_adv: {adv_name}",
                                fontsize=9, rotation=0, labelpad=70, va="center")

    fig.suptitle(f"Grad-CAM: Clean vs Adversarial ({params['attack_norm'].upper()}, "
                 f"ε={params['attack_eps']:.4f})\nModel: {params['exp_name']}",
                 fontsize=12)
    fig.tight_layout()

    os.makedirs(params["results_dir"], exist_ok=True)
    out_path = os.path.join(params["results_dir"], f"{params['exp_name']}_gradcam.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM figure saved to {out_path}  ({n_found} examples)")


def run_tsne_adversarial(
    model: nn.Module,
    params: dict,
    attack: PGDAttack,
    device: torch.device,
) -> None:
    """t-SNE plot comparing penultimate-layer features for clean vs adversarial inputs.

    Collects penultimate layer features for ``attack_n_samples`` clean examples
    and their adversarial counterparts, then runs t-SNE and saves a scatter
    plot where clean samples are shown as circles (``o``) and adversarial
    samples as crosses (``x``), both coloured by ground-truth class.

    Args:
        model: Trained model (weights loaded from ``params['save_path']``).
        params: Configuration dict.  Uses ``model``, ``attack_n_samples``,
            ``results_dir``, ``exp_name``.
        attack: Pre-configured ``PGDAttack`` instance.
        device: Torch device.
    """
    _load_weights(model, params["save_path"], device)

    n_samples = params["attack_n_samples"]
    tf = get_transforms(params, train=False)
    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    else:
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)

    subset = Subset(test_ds, range(min(n_samples, len(test_ds))))
    loader = DataLoader(subset, batch_size=params["batch_size"], shuffle=False,
                        num_workers=params["num_workers"])

    # Hook penultimate layer
    penultimate = get_penultimate_layer(model, params["model"])
    feature_store: list[torch.Tensor] = []

    def _hook(_module, _inp, output):
        feat = output.detach().cpu()
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        feature_store.append(feat)

    hook_handle = penultimate.register_forward_hook(_hook)

    clean_feats:  list[torch.Tensor] = []
    adv_feats:    list[torch.Tensor] = []
    all_labels:   list[torch.Tensor] = []

    print(f"\n=== Adversarial t-SNE (n={n_samples}) ===")

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Clean forward
        feature_store.clear()
        with torch.no_grad():
            model(imgs)
        clean_feats.append(torch.cat(feature_store, dim=0).clone())
        all_labels.append(labels.cpu())

        # Adversarial forward
        x_adv = attack.perturb(imgs, labels)
        feature_store.clear()
        with torch.no_grad():
            model(x_adv)
        adv_feats.append(torch.cat(feature_store, dim=0).clone())

    hook_handle.remove()

    clean_np  = torch.cat(clean_feats).numpy()   # (N, D)
    adv_np    = torch.cat(adv_feats).numpy()      # (N, D)
    labels_np = torch.cat(all_labels).numpy()     # (N,)

    combined  = np.concatenate([clean_np, adv_np], axis=0)  # (2N, D)
    n = len(labels_np)

    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    emb = tsne.fit_transform(combined)  # (2N, 2)

    emb_clean = emb[:n]
    emb_adv   = emb[n:]

    cmap = plt.get_cmap("tab10")
    num_classes = params["num_classes"]

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in range(num_classes):
        mask = labels_np == cls
        color = cmap(cls / max(num_classes - 1, 1))
        ax.scatter(emb_clean[mask, 0], emb_clean[mask, 1],
                   color=color, marker="o", s=8,  alpha=0.5, label=f"{cls} clean")
        ax.scatter(emb_adv[mask, 0],   emb_adv[mask, 1],
                   color=color, marker="x", s=8,  alpha=0.5, label=f"{cls} adv")

    # Legend: one entry per class (merged markers)
    handles = []
    import matplotlib.lines as mlines
    for cls in range(num_classes):
        c = cmap(cls / max(num_classes - 1, 1))
        label = CIFAR10_CLASSES[cls] if params["dataset"] == "cifar10" else str(cls)
        handles.append(mlines.Line2D([], [], color=c, marker="o", linestyle="None",
                                     markersize=6, label=label))
    ax.legend(handles=handles, title="Class", loc="best", fontsize=7, ncol=2)

    # Marker legend
    h_clean = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                             markersize=7, label="clean (o)")
    h_adv   = mlines.Line2D([], [], color="gray", marker="x", linestyle="None",
                             markersize=7, label="adversarial (x)")
    ax.legend(handles=handles + [h_clean, h_adv], title="Class / Type",
              loc="best", fontsize=7, ncol=2)

    norm_str = params["attack_norm"].upper()
    ax.set_title(f"t-SNE: Clean (o) vs Adversarial (x) — {norm_str}, ε={params['attack_eps']:.4f}\n"
                 f"Model: {params['exp_name']}", fontsize=11)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()

    os.makedirs(params["results_dir"], exist_ok=True)
    out_path = os.path.join(params["results_dir"], f"{params['exp_name']}_tsne_adversarial.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  t-SNE adversarial plot saved to {out_path}")


def run_combined_adv_eval(
    model: nn.Module,
    params: dict,
    attack: PGDAttack,
    device: torch.device,
    n_gradcam: int = 2,
) -> tuple[float, float]:
    """Run adversarial accuracy, Grad-CAM, and t-SNE in a single PGD pass.

    Generates adversarial examples exactly once and reuses them for all three
    analyses, avoiding redundant PGD computation.  Produces three output files:

    * ``{exp_name}_adv_{norm}_accuracy.txt``  — clean and adversarial accuracy
    * ``{exp_name}_adv_{norm}_gradcam.png``   — clean vs. adversarial Grad-CAM
    * ``{exp_name}_adv_{norm}_tsne.png``      — t-SNE of clean (o) and adv (x)

    Args:
        model: Trained model (weights loaded from ``params['save_path']``).
        params: Configuration dict.  Uses ``attack_n_samples``, ``batch_size``,
            ``model``, ``results_dir``, ``exp_name``, ``attack_norm``,
            ``attack_eps``, ``attack_steps``.
        attack: Pre-configured ``PGDAttack`` instance.
        device: Torch device.
        n_gradcam: Number of misclassified adversarial examples to visualise
            with Grad-CAM (HW requires 1-2).

    Returns:
        Tuple ``(clean_acc, adv_acc)``.
    """
    _load_weights(model, params["save_path"], device)

    n_samples = params["attack_n_samples"]
    norm_str  = params["attack_norm"]
    tf = get_transforms(params, train=False)
    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    else:
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)

    subset = Subset(test_ds, range(min(n_samples, len(test_ds))))
    loader = DataLoader(subset, batch_size=params["batch_size"], shuffle=False,
                        num_workers=params["num_workers"])

    # --- Hook for penultimate features (t-SNE) ---
    penultimate = get_penultimate_layer(model, params["model"])
    feat_store: list[torch.Tensor] = []

    def _feat_hook(_mod, _inp, out):
        f = out.detach().cpu()
        if f.dim() > 2:
            f = f.view(f.size(0), -1)
        feat_store.append(f)

    hook_handle = penultimate.register_forward_hook(_feat_hook)

    # --- GradCAM setup ---
    target_layer = get_target_layer(model, params["model"])
    gc = GradCAM(model, target_layer)
    class_names = CIFAR10_CLASSES if params["dataset"] == "cifar10" else [str(i) for i in range(10)]

    # --- Accumulators ---
    clean_correct = adv_correct = total = 0
    clean_feats: list[torch.Tensor] = []
    adv_feats:   list[torch.Tensor] = []
    all_labels:  list[torch.Tensor] = []
    # GradCAM candidates: (clean_img_np, cam_clean_overlay, adv_img_np, cam_adv_overlay, true_name, adv_name)
    gradcam_collected: list[tuple] = []
    # store single-image tensors for deferred GradCAM after loop
    gradcam_candidates: list[tuple] = []  # (img_1hw, x_adv_1hw, true_cls, adv_pred)

    print(f"\n=== Combined Adversarial Eval ({norm_str.upper()}, "
          f"\u03b5={params['attack_eps']:.5f}, steps={params['attack_steps']}, n={n_samples}) ===")

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # --- Clean forward ---
        feat_store.clear()
        with torch.no_grad():
            clean_logits = model(imgs)
        clean_preds = clean_logits.argmax(1)
        clean_correct += clean_preds.eq(labels).sum().item()
        clean_feats.append(torch.cat(feat_store).clone())
        all_labels.append(labels.cpu())

        # --- PGD (one call per batch) ---
        x_adv = attack.perturb(imgs, labels)

        # --- Adv forward ---
        feat_store.clear()
        with torch.no_grad():
            adv_logits = model(x_adv)
        adv_preds = adv_logits.argmax(1)
        adv_correct += adv_preds.eq(labels).sum().item()
        adv_feats.append(torch.cat(feat_store).clone())

        total += labels.size(0)

        # --- Collect GradCAM candidates (process per sample) ---
        if len(gradcam_candidates) < n_gradcam:
            for i in range(imgs.size(0)):
                if len(gradcam_candidates) >= n_gradcam:
                    break
                if clean_preds[i].item() == labels[i].item() and adv_preds[i].item() != labels[i].item():
                    gradcam_candidates.append((
                        imgs[i:i+1].detach(),
                        x_adv[i:i+1].detach(),
                        labels[i].item(),
                        adv_preds[i].item(),
                    ))

    hook_handle.remove()

    clean_acc = clean_correct / total
    adv_acc   = adv_correct   / total
    print(f"  Clean accuracy : {clean_acc:.4f}  ({clean_correct}/{total})")
    print(f"  Adv   accuracy : {adv_acc:.4f}  ({adv_correct}/{total})")
    print(f"  Accuracy drop  : {clean_acc - adv_acc:.4f}")

    os.makedirs(params["results_dir"], exist_ok=True)
    prefix = os.path.join(params["results_dir"], f"{params['exp_name']}_adv_{norm_str}")

    # --- Save accuracy ---
    with open(f"{prefix}_accuracy.txt", "w") as f:
        f.write(f"norm={norm_str}  eps={params['attack_eps']:.6f}  "
                f"steps={params['attack_steps']}  n_samples={n_samples}\n")
        f.write(f"clean_acc={clean_acc:.4f}\n")
        f.write(f"adv_acc={adv_acc:.4f}\n")
        f.write(f"accuracy_drop={clean_acc - adv_acc:.4f}\n")
    print(f"  Accuracy results saved to {prefix}_accuracy.txt")

    # --- GradCAM visualisation ---
    for img_t, adv_t, true_cls, adv_pred in gradcam_candidates:
        img_t = img_t.to(device)
        adv_t = adv_t.to(device)
        cam_c = gc(img_t, class_idx=true_cls)
        cam_a = gc(adv_t, class_idx=adv_pred)
        clean_np = _denorm(img_t.squeeze(0).cpu())
        adv_np   = _denorm(adv_t.squeeze(0).cpu())
        gradcam_collected.append((
            clean_np, GradCAM.overlay(clean_np, cam_c),
            adv_np,   GradCAM.overlay(adv_np,   cam_a),
            class_names[true_cls], class_names[adv_pred],
        ))
    gc.remove_hooks()

    if gradcam_collected:
        n_found = len(gradcam_collected)
        fig, axes = plt.subplots(n_found, 4, figsize=(14, 3.5 * n_found))
        if n_found == 1:
            axes = axes[np.newaxis, :]
        col_titles = ["Clean image", "Clean Grad-CAM", "Adv image", "Adv Grad-CAM"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=11, fontweight="bold")
        for row, (cimg, ccam, aimg, acam, tname, aname) in enumerate(gradcam_collected):
            for col, im in enumerate([cimg, ccam, aimg, acam]):
                axes[row, col].imshow(im)
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(f"true: {tname}\npred_adv: {aname}",
                                    fontsize=9, rotation=0, labelpad=70, va="center")
        fig.suptitle(f"Grad-CAM: Clean vs Adversarial ({norm_str.upper()}, "
                     f"\u03b5={params['attack_eps']:.4f})\nModel: {params['exp_name']}",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{prefix}_gradcam.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grad-CAM saved to {prefix}_gradcam.png  ({n_found} examples)")
    else:
        print("  No qualifying examples for Grad-CAM found.")

    # --- t-SNE ---
    clean_np_f  = torch.cat(clean_feats).numpy()
    adv_np_f    = torch.cat(adv_feats).numpy()
    labels_np   = torch.cat(all_labels).numpy()
    combined_f  = np.concatenate([clean_np_f, adv_np_f], axis=0)
    N = len(labels_np)

    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, N - 1), random_state=42, init="pca")
    emb  = tsne.fit_transform(combined_f)
    emb_c, emb_a = emb[:N], emb[N:]

    import matplotlib.lines as mlines
    cmap = plt.get_cmap("tab10")
    num_classes = params["num_classes"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in range(num_classes):
        mask  = labels_np == cls
        color = cmap(cls / max(num_classes - 1, 1))
        ax.scatter(emb_c[mask, 0], emb_c[mask, 1], color=color, marker="o", s=8, alpha=0.5)
        ax.scatter(emb_a[mask, 0], emb_a[mask, 1], color=color, marker="x", s=8, alpha=0.5)

    class_handles = [
        mlines.Line2D([], [], color=cmap(c / max(num_classes-1, 1)), marker="o",
                      linestyle="None", markersize=6,
                      label=CIFAR10_CLASSES[c] if params["dataset"] == "cifar10" else str(c))
        for c in range(num_classes)
    ]
    type_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=7, label="clean (o)"),
        mlines.Line2D([], [], color="gray", marker="x", linestyle="None",
                      markersize=7, label="adversarial (x)"),
    ]
    ax.legend(handles=class_handles + type_handles, title="Class / Type",
              loc="best", fontsize=7, ncol=2)
    ax.set_title(f"t-SNE: Clean (o) vs Adversarial (x) \u2014 {norm_str.upper()}, "
                 f"\u03b5={params['attack_eps']:.4f}\nModel: {params['exp_name']}", fontsize=11)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(f"{prefix}_tsne.png", dpi=150)
    plt.close(fig)
    print(f"  t-SNE saved to {prefix}_tsne.png")

    return clean_acc, adv_acc


def run_transferability_eval(
    student: nn.Module,
    teacher: nn.Module,
    params: dict,
    attack: PGDAttack,
    device: torch.device,
) -> tuple[float, float]:
    """Test adversarial transferability: craft examples on teacher, evaluate on student.

    Generates PGD adversarial examples using *teacher* as the surrogate model,
    then measures how well these examples fool the *student* model.  Both
    student and teacher must already have weights loaded before calling this
    function.

    Output file: ``{exp_name}_adv_{norm}_transferability.txt``

    Args:
        student: Student model with weights loaded, in eval mode.
        teacher: Teacher (surrogate) model with weights loaded, in eval mode.
        params: Configuration dict.  Uses ``attack_n_samples``, ``batch_size``,
            ``results_dir``, ``exp_name``, ``attack_norm``, ``attack_eps``,
            ``attack_steps``.
        attack: Pre-configured ``PGDAttack`` whose ``model`` attribute points
            to the *teacher*.  The attack will be moved to eval mode inside.
        device: Torch device.

    Returns:
        Tuple ``(student_clean_acc, student_adv_trans_acc)``.
    """
    n_samples = params["attack_n_samples"]
    norm_str  = params["attack_norm"]
    tf = get_transforms(params, train=False)
    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    else:
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)

    subset = Subset(test_ds, range(min(n_samples, len(test_ds))))
    loader = DataLoader(subset, batch_size=params["batch_size"], shuffle=False,
                        num_workers=params["num_workers"])

    clean_correct = trans_correct = total = 0

    print(f"\n=== Adversarial Transferability ({norm_str.upper()}, "
          f"\u03b5={params['attack_eps']:.5f}, steps={params['attack_steps']}, n={n_samples}) ===")
    print("  Crafting on teacher, evaluating on student...")

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Student clean accuracy
        with torch.no_grad():
            clean_preds = student(imgs).argmax(1)
        clean_correct += clean_preds.eq(labels).sum().item()

        # Craft adversarial examples on teacher
        x_adv = attack.perturb(imgs, labels)

        # Evaluate on student
        with torch.no_grad():
            trans_preds = student(x_adv).argmax(1)
        trans_correct += trans_preds.eq(labels).sum().item()

        total += labels.size(0)

    clean_acc = clean_correct / total
    trans_acc = trans_correct / total
    print(f"  Student clean accuracy          : {clean_acc:.4f}  ({clean_correct}/{total})")
    print(f"  Student accuracy (teacher adv)  : {trans_acc:.4f}  ({trans_correct}/{total})")
    print(f"  Transferability drop            : {clean_acc - trans_acc:.4f}")

    os.makedirs(params["results_dir"], exist_ok=True)
    out_path = os.path.join(params["results_dir"],
                            f"{params['exp_name']}_adv_{norm_str}_transferability.txt")
    with open(out_path, "w") as f:
        f.write(f"norm={norm_str}  eps={params['attack_eps']:.6f}  "
                f"steps={params['attack_steps']}  n_samples={n_samples}\n")
        f.write("[teacher adv samples evaluated on student]\n")
        f.write(f"student_clean_acc={clean_acc:.4f}\n")
        f.write(f"student_adv_transfer_acc={trans_acc:.4f}\n")
        f.write(f"transferability_drop={clean_acc - trans_acc:.4f}\n")
    print(f"  Results saved to {out_path}")

    return clean_acc, trans_acc