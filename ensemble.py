"""
Model Ensemble Example on CIFAR-10
------------------------------------
1. Fine-tunes ResNet-18, MobileNet-V2, VGG-16 for 5 epochs each
2. Evaluates each model individually
3. Combines them via soft and hard voting ensemble
"""

import warnings
import numpy as np
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader


# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else
                      "cpu")
print(f"Using device: {device}")


# ── Data ───────────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2)


# ── Model builders ─────────────────────────────────────────────────────────────
def load_resnet18(num_classes=10):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device)

def load_mobilenet_v2(num_classes=10):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m.to(device)

def load_vgg16(num_classes=10):
    m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m.to(device)


# ── Fine-tuning ────────────────────────────────────────────────────────────────
def fine_tune(model, model_name, epochs=10):
    """Fine-tune a model on CIFAR-10 and save the best weights."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_acc     = 0.0
    save_path    = f"best_{model_name}.pth"

    print(f"\n── Fine-tuning {model_name} for {epochs} epochs ──")
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct      += out.argmax(1).eq(labels).sum().item()
            total        += imgs.size(0)
        scheduler.step()

        train_acc  = correct / total
        train_loss = running_loss / total

        # ---- validate ----
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += model(imgs).argmax(1).eq(labels).sum().item()
                val_total   += imgs.size(0)
        val_acc = val_correct / val_total

        print(f"  Epoch {epoch}/{epochs}  "
              f"train loss: {train_loss:.4f}  train acc: {train_acc:.4f}  "
              f"val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best weights (val_acc={best_acc:.4f})")

    # Reload best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"  Best val acc for {model_name}: {best_acc:.4f}")
    return model


# ── Ensemble class ─────────────────────────────────────────────────────────────
class EnsembleModel(nn.Module):
    """
    Wraps multiple models and combines their predictions.

    Soft voting averages class probabilities (softmax outputs) across all
    models.  This is generally preferred over hard voting because it accounts
    for each model's confidence rather than just its top-1 pick.
    """
    def __init__(self, models_list):
        super().__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, x):
        probs = [torch.softmax(m(x), dim=1) for m in self.models]
        return torch.stack(probs, dim=0).mean(dim=0)   # (B, C)


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, strategy="soft"):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if strategy == "soft":
            preds = model(imgs).argmax(dim=1)

        elif strategy == "hard":
            votes = torch.stack(
                [m(imgs).argmax(dim=1) for m in model.models], dim=0
            )  # (num_models, B)
            # torch.mode not supported on MPS → run on CPU
            preds = votes.cpu().mode(dim=0).values.to(device)

        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

    return correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load pretrained backbones
    resnet    = load_resnet18()
    mobilenet = load_mobilenet_v2()
    vgg       = load_vgg16()

    # 2. Fine-tune each for 10 epochs
    resnet    = fine_tune(resnet,    "resnet18",     epochs=10)
    mobilenet = fine_tune(mobilenet, "mobilenet_v2", epochs=10)
    vgg       = fine_tune(vgg,       "vgg16",        epochs=10)

    # 3. Individual accuracy after fine-tuning
    print("\nIndividual model accuracies (after fine-tuning):")
    for name, m in [("ResNet-18", resnet), ("MobileNet-V2", mobilenet), ("VGG-16", vgg)]:
        acc = evaluate(m, test_loader, strategy="soft")
        print(f"  {name:<15}: {acc:.4f}")

    # 4. Ensemble
    ensemble = EnsembleModel([resnet, mobilenet, vgg])
    print("\nEnsemble accuracies:")
    print(f"  Soft voting : {evaluate(ensemble, test_loader, strategy='soft'):.4f}")
    print(f"  Hard voting : {evaluate(ensemble, test_loader, strategy='hard'):.4f}")


    """
Soft vs Hard Voting
--------------------
Soft voting averages the full probability distributions across models.
Hard voting lets each model cast a single vote for its top-1 class.

They produce the same result most of the time, but diverge when:
some models are weakly confident in one direction, and one model
is strongly confident in another direction.

Case 1: Soft voting CORRECT, Hard voting WRONG
    True label = dog

               cat   dog   bird
    ResNet   : [0.36, 0.34, 0.30]  -> votes cat (barely)
    MobileNet: [0.36, 0.34, 0.30]  -> votes cat (barely)
    VGG      : [0.05, 0.90, 0.05]  -> votes dog (confidently)

    Hard voting: cat (2 vs 1)          WRONG  -- two uncertain models outvote one confident model
    Soft voting: dog (0.527 vs 0.257)  RIGHT  -- VGG's confidence shifts the average

Case 2: Hard voting CORRECT, Soft voting WRONG
    True label = cat

               cat   dog   bird
    ResNet   : [0.40, 0.35, 0.25]  -> votes cat (barely)
    MobileNet: [0.40, 0.35, 0.25]  -> votes cat (barely)
    VGG      : [0.05, 0.90, 0.05]  -> votes dog (confidently)

    Hard voting: cat (2 vs 1)          RIGHT  -- two correct models outvote one wrong model
    Soft voting: dog (0.533 vs 0.283)  WRONG  -- VGG's wrong confidence drags the average

Takeaway:
    Soft voting is generally preferred because well-trained models tend to be
    confident when they are right more often than when they are wrong.
    However, a single overconfident wrong model can still hurt soft voting,
    while hard voting is immune to confidence levels entirely.
"""