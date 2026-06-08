import copy
import os
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

_HW4_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_HW4_DIR)
for _p in (_HW4_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config


class Trainer:
    def __init__(self, model, optimizer, criterion,
                 mode='regression', patience=config.PATIENCE, ckpt_path=None,
                 early_stop_metric='loss'):
        assert mode in ('regression', 'classification')
        assert early_stop_metric in ('loss', 'f1')
        if early_stop_metric == 'f1' and mode != 'classification':
            raise ValueError("early_stop_metric='f1' requires mode='classification'")
        self.model             = model
        self.optimizer         = optimizer
        self.criterion         = criterion
        self.mode              = mode
        self.patience          = patience
        self.ckpt_path         = ckpt_path
        self.early_stop_metric = early_stop_metric
        self.device            = next(model.parameters()).device
        self._best_state       = None

    def fit(self, train_loader, val_loader, max_epochs=config.MAX_EPOCHS):
        use_f1 = self.early_stop_metric == 'f1'
        best_val = float('-inf') if use_f1 else float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        best_epoch = 0

        if use_f1:
            print('Early stopping criterion: val F1 (higher is better)')

        for epoch in range(1, max_epochs + 1):
            # --- train ---
            self.model.train()
            total_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item() * len(X)
            train_loss = total_loss / len(train_loader.dataset)

            # --- val ---
            val_loss = self._eval_loss(val_loader)
            val_f1   = self._eval_f1(val_loader) if use_f1 else None

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            improved = (val_f1 > best_val) if use_f1 else (val_loss < best_val)
            if improved:
                best_val = val_f1 if use_f1 else val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                self._best_state = copy.deepcopy(self.model.state_dict())
                if self.ckpt_path:
                    torch.save(self._best_state, self.ckpt_path)
            else:
                epochs_no_improve += 1

            marker = '*' if improved else ' '
            if use_f1:
                print(f'Epoch {epoch:3d}/{max_epochs} {marker} | '
                      f'train={train_loss:.6f} | val={val_loss:.6f} | '
                      f'val_f1={val_f1:.4f} | best={best_epoch}')
            else:
                print(f'Epoch {epoch:3d}/{max_epochs} {marker} | '
                      f'train={train_loss:.6f} | val={val_loss:.6f} | best={best_epoch}')

            if epochs_no_improve >= self.patience:
                criterion_label = 'val F1' if use_f1 else 'val loss'
                print(f'Early stopping at epoch {epoch} [{criterion_label}] (best={best_epoch})')
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return {'train_losses': train_losses, 'val_losses': val_losses, 'best_epoch': best_epoch}

    def _eval_loss(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                total_loss += self.criterion(self.model(X), y).item() * len(X)
        return total_loss / len(loader.dataset)

    def _eval_f1(self, loader):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                logits = self.model(X).cpu()
                probs = torch.sigmoid(logits).squeeze(1)
                all_preds.append((probs > 0.5).int())
                all_targets.append(y.squeeze(1).int())
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        return float(f1_score(y_true, y_pred, zero_division=0))

    def evaluate(self, test_loader, threshold=0.5):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                all_preds.append(self.model(X).cpu())
                all_targets.append(y)

        preds   = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        if self.mode == 'regression':
            per_horizon = ((preds - targets) ** 2).mean(dim=0).tolist()
            return {
                'per_horizon_mse': per_horizon,
                'mean_mse':        sum(per_horizon) / len(per_horizon),
            }

        # classification
        probs  = torch.sigmoid(preds).squeeze(1)
        y_pred = (probs > threshold).int().numpy()
        y_true = targets.squeeze(1).int().numpy()

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return {
            'accuracy':         float(accuracy_score(y_true, y_pred)),
            'precision':        float(prec),
            'recall':           float(rec),
            'f1':               float(f1),
            'confusion_matrix': cm.tolist(),
        }


def plot_losses(history, save_path):
    fig, ax = plt.subplots()
    ax.plot(history['train_losses'], label='train')
    ax.plot(history['val_losses'],   label='val')
    ax.axvline(history['best_epoch'] - 1, color='gray', linestyle='--', linewidth=0.8,
               label=f"best epoch {history['best_epoch']}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
