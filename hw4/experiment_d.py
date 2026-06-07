import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HW4_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_HW4_DIR)
for _p in (_HW4_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from data_pipeline import get_loaders
from losses import FocalLoss
from trainer import Trainer, plot_losses
from models.BiStockLSTM import BiStockLSTM
from models.BiStockGRU import BiStockGRU


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_run_tag(gamma, val_metric, loss_type='bce'):
    tag = f'g{int(round(gamma * 100)):03d}_{val_metric}'
    if loss_type != 'bce':
        tag += f'_{loss_type}'
    return tag


def compute_pos_weight(train_loader):
    all_labels = train_loader.dataset.y
    num_pos = int(all_labels.sum().item())
    num_neg = len(all_labels) - num_pos
    if num_pos == 0:
        print(
            f'WARNING: no positive samples in training set '
            f'(GAMMA={config.GAMMA} threshold yields 0 buy signals). '
            f'Using pos_weight=1.0 — classification will be degenerate.'
        )
        return torch.tensor([1.0])
    pos_weight = torch.tensor([num_neg / num_pos])
    print(f'class balance — pos: {num_pos} ({100*num_pos/len(all_labels):.1f}%)  '
          f'neg: {num_neg} ({100*num_neg/len(all_labels):.1f}%)  '
          f'pos_weight: {pos_weight.item():.2f}')
    return pos_weight


def print_confusion_matrix(cm, model_name):
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f'\n  [{model_name}] Confusion Matrix:')
    print(f'               Pred 0   Pred 1')
    print(f'  Actual 0   {tn:7d}  {fp:7d}')
    print(f'  Actual 1   {fn:7d}  {tp:7d}')


def run(model_name, model, train_loader, val_loader, test_loader,
        pos_weight, device, retrain, run_tag, patience, early_stop_metric,
        loss_type='bce', fl_gamma=2.0, fl_alpha=0.75):
    ckpt_path   = os.path.join(config.CKPT_DIR,   f'part_d_{run_tag}_{model_name}.pt')
    plot_path   = os.path.join(config.PLOT_DIR,   f'part_d_{run_tag}_{model_name}_loss.png')
    metric_path = os.path.join(config.METRIC_DIR, f'part_d_{run_tag}_{model_name}.json')

    model = model.to(device)

    history = None
    if not retrain and os.path.exists(ckpt_path):
        print(f'[{model_name}] checkpoint found — skipping training')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f'[{model_name}] training '
              f'(gamma={config.GAMMA}, patience={patience}, '
              f'val_metric={early_stop_metric}, loss={loss_type})...')
        if loss_type == 'focal':
            criterion = FocalLoss(alpha=fl_alpha, fl_gamma=fl_gamma).to(device)
            print(f'  FocalLoss: alpha={fl_alpha}, fl_gamma={fl_gamma} '
                  f'(pos_weight not used — alpha handles class imbalance)')
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        trainer = Trainer(
            model, optimizer, criterion,
            mode='classification', patience=patience, ckpt_path=ckpt_path,
            early_stop_metric=early_stop_metric,
        )
        history = trainer.fit(train_loader, val_loader, max_epochs=config.MAX_EPOCHS)
        plot_losses(history, plot_path)
        print(f'[{model_name}] loss curve → {plot_path}')

    model.eval()
    criterion_eval = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    trainer_eval = Trainer(model, None, criterion_eval, mode='classification')
    metrics = trainer_eval.evaluate(test_loader)

    print_confusion_matrix(metrics['confusion_matrix'], model_name)

    result = {
        'model':            model_name,
        'run_tag':          run_tag,
        'gamma':            config.GAMMA,
        'patience':         patience,
        'val_metric':       early_stop_metric,
        'loss_type':        loss_type,
        'fl_gamma':         fl_gamma if loss_type == 'focal' else None,
        'fl_alpha':         fl_alpha if loss_type == 'focal' else None,
        'best_epoch':       history['best_epoch'] if history else 'loaded',
        'accuracy':         metrics['accuracy'],
        'precision':        metrics['precision'],
        'recall':           metrics['recall'],
        'f1':               metrics['f1'],
        'confusion_matrix': metrics['confusion_matrix'],
        'config': {
            'T': config.T, 'D': config.D, 'GAMMA': config.GAMMA,
            'hidden_dim': config.HIDDEN_DIM, 'num_layers': config.NUM_LAYERS,
            'lr': config.LR, 'batch_size': config.BATCH_SIZE,
            'patience': patience, 'val_metric': early_stop_metric,
            'loss_type': loss_type,
        },
    }

    with open(metric_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[{model_name}] metrics → {metric_path}')

    return result


def print_table(results):
    print()
    print('=' * 80)
    print('Part d — Binary Classification Results')
    print('=' * 80)
    print(f"{'Model':<16}  {'Tag':<20}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print('-' * 80)
    for r in results:
        tag = r.get('run_tag', 'unknown')
        print(
            f"{r['model']:<16}  {tag:<20}  "
            f"{r['accuracy']:>8.4f}  "
            f"{r['precision']:>9.4f}  "
            f"{r['recall']:>6.4f}  "
            f"{r['f1']:>6.4f}"
        )
    print('=' * 80)


def load_all_results():
    results = []
    for fname in sorted(os.listdir(config.METRIC_DIR)):
        if fname.startswith('part_d_g') and fname.endswith('.json'):
            with open(os.path.join(config.METRIC_DIR, fname)) as f:
                results.append(json.load(f))
    return results


def main():
    parser = argparse.ArgumentParser(description='HW4 Part d — turning point detection')
    parser.add_argument(
        '--gamma', type=float, default=config.GAMMA,
        help=f'price-ratio buy threshold (default: {config.GAMMA})',
    )
    parser.add_argument(
        '--patience', type=int, default=config.PATIENCE,
        help=f'early stopping patience (default: {config.PATIENCE})',
    )
    parser.add_argument(
        '--val-metric', choices=['loss', 'f1'], default='loss',
        help='early stopping criterion — loss (lower=better) or f1 (higher=better) (default: loss)',
    )
    parser.add_argument(
        '--loss', choices=['bce', 'focal'], default='bce',
        help='training loss function (default: bce)',
    )
    parser.add_argument(
        '--fl-gamma', type=float, default=2.0,
        help='focal loss focusing exponent (default: 2.0, used only when --loss focal)',
    )
    parser.add_argument(
        '--fl-alpha', type=float, default=0.75,
        help='focal loss positive-class weight (default: 0.75, used only when --loss focal)',
    )
    parser.add_argument(
        '--retrain', nargs='*', metavar='MODEL',
        help='retrain specified models (e.g. --retrain BiStockLSTM); '
             '--retrain with no args retrains all',
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print(f'device: {device}')

    run_tag = make_run_tag(args.gamma, args.val_metric, args.loss)
    print(f'run tag: {run_tag}  '
          f'(gamma={args.gamma}, patience={args.patience}, '
          f'val_metric={args.val_metric}, loss={args.loss})')

    if args.gamma != config.GAMMA:
        print(f'Overriding GAMMA: {config.GAMMA} → {args.gamma}')
        config.GAMMA = args.gamma

    print('loading data...')
    train_loader, val_loader, test_loader = get_loaders(
        target='binary', batch_size=config.BATCH_SIZE
    )

    pos_weight = compute_pos_weight(train_loader)

    model_specs = [
        ('BiStockLSTM', BiStockLSTM()),
        ('BiStockGRU',  BiStockGRU()),
    ]

    if args.retrain is None:
        retrain_set = set()
    elif len(args.retrain) == 0:
        retrain_set = {n for n, _ in model_specs}
    else:
        retrain_set = set(args.retrain)

    results = []
    for model_name, model in model_specs:
        print()
        result = run(
            model_name, model, train_loader, val_loader, test_loader,
            pos_weight, device,
            retrain=model_name in retrain_set,
            run_tag=run_tag,
            patience=args.patience,
            early_stop_metric=args.val_metric,
            loss_type=args.loss,
            fl_gamma=args.fl_gamma,
            fl_alpha=args.fl_alpha,
        )
        results.append(result)

    print_table(results)

    all_results = load_all_results()
    if len(all_results) > len(results):
        print('\n=== All Part d Runs (accumulated) ===')
        print_table(all_results)


if __name__ == '__main__':
    main()
