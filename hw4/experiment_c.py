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
from trainer import Trainer, plot_losses
from models.StockLSTM import StockLSTM
from models.StockGRU import StockGRU


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_part_b(model_name):
    path = os.path.join(config.METRIC_DIR, f'part_b_{model_name}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Part b metrics not found: {path}\n'
            f'Run experiment_b.py first.'
        )
    with open(path) as f:
        return json.load(f)


def stability_std(losses, n=20):
    tail = losses[-n:] if len(losses) >= n else losses
    return float(np.std(tail))


def run(model_name, model, train_loader, val_loader, test_loader,
        device, retrain):
    ckpt_path   = os.path.join(config.CKPT_DIR,   f'part_c_{model_name}.pt')
    plot_path   = os.path.join(config.PLOT_DIR,   f'part_c_{model_name}_loss.png')
    metric_path = os.path.join(config.METRIC_DIR, f'part_c_{model_name}.json')

    model = model.to(device)

    history = None
    if not retrain and os.path.exists(ckpt_path):
        print(f'[{model_name}] checkpoint found — skipping training')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f'[{model_name}] training...')
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        trainer = Trainer(
            model, optimizer, nn.MSELoss(),
            mode='regression', patience=config.PATIENCE, ckpt_path=ckpt_path,
        )
        history = trainer.fit(train_loader, val_loader, max_epochs=config.MAX_EPOCHS)
        plot_losses(history, plot_path)
        print(f'[{model_name}] loss curve → {plot_path}')

    model.eval()
    trainer_eval = Trainer(model, None, nn.MSELoss(), mode='regression')
    metrics = trainer_eval.evaluate(test_loader)

    std = stability_std(history['train_losses']) if history else None

    result = {
        'model':            model_name,
        'best_epoch':       history['best_epoch'] if history else 'loaded',
        'per_horizon_mse':  metrics['per_horizon_mse'],
        'mean_mse':         metrics['mean_mse'],
        'stability_std':    std,
        'config': {
            'T': config.T, 'D': config.D,
            'hidden_dim': config.HIDDEN_DIM, 'num_layers': config.NUM_LAYERS,
            'lr': config.LR, 'batch_size': config.BATCH_SIZE,
        },
    }

    with open(metric_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[{model_name}] metrics → {metric_path}')

    return result


def print_mse_table(results_b, results_c):
    print()
    print('=' * 52)
    print('Part b vs Part c — Test MSE Comparison')
    print('=' * 52)
    print(f"{'Model':<14}  {'Part_b_MSE':>12}  {'Part_c_MSE':>12}")
    print('-' * 52)
    for rb, rc in zip(results_b, results_c):
        print(f"{rb['model']:<14}  {rb['mean_mse']:>12.6f}  {rc['mean_mse']:>12.6f}")
    print('=' * 52)


def print_stability_table(results_b, results_c):
    print()
    print('=== Training Stability (std of last 20 epoch train losses) ===')
    print(f"{'Model':<14}  {'Part b (exact)':>16}  {'Part c (rolling)':>18}")
    print('-' * 55)
    for rb, rc in zip(results_b, results_c):
        std_b = rb.get('stability_std')
        std_c = rc.get('stability_std')
        sb = f'{std_b:.6f}' if std_b is not None else 'N/A (loaded)'
        sc = f'{std_c:.6f}' if std_c is not None else 'N/A (loaded)'
        print(f"{rb['model']:<14}  {sb:>16}  {sc:>18}")

    # overall conclusion
    valid = [(rb.get('stability_std'), rc.get('stability_std'))
             for rb, rc in zip(results_b, results_c)
             if rb.get('stability_std') is not None and rc.get('stability_std') is not None]
    if valid:
        avg_b = np.mean([v[0] for v in valid])
        avg_c = np.mean([v[1] for v in valid])
        verdict = 'more' if avg_c < avg_b else 'less'
        print(f'\n→ Rolling target training is {verdict} stable than exact target '
              f'(avg std: {avg_b:.6f} vs {avg_c:.6f})')


def main():
    parser = argparse.ArgumentParser(description='HW4 Part c — rolling average return forecasting')
    parser.add_argument(
        '--retrain', nargs='*', metavar='MODEL',
        help='retrain specified models (e.g. --retrain StockLSTM); '
             '--retrain with no args retrains all',
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print(f'device: {device}')

    print('loading data...')
    train_loader, val_loader, test_loader = get_loaders(
        target='rolling', batch_size=config.BATCH_SIZE
    )

    model_specs = [
        ('StockLSTM', StockLSTM()),
        ('StockGRU',  StockGRU()),
    ]

    # determine which models to retrain
    if args.retrain is None:
        retrain_set = set()           # no --retrain flag: use checkpoint
    elif len(args.retrain) == 0:
        retrain_set = {n for n, _ in model_specs}   # --retrain (no args): all
    else:
        retrain_set = set(args.retrain)

    # load Part b baselines (required)
    results_b = [load_part_b(name) for name, _ in model_specs]

    results_c = []
    for model_name, model in model_specs:
        print()
        result = run(model_name, model, train_loader, val_loader, test_loader,
                     device, retrain=model_name in retrain_set)
        results_c.append(result)

    print_mse_table(results_b, results_c)
    print_stability_table(results_b, results_c)


if __name__ == '__main__':
    main()
