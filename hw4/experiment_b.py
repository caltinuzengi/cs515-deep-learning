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


def _stability_std(losses, n=20):
    tail = losses[-n:] if len(losses) >= n else losses
    return float(np.std(tail))


def run(model_name, model, train_loader, val_loader, test_loader,
        device, retrain):
    ckpt_path   = os.path.join(config.CKPT_DIR,   f'part_b_{model_name}.pt')
    plot_path   = os.path.join(config.PLOT_DIR,   f'part_b_{model_name}_loss.png')
    metric_path = os.path.join(config.METRIC_DIR, f'part_b_{model_name}.json')

    model = model.to(device)

    if not retrain and os.path.exists(ckpt_path):
        print(f'[{model_name}] checkpoint found — skipping training')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        history = None
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

    result = {
        'model':            model_name,
        'best_epoch':       history['best_epoch'] if history else 'loaded',
        'per_horizon_mse':  metrics['per_horizon_mse'],
        'mean_mse':         metrics['mean_mse'],
        'stability_std':    _stability_std(history['train_losses']) if history else None,
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


def print_table(results):
    print()
    print('=' * 62)
    print('Part b — Test MSE Comparison')
    print('=' * 62)
    header = f"{'Model':<14}" + ''.join(f'  d={d}  ' for d in range(1, 6)) + '  mean'
    print(header)
    print('-' * 62)
    for r in results:
        row = f"{r['model']:<14}"
        for v in r['per_horizon_mse']:
            row += f'  {v:.4f}'
        row += f'  {r["mean_mse"]:.4f}'
        print(row)
    print('=' * 62)


def main():
    parser = argparse.ArgumentParser(description='HW4 Part b — exact return forecasting')
    parser.add_argument('--retrain', action='store_true',
                        help='force retrain even if checkpoint exists')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print(f'device: {device}')

    print('loading data...')
    train_loader, val_loader, test_loader = get_loaders(
        target='exact', batch_size=config.BATCH_SIZE
    )

    models = [
        ('StockLSTM', StockLSTM()),
        ('StockGRU',  StockGRU()),
    ]

    results = []
    for model_name, model in models:
        print()
        result = run(model_name, model, train_loader, val_loader, test_loader,
                     device, retrain=args.retrain)
        results.append(result)

    print_table(results)


if __name__ == '__main__':
    main()
