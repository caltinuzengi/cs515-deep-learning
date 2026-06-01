import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_HW4_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_HW4_DIR)
for _p in (_HW4_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config


def download_data():
    import yfinance as yf
    # yfinance end is exclusive, add one day to include TEST_END
    end = (pd.Timestamp(config.TEST_END) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    for ticker in config.TICKERS:
        path = os.path.join(config.DATA_RAW, f"{ticker}.csv")
        if os.path.exists(path):
            print(f"  {ticker}: already cached at {path}")
            continue
        df = yf.Ticker(ticker).history(start=config.TRAIN_START, end=end, auto_adjust=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.to_csv(path)
        print(f"  {ticker}: {len(df)} rows → {path}")


def _load_ticker(ticker):
    path = os.path.join(config.DATA_RAW, f"{ticker}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close']]
    df = df.ffill().bfill()
    return df


def _normalize(df):
    """Z-score using train-split stats only. Returns (norm_df, mean, std)."""
    train_mask = (df.index >= config.TRAIN_START) & (df.index <= config.TRAIN_END)
    mean = df[train_mask].mean()
    std = df[train_mask].std().replace(0, 1.0)
    return (df - mean) / std, mean, std


def _build_samples(ticker, split):
    """Returns list of (X, y_exact, y_rolling, y_binary) float32 arrays for one ticker/split."""
    split_ranges = {
        'train': (config.TRAIN_START, config.TRAIN_END),
        'val':   (config.VAL_START,   config.VAL_END),
        'test':  (config.TEST_START,  config.TEST_END),
    }
    start_date, end_date = split_ranges[split]

    raw  = _load_ticker(ticker)
    norm, _, _ = _normalize(raw)

    raw_close = raw['Close'].values
    raw_high  = raw['High'].values
    norm_ohlc = norm[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)

    split_mask = (raw.index >= start_date) & (raw.index <= end_date)
    valid_t = np.where(split_mask)[0]

    T, D = config.T, config.D
    l = config.ROLL_WINDOW
    weights = np.array(config.ROLL_WEIGHTS, dtype=np.float64)
    N = len(raw)

    samples = []
    for t in valid_t:
        # need T-day history and D-day future
        if t < T - 1 or t + D >= N:
            continue

        X = norm_ohlc[t - T + 1 : t + 1]   # (T, F) — normalized features
        close_t = raw_close[t]               # raw close, always positive

        # exact d-day return: (close[t+d] - close[t]) / close[t]
        y_exact = np.array(
            [(raw_close[t + d] - close_t) / close_t for d in range(1, D + 1)],
            dtype=np.float32,
        )

        # rolling average return: (Σ_j w_j * close[t+d-j] - close[t]) / close[t]
        # j = 0..l (l=3) → 4 terms; for d=1 this includes up to close[t-2] (past, always available)
        y_rolling = np.empty(D, dtype=np.float32)
        for i, d in enumerate(range(1, D + 1)):
            prices = np.array([raw_close[t + d - j] for j in range(l + 1)])
            y_rolling[i] = float((np.dot(weights, prices) - close_t) / close_t)

        # buy signal: any d s.t. high[t+d] / close[t] > GAMMA (price ratio, not return ratio)
        # γ=1.1 interpreted as price ratio threshold → 10% gain triggers buy
        y_binary = np.array(
            [float(any(
                raw_high[t + d] / close_t > config.GAMMA
                for d in range(1, D + 1)
            ))],
            dtype=np.float32,
        )

        samples.append((X, y_exact, y_rolling, y_binary))

    return samples


class ReturnDataset(Dataset):
    def __init__(self, tickers, split, target='exact'):
        assert target in ('exact', 'rolling'), f"Unknown target: {target}"
        self.target = target

        all_samples = []
        for ticker in tickers:
            all_samples.extend(_build_samples(ticker, split))

        self.X         = torch.from_numpy(np.stack([s[0] for s in all_samples]))
        self.y_exact   = torch.from_numpy(np.stack([s[1] for s in all_samples]))
        self.y_rolling = torch.from_numpy(np.stack([s[2] for s in all_samples]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y = self.y_exact[idx] if self.target == 'exact' else self.y_rolling[idx]
        return self.X[idx], y


class TurningPointDataset(Dataset):
    def __init__(self, tickers, split):
        all_samples = []
        for ticker in tickers:
            all_samples.extend(_build_samples(ticker, split))

        self.X = torch.from_numpy(np.stack([s[0] for s in all_samples]))
        self.y = torch.from_numpy(np.stack([s[3] for s in all_samples]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(target, batch_size=config.BATCH_SIZE):
    if target == 'binary':
        train_ds = TurningPointDataset(config.TICKERS, 'train')
        val_ds   = TurningPointDataset(config.TICKERS, 'val')
        test_ds  = TurningPointDataset(config.TICKERS, 'test')
    else:
        train_ds = ReturnDataset(config.TICKERS, 'train', target=target)
        val_ds   = ReturnDataset(config.TICKERS, 'val',   target=target)
        test_ds  = ReturnDataset(config.TICKERS, 'test',  target=target)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print('=== Downloading data ===')
    download_data()

    print('\n=== ReturnDataset (exact) ===')
    for split in ('train', 'val', 'test'):
        ds = ReturnDataset(config.TICKERS, split, target='exact')
        print(f'  {split:5s}: {len(ds)} samples')

    print('\n=== ReturnDataset (rolling) ===')
    for split in ('train', 'val', 'test'):
        ds = ReturnDataset(config.TICKERS, split, target='rolling')
        print(f'  {split:5s}: {len(ds)} samples')

    print('\n=== TurningPointDataset ===')
    for split in ('train', 'val', 'test'):
        ds = TurningPointDataset(config.TICKERS, split)
        total = len(ds)
        n_pos = int(ds.y.sum().item())
        n_neg = total - n_pos
        print(
            f'  {split:5s}: {total} samples  |  '
            f'class 1: {n_pos} ({100 * n_pos / total:.1f}%)  '
            f'class 0: {n_neg} ({100 * n_neg / total:.1f}%)'
        )

    print('\n=== Batch shapes ===')
    tl, _, _ = get_loaders('exact', batch_size=64)
    X_b, y_b = next(iter(tl))
    print(f'  X:         {tuple(X_b.shape)}')
    print(f'  y_exact:   {tuple(y_b.shape)}')

    tl, _, _ = get_loaders('rolling', batch_size=64)
    _, y_b = next(iter(tl))
    print(f'  y_rolling: {tuple(y_b.shape)}')

    tl, _, _ = get_loaders('binary', batch_size=64)
    _, y_b = next(iter(tl))
    print(f'  y_binary:  {tuple(y_b.shape)}')
