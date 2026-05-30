import os

TICKERS = ["AAPL", "MSFT", "GOOGL"]

TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-07-31"
VAL_START   = "2024-08-01"
VAL_END     = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2025-12-31"

T = 20  # lookback window
F = 4   # open, high, low, close
D = 5   # horizons d = 1..5

ROLL_WINDOW  = 3
ROLL_WEIGHTS = [0.25, 0.25, 0.25, 0.25]  # uniform, j=0..l where l=3

GAMMA = 1.1  # buy if (high[t+d] - close[t]) / close[t] > GAMMA

BATCH_SIZE   = 64
HIDDEN_DIM   = 64
NUM_LAYERS   = 2
DROPOUT      = 0.2
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 100
PATIENCE     = 10

DATA_RAW   = "data/stocks/raw"
DATA_PROC  = "data/stocks/processed"
CKPT_DIR   = "results/hw4/checkpoints"
PLOT_DIR   = "results/hw4/plots"
METRIC_DIR = "results/hw4/metrics"

for _d in [DATA_RAW, DATA_PROC, CKPT_DIR, PLOT_DIR, METRIC_DIR]:
    os.makedirs(_d, exist_ok=True)
