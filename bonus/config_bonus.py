import os
import random
import numpy as np
import torch

# Communication system
T_ROUNDS  = 4
N_SYMBOLS = 4
ALPHABET  = 8
SIGMA_SQ  = 0.25
EPS       = 1e-8

# Model dimensions
D_EMB    = 16
D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 2
FFN_DIM  = 128
DROPOUT  = 0.1

# Training & evaluation
BATCH_SIZE   = 256
LR           = 1e-3
MAX_EPOCHS   = 200
PATIENCE     = 20
NOISE_TRIALS = 100

# Engineering safeguards
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
RESULTS_DIR = "results/bonus"
CKPT_DIR    = "results/bonus/checkpoints"
PLOT_DIR    = "results/bonus/plots"
METRIC_DIR  = "results/bonus/metrics"


def setup_dirs():
    for d in [CKPT_DIR, PLOT_DIR, METRIC_DIR]:
        os.makedirs(d, exist_ok=True)


setup_dirs()
