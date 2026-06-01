import sys
import os
import json
import itertools
import math
import argparse

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from bonus.config_bonus import (
    DEVICE, SEED,
    T_ROUNDS, N_SYMBOLS, ALPHABET, SIGMA_SQ, EPS,
    D_EMB, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT,
    BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE, NOISE_TRIALS,
    RESULTS_DIR,
)
from models.TXEncoder import TXEncoder
from models.RXDecoder import RXDecoder
from bonus.comm_system import CommSystem


# ---------------------------------------------------------------------------
# Pre-training smoke tests
# ---------------------------------------------------------------------------

def run_smoke_tests(comm, enc, dec, device):
    print("=" * 60)
    print("Running pre-training smoke tests...")

    m_test = torch.randint(0, ALPHABET, (BATCH_SIZE, N_SYMBOLS), device=device)

    # 1. Forward pass shapes
    comm.eval()
    with torch.no_grad():
        logits, hx = comm(m_test)
    assert logits.shape == (BATCH_SIZE, N_SYMBOLS, ALPHABET), \
        f"[FAIL] logits shape: {logits.shape}"
    assert len(hx) == comm.t_rounds, \
        f"[FAIL] history_x length: {len(hx)}"
    assert hx[0].shape == (BATCH_SIZE, N_SYMBOLS), \
        f"[FAIL] x_t shape: {hx[0].shape}"
    print("[PASS] Forward pass shapes")

    # 2. Power constraint holds before any training
    power = comm.compute_avg_power(hx)
    assert power <= 1.0 + 1e-4, f"[FAIL] power={power:.6f} > 1.0"
    print(f"[PASS] Power constraint: {power:.6f} <= 1.0")

    # 3. Initial loss near random baseline ln(8) ≈ 2.079
    with torch.no_grad():
        init_loss = comm.compute_loss(logits, m_test).item()
    expected = math.log(ALPHABET)
    assert abs(init_loss - expected) < 0.5, \
        f"[FAIL] Init loss {init_loss:.4f} far from ln({ALPHABET})={expected:.4f}"
    print(f"[PASS] Init loss: {init_loss:.4f}  (expected ≈ {expected:.4f})")

    # 4. Initial symbol accuracy near random baseline 1/8 = 0.125
    with torch.no_grad():
        logits2, _ = comm(m_test)
    preds = logits2.argmax(dim=-1)
    sym_acc = (preds == m_test).float().mean().item()
    assert sym_acc < 0.4, f"[FAIL] Init sym_acc suspiciously high: {sym_acc:.4f}"
    print(f"[PASS] Init symbol accuracy: {sym_acc:.4f}  (expected ≈ {1/ALPHABET:.4f})")

    # 5. Single backward pass and optimizer step
    comm.train()
    opt_test = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()), lr=LR
    )
    opt_test.zero_grad()
    logits3, hx3 = comm(m_test)
    loss3 = comm.compute_loss(logits3, m_test)
    loss3.backward()
    tx_grads = [p for p in enc.parameters() if p.grad is not None]
    rx_grads = [p for p in dec.parameters() if p.grad is not None]
    assert len(tx_grads) > 0, "[FAIL] No gradients in TXEncoder"
    assert len(rx_grads) > 0, "[FAIL] No gradients in RXDecoder"
    opt_test.step()
    print(f"[PASS] Backward + optimizer step  "
          f"(TX grads: {len(tx_grads)}, RX grads: {len(rx_grads)})")

    print("All smoke tests passed.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_val_metrics(comm, val_m):
    comm.eval()
    with torch.no_grad():
        logits, hx = comm(val_m)
    loss    = comm.compute_loss(logits, val_m).item()
    preds   = logits.argmax(dim=-1)
    sym_acc = (preds == val_m).float().mean().item()
    msg_acc = (preds == val_m).all(dim=-1).float().mean().item()
    power   = comm.compute_avg_power(hx)
    comm.train()
    return loss, sym_acc, msg_acc, power


def save_plots(train_losses, val_losses, sym_accs, msg_accs, powers, plot_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses,   label='val')
    plt.xlabel('Epoch'); plt.ylabel('CE Loss')
    plt.title('Loss Curve'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/loss_curve.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, sym_accs, label='symbol accuracy')
    plt.plot(epochs, msg_accs, label='message accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Accuracy Curve'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/accuracy_curve.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, powers, label='avg power')
    plt.axhline(y=1.0, color='r', linestyle='--', label='constraint')
    plt.xlabel('Epoch'); plt.ylabel('Avg Power')
    plt.title('Power Curve'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/power_curve.png')
    plt.close()


# ---------------------------------------------------------------------------
# Full evaluation over all 4096 messages
# ---------------------------------------------------------------------------

def full_evaluation(comm, metric_dir):
    print("Running full evaluation (4096 messages x {} trials)...".format(NOISE_TRIALS))
    all_messages = torch.tensor(
        list(itertools.product(range(ALPHABET), repeat=N_SYMBOLS)),
        device=DEVICE,
    )  # (4096, 4)

    comm.eval()
    total_ce, total_sym_correct, total_msg_correct, total_power = 0.0, 0, 0, 0.0
    n_batches = 0

    for batch_msgs in all_messages.split(512):
        for _ in range(NOISE_TRIALS):
            with torch.no_grad():
                logits, hx = comm(batch_msgs)
            preds = logits.argmax(dim=-1)
            total_ce          += comm.compute_loss(logits, batch_msgs).item()
            total_sym_correct += (preds == batch_msgs).sum().item()
            total_msg_correct += (preds == batch_msgs).all(dim=-1).sum().item()
            total_power       += comm.compute_avg_power(hx)
            n_batches         += 1

    n_total_symbols  = 4096 * NOISE_TRIALS * N_SYMBOLS
    n_total_messages = 4096 * NOISE_TRIALS

    sym_acc = total_sym_correct / n_total_symbols
    msg_acc = total_msg_correct / n_total_messages

    metrics = {
        "ce_loss":          total_ce / n_batches,
        "symbol_accuracy":  sym_acc,
        "message_accuracy": msg_acc,
        "SER":              1.0 - sym_acc,
        "MER":              1.0 - msg_acc,
        "avg_power":        total_power / n_batches,
    }

    with open(f'{metric_dir}/eval_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Bonus experiment: interactive AWGN communication system'
    )
    parser.add_argument('--t-rounds',    type=int,   default=T_ROUNDS,
                        help=f'Number of communication rounds (default: {T_ROUNDS})')
    parser.add_argument('--sigma-sq',   type=float, default=SIGMA_SQ,
                        help=f'AWGN noise variance sigma^2 (default: {SIGMA_SQ})')
    parser.add_argument('--no-feedback', action='store_true', default=False,
                        help='Disable feedback: TX receives zeros instead of y_t')
    parser.add_argument('--out-dir',    type=str,   default=RESULTS_DIR,
                        help=f'Root output directory (default: {RESULTS_DIR})')
    args = parser.parse_args()

    t_rounds     = args.t_rounds
    sigma_sq     = args.sigma_sq
    use_feedback = not args.no_feedback
    out_dir      = args.out_dir
    ckpt_dir     = f'{out_dir}/checkpoints'
    plot_dir     = f'{out_dir}/plots'
    metric_dir   = f'{out_dir}/metrics'
    for d in [ckpt_dir, plot_dir, metric_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Config: T={t_rounds} | sigma_sq={sigma_sq} | "
          f"feedback={use_feedback} | out_dir={out_dir}")

    torch.manual_seed(SEED)

    enc  = TXEncoder(D_EMB, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT,
                     t_rounds, N_SYMBOLS, ALPHABET, EPS).to(DEVICE)
    dec  = RXDecoder(t_rounds, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT,
                     N_SYMBOLS, ALPHABET).to(DEVICE)
    comm = CommSystem(enc, dec, sigma_sq, t_rounds, use_feedback=use_feedback)

    run_smoke_tests(comm, enc, dec, DEVICE)

    optimizer = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()), lr=LR
    )
    val_m = torch.randint(0, ALPHABET, (BATCH_SIZE, N_SYMBOLS), device=DEVICE)

    train_losses, val_losses = [], []
    sym_accs, msg_accs, powers = [], [], []

    best_val_loss    = float('inf')
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        comm.train()
        m = torch.randint(0, ALPHABET, (BATCH_SIZE, N_SYMBOLS), device=DEVICE)
        optimizer.zero_grad()
        logits, hx = comm(m)
        loss = comm.compute_loss(logits, m)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        val_loss, sym_acc, msg_acc, power = compute_val_metrics(comm, val_m)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        sym_accs.append(sym_acc)
        msg_accs.append(msg_acc)
        powers.append(power)

        print(f"Epoch {epoch:4d} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"sym_acc={sym_acc:.4f} | msg_acc={msg_acc:.4f} | "
              f"power={power:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()},
                       f'{ckpt_dir}/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    save_plots(train_losses, val_losses, sym_accs, msg_accs, powers, plot_dir)
    print("Plots saved.")

    ckpt = torch.load(f'{ckpt_dir}/best_model.pt', map_location=DEVICE)
    enc.load_state_dict(ckpt['enc'])
    dec.load_state_dict(ckpt['dec'])

    full_evaluation(comm, metric_dir)


if __name__ == '__main__':
    main()
