import torch
import torch.nn as nn


class CommSystem(nn.Module):
    def __init__(self, tx_encoder, rx_decoder, sigma_sq, t_rounds):
        super().__init__()
        self.tx_encoder = tx_encoder
        self.rx_decoder = rx_decoder
        self.sigma_sq   = sigma_sq
        self.t_rounds   = t_rounds
        self.criterion  = nn.CrossEntropyLoss()

    def forward(self, m):
        """
        m : (batch, n_symbols) int — message indices in {0..alphabet_size-1}
        returns: logits (batch, n_symbols, alphabet_size), history_x list[T_ROUNDS x (batch, n_symbols)]
        """
        history_x, history_y, history_f = [], [], []

        for _ in range(self.t_rounds):
            x_t = self.tx_encoder(m, history_x, history_f)
            y_t = x_t + torch.randn_like(x_t) * (self.sigma_sq ** 0.5)
            f_t = y_t  # noiseless feedback relay: TX receives exactly what RX observed
            history_x.append(x_t)
            history_y.append(y_t)
            history_f.append(f_t)

        logits = self.rx_decoder(history_y)
        return logits, history_x

    def compute_loss(self, logits, m):
        """
        logits : (batch, n_symbols, alphabet_size)
        m      : (batch, n_symbols) int
        returns: scalar cross-entropy loss
        """
        return self.criterion(logits.view(-1, logits.size(-1)), m.view(-1))

    def compute_avg_power(self, history_x):
        """
        history_x : list of T_ROUNDS tensors, each (batch, n_symbols)
        returns   : average per-round per-sample power as a Python float
        """
        return torch.stack([
            (x_t ** 2).sum(dim=-1) for x_t in history_x
        ]).mean().item()
