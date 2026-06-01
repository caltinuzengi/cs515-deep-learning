import math
import torch
import torch.nn as nn


class TXEncoder(nn.Module):
    def __init__(self, d_emb, d_model, n_heads, n_layers, ffn_dim, dropout,
                 t_rounds, n_symbols, alphabet_size, eps):
        super().__init__()
        self.t_rounds = t_rounds
        self.n_symbols = n_symbols
        self.eps = eps
        buf = t_rounds - 1  # history buffer size = 3

        self.embedding = nn.Embedding(alphabet_size, d_emb)

        # Pre-processing MLP: maps per-position raw input → d_model token
        self.pre_mlp = nn.Sequential(
            nn.Linear(d_emb + buf + buf, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout, batch_first=True, norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Post-processing MLP: maps each d_model token → 1 scalar coded symbol
        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Sinusoidal positional encoding — fixed, no gradient
        pe = torch.zeros(1, n_symbols, d_model)
        pos = torch.arange(n_symbols).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def _make_buffer(self, history, pos, buf, batch, device):
        b = torch.zeros(batch, buf, device=device)
        for r, h in enumerate(history):
            b[:, r] = h[:, pos]
        return b

    def forward(self, m, history_x, history_f):
        """
        m          : (batch, n_symbols) int ∈ {0..alphabet_size-1}
        history_x  : list of t-1 tensors, each (batch, n_symbols)
        history_f  : list of t-1 tensors, each (batch, n_symbols)
        returns    : x_t (batch, n_symbols), L2-normalized per sample
        """
        batch = m.shape[0]
        device = m.device
        buf = self.t_rounds - 1

        m_emb = self.embedding(m)  # (batch, n_symbols, d_emb)

        tokens = []
        for i in range(self.n_symbols):
            emb_i = m_emb[:, i, :]                                   # (batch, d_emb)
            bx = self._make_buffer(history_x, i, buf, batch, device)  # (batch, 3)
            bf = self._make_buffer(history_f, i, buf, batch, device)  # (batch, 3)
            raw = torch.cat([emb_i, bx, bf], dim=-1)                  # (batch, d_emb+3+3)
            tokens.append(self.pre_mlp(raw))                          # (batch, d_model)

        Z = torch.stack(tokens, dim=1)      # (batch, n_symbols, d_model)
        H = self.transformer(Z + self.pe)   # (batch, n_symbols, d_model)

        raw_x = self.post_mlp(H).squeeze(-1)  # (batch, n_symbols, 1) → (batch, n_symbols)
        x_t = raw_x / (raw_x.norm(dim=-1, keepdim=True) + self.eps)
        return x_t
