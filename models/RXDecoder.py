import math
import torch
import torch.nn as nn


class RXDecoder(nn.Module):
    def __init__(self, t_rounds, d_model, n_heads, n_layers, ffn_dim, dropout,
                 n_symbols, alphabet_size):
        super().__init__()
        self.t_rounds = t_rounds
        self.n_symbols = n_symbols

        # Input MLP: summarizes T_ROUNDS observations per symbol position → d_model token
        self.input_mlp = nn.Sequential(
            nn.Linear(t_rounds, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout, batch_first=True, norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_linear = nn.Linear(d_model, alphabet_size)

        # Sinusoidal positional encoding — fixed, no gradient
        pe = torch.zeros(1, n_symbols, d_model)
        pos = torch.arange(n_symbols).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, history_y):
        """
        history_y : list of T_ROUNDS tensors, each (batch, n_symbols)
        returns   : logits (batch, n_symbols, alphabet_size)
        """
        # (batch, T_ROUNDS, n_symbols) → (batch, n_symbols, T_ROUNDS)
        Y_stacked = torch.stack(history_y, dim=1).transpose(1, 2)

        H = self.input_mlp(Y_stacked)      # (batch, n_symbols, d_model)
        H = self.transformer(H + self.pe)  # (batch, n_symbols, d_model)
        logits = self.output_linear(H)     # (batch, n_symbols, alphabet_size)
        return logits
