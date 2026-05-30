import torch
import torch.nn as nn


class StockGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2,
                 dropout=0.2, output_dim=5, use_aux_feature=False):
        super().__init__()
        self.use_aux_feature = use_aux_feature
        if use_aux_feature:
            self.aux_conv = nn.Conv1d(input_dim, 1, kernel_size=3, padding=1)
            rnn_input_dim = input_dim + 1
        else:
            rnn_input_dim = input_dim

        self.gru = nn.GRU(rnn_input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        if self.use_aux_feature:
            aux = self.aux_conv(x.permute(0, 2, 1))        # (batch, 1, T)
            x = torch.cat([x, aux.permute(0, 2, 1)], dim=2)  # (batch, T, F+1)
        out, _ = self.gru(x)                                # (batch, T, hidden_dim)
        out = self.dropout(out[:, -1, :])                   # (batch, hidden_dim)
        return self.fc(out)                                 # (batch, output_dim)
