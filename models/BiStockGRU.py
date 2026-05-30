import torch
import torch.nn as nn


class BiStockGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2,
                 dropout=0.2, use_aux_feature=False):
        super().__init__()
        self.use_aux_feature = use_aux_feature
        if use_aux_feature:
            self.aux_conv = nn.Conv1d(input_dim, 1, kernel_size=3, padding=1)
            rnn_input_dim = input_dim + 1
        else:
            rnn_input_dim = input_dim

        self.gru = nn.GRU(rnn_input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # forward + backward concat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        if self.use_aux_feature:
            aux = self.aux_conv(x.permute(0, 2, 1))        # (batch, 1, T)
            x = torch.cat([x, aux.permute(0, 2, 1)], dim=2)  # (batch, T, F+1)
        _, h_n = self.gru(x)
        # h_n: (num_layers * 2, batch, hidden_dim)
        # h_n[-2]: last forward layer final state
        # h_n[-1]: last backward layer final state
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)           # (batch, hidden_dim * 2)
        h = self.dropout(h)
        return self.fc(h)                                   # (batch, 1) — raw logits
