import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.w_1(x).relu()
        x = self.dropout(x)

        # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.w_2(x)
