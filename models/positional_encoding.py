import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pos_enc = torch.empty((seq_len, d_model))
        pos = torch.arange(0, seq_len).unsqueeze(1)
        reciprocal_den = torch.exp(
            torch.arange(0, d_model, 2) * -1 * math.log(10000) / d_model
        )  # Reciprocal of the denominator

        pos_enc[:, 0::2] = torch.sin(pos * reciprocal_den)
        pos_enc[:, 1::2] = torch.cos(pos * reciprocal_den)

        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = x + self.pos_enc[:, : x.shape[1]]
        return self.dropout(x)
