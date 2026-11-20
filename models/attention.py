import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.h = h
        self.d_model = d_model

        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Query length and key/value length can differ in cross-attention
        b, q_len, _ = query.shape
        k_len = key.shape[1]

        # (batch, len, d_model) -> (batch, h, len, d_k)
        query = query.view(b, q_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(b, k_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(b, k_len, self.h, self.d_k).transpose(1, 2)

        # (batch, h, q_len, k_len)
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        # Apply mask
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # Save attention scores for visualization
        self.attn = self.dropout(scores.softmax(dim=3))

        # (batch, h, q_len, d_k)
        out = torch.matmul(self.attn, value)

        # (batch, h, q_len, d_k) -> (batch, q_len, d_model) - Back to query shape
        out = out.transpose(1, 2).contiguous().view(b, q_len, self.d_model)
        return self.w_o(out)
