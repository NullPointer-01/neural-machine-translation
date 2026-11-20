import torch.nn as nn
from attention import MultiHeadAttention
from ffn import FeedForwardNetwork
from residual_connection import ResidualConnection
import copy


class EncoderLayer(nn.Module):
    def __init__(
        self,
        self_attn: MultiHeadAttention,
        ffn: FeedForwardNetwork,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn  # Sublayer 1
        self.ffn = ffn  # Sublayer 2

        self.res_con1 = ResidualConnection(d_model, dropout)
        self.res_con2 = ResidualConnection(d_model, dropout)

    def forward(self, x, self_attn_mask):
        x = self.res_con1(x, lambda x: self.self_attn(x, x, x, self_attn_mask))
        x = self.res_con2(x, lambda x: self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(self, x, self_attn_mask):
        for layer in self.layers:
            x = layer(x, self_attn_mask)
        return x
