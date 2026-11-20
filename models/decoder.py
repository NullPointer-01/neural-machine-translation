import torch.nn as nn
from attention import MultiHeadAttention
from ffn import FeedForwardNetwork
from residual_connection import ResidualConnection
import copy


class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attn: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        ffn: FeedForwardNetwork,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn  # Sublayer 1
        self.cross_attn = cross_attn  # Sublayer 2
        self.ffn = ffn  # Sublayer 3

        self.res_con1 = ResidualConnection(d_model, dropout)
        self.res_con2 = ResidualConnection(d_model, dropout)
        self.res_con3 = ResidualConnection(d_model, dropout)

    def forward(self, x, enc_output, self_attn_mask, cross_attn_mask):
        x = self.res_con1(x, lambda x: self.self_attn(x, x, x, self_attn_mask))
        x = self.res_con2(
            x, lambda x: self.cross_attn(x, enc_output, enc_output, cross_attn_mask)
        )
        x = self.res_con3(x, lambda x: self.ffn(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(self, x, enc_output, self_attn_mask, cross_attn_mask):
        for layer in self.layers:
            x = layer(x, enc_output, self_attn_mask, cross_attn_mask)
        return x
