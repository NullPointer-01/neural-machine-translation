import torch.nn as nn
from layer_norm import LayerNorm


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Post-Layer norm as in original Transformer.
        # Modern LLMs use Pre-Layer norm "x + self.dropout(sublayer(self.norm(x)))"
        return self.norm(x + self.dropout(sublayer(x)))
