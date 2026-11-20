import torch.nn as nn
from torch.nn.functional import log_softmax


class Projection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return log_softmax(self.linear(x), dim=-1)
