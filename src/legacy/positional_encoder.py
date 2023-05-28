import math

import torch
from torch import Tensor, nn


class PositionalEncoder(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000, add=True
    ):
        super().__init__()
        self.add = add

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        print(x.shape)
        print(self.pe[: x.size(0)].shape)
        if self.add:
            x = x + self.pe[: x.size(0)]
        else:
            x = torch.concat([x, self.pe[: x.size(0)]])
        return x
