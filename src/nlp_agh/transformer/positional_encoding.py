import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Use exp for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        PE = torch.zeros(size=(max_len, d_model))
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('PE', PE)

    def forward(self, x):
        return x + self.PE[:x.size(1), :]