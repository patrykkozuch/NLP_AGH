import math

import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Attention module that performs Scaled Dot-Product Attention
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        scaled_product = (q @ k.mT) / math.sqrt(k.size(-1))

        if mask is not None:
            scaled_product = scaled_product.masked_fill(mask, -torch.inf)

        return nn.functional.softmax(scaled_product, dim=-1) @ v

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    Calculates a Scaled Dot-Product Attention for multiple attention heads
    """
    def __init__(self, num_heads: int, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.attention = Attention()

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        out = self.attention(q, k, v, mask)

        out = self.concat(out)
        out = self.W_o(out)

        return out

    def split(self, tensor: torch.Tensor):
        """
        Create a pytorch view to calculate all heads as one matrix
        """
        return tensor.view(tensor.size(0), -1, self.num_heads, self.d_head).transpose(1, 2)

    def concat(self, tensor: torch.Tensor):
        """
        Bring back the original form of attention matrix
        """
        return tensor.transpose(1, 2).contiguous().view(tensor.size(0), -1, self.num_heads * self.d_head)