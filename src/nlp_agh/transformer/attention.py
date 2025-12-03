import math

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    """
    Attention module that performs Scaled Dot-Product Attention
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        scaled_product = (q @ k.mT) / math.sqrt(k.size(-1))
        attn_mask = torch.zeros(k.size(-2), v.size(-2), dtype=k.dtype, device=k.device)
        
        if mask is not None:
            attn_mask = attn_mask.masked_fill(mask, -1e4)
            # Use additive mask to avoid attention problems with NaN values in softmax
            scaled_product += attn_mask

        return self.dropout(nn.functional.softmax(scaled_product, dim=-1)) @ v

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    Calculates a Scaled Dot-Product Attention for multiple attention heads
    """
    def __init__(self, num_heads: int, d_model: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.attention = Attention(dropout_rate)

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
