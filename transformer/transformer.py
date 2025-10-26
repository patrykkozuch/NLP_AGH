import torch
import torch.nn as nn

from transformer.attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_ff: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.masked_attn = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.ln_masked_attn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        out = self.masked_attn(src, src, src, mask=mask)
        out = self.dropout(out)
        src = self.ln_masked_attn(src + out)
        out = self.ff(src)
        out = self.dropout(out)
        return self.ln_ff(src + out)



class Transformer(nn.Module):
    def __init__(self, vocab_size: int, n_blocks: int, num_heads: int, seq_len: int, d_model: int, d_ff: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_len=seq_len, d_model=d_model)
        self.dropout = nn.Dropout(p=0.1)

        self.decoder = nn.ModuleList([
            Decoder(
                num_heads=num_heads,
                d_model=d_model,
                d_ff=d_ff,
                *args,
                **kwargs
            )
            for _ in range(n_blocks)
        ])

        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.embedding(x)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)

        for decoder_block in self.decoder:
            x = decoder_block(x, mask)

        x = self.linear(x)

        if not self.training:
            # Apply only during prediction, CrossEntropyLoss applies softmax internally
            x = self.softmax(x)

        return x
