import torch.nn as nn
import torch

class LstmModel(nn.Module):
    def __init__(self, vocab_size: int, n_blocks: int, d_model: int, d_ff: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=0.1)

        self.ln_lstm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_blocks, batch_first=True)
        self.dropout_lstm = nn.Dropout(p=0.1)

        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout_ff = nn.Dropout(p=0.1)

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.dropout(x)

        x = self.ln_lstm(x)
        out, _ = self.lstm(x)
        x = x + self.dropout_lstm(out)

        x = self.ln_ff(x)
        out = self.ff(x)
        x = x + self.dropout_ff(out)

        return self.linear(x)