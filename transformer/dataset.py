import io
import json

import torch
import zstandard as zst

from torch.utils.data import Dataset
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


def chunk_text(text, tokenizer, chunk_size=512):
    chunk_size += 1

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None
    )

    input_ids = tokens['input_ids']

    if len(input_ids) == 0:
        return []

    chunks = []

    for start in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[start:start + chunk_size]

        if len(chunk_ids) < 2:
            continue

        actual_length = len(chunk_ids)

        input_chunk = chunk_ids[:-1]  # All except last
        label_chunk = chunk_ids[1:]  # All except first

        # Pad to chunk_size - 1 (since we removed one token for shifting)
        max_len = chunk_size - 1
        padding_length = max_len - len(input_chunk)

        if padding_length > 0:
            input_chunk = input_chunk + [tokenizer.pad_token_id] * padding_length
            label_chunk = label_chunk + [-100] * padding_length  # Ignore padding in loss

        # Attention mask
        attention_mask = [1] * (actual_length - 1) + [0] * padding_length

        chunks.append({
            "original_text": tokenizer.decode(input_chunk, skip_special_tokens=True),
            "input_ids": torch.tensor(input_chunk, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_chunk, dtype=torch.long)
        })

    return chunks

class SpeakLeashDataset(Dataset):
    def __init__(self, data_dir: str | Path, tokenizer: AutoTokenizer, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.items = []

        data_dir = Path(data_dir)
        for text in tqdm(self._get_texts(data_dir)):
            self.items.extend(chunk_text(text, tokenizer, max_len))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


    @staticmethod
    def _get_texts(data_dir: Path):
        for file in data_dir.glob("*.jsonl.zst"):
            with open(file, 'rb') as fp:
                decompressor = zst.ZstdDecompressor()
                stream_reader = decompressor.stream_reader(fp)
                stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
                for line in stream:
                    yield json.loads(line)['text']

class ManualDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: AutoTokenizer, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.items = []

        for text in tqdm(texts):
            self.items.extend(chunk_text(text, tokenizer, max_len))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]



def prepare_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(3)
    size = attention_mask.size(1)
    causal_mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return causal_mask & attention_mask
