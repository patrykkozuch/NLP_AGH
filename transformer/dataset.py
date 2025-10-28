import io
import json

import torch
import zstandard as zst

from torch.utils.data import Dataset
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


def chunk_text(tokens, tokenizer, chunk_size=512):
    input_ids = tokens

    if len(input_ids) == 0:
        return []

    chunks = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for start in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[start:start + chunk_size]

        if len(chunk_ids) < 2:
            continue

        actual_length = len(chunk_ids)

        input_chunk = chunk_ids  # All except last
        label_chunk = chunk_ids[1:]

        if start + chunk_size < len(input_ids):
            label_chunk += [input_ids[start + chunk_size]]
        else:
            label_chunk += [tokenizer.pad_token_id]

        # Pad to chunk_size - 1 (since we removed one token for shifting)
        max_len = chunk_size
        padding_length = max_len - len(input_chunk)

        if padding_length > 0:
            input_chunk = input_chunk + [tokenizer.pad_token_id] * padding_length
            label_chunk = label_chunk + [-100] * padding_length  # Ignore padding in loss

        # Attention mask
        attention_mask = [1] * actual_length + [0] * (padding_length)

        chunks['input_ids'].append(input_chunk)
        chunks['attention_mask'].append(attention_mask)
        chunks['labels'].append(label_chunk)

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
            tokens = tokenizer(
                text,
                add_special_tokens=True,
                return_tensors=None,
                return_attention_mask=False,
                truncation=True,
                max_length=16384
            )
            items = chunk_text(tokens['input_ids'], tokenizer, max_len)
            items = {k: torch.tensor(v, dtype=torch.long).squeeze(0) for k, v in items.items()}
            items['original_text'] = [text for _ in range(len(items['labels']))]
            self.items.append(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]



def prepare_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    padding_mask = ~attention_mask.bool()

    combined_mask = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(2)
    combined_mask = combined_mask.unsqueeze(1)

    return combined_mask
