import torch

from torch.utils.data import Dataset

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def chunk_text(tokens: list[int], tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, chunk_size=256):
    """
    Splits tokenized input into chunks of specified size.

    Args:
        tokens (list[int]): List of token IDs.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer for padding token
        chunk_size (int): Size of each chunk.

    Returns:
        dict: Dictionary with keys 'input_ids', 'attention_mask', and 'labels' containing lists of chunks.
    """
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
            label_chunk += [tokenizer.eos_token_id]

        # Pad to chunk_size - 1 (since we removed one token for shifting)
        max_len = chunk_size
        padding_length = max_len - len(input_chunk)

        if padding_length > 0:
            input_chunk = input_chunk + [tokenizer.pad_token_id] * padding_length
            label_chunk = label_chunk + [-100] * padding_length  # Ignore padding in loss

        # Attention mask
        attention_mask = [1] * actual_length + [0] * padding_length

        chunks['input_ids'].append(input_chunk)
        chunks['attention_mask'].append(attention_mask)
        chunks['labels'].append(label_chunk)

    return chunks

class ManualDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, max_len=256):
        self.items = []

        for text in tqdm(texts):
            tokens = tokenizer(
                text,
                add_special_tokens=True,
                return_tensors='pt',
                return_attention_mask=True,
                truncation=True,
                max_length=max_len
            )

            # Squeeze batch dimension
            items = {k: v.squeeze(0) for k, v in tokens.items()}
            items['original_text'] = [text]
            self.items.append(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]



def prepare_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Combines causal mask and padding mask for transformer attention.
    """
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    padding_mask = ~attention_mask.bool()

    combined_mask = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(2)
    combined_mask = combined_mask.unsqueeze(1)

    return combined_mask
