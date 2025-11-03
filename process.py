import pandas as pd
import datasets
from transformers import AutoTokenizer

from config import cfg
from transformer.dataset import chunk_text

tokenizer = AutoTokenizer.from_pretrained('speakleash/Bielik-1.5B-v3', use_fast=True)

def tokenize(texts):
    return tokenizer(
        texts['text'],
        add_special_tokens=True,
        return_tensors=None,
        return_attention_mask=False,
        truncation=True,
        max_length=32768
    )

def chunk(tokens):
    return chunk_text(tokens['input_ids'], tokenizer, cfg['max_len'])

def split_column(df: pd.DataFrame):
    return df.explode(["input_ids", "attention_mask", "labels"])


def process_dataset(file_path: str, output_path: str):
    dataset = (
        datasets.load_dataset('json', data_files=[file_path], num_proc=20)
        .map(tokenize, batched=True, num_proc=20, remove_columns=['meta', 'text'])
        .map(chunk, batched=False, num_proc=20, remove_columns=['input_ids'])
        # Chunk method produces lists inside the columns, we need to explode them
        .with_format('pandas')
        .map(split_column, batched=True)
        .remove_columns('__index_level_0__')
    )
    dataset['train'].to_json(output_path)


process_dataset('speakleash_dataset/plwikisource.jsonl.zst', 'chunked.plwikisource.jsonl.zst')
process_dataset('speakleash_dataset/wolne_lektury_corpus.jsonl.zst', 'chunked.wolne_lektury_corpus.jsonl.zst')
