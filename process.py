import datasets

from config import tokenizer

def tokenize(texts):
    return tokenizer(
        texts['text'],
        add_special_tokens=True,
        return_tensors=None,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        padding_side='right',
        max_length=513,
        return_overflowing_tokens=True,
        stride=64
    )

def process_dataset(file_paths: list[str], output_path: str):
    dataset = (
        datasets.load_dataset('json', data_files=file_paths, num_proc=20)
        .filter(lambda x: x['meta']['quality'] == 'HIGH', num_proc=20, desc='Filtering by quality')
        .map(tokenize, batched=True, num_proc=20, remove_columns=['meta', 'text'], desc='Tokenizing')
        .remove_columns('overflow_to_sample_mapping')
    )
    dataset['train'].to_json(output_path)
    return dataset

train_datasets = [
    'speakleash_dataset/plwikisource.jsonl.zst',
]

valid_datasets = [
    'speakleash_dataset/wolne_lektury_corpus.jsonl.zst'
]

process_dataset(train_datasets, 'chunked.train.jsonl.zst')
process_dataset(valid_datasets, 'chunked.valid.jsonl.zst')
