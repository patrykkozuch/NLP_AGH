import datasets
from transformers import PreTrainedTokenizerBase


def tokenize(tokenizer, texts):
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

def process_dataset(tokenizer: PreTrainedTokenizerBase, file_paths: list[str]):
    dataset = (
        datasets.load_dataset('json', data_files=file_paths, split='train')
        .filter(lambda x: x['meta']['quality'] == 'HIGH', num_proc=20, desc='Filtering by quality')
        .map(lambda x: tokenize(tokenizer, x), batched=True, num_proc=20, remove_columns=['meta', 'text'], desc='Tokenizing')
        .remove_columns('overflow_to_sample_mapping')
    )
    return dataset