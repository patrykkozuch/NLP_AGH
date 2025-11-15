import torch
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from config import IGNORE_INDEX
from transformer.dataset import prepare_mask


def train_step(model, loss_fn, optimizer, scheduler, acc, item):
    model.train()
    mask = prepare_mask(item['attention_mask'][..., :-1])
    inputs = item['input_ids'][..., :-1]

    targets = item['input_ids'][..., 1:].clone()
    target_attention = item['attention_mask'][..., 1:]
    targets[target_attention == 0] = IGNORE_INDEX

    output = model(inputs, mask)

    with acc.autocast():
        loss = loss_fn(output.reshape(-1, output.size(-1)), targets.reshape(-1))

    acc.backward(loss)

    if acc.sync_gradients:
        acc.clip_grad_norm_(model.parameters(), 1)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return loss

def validate(
        acc: Accelerator,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        val_dataloader: DataLoader,
        loss_fn: CrossEntropyLoss,
        steps: int
):
    model.eval()
    total_val_nll = 0.0
    total_tokens = 0
    total_words = 0
    total_chars = 0
    num_batches = 0

    for item in val_dataloader:
        mask = prepare_mask(item['attention_mask'][..., :-1])
        inputs = item['input_ids'][..., :-1]

        targets = item['input_ids'][..., 1:].clone()
        target_attention = item['attention_mask'][..., 1:]
        targets[target_attention == 0] = IGNORE_INDEX

        with acc.autocast(), torch.no_grad():
            output = model(inputs, mask)
            val_loss = loss_fn(output.reshape(-1, output.size(-1)), targets.reshape(-1)).item()

        # accumulate negative log-likelihood (NLL) weighted by number of valid tokens
        num_valid_tokens = (targets.reshape(-1) != IGNORE_INDEX).sum().item()
        total_val_nll += val_loss * num_valid_tokens
        total_tokens += num_valid_tokens

        # Decode target sequences to count words and characters
        batch_target_ids = item['input_ids'][..., 1:]
        batch_target_attn = target_attention
        for ids, attn in zip(batch_target_ids, batch_target_attn):
            valid_len = int(attn.sum().item())
            if valid_len == 0:
                continue
            decoded = tokenizer.decode(ids[:valid_len].tolist(), skip_special_tokens=True)
            words = len(decoded.split())
            chars = len(decoded.replace(" ", ""))
            total_words += max(words, 1)
            total_chars += max(chars, 1)

        num_batches += 1

    avg_val_loss = total_val_nll / total_tokens if total_tokens > 0 else 0.0

    token_ppl = torch.exp(torch.tensor(total_val_nll / total_tokens)) if total_tokens > 0 else torch.tensor(
        float('inf'))
    word_ppl = torch.exp(torch.tensor(total_val_nll / total_words)) if total_words > 0 else torch.tensor(float('inf'))
    char_ppl = torch.exp(torch.tensor(total_val_nll / total_chars)) if total_chars > 0 else torch.tensor(float('inf'))

    acc.log(
        values={
            "Validation Loss": avg_val_loss,
            "Validation Perplexity": token_ppl,
            "Word-level Perplexity": word_ppl,
            "Char-level Perplexity": char_ppl
        },
        step=steps
    )