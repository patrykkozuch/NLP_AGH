import re

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

    metrics = torch.zeros(4, device=acc.device)

    for item in val_dataloader:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']

        inputs = input_ids[..., :-1]
        targets = input_ids[..., 1:].clone()

        mask = prepare_mask(attention_mask[..., :-1])

        target_attention = attention_mask[..., 1:]
        targets[target_attention == 0] = IGNORE_INDEX

        with torch.no_grad():
            output = model(inputs, mask)

            logits = output.reshape(-1, output.size(-1))
            flat_targets = targets.reshape(-1)

            loss = loss_fn(logits, flat_targets)

        num_valid_tokens = (flat_targets != -100).sum()
        batch_nll = loss * num_valid_tokens

        metrics[0] += batch_nll
        metrics[1] += num_valid_tokens

        batch_word_count = 0
        batch_char_count = 0

        batch_target_ids = input_ids[..., 1:]

        for ids, attn in zip(batch_target_ids.cpu(), target_attention.cpu()):
            valid_len = int(attn.sum().item())
            if valid_len == 0:
                continue

            decoded = tokenizer.decode(ids[:valid_len], skip_special_tokens=True)

            words_in_seq = len(re.findall(r'\w+', decoded, re.UNICODE))
            chars_in_seq = len(decoded.replace(" ", ""))

            batch_word_count += max(words_in_seq, 1)  # Safety min 1
            batch_char_count += max(chars_in_seq, 1)

        metrics[2] += batch_word_count
        metrics[3] += batch_char_count

    # Sum metrics across all GPUs
    metrics = acc.reduce(metrics, reduction="sum")

    total_val_nll = metrics[0].item()
    total_tokens = metrics[1].item()
    total_words = metrics[2].item()
    total_chars = metrics[3].item()

    # Avoid division by zero
    if total_tokens == 0 or total_words == 0:
        acc.print("Warning: Validation set was empty or fully masked.")
        return

    avg_val_loss = total_val_nll / total_tokens
    token_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
    word_ppl = torch.exp(torch.tensor(total_val_nll / total_words)).item()
    char_ppl = torch.exp(torch.tensor(total_val_nll / total_chars)).item()

    acc.log(
        values={
            "val_loss": avg_val_loss,
            "ppl_token": token_ppl,
            "ppl_word": word_ppl,
            "ppl_char": char_ppl
        },
        step=steps
    )

    acc.print(f"Step {steps}: Token PPL: {token_ppl:.2f} | Word PPL: {word_ppl:.2f}")