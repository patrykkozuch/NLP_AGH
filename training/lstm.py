import torch
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

def train_step(model, loss_fn, optimizer, scheduler, acc, item):
    model.train()

    inputs = item['input_ids'][..., :-1]
    targets = item['input_ids'][..., 1:]

    output = model(inputs)

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
        val_dataloader: DataLoader,
        loss_fn: CrossEntropyLoss,
        steps: int
):
    model.eval()
    total_val_loss = 0.0
    num_batches = 0

    for item in val_dataloader:
        inputs = item['input_ids'][..., :-1]

        targets = item['input_ids'][..., 1:].clone()

        with acc.autocast(), torch.no_grad():
            output = model(inputs)
            val_loss = loss_fn(output.reshape(-1, output.size(-1)), targets.reshape(-1)).item()

        total_val_loss += val_loss
        num_batches += 1

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0

    acc.log(
        values={
            "Validation Loss": avg_val_loss,
            "Validation Perplexity": torch.exp(torch.tensor(avg_val_loss))
        },
        step=steps
    )