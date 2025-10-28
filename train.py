import os
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from transformer.dataset import SpeakLeashDataset, prepare_mask, ManualDataset
from transformer.transformer import Transformer


def complete_sentence(model, input_ids, attention_mask, tokenizer, max_new_tokens=64, device='cuda'):
    """
    Complete a sentence by generating tokens autoregressively.

    Args:
        model: Transformer model
        input_ids: Input token IDs (batch_size, seq_len)
        attention_mask: Attention mask (batch_size, seq_len)
        tokenizer: Tokenizer
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run on

    Returns:
        completed_tokens: Generated token IDs
        completed_text: Decoded text
    """
    model.eval()

    # Clone input to avoid modifying original
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone()
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get prediction for the entire sequence
            mask = prepare_mask(current_mask).to(device)
            output = model(current_ids, mask)  # (batch, seq_len, vocab_size)

            # Get prediction for the last real token position
            # Find last non-padded position
            real_positions = (current_mask == 1).long()
            last_real_idx = real_positions.sum(dim=1) - 1  # (batch_size,)
            batch_indices = torch.arange(current_ids.size(0), device=device)

            # Get logits for last position of each sample in batch
            next_token_logits = output[batch_indices, last_real_idx]  # (batch_size, vocab_size)

            # Sample next token (take argmax for deterministic generation)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)
            generated_tokens.append(next_token)

            # Check if we reached end-of-sequence token
            if (next_token == tokenizer.eos_token_id).all():
                break

            # Append next token to sequence
            last_real_idx += 1
            current_ids = torch.cat([current_ids[:, :last_real_idx], next_token, current_ids[:, last_real_idx:-1]], dim=1)
            # Update attention mask
            new_mask = torch.ones_like(next_token)
            current_mask = torch.cat([current_mask[:, :last_real_idx], new_mask, current_ids[:, last_real_idx:-1]], dim=1)

    # Concatenate all generated tokens
    if generated_tokens:
        all_generated = torch.cat(generated_tokens, dim=1)  # (batch_size, num_generated)
        completed_ids = torch.cat([input_ids, all_generated], dim=1)
    else:
        completed_ids = input_ids

    # Decode to text
    completed_text = tokenizer.batch_decode(completed_ids, skip_special_tokens=True)
    return completed_ids, completed_text


cfg = {
    "batch_size": 128,
    "max_len": 256,
    "n_blocks": 6,
    "num_heads": 8,
    "d_model": 512,
    "d_ff": 2048,
    "log_freq": 1000,
    "prompt_log_freq": 5000,
    "epoches": 100,
    "chkpoint_freq": 5000
}

acc = Accelerator(cpu=False, mixed_precision='bf16', log_with='wandb')
acc.init_trackers(project_name=os.getenv('WANDB_PROJECT'), config=cfg)

tokenizer = AutoTokenizer.from_pretrained('speakleash/Bielik-1.5B-v3')
dataset = SpeakLeashDataset("datasets", tokenizer, max_len=cfg["max_len"])
dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)

transformer = Transformer(vocab_size=len(tokenizer), seq_len=cfg["max_len"], n_blocks=cfg["n_blocks"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], d_model=cfg["d_model"])
loss_fn = CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(transformer.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(optimizer, cfg["epoches"])

test_data = [
    "Pamięć nie jest linią prostą. To raczej labirynt, w którym echo jednego kroku potrafi niespodziewanie",
    "Zapadło między nimi milczenie tak gęste, że można je było kroić nożem. Nie była to jednak cisza pusta – przeciwnie, była ona naładowana tym wszystkim, co",
    "Z każdą chwilą kontury pokoju stawały się coraz mniej wyraźne. Nie był pewien, czy to zmęczenie, czy może",
    "Obserwował swoje dłonie, jakby należały do kogoś zupełnie obcego. Widział, jak podnoszą filiżankę, jak obracają klucz w zamku, ale on sam znajdował się",
    "Pustka nie była jedynie brakiem. Była substancją, ciężką i zimną, która osiadała na meblach",
    "Światło lampy drżało lekko, jakby wahało się, czy pozostać. Cienie na ścianach zdawały się szeptać coś o tym, co",
    "Powietrze w pokoju stało się ciężkie, niemal namacalne. Miał wrażenie, że każdy jego oddech prowadzi go coraz bliżej miejsca, gdzie",
    "Zegar tykał z uporem, który wydawał się kpić z jego bezruchu. W każdej sekundzie kryło się coś, czego nie potrafił",
    "Krople deszczu spływały po szybie, zlewając się w krzywe linie. Przez chwilę zobaczył w nich twarze tych, których",
    "Ulica była pusta, choć miał pewność, że ktoś go obserwuje. Kroki odbijały się echem od kamieni, niosąc w sobie coś, co",
    "Na dnie filiżanki został osad, ciemny jak noc bez gwiazd. Wpatrywała się w niego, jakby próbowała odczytać z niego to, czego",
    "Zapach kurzu i starych książek otulał go niczym wspomnienie. Każda strona, którą przewracał, zdawała się przypominać mu o tym, że",
    "Świat za oknem przesuwał się powoli, jak w starym filmie. Granica między tym, co pamiętał, a tym, co sobie wyobrażał, stawała się",
    "Kiedy wypowiedziała jego imię, brzmiało ono inaczej niż kiedykolwiek wcześniej. W tym jednym słowie kryło się coś, co mogło wszystko",
    "Na chwilę wydawało mu się, że zrozumiał. Lecz myśl wymknęła się, zanim zdążył ją pochwycić, pozostawiając po sobie tylko echo, które"
]

test_dataset = ManualDataset(test_data, tokenizer, cfg["max_len"])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

columns = ["Steps", "Input", "Output"]

CHECKPOINTS_DIR = Path(f'checkpoints_' + os.getenv('SLURM_JOB_ID'))
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

transformer, loss_fn, optimizer, dataloader, test_dataloader = acc.prepare(transformer, loss_fn, optimizer, dataloader, test_dataloader)

steps = 0
table = wandb.Table(columns=columns, log_mode="INCREMENTAL")

transformer.train()

for epoch in tqdm(range(cfg["epoches"])):
    for item in tqdm(dataloader, total=len(dataloader), leave=False):
        mask = prepare_mask(item['attention_mask'])
        inputs = item['input_ids']
        targets = item['labels']

        optimizer.zero_grad()

        output = transformer(inputs, mask)
        loss = loss_fn(output.view(-1, output.size(-1)), targets.view(-1))
        acc.backward(loss)
        acc.clip_grad_norm_(transformer.parameters(), 1.0)

        optimizer.step()

        if steps % cfg['log_freq'] == 0:
            acc.log(
                values={
                    "Loss": loss,
                    "Perplexity": torch.exp(loss),
                    "Learning rate": scheduler.get_last_lr()[0]
                },
                step=steps
            )

        if steps % cfg['prompt_log_freq'] == 0:
            transformer.eval()

            for text in test_dataloader:
                inputs = text['input_ids']
                attention_mask = text['attention_mask']

                # Complete the sentence
                _, output_text = complete_sentence(
                    transformer,
                    inputs,
                    attention_mask,
                    tokenizer,
                    max_new_tokens=128,
                    device='cuda'
                )

                table.add_data(steps, text['original_text'][0], output_text[0])

            acc.log({"Example outputs": table}, step=steps)

            transformer.train()

        if steps % cfg["chkpoint_freq"] == 0:

            ckpt_path = CHECKPOINTS_DIR / f'checkpoint_epoch_{steps}.pt'
            torch.save({
                'steps': steps,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'cfg': cfg,
            }, ckpt_path)

        steps += 1

    scheduler.step()