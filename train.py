import os
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import Dataset, load_from_disk, load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from transformer.dataset import prepare_mask, ManualDataset
from transformer.transformer import Transformer


def complete_sentence(model, input_ids, attention_mask, tokenizer):
    """
    Complete a sentence by generating tokens autoregressively.

    Args:
        model: Transformer model
        input_ids: Input token IDs (batch_size, seq_len)
        attention_mask: Attention mask (batch_size, seq_len)
        tokenizer: Tokenizer

    Returns:
        completed_text: Decoded text
    """
    model.eval()

    with torch.no_grad():
        # Get prediction for the entire sequence
        mask = prepare_mask(attention_mask)
        output = model(input_ids, mask)  # (batch, seq_len, vocab_size)

    logits = torch.argmax(output, -1)

    # Decode to text
    return tokenizer.batch_decode(logits, skip_special_tokens=True)


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
dataset = load_dataset('data', data_files={'train': 'dataset.parquet'}, split='train', streaming=True).with_format('torch')
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
                output_text = complete_sentence(
                    transformer,
                    inputs,
                    attention_mask,
                    tokenizer
                )

                table.add_data(steps, text['original_text'][0][0], output_text[0])

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