import os

import torch
from accelerate import Accelerator
from datasets import load_dataset

import wandb
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg, tokenizer, CHECKPOINTS_DIR
from predict import complete_sentence
from transformer.dataset import prepare_mask, ManualDataset
from transformer.scheduler import TransformerLRScheduler
from transformer.transformer import Transformer


def setup_accelerator():
    acc = Accelerator(cpu=False, mixed_precision='bf16', log_with='wandb', gradient_accumulation_steps=cfg["gradient_accumulation_steps"])
    acc.init_trackers(project_name=os.getenv('WANDB_PROJECT'), config=cfg)
    return acc


def load_datasets():
    train_dataset = load_dataset(
        'json',
        data_files={'train': 'chunked.plwikisource.jsonl.zst'},
        split='train'
    ).with_format('torch')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], num_workers=4, persistent_workers=True, pin_memory=True)

    val_dataset = load_dataset(
        'json',
        data_files={'validation': 'chunked.wolne_lektury_corpus.jsonl.zst'},
        split='validation'
    ).with_format('torch')
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["batch_size"], num_workers=4, persistent_workers=True, pin_memory=True)

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

    return train_dataloader, val_dataloader, test_dataloader


def setup_model():
    transformer = Transformer(
        vocab_size=len(tokenizer),
        seq_len=cfg["max_len"],
        n_blocks=cfg["n_blocks"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        d_model=cfg["d_model"]
    )

    loss_fn = CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=4000)

    return transformer, loss_fn, optimizer, scheduler


def prepare_with_acc(acc, transformer, loss_fn, optimizer, train_dataloader, val_dataloader, test_dataloader):
    (
        transformer,
        loss_fn,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader
    ) = acc.prepare(
        transformer,
        loss_fn,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader
    )
    return transformer, loss_fn, optimizer, train_dataloader, val_dataloader, test_dataloader


def train_step(transformer, loss_fn, optimizer, acc, item):
    mask = prepare_mask(item['attention_mask'])
    inputs = item['input_ids']
    targets = item['labels']


    output = transformer(inputs, mask)
    loss = loss_fn(output.view(-1, output.size(-1)), targets.view(-1))
    acc.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    return loss


def log_metrics(acc, loss, scheduler, steps):
    acc.log(
        values={
            "Loss": loss,
            "Perplexity": torch.exp(loss),
            "Learning rate": scheduler.get_last_lr()[0]
        },
        step=steps
    )


def log_examples(acc, transformer, test_dataloader, tokenizer, table, steps):
    transformer.eval()

    for text in test_dataloader:
        inputs = text['input_ids']
        attention_mask = text['attention_mask']
        # Complete the sentence
        output_ids, output_text = complete_sentence(
            transformer,
            inputs,
            attention_mask,
            tokenizer
        )

        table.add_data(steps, text['original_text'][0][0], output_text[0], output_ids[0].cpu().numpy().tolist())

    acc.log({"Example outputs": table}, step=steps)

    transformer.train()


def save_checkpoint(transformer, optimizer, steps):
    ckpt_path = CHECKPOINTS_DIR / f'checkpoint_epoch_{steps}.pt'
    torch.save({
        'steps': steps,
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'cfg': cfg,
    }, ckpt_path)


def validate(acc, transformer, val_dataloader, loss_fn, steps):
    total_val_loss = 0.0
    num_batches = 0

    for item in val_dataloader:
        mask = prepare_mask(item['attention_mask'])
        inputs = item['input_ids']
        targets = item['labels']

        with torch.no_grad():
            output = transformer(inputs, mask)
            val_loss = loss_fn(output.view(-1, output.size(-1)), targets.view(-1))

        total_val_loss += val_loss.item()
        num_batches += 1

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0

    acc.log(
        values={
            "Validation Loss": avg_val_loss,
            "Validation Perplexity": torch.exp(torch.tensor(avg_val_loss))
        },
        step=steps
    )


def main():
    acc = setup_accelerator()
    train_dataloader, val_dataloader, test_dataloader = load_datasets()
    transformer, loss_fn, optimizer, scheduler = setup_model()
    transformer, loss_fn, optimizer, train_dataloader, val_dataloader, test_dataloader = prepare_with_acc(
        acc, transformer, loss_fn, optimizer, train_dataloader, val_dataloader, test_dataloader
    )

    steps = 0
    table = wandb.Table(columns=["Steps", "Input", "Output", "Output tokens"], log_mode="INCREMENTAL")
    transformer.train()

    for epoch in tqdm(range(cfg["epoches"])):
        for item in tqdm(train_dataloader, total=len(train_dataloader), leave=False):
            with acc.accumulate(transformer):
                loss = train_step(transformer, loss_fn, optimizer, acc, item)

                if steps % cfg['log_freq'] == 0:
                    log_metrics(acc, loss, scheduler, steps)

                if steps % cfg['prompt_log_freq'] == 0:
                    log_examples(acc, transformer, test_dataloader, tokenizer, table, steps)

                if steps % cfg["chkpoint_freq"] == 0:
                    save_checkpoint(transformer, optimizer, steps)

                steps += 1
                scheduler.step()

        validate(acc, transformer, val_dataloader, loss_fn, steps)

    acc.end_training()


if __name__ == "__main__":
    main()
