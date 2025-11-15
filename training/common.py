import os
from typing import Callable

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

import wandb
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from config import IGNORE_INDEX, CHECKPOINTS_DIR
from predict import complete_sentence
from process import process_dataset

from transformer.dataset import ManualDataset, prepare_mask
from transformer.scheduler import TransformerLRScheduler


def setup_accelerator(cfg: dict):
    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
        mode="max-autotune",  # Options: "default", "reduce-overhead", "max-autotune"
        use_regional_compilation=True,
        fullgraph=True,
        dynamic=False
    )

    acc = Accelerator(
        cpu=False,
        log_with='wandb',
        mixed_precision='bf16',
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        dynamo_plugin=dynamo_plugin
    )
    acc.init_trackers(project_name=os.getenv('WANDB_PROJECT'), config=cfg)
    return acc


def load_datasets(tokenizer: PreTrainedTokenizerBase, training_dataset_path: str, cfg: dict):
    split = process_dataset(tokenizer, [training_dataset_path]).train_test_split(test_size=0.1)
    train_dataset = split['train'].with_format('torch')
    valid_dataset = split['test'].with_format('torch')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True
    )

    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["batch_size"],
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )

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


def setup_utils(model: torch.nn.Module, cfg: dict):
    loss_fn = CrossEntropyLoss(label_smoothing=0.1, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=4000)

    return loss_fn, optimizer, scheduler


def log_metrics(acc, loss, scheduler, steps):
    acc.log(
        values={
            "Loss": loss,
            "Perplexity": torch.exp(loss),
            "Learning rate": scheduler.get_last_lr()[0]
        },
        step=steps
    )


def log_examples(acc, model, test_dataloader, tokenizer, table, steps):
    model.eval()

    for text in test_dataloader:
        inputs = text['input_ids']
        attention_mask = text['attention_mask']
        # Complete the sentence
        output_ids, output_text = complete_sentence(
            model,
            inputs,
            attention_mask,
            tokenizer
        )

        table.add_data(steps, text['original_text'][0][0], output_text[0], output_ids[0].cpu().numpy().tolist())

    acc.log({"Example outputs": table}, step=steps)


def save_checkpoint(model, optimizer, steps, cfg: dict):
    ckpt_path = CHECKPOINTS_DIR / f'checkpoint_epoch_{steps}.pt'
    torch.save({
        'steps': steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg,
    }, ckpt_path)


def train(model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, cfg: dict, train_step: Callable,
          val_step: Callable):
    acc = setup_accelerator(cfg)
    train_dataloader, val_dataloader, test_dataloader = load_datasets(tokenizer, 'train.jsonl.zst', cfg)
    loss_fn, optimizer, scheduler = setup_utils(model, cfg)
    model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader = acc.prepare(
        model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader
    )

    steps = 0
    table = wandb.Table(columns=["Steps", "Input", "Output", "Output tokens"], log_mode="INCREMENTAL")

    for epoch in tqdm(range(cfg["epoches"]), disable=not acc.is_local_main_process):
        for item in train_dataloader:
            with acc.accumulate(model):
                loss = train_step(model, loss_fn, optimizer, scheduler, acc, item)
                steps += 1

            if steps % cfg['log_freq'] == 0:
                log_metrics(acc, loss, scheduler, steps)

            if steps % cfg['prompt_log_freq'] == 0:
                log_examples(acc, model, test_dataloader, tokenizer, table, steps)

            if steps % cfg["chkpoint_freq"] == 0:
                save_checkpoint(model, optimizer, steps, cfg)

            if steps % cfg["val_freq"] == 0:
                val_step(acc, model, tokenizer, val_dataloader, loss_fn, steps)

    acc.end_training()
