from pathlib import Path

import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from transformer.dataset import SpeakLeashDataset, prepare_mask, ManualDataset
from transformer.scheduler import TransformerLRScheduler
from transformer.transformer import Transformer

from accelerate import Accelerator

cfg = {
    "batch_size": 64,
    "max_len": 512,
    "n_blocks": 6,
    "num_heads": 8,
    "d_model": 512,
    "d_ff": 2048,
    "log_freq": 10,
    "prompt_log_freq": 10,
    "epoches": 1000,
    "chkpoint_freq": 50
}


tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
dataset = SpeakLeashDataset("datasets", tokenizer, max_len=cfg["max_len"])
dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

accelerator = Accelerator()
device = accelerator.device

transformer = Transformer(vocab_size=len(tokenizer), seq_len=cfg["max_len"], n_blocks=cfg["n_blocks"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], d_model=cfg["d_model"])

loss_fn = CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerLRScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=4000)

test_data = [
    "Pamięć nie jest linią prostą. To raczej labirynt, w którym echo jednego kroku potrafi niespodziewanie",
    "Zapadło między nimi milczenie tak gęste, że można je było kroić nożem. Nie była to jednak cisza pusta – przeciwnie, była ona naładowana tym wszystkim, co",
    "Z każdą chwilą kontury pokoju stawały się coraz mniej wyraźne. Nie był pewien, czy to zmęczenie, czy może",
    "Obserwował swoje dłonie, jakby należały do kogoś zupełnie obcego. Widział, jak podnoszą filiżankę, jak obracają klucz w zamku, ale on sam znajdował się",
    "Pustka nie była jedynie brakiem. Była substancją, ciężką i zimną, która osiadała na meblach"
]

test_dataset = ManualDataset(test_data, tokenizer, cfg["max_len"])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

transformer, optimizer, dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, dataloader, test_dataloader)

columns = ["Epoch", "Input", "Output"]

CHECKPOINTS_DIR = Path('checkpoints')
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


with wandb.init(config=cfg) as run:
    table = wandb.Table(columns=columns, log_mode="INCREMENTAL")
    run.watch(transformer, loss_fn, log_freq=10)

    transformer.train()
    for epoch in tqdm(range(cfg["epoches"])):
        for item in tqdm(dataloader, total=len(dataloader), leave=False):
            mask = prepare_mask(item['attention_mask'])
            inputs = item['input_ids']
            targets = item['labels']

            optimizer.zero_grad()

            output = transformer(inputs, mask)
            loss = loss_fn(output.view(-1, output.size(-1)), targets.view(-1))
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()

        if epoch % cfg['log_freq'] == 0:
            run.log({"Loss": loss.item(), "Perplexity": torch.exp(loss).item()}, step=epoch)

        if epoch % cfg['prompt_log_freq'] == 0:
            transformer.eval()

            for text in test_dataloader:
                inputs = text['input_ids']
                mask = prepare_mask(text['attention_mask'])
                output = transformer(inputs, mask)
                out_token_ids = torch.argmax(output, -1)
                output_text = tokenizer.batch_decode(out_token_ids, skip_special_tokens=True)[0]
                table.add_data(epoch, text['original_text'][0], output_text)

            run.log({"Example outputs": table}, step=epoch)

            transformer.train()

        if epoch % cfg["chkpoint_freq"] == 0:

            ckpt_path = CHECKPOINTS_DIR / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'cfg': cfg,
            }, ckpt_path)
