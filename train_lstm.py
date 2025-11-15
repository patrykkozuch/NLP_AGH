import argparse

from transformers import AutoTokenizer

import training.lstm
from config import base_cfg
from lstm.lstm import LstmModel
from training.common import train


def main(cfg: dict = None):
    if cfg is None:
        cfg = base_cfg
    else:
        cfg = base_cfg | cfg

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer"],
        use_fast=True,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
        add_pad_token=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = LstmModel(
        vocab_size=len(tokenizer),
        seq_len=cfg["max_len"],
        n_blocks=cfg["n_blocks"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        d_model=cfg["d_model"]
    )

    train(model, tokenizer, cfg, training.lstm.train_step, training.lstm.validate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument('--tokenizer', type=str, help='Tokenizer to use for training.', default=base_cfg['tokenizer'])
    parser.add_argument('--d_model', type=int, help='Dimension of the model.', default=base_cfg['d_model'])
    parser.add_argument('--d_ff', type=int, help='Dimension of the feedforward network.', default=base_cfg['d_ff'])
    parser.add_argument('--n_blocks', type=int, help='Number of transformer blocks.', default=base_cfg['n_blocks'])
    parser.add_argument('--num_heads', type=int, help='Number of attention heads.', default=base_cfg['num_heads'])

    custom_cfg = vars(parser.parse_args())
    main(custom_cfg)
