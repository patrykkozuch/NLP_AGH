import os
from pathlib import Path

from transformers import AutoTokenizer

BASE_DIR = Path()

DATASETS_DIR = BASE_DIR / "speakleash_dataset"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = BASE_DIR / (f'checkpoints_' + os.getenv('SLURM_JOB_ID'))
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

IGNORE_INDEX = -100

cfg = {
    "batch_size": 64,
    "max_len": 512,
    "n_blocks": 12,
    "num_heads": 12,
    "d_model": 768,
    "d_ff": 3072,
    "log_freq": 1000,
    "prompt_log_freq": 5000,
    "val_freq": 10000,
    "epoches": 50,
    "chkpoint_freq": 5000,
    "slurm_job_id": os.getenv('SLURM_JOB_ID', 'local_run')
}

tokenizer = AutoTokenizer.from_pretrained('speakleash/Bielik-1.5B-v3', use_fast=True)
