import os
from pathlib import Path

from transformers import AutoTokenizer

BASE_DIR = Path()

DATASETS_DIR = BASE_DIR / "speakleash_dataset"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = BASE_DIR / (f'checkpoints_' + os.getenv('SLURM_JOB_ID'))
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

cfg = {
    "batch_size": 128,
    "max_len": 256,
    "n_blocks": 6,
    "num_heads": 8,
    "d_model": 512,
    "d_ff": 2048,
    "log_freq": 1000,
    "prompt_log_freq": 5000,
    "epoches": 50,
    "chkpoint_freq": 5000,
    "slurm_job_id": os.getenv('SLURM_JOB_ID', 'local_run')
}

tokenizer = AutoTokenizer.from_pretrained('speakleash/Bielik-1.5B-v3', use_fast=True)
