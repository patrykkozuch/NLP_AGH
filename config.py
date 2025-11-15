import os
from pathlib import Path

BASE_DIR = Path()

DATASETS_DIR = BASE_DIR / "speakleash_dataset"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = BASE_DIR / (f'checkpoints_' + os.getenv('SLURM_JOB_ID', 'local_run'))
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

IGNORE_INDEX = -100

base_cfg = {
    "batch_size": 1,
    "max_len": 512,
    "n_blocks": 12,
    "num_heads": 8,
    "d_model": 1024,
    "d_ff": 4096,
    "log_freq": 1000,
    "prompt_log_freq": 5000,
    "val_freq": 10000,
    "epoches": 50,
    "chkpoint_freq": 5000,
    "gradient_accumulation_steps": 2,
    "tokenizer": 'speakleash/Bielik-1.5B-v3',
    "slurm_job_id": os.getenv('SLURM_JOB_ID', 'local_run'),
}
