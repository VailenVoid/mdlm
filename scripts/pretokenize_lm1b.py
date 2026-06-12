"""
Pre-tokenize LM1B (train + test splits, block_size=128, unwrapped)
once, on CPU, so the cached .dat files already exist before the
multi-GPU training job starts.

Run with:
  micromamba run -p /home/vasilije_ivanovic/envs/mdlm \
    python -u scripts/pretokenize_lm1b.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omegaconf

import dataloader

CACHE_DIR = '/home/vasilije_ivanovic/data'
BLOCK_SIZE = 128
NUM_PROC = 64

config = omegaconf.OmegaConf.create(
  {'data': {'tokenizer_name_or_path': 'bert-base-uncased'}})
tokenizer = dataloader.get_tokenizer(config)

for mode in ['train', 'test']:
  print(f'--- tokenizing lm1b/{mode} ---', flush=True)
  dataloader.get_dataset(
    'lm1b', tokenizer, wrap=False, mode=mode,
    cache_dir=CACHE_DIR, block_size=BLOCK_SIZE, num_proc=NUM_PROC)

print('done')
