"""
Inspect the pre-tokenized LM1B .dat dataset (Arrow shards produced by
scripts/pretokenize_lm1b.py).

Run with:
  micromamba run -p /home/vasilije_ivanovic/envs/mdlm \
    python -u scripts/inspect_lm1b_dat.py [path/to/*.dat]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
from datasets import load_from_disk
from transformers import AutoTokenizer

DEFAULT_PATH = '/home/vasilije_ivanovic/data/lm1b_train_bs128_unwrapped.dat'


def main(path):
  print(f'=== Loading dataset from {path} ===')
  ds = load_from_disk(path)
  print(ds)
  print()
  print('Features:', ds.features)
  print('Num rows:', len(ds))
  print()

  # Peek at the raw Arrow file directly (first shard).
  arrow_files = sorted(f for f in os.listdir(path) if f.endswith('.arrow'))
  first_shard = os.path.join(path, arrow_files[0])
  print(f'=== Raw schema of {arrow_files[0]} ===')
  with pa.memory_map(first_shard, 'r') as source:
    reader = pa.ipc.open_stream(source)
    print(reader.schema)

  # Show a few decoded examples.
  print()
  print('=== First 3 examples ===')
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  for i in range(3):
    row = ds[i]
    print(f'-- example {i} --')
    print('input_ids       :', row['input_ids'][:20], '...')
    print('attention_mask  :', row['attention_mask'][:20], '...')
    print('token_type_ids  :', row['token_type_ids'][:20], '...')
    print('len(input_ids)  :', len(row['input_ids']))
    print('decoded text    :', tokenizer.decode(row['input_ids']))
    print()


if __name__ == '__main__':
  path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
  main(path)
