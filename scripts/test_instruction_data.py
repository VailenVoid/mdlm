"""CPU smoke test for the instruction dataloader path."""
import sys

sys.path.insert(0, '/home/vasilije_ivanovic/mdlm')

import torch
from omegaconf import OmegaConf

import dataloader

config = OmegaConf.create({
  'data': {
    'train': 'instruction',
    'valid': 'instruction',
    'tokenizer_name_or_path': 't5-small',
    'cache_dir': '/home/vasilije_ivanovic/data',
    'wrap': False,
    'streaming': False,
    'instruction_train_dir': '/home/vasilije_ivanovic/sde_llm/data/instruction_data/instruction_train',
    'instruction_valid_dir': '/home/vasilije_ivanovic/sde_llm/data/instruction_data/instruction_valid',
  },
  'model': {'length': 1024},
})

tokenizer = dataloader.get_tokenizer(config)
print('eos:', tokenizer.eos_token, tokenizer.eos_token_id,
      '| pad:', tokenizer.pad_token, tokenizer.pad_token_id,
      '| vocab:', tokenizer.vocab_size)

train_set, valid_set = dataloader.get_instruction_datasets(config, tokenizer)
print('train len:', len(train_set), '| valid len:', len(valid_set))

for name, ds, indices in [('train', train_set, [0, 1, len(train_set) - 1]),
                          ('valid', valid_set, [0, len(valid_set) - 1])]:
  for i in indices:
    ex = ds[i]
    ids, cm = ex['input_ids'], ex['cond_mask']
    assert ids.shape == (1024,) and ids.dtype == torch.long, ids.shape
    assert cm.shape == (1024,) and cm.dtype == torch.bool, cm.shape
    cond_len = int(cm.sum())
    # cond_mask must be a contiguous prefix
    assert cm[:cond_len].all() and not cm[cond_len:].any()
    n_eos_tail = int((ids == tokenizer.eos_token_id).flip(0).cummin(0)[0].sum()) \
      if hasattr(torch, 'cummin') else -1
    print(f'{name}[{i}]: cond_len={cond_len}, '
          f'response+pad={1024 - cond_len}, eos_tail={n_eos_tail}')

# Default-collate a batch exactly like Lightning will after on_train_start
loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False)
batch = next(iter(loader))
print('batch input_ids:', batch['input_ids'].shape, batch['input_ids'].dtype)
print('batch cond_mask:', batch['cond_mask'].shape, batch['cond_mask'].dtype)

ex = train_set[0]
cond_len = int(ex['cond_mask'].sum())
print('\n--- decoded condition ---')
print(tokenizer.decode(ex['input_ids'][:cond_len]))
print('--- decoded response (first 80 tok) ---')
print(tokenizer.decode(ex['input_ids'][cond_len:cond_len + 80]))

# Simulate the q_xt clamp: condition must survive full masking
mask_index = tokenizer.vocab_size  # T5 has no mask token -> appended id
x0 = batch['input_ids']
move_chance = torch.ones(x0.shape[0], 1)
move_indices = torch.rand(*x0.shape) < move_chance
xt = torch.where(move_indices, mask_index, x0)
xt = torch.where(batch['cond_mask'], x0, xt)
assert (xt[batch['cond_mask']] != mask_index).all()
assert (xt[~batch['cond_mask']] == mask_index).all()
print('\nq_xt clamp OK: condition never masked, response fully maskable')
print('SMOKE TEST PASSED')
