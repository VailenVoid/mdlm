"""
Quick standalone probe: does the model's prediction for a masked
position depend on the *values* of currently-unmasked tokens
elsewhere in the sequence?

Run on a GPU node (flash_attn requires CUDA):

  micromamba run -p /home/vasilije_ivanovic/envs/mdlm \
    python -u scripts/test_context_dependence.py
"""
import os
import sys

import hydra
import omegaconf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataloader
import diffusion

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


with hydra.initialize(version_base=None, config_path='../configs'):
  config = hydra.compose(config_name='config', overrides=[
    'data=lm1b',
    'model=small',
    'backbone=dit',
    'parameterization=subs',
    'model.length=16',
    'loader.batch_size=1',
    'loader.eval_batch_size=1',
  ])

tokenizer = dataloader.get_tokenizer(config)

# Random init, no checkpoint needed -- we're only probing the
# architecture's information flow, not trained predictions.
model = diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')
model.eval()

# The DiT architecture zero-initializes the adaLN gate params and the
# final output layer (standard diffusion-model init trick for stable
# training). At pure init this makes attention/MLP contribute nothing
# to the residual stream, and the final logits are identically 0
# regardless of input. That makes a freshly-initialized model useless
# for probing information flow, so we overwrite those zero'd params
# with small random noise -- purely for this structural test, not
# training.
with torch.no_grad():
  for module in model.backbone.modules():
    if isinstance(module, torch.nn.Linear):
      if torch.all(module.weight == 0):
        module.weight.normal_(mean=0.0, std=0.02)
      if module.bias is not None and torch.all(module.bias == 0):
        module.bias.normal_(mean=0.0, std=0.02)

L = config.model.length
print(f'mask_index = {model.mask_index}, vocab_size = {model.vocab_size}')

# ---- 1. Inspect the shape/dtype of x ----
x = model._sample_prior(1, L).to('cuda')
print(f'x.shape = {tuple(x.shape)}, x.dtype = {x.dtype}')
print(f'x (all-mask prior) = {x}')

# ---- 2. Inspect the shape of the forward output ----
sigma_t, _ = model.noise(torch.tensor([0.5], device='cuda'))
with torch.no_grad():
  logits = model.forward(x, sigma_t)
print(f'logits.shape = {tuple(logits.shape)}')  # (B, L, vocab_size)

# ---- 3. Content-dependence test ----
# Build a sequence where every position EXCEPT `target_pos` is a
# concrete (unmasked) token, and `target_pos` is MASK. Compare the
# model's predicted distribution at `target_pos` before and after
# changing the token at a different position `other_pos`.
torch.manual_seed(0)
target_pos = 5
other_pos = 10

base = torch.randint(0, model.vocab_size - 1, (1, L), device='cuda')
base[0, target_pos] = model.mask_index

variant = base.clone()
# pick a different token id for other_pos
new_token = (base[0, other_pos] + 1) % (model.vocab_size - 1)
variant[0, other_pos] = new_token

print(f'base[other_pos]    = {base[0, other_pos].item()}')
print(f'variant[other_pos] = {variant[0, other_pos].item()}')

with torch.no_grad():
  logits_base = model.forward(base, sigma_t)
  logits_variant = model.forward(variant, sigma_t)

p_base = logits_base[0, target_pos]
p_variant = logits_variant[0, target_pos]

max_abs_diff = (p_base - p_variant).abs().max().item()
print(f'max |delta logit| at target_pos from changing other_pos: '
      f'{max_abs_diff:.6f}')
print('-> If this is ~0, the prediction at target_pos is independent '
      'of other_pos.\n'
      '-> If this is large (it will be, for an untrained model with '
      'random weights, full self-attention mixes all positions), the '
      'prediction at target_pos DOES depend on the value at other_pos.')

# ---- 4. Where does the per-step UNMASKING PROBABILITY come from? ----
# move_chance is computed purely from sigma(t), independent of x.
move_chance_t = 1 - torch.exp(-sigma_t)
print(f'\nmove_chance(t=0.5) = {move_chance_t.item():.6f}')
print('This scalar is identical for every position/batch element -- '
      'it does not depend on x at all (see _ddpm_update in diffusion.py).')
