"""
Load the official pretrained MDLM checkpoint (kuleshov-group/mdlm-owt,
trained on OpenWebText for 1M steps) and run zero-shot conditional
generation: feed a text prompt, mask out the rest of the sequence, and
let the reverse diffusion process fill in the remainder.

This relies on a property of absorbing-state diffusion: in
`_ddpm_update` / `_ddpm_caching_update`, any position that is NOT
currently `[MASK]` is copied through unchanged
(`copy_flag = (x != mask_index)`). So if we initialize `x` with the
prompt tokens already "revealed" and everything else masked, the
sampler naturally treats the prompt as fixed context and only fills
in the masked continuation -- no special infilling code needed.

Run on a GPU node:

  micromamba run -p /home/vasilije_ivanovic/envs/mdlm \
    python -u scripts/test_pretrained_owt.py
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
    'data=openwebtext-split',
    'parameterization=subs',
    'backbone=hf_dit',
    'model.length=64',
    'sampling.steps=64',
    'sampling.predictor=ddpm_cache',
    'loader.batch_size=1',
    'loader.eval_batch_size=1',
    'eval.checkpoint_path=kuleshov-group/mdlm-owt',
  ])

tokenizer = dataloader.get_tokenizer(config)

print('Loading pretrained checkpoint kuleshov-group/mdlm-owt ...')
model = diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')
model.eval()

L = config.model.length
mask_index = model.mask_index
print(f'model.length = {L}, mask_index = {mask_index}, '
      f'vocab_size = {model.vocab_size}')

PROMPTS = [
  'Paris is the capital and most populous city of France. It is located on the',
  'Q: France is a country in Europe. Its capital city, known for the Eiffel Tower, is\nA:',
]


@torch.no_grad()
def generate(prompt):
  prompt_ids = tokenizer(prompt)['input_ids']
  prompt_ids = prompt_ids[:L]

  x = torch.full((1, L), mask_index, dtype=torch.long, device='cuda')
  x[0, :len(prompt_ids)] = torch.tensor(prompt_ids, device='cuda')

  num_steps = config.sampling.steps
  eps = 1e-5
  timesteps = torch.linspace(1, eps, num_steps + 1, device='cuda')
  dt = (1 - eps) / num_steps

  p_x0_cache = None
  for i in range(num_steps):
    t = timesteps[i] * torch.ones(x.shape[0], 1, device='cuda')
    p_x0_cache, x_next = model._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
    if not torch.allclose(x_next, x):
      p_x0_cache = None
    x = x_next

  # noise removal: argmax cleanup. Already-unmasked positions
  # (including the prompt) are preserved by _subs_parameterization,
  # which forces logit=0 at the existing token and -inf elsewhere.
  t = timesteps[-1] * torch.ones(x.shape[0], 1, device='cuda')
  unet_conditioning = model.noise(t)[0]
  x = model.forward(x, unet_conditioning).argmax(dim=-1)

  remaining_mask = (x == mask_index)
  if remaining_mask.any():
    x = torch.where(remaining_mask, torch.zeros_like(x), x)

  return tokenizer.decode(x[0].tolist())


for prompt in PROMPTS:
  print('=' * 60)
  print(f'PROMPT: {prompt!r}')
  completion = generate(prompt)
  print(f'COMPLETION: {completion!r}')
