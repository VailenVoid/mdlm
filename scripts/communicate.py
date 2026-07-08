#!/usr/bin/env python
"""Prompted infilling with a trained MDLM checkpoint.

MDLM is not an instruction-tuned chat model: it was trained to denoise
fully- or partially-masked LM1B sentences. "Talking" to it means seeding
the sequence with your text as an unmasked prefix and letting the reverse
diffusion process fill in the remaining masked positions.
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

import dataloader
import diffusion

CONFIG_PATH = (
  '/home/vasilije_ivanovic/mdlm/runs/mdlm-lm1b/.hydra/config.yaml')
CKPT_PATH = (
  '/home/vasilije_ivanovic/mdlm/runs/mdlm-lm1b/checkpoints/best.ckpt')


def load_model():
  config = OmegaConf.load(CONFIG_PATH)
  config.eval.checkpoint_path = CKPT_PATH
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion.load_from_checkpoint(
    CKPT_PATH, tokenizer=tokenizer, config=config,
    map_location=device)
  model = model.to(device)
  model.eval()
  if model.ema:
    model.ema.store(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))
    model.ema.copy_to(itertools.chain(
      model.backbone.parameters(), model.noise.parameters()))
  return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, device, prompt, length=128, steps=500,
             eps=1e-5):
  bos = tokenizer.encode(tokenizer.bos_token)[0]
  prompt_ids = [bos] + tokenizer.encode(prompt, add_special_tokens=False)
  prompt_ids = prompt_ids[:length]

  x = torch.full((1, length), model.mask_index, dtype=torch.long,
                 device=device)
  x[0, :len(prompt_ids)] = torch.tensor(prompt_ids, device=device)

  timesteps = torch.linspace(1, eps, steps + 1, device=device)
  dt = (1 - eps) / steps
  for i in range(steps):
    t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
    x = model._ddpm_update(x, t, dt)

  # Final cleanup pass for any still-masked tokens; copy_flag guarantees
  # the prompt itself is never touched by this un-masked argmax step.
  t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
  unet_conditioning = model.noise(t)[0]
  new_x = model.forward(x, unet_conditioning).argmax(dim=-1)
  copy_flag = (x != model.mask_index).long()
  x = copy_flag * x + (1 - copy_flag) * new_x
  return tokenizer.decode(x[0], skip_special_tokens=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('prompt', nargs='?', default='')
  parser.add_argument('--steps', type=int, default=500)
  parser.add_argument('--length', type=int, default=128)
  args = parser.parse_args()

  model, tokenizer, device = load_model()
  print(f'[device={device}, steps={args.steps}, length={args.length}]',
        file=sys.stderr)

  if args.prompt:
    text = generate(model, tokenizer, device, args.prompt,
                     length=args.length, steps=args.steps)
    print('MODEL:', text)
    return

  print("Interactive mode. Type a prompt and press Enter "
        "(Ctrl-D or empty 'quit' to exit).", file=sys.stderr)
  while True:
    try:
      prompt = input('> ')
    except EOFError:
      break
    if prompt.strip().lower() in ('quit', 'exit'):
      break
    if not prompt.strip():
      continue
    text = generate(model, tokenizer, device, prompt,
                     length=args.length, steps=args.steps)
    print('MODEL:', text)
    print()


if __name__ == '__main__':
  main()
