#!/usr/bin/env python
"""Prompted generation with the trained AR (autoregressive) baseline.

Unlike MDLM, this model is a standard left-to-right transformer LM, so
"talking" to it is the usual game: feed it your prompt, sample one token
at a time from its output distribution, and stop at EOS or max length.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

import dataloader
import diffusion

CONFIG_PATH = (
  '/home/vasilije_ivanovic/mdlm/runs/ar-lm1b/.hydra/config.yaml')
CKPT_PATH = (
  '/home/vasilije_ivanovic/mdlm/runs/ar-lm1b/checkpoints/best.ckpt')


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
  return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, device, prompt, length=128,
             temperature=1.0):
  bos = tokenizer.encode(tokenizer.bos_token)[0]
  eos = tokenizer.encode(tokenizer.eos_token)[0]
  prompt_ids = [bos] + tokenizer.encode(prompt, add_special_tokens=False)
  prompt_ids = prompt_ids[:length]

  x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
  for _ in range(len(prompt_ids), length):
    logits = model.forward(x, None)[:, -1]
    if temperature <= 0:
      next_id = logits.argmax(-1)
    else:
      noise = torch.distributions.Gumbel(0, 1).sample(
        logits.shape).to(device)
      next_id = (logits / temperature + noise).argmax(-1)
    x = torch.cat([x, next_id[:, None]], dim=1)
    if next_id.item() == eos:
      break
  return tokenizer.decode(x[0], skip_special_tokens=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('prompt', nargs='?', default='')
  parser.add_argument('--length', type=int, default=128)
  parser.add_argument('--temperature', type=float, default=1.0)
  args = parser.parse_args()

  model, tokenizer, device = load_model()
  print(f'[device={device}, length={args.length}, '
        f'temperature={args.temperature}]', file=sys.stderr)

  if args.prompt:
    text = generate(model, tokenizer, device, args.prompt,
                     length=args.length, temperature=args.temperature)
    print('MODEL:', text)
    return

  print("Interactive mode. Type a prompt and press Enter "
        "(Ctrl-D or 'quit' to exit).", file=sys.stderr)
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
                     length=args.length, temperature=args.temperature)
    print('MODEL:', text)
    print()


if __name__ == '__main__':
  main()
