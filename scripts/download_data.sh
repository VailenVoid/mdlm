#!/bin/bash
#SBATCH -J download_data
#SBATCH -o /home/vasilije_ivanovic/mdlm/logs/download_%j.out
#SBATCH -N 1
#SBATCH --partition=non-reserved
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --open-mode=append

mkdir -p /home/vasilije_ivanovic/mdlm/logs

echo "Starting dataset download on $(hostname) at $(date)"

micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -c "
from datasets import load_dataset

print('Downloading OpenWebText train split...')
load_dataset('openwebtext', split='train[:-100000]', cache_dir='/home/vasilije_ivanovic/data')
print('Done: OpenWebText train')

print('Downloading OpenWebText valid split...')
load_dataset('openwebtext', split='train[-100000:]', cache_dir='/home/vasilije_ivanovic/data')
print('Done: OpenWebText valid')

print('All downloads complete!')
"

echo "Finished at $(date)"
