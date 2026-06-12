#!/bin/bash
#SBATCH -J pretok_lm1b
#SBATCH -o /home/vasilije_ivanovic/mdlm/logs/pretok_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=128000
#SBATCH -t 12:00:00
#SBATCH --partition=non-reserved
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --open-mode=append

mkdir -p /home/vasilije_ivanovic/mdlm/logs

srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u /home/vasilije_ivanovic/mdlm/scripts/pretokenize_lm1b.py
