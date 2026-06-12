#!/bin/bash
#SBATCH -J test_context_dep
#SBATCH -o /home/vasilije_ivanovic/mdlm/logs/test_context_dep_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=16000
#SBATCH -t 00:10:00
#SBATCH --partition=non-reserved
#SBATCH --exclude=hala,msp3-0,msp3-1,msp3-2,msp3-3,msp3-4,msp3-5,msp3-6,msp3-7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --open-mode=append

mkdir -p /home/vasilije_ivanovic/mdlm/logs

cd /home/vasilije_ivanovic/mdlm

srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u scripts/test_context_dependence.py
