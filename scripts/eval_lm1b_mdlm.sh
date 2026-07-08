#!/bin/bash
#SBATCH -J train_mdlm_lm1b
#SBATCH -o /home/vasilije_ivanovic/mdlm/logs/train_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64000
#SBATCH -t 6-00:00:00
#SBATCH --partition=non-reserved
#SBATCH --constraint=zone-sof1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --open-mode=append
#SBATCH --requeue

export WANDB_MODE=offline
CKPT_PATH=${1:-/home/vasilije_ivanovic/mdlm/runs/mdlm-lm1b/checkpoints/last.ckpt}
mkdir -p /home/vasilije_ivanovic/mdlm/logs

srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u -m main \
  mode=ppl_eval\
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small \
  data=lm1b \
  backbone=dit \
  parameterization=subs \
  model.length=128 \
  time_conditioning=True \
  eval.checkpoint_path=${CKPT_PATH} \
  +wandb.offline=true
