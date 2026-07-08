#!/bin/bash
#SBATCH -J train_mdlm_lm1b
#SBATCH -o /home/vasilije_ivanovic/mdlm/logs/train_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64000
#SBATCH -t 6-00:00:00
#SBATCH --partition=non-reserved
#SBATCH --constraint=zone-sof1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h200:4
#SBATCH --open-mode=append
#SBATCH --requeue

export WANDB_MODE=offline

RUN_NAME=mdlm-lm1b
RUN_DIR=/home/vasilije_ivanovic/mdlm/runs/${RUN_NAME}
mkdir -p ${RUN_DIR}
mkdir -p /home/vasilije_ivanovic/mdlm/logs

srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small \
  data=lm1b \
  wandb.name=${RUN_NAME} \
  parameterization=subs \
  model.length=128 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  time_conditioning=True \
  hydra.run.dir=${RUN_DIR} \
  checkpointing.save_dir=${RUN_DIR}
