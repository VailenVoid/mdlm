#!/bin/bash
#SBATCH -J train_mdlm
#SBATCH -o /home/vasilije_ivanovic/mdlm/logs/train_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64000
#SBATCH -t 960:00:00
#SBATCH --partition=non-reserved
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --open-mode=append
#SBATCH --requeue

RUN_NAME=mdlm-owt
RUN_DIR=/home/vasilije_ivanovic/mdlm/runs/${RUN_NAME}
mkdir -p ${RUN_DIR}
mkdir -p /home/vasilije_ivanovic/mdlm/logs

srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small \
  data=openwebtext-split \
  wandb.name=${RUN_NAME} \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  hydra.run.dir=${RUN_DIR} \
  checkpointing.save_dir=${RUN_DIR}