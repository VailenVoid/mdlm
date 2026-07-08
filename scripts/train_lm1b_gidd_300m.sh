#!/bin/bash
#SBATCH -J train_gidd300m_lm1b
#SBATCH -o /home/vasilije_ivanovic/mdlm/logs/train_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64000
#SBATCH -t 2-00:00:00
#SBATCH --partition=non-reserved
#SBATCH --exclude=hala,msp3-0,msp3-1,msp3-2,msp3-3,msp3-4,msp3-5,msp3-6,msp3-7
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h200:4
#SBATCH --open-mode=append
#SBATCH --requeue

export WANDB_MODE=offline

RUN_NAME=gidd300m-lm1b
RUN_DIR=/home/vasilije_ivanovic/mdlm/runs/${RUN_NAME}
mkdir -p ${RUN_DIR}
mkdir -p /home/vasilije_ivanovic/mdlm/logs

# GIDD+ setup (arXiv 2503.04482): hybrid noise p_uniform=0.1,
# dynamic loss weights (configs/config.yaml gidd section) and
# weight_decay=0.02. Backbone is dit-300m (~304M): the same
# transformer+FFN blocks as the mdlm300m run plus adaLN time
# conditioning, matching the paper, which conditions on raw t
# (time_conditioning=True passes t through to the backbone).
srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=dit-300m \
  backbone=dit \
  data=lm1b \
  wandb.name=${RUN_NAME} \
  parameterization=gidd \
  gidd.p_uniform=0.1 \
  optim.weight_decay=0.02 \
  model.length=128 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  time_conditioning=True \
  hydra.run.dir=${RUN_DIR} \
  checkpointing.save_dir=${RUN_DIR}
