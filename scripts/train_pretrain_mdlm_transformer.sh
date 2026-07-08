#!/bin/bash
#SBATCH --job-name=pretrain-mdlm-transformer
#SBATCH --partition=batch
#SBATCH --gres=gpu:h200:4
#SBATCH --constraint=zone-sof1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=7-00:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --output=/home/vasilije_ivanovic/mdlm/logs/pretrain-mdlm-transformer-%j.out
#SBATCH --error=/home/vasilije_ivanovic/mdlm/logs/pretrain-mdlm-transformer-%j.error

# Pretrain MDLM with our plain transformer backbone (models/transformer.py --
# same block architecture as the AR-M model) on the SAME corpus the AR-M
# baseline (sde_llm) is pretraining on: the Arrow IPC shards under
#   sde_llm/data/pretrain_data/pretrain_train/
# (T5-small tokenized, packed into model.length windows). See
# configs/data/pretrain.yaml.
#
# time_conditioning=False: the transformer backbone has no sigma pathway
# (it ignores the diffusion timestep), which also lets ddpm_cache reuse p_x0.
#
# global_batch_size 512 over 4x H200 -> per-GPU batch 16, grad accum 8.
# ~33.16M packed windows in the train split => one epoch ~= 64.8k steps;
# max_steps=65000 mirrors the AR-M single-epoch pretrain. Bump it for more
# passes. Checkpoints (last.ckpt) land in ${RUN_DIR}/checkpoints and the job
# auto-resumes on requeue.

export WANDB_MODE=offline

RUN_NAME=mdlm-transformer-pretrain
RUN_DIR=/home/vasilije_ivanovic/mdlm/runs/${RUN_NAME}
mkdir -p ${RUN_DIR}
mkdir -p /home/vasilije_ivanovic/mdlm/logs

cd /home/vasilije_ivanovic/mdlm

echo "====== [$(date)] MDLM (transformer backbone) pretrain on AR corpus ======"
srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u -m main \
  model=transformer-300m \
  backbone=transformer \
  data=pretrain \
  model.length=1024 \
  parameterization=subs \
  time_conditioning=False \
  loader.global_batch_size=512 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  trainer.max_steps=65000 \
  trainer.val_check_interval=10000 \
  trainer.limit_val_batches=25 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  wandb.name=${RUN_NAME} \
  hydra.run.dir=${RUN_DIR} \
  checkpointing.save_dir=${RUN_DIR}
echo "====== [$(date)] Pretrain complete -- checkpoints at ${RUN_DIR}/checkpoints ======"
