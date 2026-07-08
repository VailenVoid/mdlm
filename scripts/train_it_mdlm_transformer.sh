#!/bin/bash
#SBATCH --job-name=it-mdlm-transformer
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
#SBATCH --output=/home/vasilije_ivanovic/mdlm/logs/it-mdlm-transformer-%j.out
#SBATCH --error=/home/vasilije_ivanovic/mdlm/logs/it-mdlm-transformer-%j.error

# Stage 2 -- Instruction tuning: MDLM (transformer backbone) on
# smol-smoltalk / ultrachat_200k / tulu-3-sft, the exact data the AR-M
# baseline's IT stage (sde_llm it_AR-M.yml) trains on:
#   sde_llm/data/instruction_data/instruction_train/   (Arrow IPC shards)
#   sde_llm/data/instruction_data/instruction_valid/   (HF format, 1000 ex.)
# See configs/data/instruction.yaml.
#
# Objective (mirrors the AR-M IT loss masking, adapted to masked diffusion):
#   * input = condition + target, right-padded to model.length=1024 with EOS
#   * condition/prompt tokens: never noised, never scored
#   * response + EOS padding: masked by the forward process and scored,
#     so the model learns both the response and where to stop
#
# Warm-starts from the MDLM pretrain checkpoint (raw weights + EMA copied;
# optimizer / LR schedule / step counters start fresh) -- the pretrain
# checkpoints are only read, never modified. Recipe mirrors it_AR-M.yml:
# same base LR as pretrain (3e-4) on a constant schedule with a fresh
# 4000-step warmup, global batch 512, 10 epochs, validation once per epoch
# with best.ckpt tracking val/nll.
#
# 1,953,215 train examples / gbs 512 => ~3815 steps per epoch, ~38.2k steps
# for 10 epochs. Checkpoints land in ${RUN_DIR}/checkpoints and the job
# auto-resumes on requeue (init_from_ckpt is ignored once last.ckpt exists).

export WANDB_MODE=offline

RUN_NAME=mdlm-transformer-it
RUN_DIR=/home/vasilije_ivanovic/mdlm/runs/${RUN_NAME}
INIT_CKPT=/home/vasilije_ivanovic/mdlm/runs/mdlm-transformer-pretrain/checkpoints/last.ckpt
mkdir -p ${RUN_DIR}
mkdir -p /home/vasilije_ivanovic/mdlm/logs

if [ ! -f "${INIT_CKPT}" ]; then
  echo "ERROR: pretrain checkpoint not found at ${INIT_CKPT}" >&2
  exit 1
fi

cd /home/vasilije_ivanovic/mdlm

echo "====== [$(date)] MDLM (transformer backbone) instruction tuning ======"
srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u -m main \
  model=transformer-300m \
  backbone=transformer \
  data=instruction \
  model.length=1024 \
  parameterization=subs \
  time_conditioning=False \
  loader.global_batch_size=512 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  +trainer.max_epochs=10 \
  trainer.max_steps=-1 \
  trainer.val_check_interval=1.0 \
  optim.lr=3e-4 \
  lr_scheduler.num_warmup_steps=4000 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  checkpointing.init_from_ckpt=${INIT_CKPT} \
  wandb.name=${RUN_NAME} \
  hydra.run.dir=${RUN_DIR} \
  checkpointing.save_dir=${RUN_DIR}
echo "====== [$(date)] IT complete -- checkpoints at ${RUN_DIR}/checkpoints ======"
