#!/bin/bash
#SBATCH --job-name=smoke-it-mdlm
#SBATCH --partition=debug
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=0:30:00
#SBATCH --output=/home/vasilije_ivanovic/mdlm/logs/smoke-it-mdlm-%j.out
#SBATCH --error=/home/vasilije_ivanovic/mdlm/logs/smoke-it-mdlm-%j.error

# Smoke test for the MDLM instruction-tuning pipeline
# (scripts/train_it_mdlm_transformer.sh): warm-start from the real pretrain
# checkpoint, run a handful of optimizer steps on the instruction data and a
# small validation pass. Writes to a throwaway run dir.

export WANDB_MODE=offline

RUN_NAME=smoke-mdlm-transformer-it
RUN_DIR=/home/vasilije_ivanovic/mdlm/runs/_${RUN_NAME}
INIT_CKPT=/home/vasilije_ivanovic/mdlm/runs/mdlm-transformer-pretrain/checkpoints/last.ckpt
rm -rf ${RUN_DIR}
mkdir -p ${RUN_DIR}

cd /home/vasilije_ivanovic/mdlm

echo "====== [$(date)] MDLM IT smoke test ======"
srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u -m main \
  model=transformer-300m \
  backbone=transformer \
  data=instruction \
  model.length=1024 \
  parameterization=subs \
  time_conditioning=False \
  loader.global_batch_size=8 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=4 \
  trainer.max_steps=6 \
  trainer.val_check_interval=4 \
  trainer.limit_val_batches=2 \
  optim.lr=3e-4 \
  lr_scheduler.num_warmup_steps=4000 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  checkpointing.init_from_ckpt=${INIT_CKPT} \
  wandb.name=${RUN_NAME} \
  hydra.run.dir=${RUN_DIR} \
  checkpointing.save_dir=${RUN_DIR}
STATUS=$?
echo "====== [$(date)] Smoke test exit status: ${STATUS} ======"
exit ${STATUS}
