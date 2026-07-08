#!/bin/bash
#SBATCH --job-name=eval-mdlm-ppl
#SBATCH --partition=batch
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=zone-sof1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=04:00:00
#SBATCH --output=/home/vasilije_ivanovic/mdlm/logs/eval-mdlm-ppl-%j.out
#SBATCH --error=/home/vasilije_ivanovic/mdlm/logs/eval-mdlm-ppl-%j.error

# Full-validation ppl eval of the mdlm-transformer-pretrain checkpoint on the
# entire held-out shard (trainer.limit_val_batches defaults to 1.0 = full
# split, vs. 25 batches during training). Model/data flags mirror
# train_pretrain_mdlm_transformer.sh exactly.

export WANDB_MODE=offline

CKPT_PATH=${1:-/home/vasilije_ivanovic/mdlm/runs/mdlm-transformer-pretrain/checkpoints/last.ckpt}
mkdir -p /home/vasilije_ivanovic/mdlm/logs

cd /home/vasilije_ivanovic/mdlm

echo "====== [$(date)] MDLM (transformer backbone) full-val ppl eval: ${CKPT_PATH} ======"
srun micromamba run -p /home/vasilije_ivanovic/envs/mdlm python -u -m main \
  mode=ppl_eval \
  model=transformer-300m \
  backbone=transformer \
  data=pretrain \
  model.length=1024 \
  parameterization=subs \
  time_conditioning=False \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  eval.checkpoint_path=${CKPT_PATH} \
  wandb.name=mdlm-transformer-pretrain-fullval-eval
echo "====== [$(date)] MDLM full-val ppl eval complete ======"
