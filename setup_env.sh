#!/bin/bash
set -e

ENV_PATH="/home/vasilije_ivanovic/envs/mdlm"

echo "=== Step 1: Creating base environment ==="
micromamba create -p $ENV_PATH python=3.9 pip=23.3.1 cuda-nvcc=12.4.99 jupyter=1.0.0 -c nvidia -c conda-forge -y

echo "=== Step 2: Installing PyTorch (cu121) ==="
micromamba run -p $ENV_PATH pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 3: Installing remaining packages ==="
micromamba run -p $ENV_PATH pip install \
    datasets==2.18.0 \
    einops==0.7.0 \
    fsspec==2024.2.0 \
    h5py==3.10.0 \
    hydra-core==1.3.2 \
    ipdb==0.13.13 \
    lightning==2.2.1 \
    notebook==7.1.1 \
    nvitop==1.3.2 \
    omegaconf==2.3.0 \
    packaging==23.2 \
    pandas==2.2.1 \
    rich==13.7.1 \
    seaborn==0.13.2 \
    scikit-learn==1.4.0 \
    timm==0.9.16 \
    transformers==4.38.2 \
    wandb==0.13.5

echo "=== Step 4: Installing flash-attn (requires torch, compiled for H200) ==="
micromamba run -p $ENV_PATH pip install flash-attn==2.5.6 --no-build-isolation

echo "=== Done! Activate with: micromamba activate $ENV_PATH ==="
