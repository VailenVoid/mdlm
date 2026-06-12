#!/bin/bash
#SBATCH -J diag_msp3
#SBATCH -o /home/vasilije_ivanovic/mdlm/logs/diag_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=1000
#SBATCH -t 00:02:00
#SBATCH --partition=non-reserved
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=msp3-5

hostname
echo "HOME=$HOME"
ls -la /home/vasilije_ivanovic/mdlm/logs | head -5
df -h /home/vasilije_ivanovic
ls /home/vasilije_ivanovic/envs/mdlm | head -3
