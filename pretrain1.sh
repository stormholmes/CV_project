#!/bin/bash
#SBATCH --job-name=GeCo
#SBATCH --output=train/GeCo_pretrain_%j.txt
#SBATCH --error=train/GeCo_pretrain_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=34:00:00
#SBATCH --account=mscaisuperpod
#SBATCH --partition=normal

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=50197
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

srun --unbuffered python pretrain.py \
--model_name GeCo_PRETRAIN \
--data_path  DATA \
--epochs 150 \
--lr 1e-4 \
--backbone_lr 0 \
--lr_drop 150 \
--weight_decay 1e-4 \
--batch_size 4 \
--tiling_p 0.2 \
--model_path MODEL