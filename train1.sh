#!/bin/bash
#SBATCH --job-name=GECO
#SBATCH --output=train/GeCo_%j.txt
#SBATCH --error=train/GeCo_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --account=mscaisuperpod
#SBATCH --partition=normal

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=50197
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun --unbuffered python train.py \
--resume_training \
--model_name GeCo_best \
--model_name_resumed GeCo1 \
--data_path  DATA \
--epochs 30 \
--lr 5e-6 \
--backbone_lr 0 \
--lr_drop 200 \
--weight_decay 1e-4 \
--batch_size 4 \
--tiling_p 0.5 \
--model_path MODEL