#!/bin/bash
#SBATCH --account=mscaisuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00

# Your commands here
python utils/data.py --data_path DATA
python evaluate.py --data_path DATA --model_path MODEL --model_name GeCo1
python evaluate_bboxes.py --data_path DATA