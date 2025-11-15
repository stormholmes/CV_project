#!/bin/bash
#SBATCH --account=mscaisuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00

# Your commands here
python utils/data.py --data_path DATA