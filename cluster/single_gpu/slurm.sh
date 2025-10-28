#!/bin/bash

#SBATCH --account=hk-project-p0023364
#SBATCH --job-name=test
#SBATCH --error=/home/hk-project-p0023364/hgf_omt7140/log/%x_%A_%a.err
#SBATCH --output=/home/hk-project-p0023364/hgf_omt7140/log/%x_%A_%a.out
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --partition=dev_accelerated
#SBATCH --array=1

# Activate environment
cd ~/plainLM
source .venv/bin/activate

# Hyperparmeters are specified in a YAML configuration file
config=config/70M_10BT.yaml

# SLURM job arrays range from 1 to n
job_idx=$((SLURM_ARRAY_TASK_ID - 1))

# Launch torch distributed run on 8 devices
python train.py --config=$config --job_idx=$job_idx
