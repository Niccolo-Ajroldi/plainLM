#!/bin/bash

#SBATCH --job-name=plainLM_test
#SBATCH --error=/ptmp/najroldi/logs/plainLM/err/%x_%A_%a.err
#SBATCH --output=/ptmp/najroldi/logs/plainLM/out/%x_%A_%a.out
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=125000
#SBATCH --ntasks=1
#SBATCH --array=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plainLM

# Hyperparmeters are specified in a YAML configuration file
config=config/config.yaml

# SLURM job arrays range from 1 to n
job_idx=$((SLURM_ARRAY_TASK_ID - 1))

# Execute python script
python train.py --config=$config --job_idx=$job_idx
