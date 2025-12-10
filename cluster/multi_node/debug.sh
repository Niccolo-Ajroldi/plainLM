#!/bin/bash

#SBATCH --account=hk-project-p0023364
#SBATCH --job-name=multinode
#SBATCH --error=/home/hk-project-p0023364/hgf_omt7140/log/%x_%A_%a.err
#SBATCH --output=/home/hk-project-p0023364/hgf_omt7140/log/%x_%A_%a.out
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --mem=200000mb
#SBATCH --partition=accelerated
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --array=1

# Activate environment
cd ~/plainLM
source .venv/bin/activate

# Hyperparmeters are specified in a YAML configuration file
config=config/dev/debug.yaml

# SLURM job arrays range from 1 to n
job_idx=$((SLURM_ARRAY_TASK_ID - 1))

# Get list of allocated nodes; the first one becomes rank 0 (master)
# Query the master node for its IPv4 address
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)

# Select the first node as the rendezvous host
MASTER_ADDR=$head_node_ip
MASTER_PORT=29500

echo "I am node $(hostname), rank $SLURM_PROCID"
echo "MASTER_ADDR:MASTER_PORT set to: ${MASTER_ADDR}:${MASTER_PORT}"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_GPUS_PER_NODE=$SLURM_GPUS_PER_NODE"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURM_NODEID=$SLURM_NODEID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

# Launch torch distributed run
srun torchrun \
  --nnodes $SLURM_NNODES \
  --nproc-per-node $SLURM_GPUS_PER_NODE \
  --node_rank $SLURM_PROCID \
  --rdzv_id=$SLURM_ARRAY_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py --config=$config --job_idx=$job_idx

##############
# Notes
#
# NODEID vs PROCID
#   - SLURM_NODEID = node index in the job (0..nodes-1), shared by tasks on same node.
#   - SLURM_PROCID = global task rank (0..ntasks-1), unique per task across the job.
#   When `ntasks-per-node=1`, each node has one task -> SLURM_NODEID == SLURM_PROCID.
#
# TORCHRUN will set some env vars
#   - LOCAL_RANK: uniquely identifies each GPU-process on a node. 
#   - RANK: unique identifier across all the nodes (globally unique)
#
##############
