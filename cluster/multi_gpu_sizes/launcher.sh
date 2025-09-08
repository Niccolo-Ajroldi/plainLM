#!/bin/bash
SUB_FILE="/home/atatjer/src/plainLM/cluster/multi_gpu_sizes/condor4template.sub"
CONFIG_DIR="/home/atatjer/src/plainLM/config/kumar"

# Array of "config nrun" pairs
experiments=(
    'cooldown_00.yaml 4'
    'cooldown_01.yaml 4'
    'cooldown_02.yaml 4'
)

for exp in "${experiments[@]}"; do
    # Read into two variables: cfg and nrun
    read -r cfg nrun <<< "$exp"
    full_path="${CONFIG_DIR}/${cfg}"
    cfg_name="${cfg%.*}"

    echo "Submitting: $cfg_name with nrun=$nrun"
    condor_submit_bid 1000 "$SUB_FILE" \
        -append "config $full_path" \
        -append "queue $nrun"
done
