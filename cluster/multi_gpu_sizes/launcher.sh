#!/bin/bash
SUB_FILE="/home/atatjer/src/plainLM/cluster/multi_gpu_sizes/condor4template.sub"
CONFIG_DIR="/home/atatjer/src/plainLM/config/pythia100BT"

# Array of "config nrun" pairs
experiments=(
    # 'cooldown_05.yaml 6'
    'cosine_01.yaml 2'
)

for exp in "${experiments[@]}"; do
    # Read into two variables: cfg and nrun
    read -r cfg nrun <<< "$exp"
    full_path="${CONFIG_DIR}/${cfg}"
    cfg_name="${cfg%.*}"

    echo "Submitting: $cfg_name with nrun=$nrun"
    condor_submit_bid 45 "$SUB_FILE" \
        -append "config $full_path" \
        -append "queue $nrun"
done