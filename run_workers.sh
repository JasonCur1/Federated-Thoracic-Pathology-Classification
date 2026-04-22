#!/bin/bash

# usage: ./run_workers.sh --hospital-name <name> --data-path <path> --global-weights <path> --local-weights <path> --epochs <epochs>

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hospital-name) HOSPITAL_NAME="$2"; shift ;;
        --data-path) DATA_PATH="$2"; shift ;;
        --global-weights) GLOBAL_WEIGHTS="$2"; shift ;;
        --local-weights) LOCAL_WEIGHTS="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Map hospitals to their specific worker nodes
if [ "$HOSPITAL_NAME" == "hospital_a" ]; then
    WORKERS=("flounder" "blowfish" "herring")
elif [ "$HOSPITAL_NAME" == "hospital_b" ]; then
    WORKERS=("bullhead" "wahoo" "tuna")
elif [ "$HOSPITAL_NAME" == "hospital_c" ]; then
    WORKERS=("grouper" "marlin" "perch")
else
    echo "Error: Unknown hospital $HOSPITAL_NAME"
    exit 1
fi

# The first worker in the array acts as the DDP communication endpoint
DDP_MASTER=${WORKERS[0]}
DDP_PORT=29500 # Default NCCL port

echo "[$HOSPITAL_NAME] Starting DDP group. Master node: $DDP_MASTER"

PIDS=()

# Launch torchrun on each of the 3 workers via SSH
for i in "${!WORKERS[@]}"; do
    NODE=${WORKERS[$i]}
    
    echo "[$HOSPITAL_NAME] Launching Rank $i on $NODE..."
    
    # Assuming 1 GPU per worker machine (nproc_per_node=1)
    ssh "$NODE" "cd $PWD && torchrun \
        --nnodes=3 \
        --nproc_per_node=1 \
        --node_rank=$i \
        --master_addr=$DDP_MASTER \
        --master_port=$DDP_PORT \
        src/hierarchical/worker.py \
        --data-path \"$DATA_PATH\" \
        --global-weights-path \"$GLOBAL_WEIGHTS\" \
        --local-weights-path \"$LOCAL_WEIGHTS\" \
        --epochs $EPOCHS" &
        
    PIDS+=($!)
done

# Wait for all 3 SSH commands (the training round) to finish
wait "${PIDS[@]}"

echo "[$HOSPITAL_NAME] DDP training round complete."