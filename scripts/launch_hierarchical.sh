#!/bin/bash

# ==============================================================================
# Hierarchical FL Launch Script (1-3-6 Topology) using nohup
# ==============================================================================

# 1. Configuration (Update these to match your environment)
PROJECT_DIR="/s/chopin/k/grad/jasoncur/cs535/termProject"
ENV_CMD="source .venv/bin/activate"
LOG_DIR="$PROJECT_DIR/logs/hierarchical"

MASTER_NODE="boston"
MASTER_PORT=8084
NUM_ROUNDS=30
LOCAL_EPOCHS=1
DDP_PORT=12346

# Hospital A
COORD_A="jupiter"
WORKERS_A=("neptune" "earth")

# Hospital B
COORD_B="uranus"
WORKERS_B=("mercury" "sardine")

# Hospital C
COORD_C="sole"
WORKERS_C=("char" "barracuda")

# ==============================================================================

# Ensure local log directory exists
mkdir -p "$LOG_DIR"

# 2. Cleanup Function (kills previous lingering python processes from this project)
cleanup_old_processes() {
    echo "Cleaning up old Python FL processes across the cluster..."
    ALL_NODES=($MASTER_NODE $COORD_A "${WORKERS_A[@]}" $COORD_B "${WORKERS_B[@]}" $COORD_C "${WORKERS_C[@]}")
    for node in "${ALL_NODES[@]}"; do
        # Create log dir on remote nodes just in case
        ssh "$node" "mkdir -p $LOG_DIR"
        # Gracefully kill any process running scripts from our hierarchical directory
        ssh "$node" "pkill -f 'python src/hierarchical/' 2>/dev/null"
    done
    echo "Cleanup complete."
    sleep 2
}

# Comment this out if you don't want automatic cleanup before launching
cleanup_old_processes

echo "============================================================"
echo " Launching Tier 1: Master Coordinator (FL Server)"
echo "============================================================"

echo "Starting master: $MASTER_NODE"

ssh -f "$MASTER_NODE" "bash -c 'cd $PROJECT_DIR && $ENV_CMD && \
    nohup python src/hierarchical/master_coordinator.py \
    --port $MASTER_PORT \
    --num_rounds $NUM_ROUNDS \
    --min_clients 3 \
    --local_epochs $LOCAL_EPOCHS \
    > $LOG_DIR/${MASTER_NODE}_master.log 2>&1 &'"

sleep 80

echo "============================================================"
echo " Launching Tier 2: Hospital Coordinators (FL Clients & DDP Masters)"
echo "============================================================"

launch_coordinator() {
    local coord=$1
    local hospital_id=$2
    
    echo "Starting coordinator: $coord ($hospital_id)"
    
    ssh -f "$coord" "bash -c 'cd $PROJECT_DIR && $ENV_CMD && \
        nohup python src/hierarchical/hospital_coordinator.py \
        --master_address $MASTER_NODE:$MASTER_PORT \
        --hospital_id $hospital_id \
        --num_nodes 3 \
        --ddp_port $DDP_PORT \
        > $LOG_DIR/${coord}_coordinator.log 2>&1 &'"
}

launch_coordinator "$COORD_A" "hospital_a"
launch_coordinator "$COORD_B" "hospital_b"
launch_coordinator "$COORD_C" "hospital_c"

sleep 5

echo "============================================================"
echo " Launching Tier 3: GPU Workers (DDP Nodes)"
echo "============================================================"

launch_workers() {
    local coord=$1
    local hospital_id=$2
    shift 2
    local workers=("$@")

    local rank=1
    for worker in "${workers[@]}"; do
        echo "Starting worker: $worker (Rank $rank) -> connecting to $coord"
        
        # -f tells ssh to go to the background just before command execution
        ssh -f "$worker" "bash -c 'cd $PROJECT_DIR && $ENV_CMD && \
            nohup python src/hierarchical/ddp_worker.py \
            --master_address $coord \
            --ddp_port $DDP_PORT \
            --hospital_id $hospital_id \
            --num_nodes 3 \
            --node_rank $rank \
            --num_rounds $NUM_ROUNDS \
            --local_epochs $LOCAL_EPOCHS \
            > $LOG_DIR/${worker}_worker.log 2>&1 &'"
        
        ((rank++))
    done
}

launch_workers "$COORD_A" "hospital_a" "${WORKERS_A[@]}"
launch_workers "$COORD_B" "hospital_b" "${WORKERS_B[@]}"
launch_workers "$COORD_C" "hospital_c" "${WORKERS_C[@]}"

echo "============================================================"
echo "Cluster launched successfully in the background!"
echo "All outputs are being redirected to: $LOG_DIR/"
echo ""
echo "To monitor the master node live, run:"
echo "  ssh $MASTER_NODE \"tail -f $LOG_DIR/${MASTER_NODE}_master.log\""
echo "============================================================"