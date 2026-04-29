#!/bin/bash

# ==============================================================================
# Hierarchical FL Launch Script: FULL DEBUG MODE
# 3 Hospitals + 2 DDP Workers per Hospital
#
# Topology:
#   herring     = Master FL Server
#
#   Hospital A:
#     perch     = Coordinator / DDP rank 0
#     barracuda = DDP worker rank 1
#     sardine   = DDP worker rank 2
#
#   Hospital B:
#     sole      = Coordinator / DDP rank 0
#     blowfish  = DDP worker rank 1
#     turbot    = DDP worker rank 2
#
#   Hospital C:
#     bonito    = Coordinator / DDP rank 0
#     flounder  = DDP worker rank 1
#     shark     = DDP worker rank 2
#
# Folder structure:
# project_root/
#   ├── src/
#   └── scripts/
#       └── launch_hierarchical.sh
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$PROJECT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"

LOG_DIR="$PROJECT_DIR/logs/hierarchical"
mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------------------
# 2. Cluster configuration: 3 hospitals + 2 DDP workers each
# ------------------------------------------------------------------------------

MASTER_NODE="herring"
MASTER_PORT=8084

# Full debug run: all hospitals, but only 1 FL round first
NUM_ROUNDS=5
LOCAL_EPOCHS=2
MIN_CLIENTS=3

# Each hospital has:
# coordinator rank 0 + worker rank 1 + worker rank 2
NUM_NODES=3

# Use different DDP ports per hospital to avoid conflicts
DDP_PORT_A=12346
DDP_PORT_B=12446
DDP_PORT_C=12546

# Hospital A
COORD_A="perch"
WORKERS_A=("barracuda" "sardine")

# Hospital B
COORD_B="sole"
WORKERS_B=("blowfish" "turbot")

# Hospital C
COORD_C="bonito"
WORKERS_C=("flounder" "shark")

# ------------------------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------------------------

check_python() {
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "ERROR: Python venv not found at: $PYTHON_BIN"
        echo "Create it using:"
        echo "  cd $PROJECT_DIR"
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
}

remote_run() {
    local node=$1
    local cmd=$2

    ssh -f "$node" "bash -lc 'cd $PROJECT_DIR && mkdir -p $LOG_DIR && $cmd'"
}

cleanup_old_processes() {
    echo "Cleaning old hierarchical FL processes..."

    ALL_NODES=(
        "$MASTER_NODE"
        "$COORD_A" "${WORKERS_A[@]}"
        "$COORD_B" "${WORKERS_B[@]}"
        "$COORD_C" "${WORKERS_C[@]}"
    )

    for node in "${ALL_NODES[@]}"; do
        echo "Cleaning $node..."

        ssh "$node" "mkdir -p '$LOG_DIR'" || {
            echo "WARNING: Could not create log directory on $node"
            continue
        }

        ssh "$node" "
            pids=\$(pgrep -f '[p]ython .*src/hierarchical' || true)
            if [ -n \"\$pids\" ]; then
                echo \"Killing old hierarchical processes on $node: \$pids\"
                kill \$pids 2>/dev/null || true
                sleep 1
                kill -9 \$pids 2>/dev/null || true
            else
                echo \"No old hierarchical processes found on $node\"
            fi
        " || {
            echo "WARNING: Cleanup command failed on $node, continuing..."
        }
    done

    echo "Cleanup complete."
    sleep 2
}

wait_for_master() {
    echo "Waiting for master server to listen on $MASTER_NODE:$MASTER_PORT..."

    for i in {1..120}; do
        if ssh "$MASTER_NODE" "ss -ltn | grep -q ':$MASTER_PORT'"; then
            echo "Master is ready on $MASTER_NODE:$MASTER_PORT"
            return 0
        fi

        echo "Master not ready yet... attempt $i/120"
        sleep 3
    done

    echo "ERROR: Master did not start listening on port $MASTER_PORT"
    echo "Check master log:"
    echo "ssh $MASTER_NODE \"tail -n 100 $LOG_DIR/${MASTER_NODE}_master.log\""
    exit 1
}

launch_master() {
    echo "Launching master coordinator on $MASTER_NODE..."

    remote_run "$MASTER_NODE" "
        nohup $PYTHON_BIN src/hierarchical/master_coordinator.py \
            --port $MASTER_PORT \
            --num_rounds $NUM_ROUNDS \
            --min_clients $MIN_CLIENTS \
            --local_epochs $LOCAL_EPOCHS \
            > $LOG_DIR/${MASTER_NODE}_master.log 2>&1 &
    "
}

launch_workers() {
    local coord=$1
    local hospital_id=$2
    local ddp_port=$3
    shift 3
    local workers=("$@")

    local rank=1

    for worker in "${workers[@]}"; do
        echo "Launching DDP worker $worker for $hospital_id with rank $rank..."

        remote_run "$worker" "
            nohup $PYTHON_BIN src/hierarchical/ddp_worker.py \
                --master_address $coord \
                --ddp_port $ddp_port \
                --hospital_id $hospital_id \
                --num_nodes $NUM_NODES \
                --node_rank $rank \
                --num_rounds $NUM_ROUNDS \
                --local_epochs $LOCAL_EPOCHS \
                > $LOG_DIR/${worker}_${hospital_id}_worker_rank${rank}.log 2>&1 &
        "

        rank=$((rank + 1))
    done
}

launch_coordinator() {
    local coord=$1
    local hospital_id=$2
    local ddp_port=$3

    echo "Launching hospital coordinator $coord for $hospital_id..."

    remote_run "$coord" "
        nohup $PYTHON_BIN src/hierarchical/hospital_coordinator.py \
            --master_address $MASTER_NODE:$MASTER_PORT \
            --hospital_id $hospital_id \
            --num_nodes $NUM_NODES \
            --ddp_port $ddp_port \
            > $LOG_DIR/${coord}_${hospital_id}_coordinator.log 2>&1 &
    "
}

# ------------------------------------------------------------------------------
# 4. Launch
# ------------------------------------------------------------------------------

check_python
cleanup_old_processes

echo "============================================================"
echo "Launching Tier 1: Master Coordinator"
echo "============================================================"

launch_master
wait_for_master

echo "============================================================"
echo "Launching Tier 3: DDP Workers"
echo "============================================================"

# Launch workers first so they are ready when coordinators start DDP.
launch_workers "$COORD_A" "hospital_a" "$DDP_PORT_A" "${WORKERS_A[@]}"
launch_workers "$COORD_B" "hospital_b" "$DDP_PORT_B" "${WORKERS_B[@]}"
launch_workers "$COORD_C" "hospital_c" "$DDP_PORT_C" "${WORKERS_C[@]}"

echo "Waiting for DDP workers to initialize..."
sleep 15

echo "============================================================"
echo "Launching Tier 2: Hospital Coordinators"
echo "============================================================"

launch_coordinator "$COORD_A" "hospital_a" "$DDP_PORT_A"
launch_coordinator "$COORD_B" "hospital_b" "$DDP_PORT_B"
launch_coordinator "$COORD_C" "hospital_c" "$DDP_PORT_C"

echo "============================================================"
echo "Hierarchical FL full debug cluster launched."
echo "Project directory: $PROJECT_DIR"
echo "Logs directory:    $LOG_DIR"
echo ""
echo "Monitor master:"
echo "  ssh $MASTER_NODE \"tail -f $LOG_DIR/${MASTER_NODE}_master.log\""
echo ""
echo "Monitor coordinators:"
echo "  ssh $COORD_A \"tail -f $LOG_DIR/${COORD_A}_hospital_a_coordinator.log\""
echo "  ssh $COORD_B \"tail -f $LOG_DIR/${COORD_B}_hospital_b_coordinator.log\""
echo "  ssh $COORD_C \"tail -f $LOG_DIR/${COORD_C}_hospital_c_coordinator.log\""
echo ""
echo "Monitor workers:"
echo "  ssh ${WORKERS_A[0]} \"tail -f $LOG_DIR/${WORKERS_A[0]}_hospital_a_worker_rank1.log\""
echo "  ssh ${WORKERS_A[1]} \"tail -f $LOG_DIR/${WORKERS_A[1]}_hospital_a_worker_rank2.log\""
echo "  ssh ${WORKERS_B[0]} \"tail -f $LOG_DIR/${WORKERS_B[0]}_hospital_b_worker_rank1.log\""
echo "  ssh ${WORKERS_B[1]} \"tail -f $LOG_DIR/${WORKERS_B[1]}_hospital_b_worker_rank2.log\""
echo "  ssh ${WORKERS_C[0]} \"tail -f $LOG_DIR/${WORKERS_C[0]}_hospital_c_worker_rank1.log\""
echo "  ssh ${WORKERS_C[1]} \"tail -f $LOG_DIR/${WORKERS_C[1]}_hospital_c_worker_rank2.log\""
echo "============================================================"