#!/bin/bash

# ==============================================================================
# DP Hierarchical FL Launch Script
# PyTorch Distributed + Opacus DPDDP
#
# 3 Hospitals + 2 DDP Workers per Hospital
#
# Topology:
#   herring = Master FL Server
#
#   Hospital A:
#     marlin  = Coordinator / DDP rank 0
#     turbot  = DDP worker rank 1
#     bonito  = DDP worker rank 2
#
#   Hospital B:
#     sardine   = Coordinator / DDP rank 0
#     blowfish  = DDP worker rank 1
#     swordfish = DDP worker rank 2
#
#   Hospital C:
#     char    = Coordinator / DDP rank 0
#     pollock = DDP worker rank 1
#     grouper = DDP worker rank 2
#
# Folder structure:
# project_root/
#   ├── src/
#   │   └── DP_hierarchial/
#   │       ├── master_coordinator.py
#   │       ├── hospital_coordinator.py
#   │       ├── ddp_worker.py
#   │       └── dp_utils.py
#   └── scripts/
#       └── launch_dp_hierarchial.sh
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$PROJECT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"

LOG_DIR="$PROJECT_DIR/logs/DP_hierarchial"
mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------------------
# 2. Cluster configuration
# ------------------------------------------------------------------------------

MASTER_NODE="herring"
MASTER_PORT=8084

NUM_ROUNDS=10
LOCAL_EPOCHS=1
MIN_CLIENTS=3

# Each hospital has:
# coordinator rank 0 + worker rank 1 + worker rank 2
NUM_NODES=3

# Use different DDP ports per hospital to avoid conflicts
DDP_PORT_A=12346
DDP_PORT_B=12446
DDP_PORT_C=12546

# Hospital A
COORD_A="marlin"
WORKERS_A=("turbot" "bonito")

# Hospital B
COORD_B="sardine"
WORKERS_B=("blowfish" "swordfish")

# Hospital C
COORD_C="char"
WORKERS_C=("pollock" "grouper")

# ------------------------------------------------------------------------------
# 3. Differential Privacy configuration
# ------------------------------------------------------------------------------

# Higher noise_multiplier -> stronger privacy, lower epsilon, usually lower accuracy
DP_NOISE_MULTIPLIER=1.0

# Per-sample gradient clipping norm
DP_MAX_GRAD_NORM=1.0

# Delta for epsilon accounting
DP_DELTA=1e-5

# Set to 1 if you want Opacus secure RNG.
# Secure RNG is slower. For debugging/experiments, keep it 0.
SECURE_RNG=0

# ------------------------------------------------------------------------------
# 4. Helper functions
# ------------------------------------------------------------------------------

check_python() {
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "ERROR: Python venv not found at: $PYTHON_BIN"
        echo "Create it using:"
        echo "  cd $PROJECT_DIR"
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
        echo "  pip install opacus"
        exit 1
    fi
}

remote_run() {
    local node=$1
    local cmd=$2

    ssh -f "$node" "bash -lc 'cd $PROJECT_DIR && mkdir -p $LOG_DIR && $cmd'"
}

cleanup_old_processes() {
    echo "Cleaning old DP hierarchical FL processes..."

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
            pids=\$(pgrep -f '[p]ython .*src/DP_hierarchial' || true)
            if [ -n \"\$pids\" ]; then
                echo \"Killing old DP_hierarchial processes on $node: \$pids\"
                kill \$pids 2>/dev/null || true
                sleep 1
                kill -9 \$pids 2>/dev/null || true
            else
                echo \"No old DP_hierarchial processes found on $node\"
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
    echo "ssh $MASTER_NODE \"tail -n 100 $LOG_DIR/${MASTER_NODE}_dp_master.log\""
    exit 1
}

secure_rng_arg() {
    if [ "$SECURE_RNG" -eq 1 ]; then
        echo "--secure_rng"
    else
        echo ""
    fi
}

launch_master() {
    echo "Launching DP master coordinator on $MASTER_NODE..."

    remote_run "$MASTER_NODE" "
        nohup $PYTHON_BIN src/DP_hierarchial/master_coordinator.py \
            --port $MASTER_PORT \
            --num_rounds $NUM_ROUNDS \
            --min_clients $MIN_CLIENTS \
            --local_epochs $LOCAL_EPOCHS \
            > $LOG_DIR/${MASTER_NODE}_dp_master.log 2>&1 &
    "
}

launch_workers() {
    local coord=$1
    local hospital_id=$2
    local ddp_port=$3
    shift 3
    local workers=("$@")

    local rank=1
    local secure_arg
    secure_arg=$(secure_rng_arg)

    for worker in "${workers[@]}"; do
        echo "Launching DP DDP worker $worker for $hospital_id with rank $rank..."

        remote_run "$worker" "
            nohup $PYTHON_BIN src/DP_hierarchial/ddp_worker.py \
                --master_address $coord \
                --ddp_port $ddp_port \
                --hospital_id $hospital_id \
                --num_nodes $NUM_NODES \
                --node_rank $rank \
                --num_rounds $NUM_ROUNDS \
                --local_epochs $LOCAL_EPOCHS \
                --dp_noise_multiplier $DP_NOISE_MULTIPLIER \
                --dp_max_grad_norm $DP_MAX_GRAD_NORM \
                --dp_delta $DP_DELTA \
                $secure_arg \
                > $LOG_DIR/${worker}_${hospital_id}_dp_worker_rank${rank}.log 2>&1 &
        "

        rank=$((rank + 1))
    done
}

launch_coordinator() {
    local coord=$1
    local hospital_id=$2
    local ddp_port=$3

    local secure_arg
    secure_arg=$(secure_rng_arg)

    echo "Launching DP hospital coordinator $coord for $hospital_id..."

    remote_run "$coord" "
        nohup $PYTHON_BIN src/DP_hierarchial/hospital_coordinator.py \
            --master_address $MASTER_NODE:$MASTER_PORT \
            --hospital_id $hospital_id \
            --num_nodes $NUM_NODES \
            --ddp_port $ddp_port \
            --dp_noise_multiplier $DP_NOISE_MULTIPLIER \
            --dp_max_grad_norm $DP_MAX_GRAD_NORM \
            --dp_delta $DP_DELTA \
            $secure_arg \
            > $LOG_DIR/${coord}_${hospital_id}_dp_coordinator.log 2>&1 &
    "
}

print_config() {
    echo "============================================================"
    echo "DP Hierarchical FL Configuration"
    echo "============================================================"
    echo "Project directory:       $PROJECT_DIR"
    echo "Python:                  $PYTHON_BIN"
    echo "Logs:                    $LOG_DIR"
    echo "Master:                  $MASTER_NODE:$MASTER_PORT"
    echo "Rounds:                  $NUM_ROUNDS"
    echo "Local epochs:            $LOCAL_EPOCHS"
    echo "Min clients:             $MIN_CLIENTS"
    echo "Num DDP nodes/hospital:  $NUM_NODES"
    echo ""
    echo "DP noise multiplier:     $DP_NOISE_MULTIPLIER"
    echo "DP max grad norm:        $DP_MAX_GRAD_NORM"
    echo "DP delta:                $DP_DELTA"
    echo "Secure RNG:              $SECURE_RNG"
    echo "============================================================"
}

# ------------------------------------------------------------------------------
# 5. Launch
# ------------------------------------------------------------------------------

check_python
print_config
cleanup_old_processes

echo "============================================================"
echo "Launching Tier 1: DP Master Coordinator"
echo "============================================================"

launch_master
wait_for_master

echo "============================================================"
echo "Launching Tier 3: DP DDP Workers"
echo "============================================================"

# Launch workers first so they are ready when coordinators start DDP.
launch_workers "$COORD_A" "hospital_a" "$DDP_PORT_A" "${WORKERS_A[@]}"
launch_workers "$COORD_B" "hospital_b" "$DDP_PORT_B" "${WORKERS_B[@]}"
launch_workers "$COORD_C" "hospital_c" "$DDP_PORT_C" "${WORKERS_C[@]}"

echo "Waiting for DP DDP workers to initialize..."
sleep 15

echo "============================================================"
echo "Launching Tier 2: DP Hospital Coordinators"
echo "============================================================"

launch_coordinator "$COORD_A" "hospital_a" "$DDP_PORT_A"
launch_coordinator "$COORD_B" "hospital_b" "$DDP_PORT_B"
launch_coordinator "$COORD_C" "hospital_c" "$DDP_PORT_C"

echo "============================================================"
echo "DP Hierarchical FL cluster launched."
echo "Project directory: $PROJECT_DIR"
echo "Logs directory:    $LOG_DIR"
echo ""
echo "Monitor master:"
echo "  ssh $MASTER_NODE \"tail -f $LOG_DIR/${MASTER_NODE}_dp_master.log\""
echo ""
echo "Monitor coordinators:"
echo "  ssh $COORD_A \"tail -f $LOG_DIR/${COORD_A}_hospital_a_dp_coordinator.log\""
echo "  ssh $COORD_B \"tail -f $LOG_DIR/${COORD_B}_hospital_b_dp_coordinator.log\""
echo "  ssh $COORD_C \"tail -f $LOG_DIR/${COORD_C}_hospital_c_dp_coordinator.log\""
echo ""
echo "Monitor workers:"
echo "  ssh ${WORKERS_A[0]} \"tail -f $LOG_DIR/${WORKERS_A[0]}_hospital_a_dp_worker_rank1.log\""
echo "  ssh ${WORKERS_A[1]} \"tail -f $LOG_DIR/${WORKERS_A[1]}_hospital_a_dp_worker_rank2.log\""
echo "  ssh ${WORKERS_B[0]} \"tail -f $LOG_DIR/${WORKERS_B[0]}_hospital_b_dp_worker_rank1.log\""
echo "  ssh ${WORKERS_B[1]} \"tail -f $LOG_DIR/${WORKERS_B[1]}_hospital_b_dp_worker_rank2.log\""
echo "  ssh ${WORKERS_C[0]} \"tail -f $LOG_DIR/${WORKERS_C[0]}_hospital_c_dp_worker_rank1.log\""
echo "  ssh ${WORKERS_C[1]} \"tail -f $LOG_DIR/${WORKERS_C[1]}_hospital_c_dp_worker_rank2.log\""
echo ""
echo "DP privacy logs should be saved under:"
echo "  fl_logs/DP_coordinator_hospital_a/dp_privacy_metrics.json"
echo "  fl_logs/DP_coordinator_hospital_b/dp_privacy_metrics.json"
echo "  fl_logs/DP_coordinator_hospital_c/dp_privacy_metrics.json"
echo "============================================================"