#!/usr/bin/env bash
# =============================================================================
# launch_clients.sh
# Run on: shark (the server) – SSH-launches clients on perch, turbot, wahoo
#
# Usage:
#   bash scripts/launch_clients.sh [--server_address shark:8080] [--local_epochs 3]
#
# Pre-requisites:
#   - Passwordless SSH from shark → perch, turbot, wahoo
#   - Same conda/venv path on all nodes (edit ACTIVATE_CMD below)
#   - Project directory mounted at the same path on all nodes (edit PROJECT_DIR)
# =============================================================================
set -euo pipefail

# ---------- Configuration – edit these for your cluster ----------------------
NODES=("perch" "turbot" "wahoo")
HOSPITAL_IDS=("hospital_a" "hospital_b" "hospital_c")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Command to activate the Python environment on the remote nodes.
# Examples:
#   conda: "source /opt/conda/etc/profile.d/conda.sh && conda activate fl_env"
#   venv:  "source /home/user/fl_env/bin/activate"
ACTIVATE_CMD="source /s/chopin/k/grad/jasoncur/cs535/termProject/.venv/bin/activate"

# ---------- Defaults (override with CLI flags) --------------------------------
SERVER_ADDRESS="shark:8080"
LOCAL_EPOCHS=3
LOG_DIR_BASE="fl_logs"

# ---------- Parse args --------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --server_address) SERVER_ADDRESS="$2"; shift 2 ;;
    --local_epochs)   LOCAL_EPOCHS="$2";   shift 2 ;;
    --log_dir)        LOG_DIR_BASE="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo " Launching FL clients"
echo "  server_address : $SERVER_ADDRESS"
echo "  local_epochs   : $LOCAL_EPOCHS"
echo "  nodes          : ${NODES[*]}"
echo "============================================================"

# Keep track of child PIDs for cleanup
PIDS=()

cleanup() {
  echo ""
  echo "Caught signal – terminating client processes …"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait
  echo "All clients stopped."
}
trap cleanup SIGINT SIGTERM

echo "Performing pre-flight cleanup on remote nodes..."
for NODE in "${NODES[@]}"; do
  ssh -o StrictHostKeyChecking=no "$NODE" "pkill -f federated_client.py || true" &
done
wait
echo "Remote nodes cleared."


for i in "${!NODES[@]}"; do
  NODE="${NODES[$i]}"
  HOSPITAL="${HOSPITAL_IDS[$i]}"
  LOG_DIR="${LOG_DIR_BASE}/client_${HOSPITAL}"

  echo "  → SSH to ${NODE} (hospital=${HOSPITAL})"

  ssh -o StrictHostKeyChecking=no "$NODE" \
    "cd ${PROJECT_DIR}/src/federated && \
     ${ACTIVATE_CMD} && \
     python federated_client.py \
       --server_address ${SERVER_ADDRESS} \
       --local_epochs   ${LOCAL_EPOCHS} \
       --hospital_id    ${HOSPITAL}" \
    > "${LOG_DIR_BASE}/ssh_${NODE}.log" 2>&1 &

  PIDS+=($!)
done

echo ""
echo "Clients launched (PIDs: ${PIDS[*]}). Waiting …"
echo "Log files: ${LOG_DIR_BASE}/ssh_<node>.log"
echo ""

# Wait for all background jobs
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "Client PID $pid exited with non-zero status."
done

echo "All clients have finished."


# =============================================================================
# Alternative: launch each client manually on its own node
# =============================================================================
#
# On perch:
#   python federated_client.py --server_address shark:8080 --local_epochs 3 --hospital_id hospital_a
#
# On turbot:
#   python federated_client.py --server_address shark:8080 --local_epochs 3 --hospital_id hospital_b
#
# On wahoo:
#   python federated_client.py --server_address shark:8080 --local_epochs 3 --hospital_id hospital_c
