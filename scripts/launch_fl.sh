#!/usr/bin/env bash
# =============================================================================
# launch_federated.sh
# Combines server and client launch into a single execution.
# Runs the server locally, waits 80s, then launches clients across the cluster.
# Everything is safely backgrounded using nohup to survive terminal closure.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------- Configuration – edit these for your cluster ----------------------
NODES=("perch" "turbot" "wahoo")
HOSPITAL_IDS=("hospital_a" "hospital_b" "hospital_c")
ACTIVATE_CMD="source /s/chopin/k/grad/jasoncur/cs535/termProject/.venv/bin/activate"

# ---------- Internal Background Launcher -------------------------------------
# This block intercepts an internal call to handle the 80-second delay
# and SSH client spawning in the background without hanging the user's prompt.
if [[ "${1:-}" == "--internal_client_launcher" ]]; then
  shift
  SERVER_ADDRESS=$1
  LOCAL_EPOCHS=$2
  LOG_DIR=$3

  echo "[Background Task] Waiting 80 seconds for the server to fully initialize..."
  sleep 80

  echo "[Background Task] Launching clients on remote nodes..."
  for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    HOSPITAL="${HOSPITAL_IDS[$i]}"

    # -n prevents reading stdin, -f puts ssh in background
    # nohup on the remote command ensures it survives disconnected SSH
    ssh -n -f -o StrictHostKeyChecking=no "$NODE" \
      "cd ${PROJECT_DIR}/src/federated && \
       ${ACTIVATE_CMD} && \
       mkdir -p ${PROJECT_DIR}/${LOG_DIR}/client_${HOSPITAL} && \
       nohup python federated_client.py \
         --server_address ${SERVER_ADDRESS} \
         --local_epochs   ${LOCAL_EPOCHS} \
         --hospital_id    ${HOSPITAL} \
         > ${PROJECT_DIR}/${LOG_DIR}/client_${HOSPITAL}/client_stdout.log 2>&1 &"
  done
  
  echo "[Background Task] All remote clients launched."
  exit 0
fi

# ---------- Defaults ---------------------------------------------------------
NUM_ROUNDS=100
MIN_CLIENTS=3
PORT=8080
SERVER_ADDRESS_OVERRIDE=""
LOCAL_EPOCHS=3
LOG_DIR="fl_logs"

# ---------- Parse args -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_rounds)     NUM_ROUNDS="$2";     shift 2 ;;
    --min_clients)    MIN_CLIENTS="$2";    shift 2 ;;
    --port)           PORT="$2";           shift 2 ;;
    --server_address) SERVER_ADDRESS_OVERRIDE="$2"; shift 2 ;;
    --local_epochs)   LOCAL_EPOCHS="$2";   shift 2 ;;
    --log_dir)        LOG_DIR="$2";        shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Resolve server address dynamically if not explicitly provided
if [[ -z "$SERVER_ADDRESS_OVERRIDE" ]]; then
  SERVER_ADDRESS="$(hostname):$PORT"
else
  SERVER_ADDRESS="$SERVER_ADDRESS_OVERRIDE"
fi

echo "============================================================"
echo " Launching Federated Learning Setup"
echo "  num_rounds     : $NUM_ROUNDS"
echo "  min_clients    : $MIN_CLIENTS"
echo "  port           : $PORT"
echo "  server_address : $SERVER_ADDRESS"
echo "  local_epochs   : $LOCAL_EPOCHS"
echo "  log_dir        : $LOG_DIR"
echo "============================================================"

mkdir -p "$PROJECT_DIR/$LOG_DIR/server"

echo "Performing pre-flight cleanup on all nodes..."
pkill -f "federated_server.py" || true
for NODE in "${NODES[@]}"; do
  ssh -n -o StrictHostKeyChecking=no "$NODE" "pkill -f federated_client.py || true" &
done
wait
echo "Cleanup finished."

cd "$PROJECT_DIR/src/federated"

echo "1) Starting FL Server via nohup..."
nohup python federated_server.py \
  --num_rounds   "$NUM_ROUNDS" \
  --min_clients  "$MIN_CLIENTS" \
  --port         "$PORT" \
  --local_epochs "$LOCAL_EPOCHS" \
  --log_dir      "$PROJECT_DIR/$LOG_DIR/server" \
  > "$PROJECT_DIR/$LOG_DIR/server/server_nohup.log" 2>&1 &

echo "2) Spawning background client launcher (will wait 80 seconds)..."
nohup bash "$0" --internal_client_launcher "$SERVER_ADDRESS" "$LOCAL_EPOCHS" "$LOG_DIR" \
  > "$PROJECT_DIR/$LOG_DIR/launcher_nohup.log" 2>&1 &

echo ""
echo "Successfully dispatched background tasks!"
echo "Server logs   : tail -f $PROJECT_DIR/$LOG_DIR/server/server_nohup.log"
echo "Launcher logs : tail -f $PROJECT_DIR/$LOG_DIR/launcher_nohup.log"
echo "You may now safely close this terminal."