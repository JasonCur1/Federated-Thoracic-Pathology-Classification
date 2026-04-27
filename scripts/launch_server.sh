#!/usr/bin/env bash
# =============================================================================
# launch_server.sh
# Run on: shark
# Usage:  bash scripts/launch_server.sh [--num_rounds 20] [--port 8080]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------- Defaults (override with CLI flags) --------------------------------
NUM_ROUNDS=20
MIN_CLIENTS=3
PORT=8080
LOCAL_EPOCHS=3
LOG_DIR="fl_logs/server"

# ---------- Parse args --------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_rounds)    NUM_ROUNDS="$2";    shift 2 ;;
    --min_clients)   MIN_CLIENTS="$2";   shift 2 ;;
    --port)          PORT="$2";          shift 2 ;;
    --local_epochs)  LOCAL_EPOCHS="$2";  shift 2 ;;
    --log_dir)       LOG_DIR="$2";       shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo " Flower FL Server – node: $(hostname)"
echo "  rounds       : $NUM_ROUNDS"
echo "  min_clients  : $MIN_CLIENTS"
echo "  port         : $PORT"
echo "  local_epochs : $LOCAL_EPOCHS"
echo "  log_dir      : $LOG_DIR"
echo "============================================================"

cd "$PROJECT_DIR/src/federated"

# Activate conda/venv if needed – edit the line below
# source /opt/conda/etc/profile.d/conda.sh && conda activate fl_env

echo "Cleaning up old server processes..."
pkill -f "federated_server.py" || true
sleep 1 # Give it a second to release the port

python federated_server.py \
  --num_rounds   "$NUM_ROUNDS" \
  --min_clients  "$MIN_CLIENTS" \
  --port         "$PORT" \
  --local_epochs "$LOCAL_EPOCHS" \
  --log_dir      "$LOG_DIR"
