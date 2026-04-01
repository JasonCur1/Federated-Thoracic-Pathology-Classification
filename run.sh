#!/bin/bash

# Usage: ./run.sh <base_data_dir>
# Expected layout:
#   <base_data_dir>/hospital1/train-*.parquet, eval-*.parquet
#   <base_data_dir>/hospital2/train-*.parquet, eval-*.parquet
#   <base_data_dir>/hospital3/train-*.parquet, eval-*.parquet

# Track child PIDs for cleanup
PIDS=()

cleanup() {
    echo "Shutting down all processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    echo "Done."
}
trap cleanup EXIT INT TERM

# Start server
echo "Starting server on little-rock..."
ssh little-rock "cd $PWD && python src/train/server.py" > server.log 2>&1 &
PIDS+=($!)

# Give the server a moment to bind before clients connect
sleep 5

declare -A CLIENT_MAP=(
    [warhol]="hospital_a"
    [raphael]="hospital_b"
    [lincoln]="hospital_c"
)

# Start 3 clients, one per hospital
for machine in "${!CLIENT_MAP[@]}"; do
    hospital="${CLIENT_MAP[$machine]}"
    echo "Starting client on $machine ($hospital)..."
    # Added output redirection for easier debugging
    ssh "$machine" "cd $PWD && python src/train/client.py --data-path \"data/$hospital\"" > "client_$machine.log" 2>&1 &
    PIDS+=($!)
done

# Wait for all processes to finish
wait