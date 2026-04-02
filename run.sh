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
    
    # 1. Kill local SSH processes
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    
    # 2. Explicitly kill the remote server on little-rock
    # -u $(whoami) ensures you only kill your own processes
    echo "Cleaning up little-rock..."
    ssh little-rock "pkill -u \$(whoami) -f 'src/train/server.py'" 2>/dev/null
    
    # 3. Explicitly kill the remote clients
    for machine in "${!CLIENT_MAP[@]}"; do
        echo "Cleaning up $machine..."
        ssh "$machine" "pkill -u \$(whoami) -f 'src/train/client.py'" 2>/dev/null
    done
    
    wait
    echo "All processes cleared."
}
trap cleanup EXIT INT TERM

# Start server
echo "Starting server on little-rock..."
ssh little-rock "cd $PWD && python -u src/train/server.py" > server.log 2>&1 &
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
    ssh "$machine" "cd $PWD && python -u src/train/client.py --data-path \"data/$hospital\"" > "client_$machine.log" 2>&1 &
    PIDS+=($!)
done

# Wait for all processes to finish
wait