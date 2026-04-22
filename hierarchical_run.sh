#!/bin/bash

# Usage: ./hierarchical_run.sh <base_data_dir>

PIDS=()

cleanup() {
    echo "Shutting down all processes across the cluster..."
    
    # 1. Kill local SSH processes
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    
    # 2. Kill Master
    echo "Cleaning up Master (little-rock)..."
    ssh little-rock "pkill -u \$(whoami) -f 'server.py'" 2>/dev/null
    
    # 3. Kill Coordinators
    COORDINATORS=(warhol raphael lincoln)
    for machine in "${COORDINATORS[@]}"; do
        echo "Cleaning up Coordinator: $machine..."
        ssh "$machine" "pkill -u \$(whoami) -f 'coordinator.py'" 2>/dev/null
    done

    # 4. Kill All Workers
    WORKERS=(flounder blowfish herring bullhead wahoo tuna grouper marlin perch)
    for machine in "${WORKERS[@]}"; do
        echo "Cleaning up Worker: $machine..."
        # Kill both the worker script and the torchrun daemon
        ssh "$machine" "pkill -u \$(whoami) -f 'worker.py'; pkill -u \$(whoami) -f 'torchrun'" 2>/dev/null
    done
    
    wait
    echo "All processes cleared."
}
trap cleanup EXIT INT TERM

# Start server
echo "Starting Master on little-rock..."
ssh little-rock "cd $PWD && python -u src/hierarchical/server.py" > hierarchical/server.log 2>&1 &
PIDS+=($!)

sleep 5

declare -A CLIENT_MAP=(
    [warhol]="hospital_a"
    [raphael]="hospital_b"
    [lincoln]="hospital_c"
)

# Start 3 Coordinators
for machine in "${!CLIENT_MAP[@]}"; do
    hospital="${CLIENT_MAP[$machine]}"
    echo "Starting Coordinator on $machine ($hospital)..."
    ssh "$machine" "cd $PWD && python -u src/hierarchical/coordinator.py --hospital-name $hospital --data-path \"data/$hospital\"" > "hierarchical/coordinator_$machine.log" 2>&1 &
    PIDS+=($!)
done

wait