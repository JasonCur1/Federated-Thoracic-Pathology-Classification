# Federated-Thoracic-Pathology-Classification

# Notes
- venv packages can be tricky. More recent versions of numpy experience lots of deprecated functionality due to switching from C to C++.
    - May have to downgrade numpy for torch compatibility

# Baseline Instructions
python src/baseline/train.py

# Federated Instructions
- Launch_server.sh can be ran anywhere.
- The server_address passed to launch_clients.sh must match the machine running the server node.
- See code for configurable parameters

## Terminal 1:
bash scripts/launch_server.sh --num_rounds 30 --port 8084

## Terminal 2:
bash scripts/launch_clients.sh --server_address boston:8084 --local_epochs 1