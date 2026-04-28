# Federated-Thoracic-Pathology-Classification

# Notes
- venv packages can be tricky. More recent versions of numpy experience lots of deprecated functionality due to switching from C to C++.
    - May have to downgrade numpy for torch compatibility

# Baseline Instructions
python src/baseline/train.py

# Federated Instructions
- launch_server.sh can be ran anywhere.
- The server_address passed to launch_clients.sh must match the machine running the server node.
- See code for configurable parameters

bash scripts/launch_fl.sh --num_rounds 100 --port 8084 --server_address boston:8084 --local_epochs 1

# Hierarchical Instructions
bash scripts/launch_hierarchical.sh