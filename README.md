# Federated-Thoracic-Pathology-Classification


## How to run:
Terminal 1 (Server):
    - python server.py

Terminal 2 (Hospital A)
    - python client.py --data-path "../../data/hospital_a"

Terminal 3 (Hospital B)
    - python client.py --data-path "../../data/hospital_b"

Terminal 4 (Hospital C)
    - python client.py --data-path "../../data/hospital_c"

TODO: Make a script to automate this. Each hospital could have its own cluster, which would use Pytorch DDP.