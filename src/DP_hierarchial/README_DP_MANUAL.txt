DP_hierarchial manual PyTorch version
====================================

Files:
  dp_utils.py
    New Opacus helper functions.

  hospital_coordinator.py
    CHANGED. Uses manual PyTorch training with Opacus PrivacyEngine.
    It does not import pytorch_lightning and does not use pl.Trainer/DDP.

  master_coordinator.py
    MOSTLY SAME as the DP master version. It runs Flower FedAvg and logs
    epsilon/delta/noise/clip values returned by each hospital coordinator.

Removed/not used:
  ddp_worker.py
    Not needed in this manual PyTorch DP version because we are not using
    Lightning DDP local workers.

Run example:
  python src/DP_hierarchial/master_coordinator.py --port 8080 --num_rounds 20 --min_clients 3 --local_epochs 1

  python src/DP_hierarchial/hospital_coordinator.py --master_address boston:8080 --hospital_id HOSPITAL_1 --dp_noise_multiplier 1.0 --dp_max_grad_norm 1.0 --dp_delta 1e-5

Try privacy-accuracy tradeoff runs by changing --dp_noise_multiplier, for example:
  0.5, 1.0, 1.5, 2.0
