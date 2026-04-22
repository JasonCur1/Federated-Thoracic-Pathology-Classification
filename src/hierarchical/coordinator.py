import argparse
import time
import torch
import flwr as fl
import os
import subprocess
from collections import OrderedDict
import task
import pandas as pd
import glob
import pyarrow.parquet as pq

def get_exact_row_count(data_path):
    total_rows = 0
    for f in glob.glob(os.path.join(data_path, 'train-*.parquet')):
        total_rows += pq.ParquetFile(f).metadata.num_rows
    return total_rows

class HospitalCoordinator(fl.client.NumPyClient):
    def __init__(self, hospital_name, data_path, shared_dir):
        self.hospital_name = hospital_name
        self.data_path = data_path
        self.global_weights_path = os.path.join(shared_dir, f"{hospital_name}_global.pt")
        self.local_weights_path = os.path.join(shared_dir, f"{hospital_name}_local.pt")
        
        # Load eval data locally on coordinator just to return eval metrics to Flower
        all_eval_files = glob.glob(os.path.join(self.data_path, 'eval-*.parquet'))
        df_eval = pd.concat((pd.read_parquet(f) for f in all_eval_files), ignore_index=True)

        _, self.eval_loader = task.load_data(df_eval.head(1), df_eval) 
        self.model = task.load_model(num_diseases=14)
        self.device = torch.device("cpu") # Eval is fine on CPU for the coordinator

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)



    def fit(self, parameters, config):
        self.set_parameters(parameters)
        torch.save(self.model.state_dict(), self.global_weights_path)

        epochs = config.get("local_epochs", 1)

        # Updated to use the unified bash script
        launch_script = "./run_workers.sh"
        print(f"[{self.hospital_name}] Launching DDP workers...")
        
        subprocess.run([
            "bash", launch_script, 
            "--hospital-name", self.hospital_name,
            "--data-path", self.data_path,
            "--global-weights", self.global_weights_path,
            "--local-weights", self.local_weights_path,
            "--epochs", str(epochs)
        ], check=True)

        trained_state_dict = torch.load(self.local_weights_path, map_location="cpu")
        self.model.load_state_dict(trained_state_dict, strict=True)
        
        dataset_size = get_exact_row_count(self.data_path)
        return self.get_parameters(config={}), dataset_size, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = task.test(self.model, self.eval_loader, self.device, is_distributed=False)
        return loss, len(self.eval_loader.dataset), metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital-name", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--shared-dir", type=str, default="./shared_weights")
    args = parser.parse_args()

    os.makedirs(args.shared_dir, exist_ok=True)

    for attempt in range(10):
        try:
            print(f"Connecting to master at little-rock:8080 (attempt {attempt + 1})...")
            fl.client.start_client(
                server_address="little-rock:8080",
                client=HospitalCoordinator(args.hospital_name, args.data_path, args.shared_dir).to_client()
            )
            break
        except Exception as e:
            time.sleep(5)