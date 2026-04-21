import argparse
import time
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
import pandas as pd
import task
import glob
import os

class Client(fl.client.NumPyClient):
    def __init__(self, model, train_loader, eval_loader, test_loader, device, pos_weight):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.device = device
        self.pos_weight = pos_weight

    def get_parameters(self, config):
        weights = []

        for _, val in self.model.state_dict().items():
            weights.append(val.cpu().numpy())

        return weights
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        start_time = time.time()
        self.set_parameters(parameters)

        epochs = config.get("local_epochs", 1)
        total_train_loss = task.train(self.model, self.train_loader, epochs, self.device, self.pos_weight) # pass a list of loss values now to use in future plotting functions
        avg_loss = sum(total_train_loss) / len(total_train_loss)

        end_time = time.time()
        latency = end_time - start_time

        new_parameters = self.get_parameters(config={})

        divergence = 0.0
        for old_layer, new_layer in zip(parameters, new_parameters):
            divergence += np.sum(np.square(np.array(old_layer) - np.array(new_layer)))
        divergence = float(np.sqrt(divergence))

        metrics = {
            "train_loss": float(avg_loss),
            "latency": float(latency),
            "weight_divergence": divergence
        }

        return new_parameters, len(self.train_loader.dataset), metrics
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = task.test(self.model, self.eval_loader, self.test_loader, self.device)
        return loss, len(self.eval_loader.dataset), metrics

# I want to use args for the client so that the run script is simplified. These need to be path agnostic
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--data-path", type=str, required=True, help="Path to hospital x data (e.g., data/hospital_a)")
    args = parser.parse_args()

    all_training_files = glob.glob(os.path.join(args.data_path, 'train*.parquet'))
    df_train = pd.concat((pd.read_parquet(f) for f in all_training_files), ignore_index=True)

    all_eval_files = glob.glob(os.path.join(args.data_path, 'eval*.parquet'))
    df_eval = pd.concat((pd.read_parquet(f) for f in all_eval_files), ignore_index=True)

    local_test_files = glob.glob(os.path.join(args.data_path, 'test_*.parquet'))
    
    parent_dir = os.path.dirname(os.path.normpath(args.data_path))
    global_test_files = glob.glob(os.path.join(parent_dir, 'test_D*.parquet'))
    
    # Combine both test sets
    all_test_files = local_test_files + global_test_files
    df_test = pd.concat((pd.read_parquet(f) for f in all_test_files), ignore_index=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    train_loader, eval_loader, test_loader = task.load_data(df_train, df_eval, df_test)

    model = task.load_model(num_diseases=14)
    pos_weight = task.compute_pos_weights(df_train, cap=50.0)

    for attempt in range(10):
        try:
            print(f"Connecting to server (attempt {attempt + 1}/10)...")
            fl.client.start_client(
                server_address="little-rock:8080",
                client=Client(model, train_loader, eval_loader, test_loader, device, pos_weight).to_client()
            )
            break
        except Exception as e:
            print(f"Failed: {e}")
            time.sleep(5)