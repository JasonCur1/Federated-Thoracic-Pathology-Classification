import argparse
import torch
import flwr as fl
from collections import OrderedDict
import pandas as pd
import task
import glob
import os

class Client(fl.client.NumPyClient):
    def __init__(self, model, train_loader, eval_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device

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
        self.set_parameters(parameters)

        # Train locally from num epochs from config. Default = 1
        epochs = config.get("local_epochs", 1)
        task.train(self.model, self.train_loader, epochs, self.device)

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = task.test(self.model, self.eval_loader, self.device)
        return loss, len(self.eval_loader.dataset), metrics

# I want to use args for the client so that the run script is simplified. These need to be path agnostic
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--data-path", type=str, required=True, help="Path to hospital x data")
    args = parser.parse_args()

    all_training_files = glob.glob(os.path.join(args.data_path, 'train-*.parquet'))
    df_train = pd.concat((pd.read_parquet(f) for f in all_training_files), ignore_index=True)

    all_eval_files = glob.glob(os.path.join(args.data_path, 'eval-*.parquet'))
    df_eval = pd.concat((pd.read_parquet(f) for f in all_eval_files), ignore_index=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, eval_loader = task.load_data(df_train, df_eval)

    model = task.load_model(num_diseases=14)

    # TODO: This address needs to change. Maybe automate with a start script. Idk quite how the setup will look like yet
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=Client(model, train_loader, eval_loader, device).to_client()
    )