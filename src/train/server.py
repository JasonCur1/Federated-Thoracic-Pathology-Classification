import flwr as fl
import task
import torch

if __name__ == "__main__":
    device = torch.device("cpu")
    global_model = task.load_model(num_diseases=14)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # Sample 100% of available clients for training
        fraction_evaluate=1.0,      # Sample 100% of available clients for evaluation
        min_fit_clients=2,          # Minimum number of clients required to start a training round
        min_evaluate_clients=3,     # Minimum number of clients required to start an eval round
        min_available_clients=3,    # Wait until all clients are connected
        on_fit_config_fn=lambda server_round: {"local_epochs": 1}, # Send config to clients
    )

    print("Starting Flower Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5), # Run for 5 global rounds
        strategy=strategy,
    )