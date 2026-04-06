import flwr as fl
import task
import torch

def eval_metrics(metrics):
    # metrics is a list of (num_examples, metric_dict)
    total_examples = sum([num for num, _ in metrics])
    
    aggregated_metrics = {}
    
    # Get all metric keys from the first client (global_auc, auc_Atelectasis, etc.)
    all_keys = metrics[0][1].keys()
    
    for key in all_keys:
        weighted_sum = sum([num * m[key] for num, m in metrics])
        aggregated_metrics[key] = weighted_sum / total_examples
        
    return aggregated_metrics

def fit_metrics(metrics):
    # metrics is a list of tuples: (num_examples, {"train_loss": value})
    total_examples = sum([num for num, _ in metrics])
    weighted_losses = [num * m["train_loss"] for num, m in metrics]
    return {"avg_train_loss": sum(weighted_losses) / total_examples}

if __name__ == "__main__":
    device = torch.device("cpu")
    global_model = task.load_model(num_diseases=14)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # Sample 100% of available clients for training
        fraction_evaluate=1.0,      # Sample 100% of available clients for evaluation
        min_fit_clients=2,          # Minimum number of clients required to start a training round
        min_evaluate_clients=3,     # Minimum number of clients required to start an eval round
        min_available_clients=3,    # Wait until all clients are connected
        on_fit_config_fn=lambda server_round: {"local_epochs": 5}, # Send config to clients
        evaluate_metrics_aggregation_fn=eval_metrics,
        fit_metrics_aggregation_fn=fit_metrics,
    )

    print("Starting Flower Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5), # Run for 5 global rounds
        strategy=strategy,
    )