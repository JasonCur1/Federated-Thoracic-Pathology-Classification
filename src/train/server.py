import flwr as fl
import task
import torch
import json

def eval_metrics(metrics):
    # metrics is a list of (num_examples, metric_dict)
    total_examples = sum([num for num, _ in metrics])
    aggregated_metrics = {}
    
    # Get all metric keys from the first client (global_auc, auc_Atelectasis, etc.)
    all_keys = metrics[0][1].keys()
    
    for key in all_keys:
        weighted_sum = sum([num * m[key] for num, m in metrics])
        mean_val = weighted_sum / total_examples
        aggregated_metrics[key] = mean_val
        
        # cross-client variance
        variance_sum = sum([num * ((m[key] - mean_val) ** 2) for num, m in metrics])
        aggregated_metrics[f"{key}_variance"] = variance_sum / total_examples

    return aggregated_metrics

def fit_metrics(metrics):
    total_examples = sum([num for num, _ in metrics])
    
    weighted_losses = sum([num * m["train_loss"] for num, m in metrics])
    avg_train_loss = weighted_losses / total_examples
    
    # latency and straggler data
    latencies = [m["latency"] for _, m in metrics]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    # weight divergence
    divergences = [m["weight_divergence"] for _, m in metrics]
    avg_divergence = sum(divergences) / len(divergences)
    max_divergence = max(divergences)

    return {
        "avg_train_loss": avg_train_loss,
        "avg_latency": avg_latency,
        "max_latency": max_latency, # Identifies the round's straggler bottleneck
        "avg_weight_divergence": avg_divergence,
        "max_weight_divergence": max_divergence
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = task.load_model(num_diseases=14)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           
        fraction_evaluate=1.0,      
        min_fit_clients=3,          
        min_evaluate_clients=3,     
        min_available_clients=3,    
        on_fit_config_fn=lambda server_round: {"local_epochs": 3}, 
        evaluate_metrics_aggregation_fn=eval_metrics,
        fit_metrics_aggregation_fn=fit_metrics,
    )

    print("Starting Flower Server...")
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )
    
    metrics_to_save = {
        "losses_distributed": history.losses_distributed,
        "metrics_distributed": history.metrics_distributed,
        "metrics_distributed_fit": history.metrics_distributed_fit,
    }
    
    with open("src/train/federated_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print("Federated metrics saved to federated_metrics.json")