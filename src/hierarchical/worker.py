import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import argparse
import pandas as pd
import glob
import task

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--global-weights-path", type=str, required=True)
    parser.add_argument("--local-weights-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # NCCL is for GPU training
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Load Parquet files
    all_training_files = glob.glob(os.path.join(args.data_path, 'train-*.parquet'))
    df_train = pd.concat((pd.read_parquet(f) for f in all_training_files), ignore_index=True)
    
    # We only evaluate on Rank 0 for simplicity
    all_eval_files = glob.glob(os.path.join(args.data_path, 'eval-*.parquet'))
    df_eval = pd.concat((pd.read_parquet(f) for f in all_eval_files), ignore_index=True)

    train_loader, _ = task.load_data(df_train, df_eval, is_distributed=True)

    model = task.load_model(num_diseases=14)
    global_state_dict = torch.load(args.global_weights_path, map_location="cpu")
    model.load_state_dict(global_state_dict, strict=True)
    
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    task.train(model, train_loader, args.epochs, device, is_distributed=True)

    # Rank 0 saves the newly trained weights for the coordinator to pick up
    if dist.get_rank() == 0:
        # Save the model state dict
        torch.save(model.module.state_dict(), args.local_weights_path)
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()