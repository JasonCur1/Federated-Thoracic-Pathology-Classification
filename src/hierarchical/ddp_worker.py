from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch.distributed as dist
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.config import CFG
from baseline.datamodule import FederatedDataModule
from baseline.model import ChestXrayClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master_address", type=str, required=True)
    p.add_argument("--ddp_port", type=int, default=12345)
    p.add_argument("--hospital_id", type=str, required=True)
    p.add_argument("--num_nodes", type=int, default=3)
    p.add_argument("--node_rank", type=int, required=True)
    p.add_argument("--num_rounds", type=int, default=20)
    p.add_argument("--local_epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=CFG.seed)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.node_rank == 0:
        raise ValueError("Worker node_rank must be > 0. Rank 0 is reserved for the Hospital Coordinator.")

    pl.seed_everything(args.seed, workers=True)

    # Setup PyTorch Distributed Environment Variables
    # This instructs PyTorch Lightning how to connect to the Tier 2 Coordinator
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = str(args.ddp_port)
    os.environ["WORLD_SIZE"] = str(args.num_nodes)
    os.environ["NODE_RANK"] = str(args.node_rank)

    logger.info(
        "Configured as DDP Worker (Rank %d/%d) for %s. Waiting for Master at %s:%d...",
        args.node_rank, args.num_nodes - 1, args.hospital_id, args.master_address, args.ddp_port
    )

    datamodule = FederatedDataModule(cfg=CFG, hospital_id=args.hospital_id)
    datamodule.setup()

    # Initialize Model
    # We do not need to worry about loading the global FL weights manually
    # PyTorch DDP automatically broadcasts the model state from Rank 0 to all workers at the beginning of trainer.fit().
    model = ChestXrayClassifier(pos_weight=datamodule.pos_weight, cfg=CFG)
    model.cfg.max_epochs = args.local_epochs

    # Enter Federated Round Loop
    for fl_round in range(1, args.num_rounds + 1):
        current_port = args.ddp_port + fl_round
        os.environ["MASTER_PORT"] = str(current_port)
        
        logger.info("\n" + "="*50)
        logger.info("[Worker Rank %d] Starting Federated Round %d / %d on port %d", 
                    args.node_rank, fl_round, args.num_rounds, current_port)
        logger.info("="*50)

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            num_nodes=args.num_nodes,
            strategy="ddp",
            max_epochs=args.local_epochs,
            enable_checkpointing=False,
            logger=False,
            sync_batchnorm=True,
            use_distributed_sampler=True,
        )

        trainer.fit(model=model, datamodule=datamodule)
        
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("[Worker Rank %d] DDP process group destroyed.", args.node_rank)
        
        logger.info("[Worker Rank %d] Finished local training for Round %d", args.node_rank, fl_round)


if __name__ == "__main__":
    main()