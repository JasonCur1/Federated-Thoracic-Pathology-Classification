"""
DP DDP Worker for one hospital.

This file replaces PyTorch Lightning DDP worker logic with direct torch.distributed
training using Opacus DPDDP.

Important:
- This script does not connect to Flower.
- The hospital coordinator is rank 0 and connects to Flower.
- This worker joins the local hospital-level DDP group each FL round.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.config import CFG
from baseline.datamodule import FederatedDataModule
from baseline.model import ChestXrayClassifier

from DP_hierarchial.dp_utils import (
    attach_dpddp_privacy_engine,
    broadcast_model_state,
    cleanup_process_group,
    get_learning_rate,
    get_weight_decay,
    make_opacus_compatible,
    seed_everything,
    setup_process_group,
    unwrap_private_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DP DDP Worker for hierarchical FL")
    p.add_argument("--master_address", type=str, required=True)
    p.add_argument("--ddp_port", type=int, default=12345)
    p.add_argument("--hospital_id", type=str, required=True)
    p.add_argument("--num_nodes", type=int, default=3)
    p.add_argument("--node_rank", type=int, required=True)
    p.add_argument("--num_rounds", type=int, default=20)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=CFG.seed)

    p.add_argument("--dp_noise_multiplier", type=float, default=1.0)
    p.add_argument("--dp_max_grad_norm", type=float, default=1.0)
    p.add_argument("--dp_delta", type=float, default=1e-5)
    p.add_argument("--secure_rng", action="store_true")

    return p.parse_args()


def run_one_round(
    args: argparse.Namespace,
    model: ChestXrayClassifier,
    datamodule: FederatedDataModule,
    fl_round: int,
) -> None:
    rank = int(args.node_rank)
    world_size = int(args.num_nodes)
    current_port = int(args.ddp_port) + int(fl_round)

    setup_process_group(
        rank=rank,
        world_size=world_size,
        master_addr=args.master_address,
        master_port=current_port,
    )

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = make_opacus_compatible(model)

        model.to(device)
        model.train()

        # Rank 0 broadcasts the Flower global weights here.
        # Worker ranks receive those weights before training starts.
        broadcast_model_state(model, src=0)

        train_loader = datamodule.fl_train_dataloader()
        loss_fn = model._get_loss_fn()

        lr = get_learning_rate(CFG, default=1e-4)
        weight_decay = get_weight_decay(CFG, default=0.0)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        private_model, private_optimizer, private_loader, privacy_engine = attach_dpddp_privacy_engine(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            noise_multiplier=args.dp_noise_multiplier,
            max_grad_norm=args.dp_max_grad_norm,
            secure_mode=args.secure_rng,
        )

        private_model.to(device)
        private_model.train()

        total_loss = 0.0
        total_steps = 0

        for epoch in range(int(args.local_epochs)):
            epoch_loss = 0.0
            epoch_steps = 0

            for images, targets in private_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                private_optimizer.zero_grad(set_to_none=True)

                logits = private_model(images)
                loss = loss_fn(logits, targets)

                loss.backward()
                private_optimizer.step()

                loss_value = float(loss.detach().cpu().item())
                epoch_loss += loss_value
                total_loss += loss_value
                epoch_steps += 1
                total_steps += 1

            epsilon = float(privacy_engine.get_epsilon(delta=args.dp_delta))
            logger.info(
                "[Worker rank %d] Round %d Epoch %d/%d | loss=%.4f epsilon=%.4f",
                rank,
                fl_round,
                epoch + 1,
                args.local_epochs,
                epoch_loss / max(epoch_steps, 1),
                epsilon,
            )

        # Keep worker model state updated for the next round.
        trained_base_model = unwrap_private_model(private_model)
        model.load_state_dict(trained_base_model.state_dict(), strict=True)

        logger.info(
            "[Worker rank %d] Finished round %d | mean_loss=%.4f epsilon=%.4f",
            rank,
            fl_round,
            total_loss / max(total_steps, 1),
            float(privacy_engine.get_epsilon(delta=args.dp_delta)),
        )

    finally:
        cleanup_process_group()


def main() -> None:
    args = parse_args()

    if args.node_rank == 0:
        raise ValueError("node_rank=0 is reserved for hospital_coordinator.py")

    seed_everything(args.seed)

    logger.info(
        "Configured as DP DDP worker | hospital=%s rank=%d/%d coordinator=%s base_port=%d hostname=%s",
        args.hospital_id,
        args.node_rank,
        args.num_nodes,
        args.master_address,
        args.ddp_port,
        socket.gethostname(),
    )

    datamodule = FederatedDataModule(cfg=CFG, hospital_id=args.hospital_id)
    datamodule.setup()

    model = ChestXrayClassifier(pos_weight=datamodule.pos_weight, cfg=CFG)
    model = make_opacus_compatible(model)

    for fl_round in range(1, int(args.num_rounds) + 1):
        logger.info("=" * 80)
        logger.info(
            "[Worker rank %d] Waiting for coordinator for FL round %d/%d on port %d",
            args.node_rank,
            fl_round,
            args.num_rounds,
            args.ddp_port + fl_round,
        )
        logger.info("=" * 80)

        run_one_round(
            args=args,
            model=model,
            datamodule=datamodule,
            fl_round=fl_round,
        )


if __name__ == "__main__":
    main()