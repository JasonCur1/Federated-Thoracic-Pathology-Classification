"""
Acts as a Flower Client to the Master Coordinator.
Acts as the DDP Master for its local GPU Workers.
"""

from __future__ import annotations

import argparse
import logging
import socket
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.config import CFG, DISEASE_LABELS
from baseline.datamodule import FederatedDataModule
from baseline.model import ChestXrayClassifier
from federated.metrics_logger import FLMetricsLogger, RoundClientStats, compute_model_size_bytes
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
from torchmetrics.classification import MultilabelRecall, MultilabelPrecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# Helpers
def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


class HospitalCoordinatorClient(fl.client.Client):
    def __init__(
        self,
        model: ChestXrayClassifier,
        datamodule: FederatedDataModule,
        hospital_id: str,
        num_nodes: int = 3, # 1 Coordinator + 2 Workers
        ddp_port: int = 12345,
    ) -> None:
        self.model = model
        self.datamodule = datamodule
        self.hospital_id = hospital_id
        self.num_nodes = num_nodes
        self.base_ddp_port = ddp_port
        self._local_metrics_logger = FLMetricsLogger(log_dir=f"fl_logs/coordinator_{hospital_id}")
        self._round = 0

    def fit(self, ins: FitIns) -> FitRes:
        self._round += 1
        logger.info("[Coordinator %s] Round %d – fit() called by Master", self.hospital_id, self._round)

        # Apply global weights from Master
        global_params = parameters_to_ndarrays(ins.parameters)
        download_bytes = compute_model_size_bytes(global_params)
        set_parameters(self.model, global_params)

        local_epochs = ins.config.get("local_epochs", CFG.max_epochs)

        # Local DDP Training
        t0 = time.perf_counter()
        train_loss = self._run_ddp_training(local_epochs)
        compute_time = time.perf_counter() - t0

        # Collect updated weights to send back
        updated_params = get_parameters(self.model)
        upload_bytes = compute_model_size_bytes(updated_params)
        n_examples = len(self.datamodule.train_df)

        self._local_metrics_logger.start_round(self._round)
        self._local_metrics_logger.log_client_stats(
            RoundClientStats(
                client_id=self.hospital_id,
                round_num=self._round,
                compute_time_s=compute_time,
                upload_bytes=upload_bytes,
                download_bytes=download_bytes,
                train_loss=train_loss,
                num_examples=n_examples,
            )
        )
        self._local_metrics_logger.end_round()
        self._local_metrics_logger.save(f"coordinator_{self.hospital_id}_metrics.json")

        metrics: Dict[str, float] = {
            "train_loss": train_loss,
            "compute_time_s": compute_time,
            "upload_bytes": float(upload_bytes),
            "download_bytes": float(download_bytes),
            "client_id_hash": float(abs(hash(self.hospital_id)) % 1_000_000),
        }

        logger.info(
            "[Coordinator %s] DDP fit done – loss=%.4f  compute=%.1fs  upload=%.2fMB",
            self.hospital_id, train_loss, compute_time, upload_bytes / 1e6,
        )

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=n_examples,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        logger.info("[Coordinator %s] Round %d – evaluate() called", self.hospital_id, self._round)

        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))

        # We can evaluate just on the coordinator to save DDP sync overhead during validation
        loss, metrics = self._local_evaluate()

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=loss,
            num_examples=len(self.datamodule.val_df),
            metrics=metrics,
        )

    def _run_ddp_training(self, local_epochs: int) -> float:
        """
        Orchestrates the DDP training. The PL Trainer will block here until
        the other workers also instantiate their Trainers and join the group.
        """
        self.model.train()

        current_port = self.base_ddp_port + self._round
        os.environ["MASTER_PORT"] = str(current_port)

        logger.info("Initializing DDP Trainer for %d nodes on port %d...", self.num_nodes, current_port)

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=-1,
            num_nodes=self.num_nodes,
            strategy="ddp",
            max_epochs=local_epochs,
            enable_checkpointing=False,
            logger=False,
            sync_batchnorm=True,
            use_distributed_sampler=True,
        )

        self.model.cfg.max_epochs = local_epochs
        trainer.fit(model=self.model, datamodule=self.datamodule)

        loss_tensor = trainer.callback_metrics.get("train/loss")
        train_loss = float(loss_tensor.item()) if loss_tensor is not None else float("nan")

        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("DDP process group destroyed. Port %d released.", current_port)

        return train_loss

    @torch.no_grad()
    def _local_evaluate(self) -> Tuple[float, Dict[str, float]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        val_loader = self.datamodule.fl_val_dataloader()
        loss_fn = self.model._get_loss_fn()

        total_loss = 0.0
        all_probs: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            logits = self.model(images)
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).cpu())
            all_targets.append(targets.cpu())

        mean_loss = total_loss / max(len(val_loader), 1)

        probs_cat = torch.cat(all_probs, dim=0)
        targets_cat = torch.cat(all_targets, dim=0).int()

        n = len(DISEASE_LABELS)
        auroc = MultilabelAUROC(num_labels=n, average="macro")(probs_cat, targets_cat).item()
        f1 = MultilabelF1Score(num_labels=n, average="macro")(probs_cat, targets_cat).item()
        recall = MultilabelRecall(num_labels=n, average="macro")(probs_cat, targets_cat).item()
        precision = MultilabelPrecision(num_labels=n, average="macro")(probs_cat, targets_cat).item()

        metrics = {
            "val_loss": mean_loss,
            "auroc_macro": auroc,
            "f1_macro": f1,
            "recall_macro": recall,
            "precision_macro": precision,
        }
        logger.info("[Coordinator %s] eval – loss=%.4f  AUROC=%.4f  F1=%.4f", self.hospital_id, mean_loss, auroc, f1)
        return mean_loss, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hospital Coordinator for Hierarchical FL")
    p.add_argument("--master_address", type=str, default="boston:8080")
    p.add_argument("--hospital_id", type=str, required=True)
    p.add_argument("--num_nodes", type=int, default=3)
    p.add_argument("--ddp_port", type=int, default=12345)
    p.add_argument("--seed", type=int, default=CFG.seed)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    os.environ["MASTER_ADDR"] = socket.gethostname()

    logger.info("Configured as DDP Master (Node 0) for %d nodes on port %d", args.num_nodes, args.ddp_port)

    datamodule = FederatedDataModule(cfg=CFG, hospital_id=args.hospital_id)
    datamodule.setup()

    model = ChestXrayClassifier(pos_weight=datamodule.pos_weight, cfg=CFG)

    client = HospitalCoordinatorClient(
        model=model,
        datamodule=datamodule,
        hospital_id=args.hospital_id,
        num_nodes=args.num_nodes,
        ddp_port=args.ddp_port,
    )

    logger.info("Connecting to Tier 1 Master Coordinator at %s …", args.master_address)
    fl.client.start_client(
        server_address=args.master_address,
        client=client,
    )


if __name__ == "__main__":
    main()