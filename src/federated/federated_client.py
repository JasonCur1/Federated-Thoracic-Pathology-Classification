"""
Run this on each GPU machines
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pytorch_lightning as pl
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# Helpers
def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a flat list of numpy arrays into model state dict."""
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


# Flower Client

class ChestXrayFlowerClient(fl.client.Client):
    """
    Flower client that wraps a PyTorch Lightning training loop.

    Each round:
      1. Receives global weights from the server.
      2. Runs local_epochs of training on the local hospital data.
      3. Returns updated weights + metadata (compute time, loss, n_examples).
    """

    def __init__(
        self,
        model: ChestXrayClassifier,
        datamodule: FederatedDataModule,
        cfg=CFG,
        local_epochs: int = 3,
        client_id: str = "unknown",
    ) -> None:
        self.model = model
        self.datamodule = datamodule
        self.cfg = cfg
        self.local_epochs = local_epochs
        self.client_id = client_id
        self._local_metrics_logger = FLMetricsLogger(log_dir=f"fl_logs/client_{client_id}")
        self._round = 0

    def fit(self, ins: FitIns) -> FitRes:
        self._round += 1
        logger.info("[Client %s] Round %d – fit() called", self.client_id, self._round)

        # Apply global weights
        global_params = parameters_to_ndarrays(ins.parameters)
        download_bytes = compute_model_size_bytes(global_params)
        set_parameters(self.model, global_params)

        # Local training
        t0 = time.perf_counter()
        train_loss = self._local_train()
        compute_time = time.perf_counter() - t0

        # Collect updated weights
        updated_params = get_parameters(self.model)
        upload_bytes = compute_model_size_bytes(updated_params)

        n_examples = len(self.datamodule.train_df)

        # Log client-side stats
        self._local_metrics_logger.start_round(self._round)
        self._local_metrics_logger.log_client_stats(
            RoundClientStats(
                client_id=self.client_id,
                round_num=self._round,
                compute_time_s=compute_time,
                upload_bytes=upload_bytes,
                download_bytes=download_bytes,
                train_loss=train_loss,
                num_examples=n_examples,
            )
        )
        self._local_metrics_logger.end_round()
        self._local_metrics_logger.save(f"client_{self.client_id}_metrics.json")

        metrics: Dict[str, float] = {
            "train_loss": train_loss,
            "compute_time_s": compute_time,
            "upload_bytes": float(upload_bytes),
            "download_bytes": float(download_bytes),
            "client_id_hash": float(abs(hash(self.client_id)) % 1_000_000),
        }

        logger.info(
            "[Client %s] fit done – loss=%.4f  compute=%.1fs  upload=%.2fMB",
            self.client_id, train_loss, compute_time, upload_bytes / 1e6,
        )

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=n_examples,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        logger.info("[Client %s] Round %d – evaluate() called", self.client_id, self._round)

        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))

        loss, metrics = self._local_evaluate()

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=loss,
            num_examples=len(self.datamodule.val_df),
            metrics=metrics,
        )

    def _local_train(self) -> float:
        """
        Run local_epochs of training using a PyTorch loop
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        train_loader = self.datamodule.fl_train_dataloader()
        epoch_losses: List[float] = []

        for epoch in range(self.local_epochs):
            batch_losses: List[float] = []
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                # MixUp (augmentation. Only does it if flag is set in config)
                if self.cfg.use_mixup:
                    lam = float(np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha))
                    idx = torch.randperm(images.size(0), device=device)
                    images = lam * images + (1 - lam) * images[idx]
                    targets = lam * targets + (1 - lam) * targets[idx]

                logits = self.model(images)
                loss = self.model._get_loss_fn()(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_val)
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_loss = float(np.mean(batch_losses))
            epoch_losses.append(epoch_loss)
            logger.info(
                "[Client %s] Epoch %d/%d  loss=%.4f",
                self.client_id, epoch + 1, self.local_epochs, epoch_loss,
            )

        return float(np.mean(epoch_losses))

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

        from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
        from torchmetrics.classification import MultilabelRecall, MultilabelPrecision

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
        logger.info("[Client %s] eval – loss=%.4f  AUROC=%.4f  F1=%.4f", self.client_id, mean_loss, auroc, f1)
        return mean_loss, metrics

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flower FL client for Chest X-ray classification")
    p.add_argument("--server_address", type=str, default="shark:8080",
                   help="Server address in host:port format")
    p.add_argument("--local_epochs", type=int, default=3,
                   help="Number of local training epochs per FL round")
    p.add_argument("--hospital_id", type=str, default=None,
                   help="Override hospital ID (default: auto-detect from hostname)")
    p.add_argument("--seed", type=int, default=CFG.seed)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    if args.hospital_id:
        from baseline.datamodule import FederatedDataModule
        datamodule = FederatedDataModule(cfg=CFG, hospital_id=args.hospital_id)
    else:
        datamodule = FederatedDataModule.from_hostname(cfg=CFG)

    datamodule.setup()

    # Model
    model = ChestXrayClassifier(pos_weight=datamodule.pos_weight, cfg=CFG)

    import socket
    client_id = args.hospital_id or datamodule.hospital_id or socket.gethostname()

    client = ChestXrayFlowerClient(
        model=model,
        datamodule=datamodule,
        cfg=CFG,
        local_epochs=args.local_epochs,
        client_id=client_id,
    )

    logger.info("Connecting to server at %s …", args.server_address)
    fl.client.start_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
