"""
Hospital Coordinator for DP Hierarchical FL.

This file replaces PyTorch Lightning DDP with direct torch.distributed training.

Role:
- Acts as a Flower client to the Tier-1 Master Coordinator.
- Acts as rank 0 in local hospital-level PyTorch Distributed training.
- Receives global FL weights from Flower.
- Broadcasts those weights to local DDP workers.
- Runs local DP-SGD training using Opacus DPDDP.
- Sends updated weights and epsilon back to the Tier-1 Master.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.config import CFG, DISEASE_LABELS
from baseline.datamodule import FederatedDataModule
from baseline.model import ChestXrayClassifier
from federated.metrics_logger import (
    FLMetricsLogger,
    RoundClientStats,
    compute_model_size_bytes,
)

from DP_hierarchial.dp_utils import (
    attach_dpddp_privacy_engine,
    broadcast_model_state,
    cleanup_process_group,
    get_learning_rate,
    get_parameters,
    get_weight_decay,
    make_opacus_compatible,
    seed_everything,
    set_parameters,
    setup_process_group,
    unwrap_private_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

logger = logging.getLogger(__name__)


class HospitalCoordinatorClient(fl.client.Client):
    """
    Flower client representing one hospital coordinator.

    This process is rank 0 for local hospital-level torch.distributed training.
    Other hospital machines should run DP_hierarchial/ddp_worker.py as ranks 1, 2, ...
    """

    def __init__(
        self,
        model: ChestXrayClassifier,
        datamodule: FederatedDataModule,
        hospital_id: str,
        num_nodes: int = 3,
        ddp_port: int = 12345,
        dp_noise_multiplier: float = 1.0,
        dp_max_grad_norm: float = 1.0,
        dp_delta: float = 1e-5,
        secure_rng: bool = False,
    ) -> None:
        self.model = model
        self.datamodule = datamodule
        self.hospital_id = hospital_id

        self.num_nodes = int(num_nodes)
        self.base_ddp_port = int(ddp_port)

        self.dp_noise_multiplier = float(dp_noise_multiplier)
        self.dp_max_grad_norm = float(dp_max_grad_norm)
        self.dp_delta = float(dp_delta)
        self.secure_rng = bool(secure_rng)

        self._round = 0

        self._local_metrics_logger = FLMetricsLogger(
            log_dir=f"fl_logs/DP_coordinator_{hospital_id}"
        )

        self.dp_log_dir = Path(f"fl_logs/DP_coordinator_{hospital_id}")
        self.dp_log_dir.mkdir(parents=True, exist_ok=True)

        self.dp_round_metrics: List[Dict[str, float]] = []

    def fit(self, ins: FitIns) -> FitRes:
        """
        Called by the Tier-1 master.

        Steps:
        1. Receive global model parameters.
        2. Load them into this hospital model.
        3. Start local torch.distributed DP training as rank 0.
        4. Return updated hospital model weights to the master.
        """
        self._round += 1

        logger.info(
            "[DP Coordinator %s] Round %d – fit() called by Master",
            self.hospital_id,
            self._round,
        )

        global_params = parameters_to_ndarrays(ins.parameters)
        download_bytes = compute_model_size_bytes(global_params)

        set_parameters(self.model, global_params)

        local_epochs = int(ins.config.get("local_epochs", CFG.max_epochs))

        t0 = time.perf_counter()

        train_loss, epsilon = self._run_distributed_dp_training(
            local_epochs=local_epochs
        )

        compute_time = time.perf_counter() - t0

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
        self._local_metrics_logger.save(
            f"coordinator_{self.hospital_id}_metrics_dp.json"
        )

        dp_record = {
            "round": int(self._round),
            "hospital_id": self.hospital_id,
            "epsilon": float(epsilon),
            "delta": float(self.dp_delta),
            "noise_multiplier": float(self.dp_noise_multiplier),
            "max_grad_norm": float(self.dp_max_grad_norm),
            "local_epochs": int(local_epochs),
            "train_loss": float(train_loss),
            "compute_time_s": float(compute_time),
            "num_examples": int(n_examples),
            "upload_bytes": int(upload_bytes),
            "download_bytes": int(download_bytes),
        }

        self.dp_round_metrics.append(dp_record)

        with open(self.dp_log_dir / "dp_privacy_metrics_dp.json", "w") as f:
            json.dump(self.dp_round_metrics, f, indent=2)

        metrics: Dict[str, float] = {
            "train_loss": float(train_loss),
            "compute_time_s": float(compute_time),
            "upload_bytes": float(upload_bytes),
            "download_bytes": float(download_bytes),
            "client_id_hash": float(abs(hash(self.hospital_id)) % 1_000_000),
            "epsilon": float(epsilon),
            "delta": float(self.dp_delta),
            "noise_multiplier": float(self.dp_noise_multiplier),
            "max_grad_norm": float(self.dp_max_grad_norm),
        }

        logger.info(
            "[DP Coordinator %s] Round %d done | loss=%.4f epsilon=%.4f delta=%g compute=%.1fs upload=%.2fMB",
            self.hospital_id,
            self._round,
            train_loss,
            epsilon,
            self.dp_delta,
            compute_time,
            upload_bytes / 1e6,
        )

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=n_examples,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Called by the Tier-1 master to evaluate this hospital model locally.
        """
        logger.info(
            "[DP Coordinator %s] Round %d – evaluate() called",
            self.hospital_id,
            self._round,
        )

        set_parameters(self.model, parameters_to_ndarrays(ins.parameters))

        loss, metrics = self._local_evaluate()

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss),
            num_examples=len(self.datamodule.val_df),
            metrics=metrics,
        )

    def _run_distributed_dp_training(self, local_epochs: int) -> Tuple[float, float]:
        """
        Runs local hospital-level DP distributed training.

        This process:
        - is rank 0
        - receives global weights from Flower
        - broadcasts global weights to worker ranks
        - trains using Opacus DPDDP
        - keeps the trained model state for returning to Flower
        """
        rank = 0
        world_size = self.num_nodes
        current_port = self.base_ddp_port + self._round
        master_addr = socket.gethostname()

        setup_process_group(
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=current_port,
        )

        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


            self.model.to(device)
            self.model.train()

            # Only rank 0 received global FL weights from Flower.
            # This sends those weights to all local DDP worker ranks.
            broadcast_model_state(self.model, src=0)

            train_loader = self.datamodule.fl_train_dataloader()
            loss_fn = self.model._get_loss_fn()

            lr = get_learning_rate(CFG, default=1e-4)
            weight_decay = get_weight_decay(CFG, default=0.0)

            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

            private_model, private_optimizer, private_loader, privacy_engine = (
                attach_dpddp_privacy_engine(
                    model=self.model,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    noise_multiplier=self.dp_noise_multiplier,
                    max_grad_norm=self.dp_max_grad_norm,
                    secure_mode=self.secure_rng,
                )
            )

            private_model.to(device)
            private_model.train()

            total_loss = 0.0
            total_steps = 0

            for epoch in range(local_epochs):
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

                mean_epoch_loss = epoch_loss / max(epoch_steps, 1)
                epsilon = float(
                    privacy_engine.get_epsilon(delta=self.dp_delta)
                )

                logger.info(
                    "[DP Coordinator %s] Round %d Epoch %d/%d | loss=%.4f epsilon=%.4f",
                    self.hospital_id,
                    self._round,
                    epoch + 1,
                    local_epochs,
                    mean_epoch_loss,
                    epsilon,
                )

            final_epsilon = float(
                privacy_engine.get_epsilon(delta=self.dp_delta)
            )

            mean_train_loss = total_loss / max(total_steps, 1)

            # Copy trained private model weights back into the normal model.
            # Flower will serialize self.model, not the wrapped private model.
            trained_base_model = unwrap_private_model(private_model)
            self.model.load_state_dict(
                trained_base_model.state_dict(),
                strict=True,
            )

            return float(mean_train_loss), float(final_epsilon)

        finally:
            cleanup_process_group()

    @torch.no_grad()
    def _local_evaluate(self) -> Tuple[float, Dict[str, float]]:
        """
        Local validation on this hospital's validation split.
        This does not use DP because privacy cost comes from training access.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.model.eval()

        val_loader = self.datamodule.fl_val_dataloader()
        loss_fn = self.model._get_loss_fn()

        total_loss = 0.0

        all_probs: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = self.model(images)
            loss = loss_fn(logits, targets)

            total_loss += float(loss.detach().cpu().item())

            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_targets.append(targets.detach().cpu())

        mean_loss = total_loss / max(len(val_loader), 1)

        probs_cat = torch.cat(all_probs, dim=0)
        targets_cat = torch.cat(all_targets, dim=0).int()

        n_labels = len(DISEASE_LABELS)

        metrics = {
            "val_loss": float(mean_loss),
            "auroc_macro": float(
                MultilabelAUROC(num_labels=n_labels, average="macro")(
                    probs_cat, targets_cat
                ).item()
            ),
            "f1_macro": float(
                MultilabelF1Score(num_labels=n_labels, average="macro")(
                    probs_cat, targets_cat
                ).item()
            ),
            "recall_macro": float(
                MultilabelRecall(num_labels=n_labels, average="macro")(
                    probs_cat, targets_cat
                ).item()
            ),
            "precision_macro": float(
                MultilabelPrecision(num_labels=n_labels, average="macro")(
                    probs_cat, targets_cat
                ).item()
            ),
        }

        logger.info(
            "[DP Coordinator %s] eval | loss=%.4f AUROC=%.4f F1=%.4f",
            self.hospital_id,
            mean_loss,
            metrics["auroc_macro"],
            metrics["f1_macro"],
        )

        return float(mean_loss), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DP Hospital Coordinator using torch.distributed + Opacus"
    )

    parser.add_argument(
        "--master_address",
        type=str,
        default="boston:8080",
        help="Flower Tier-1 master address, for example boston:8080",
    )

    parser.add_argument(
        "--hospital_id",
        type=str,
        required=True,
        help="Hospital/client id used by FederatedDataModule",
    )

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=3,
        help="Total local DDP processes for this hospital, including coordinator rank 0",
    )

    parser.add_argument(
        "--ddp_port",
        type=int,
        default=12345,
        help="Base torch.distributed port. Actual port is ddp_port + FL round.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CFG.seed,
    )

    parser.add_argument(
        "--dp_noise_multiplier",
        type=float,
        default=1.0,
        help="Opacus DP noise multiplier. Higher value gives stronger privacy but may reduce accuracy.",
    )

    parser.add_argument(
        "--dp_max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum per-sample gradient norm for DP-SGD clipping.",
    )

    parser.add_argument(
        "--dp_delta",
        type=float,
        default=1e-5,
        help="Delta value for epsilon accounting.",
    )

    parser.add_argument(
        "--secure_rng",
        action="store_true",
        help="Use secure RNG in Opacus. Slower, but stronger for real privacy guarantees.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)

    logger.info(
        "Configured as DP DDP rank 0 coordinator | hospital=%s num_nodes=%d base_port=%d",
        args.hospital_id,
        args.num_nodes,
        args.ddp_port,
    )

    datamodule = FederatedDataModule(
        cfg=CFG,
        hospital_id=args.hospital_id,
    )
    datamodule.setup()

    model = ChestXrayClassifier(
        pos_weight=datamodule.pos_weight,
        cfg=CFG,
    )
    model = make_opacus_compatible(model)

    # Important:
    # Master, coordinator, and workers should all apply this same compatibility fix.
    # Otherwise state_dict keys/shapes may not match.
    model = make_opacus_compatible(model)

    client = HospitalCoordinatorClient(
        model=model,
        datamodule=datamodule,
        hospital_id=args.hospital_id,
        num_nodes=args.num_nodes,
        ddp_port=args.ddp_port,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_max_grad_norm=args.dp_max_grad_norm,
        dp_delta=args.dp_delta,
        secure_rng=args.secure_rng,
    )

    logger.info(
        "Connecting DP hospital coordinator %s to Tier-1 Master at %s",
        args.hospital_id,
        args.master_address,
    )

    fl.client.start_client(
        server_address=args.master_address,
        client=client,
    )


if __name__ == "__main__":
    main()