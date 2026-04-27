from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline.config import CFG, DISEASE_LABELS
from baseline.datamodule import ChestXrayDataModule
from baseline.model import ChestXrayClassifier
from federated.metrics_logger import (
    FLMetricsLogger,
    MIAEvaluator,
    RoundClientStats,
    compute_model_size_bytes,
    make_small_loader,
)

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


def weighted_average(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Weighted average of scalar metrics from multiple clients.
    Weight = number of examples for that client.
    """
    if not metrics_list:
        return {}

    total_examples = sum(n for n, _ in metrics_list)
    if total_examples == 0:
        return {}

    keys = metrics_list[0][1].keys()
    aggregated: Dict[str, float] = {}

    for key in keys:
        weighted_sum = sum(n * m.get(key, 0.0) for n, m in metrics_list)
        aggregated[key] = weighted_sum / total_examples

    return aggregated


class ChestXrayFedAvg(FedAvg):
    """
    Extends FedAvg with:
      - Weighted metric aggregation (AUROC, F1, Recall, Precision, Loss)
      - Per-round FL metadata logging (compute time, bandwidth)
      - Server-side MIA evaluation using the OOD (hospital_d) dataset
      - JSON metric persistence via FLMetricsLogger
    """

    def __init__(
        self,
        global_model: ChestXrayClassifier,
        ood_datamodule: Optional[ChestXrayDataModule],
        metrics_logger: FLMetricsLogger,
        device: torch.device,
        mia_member_loader=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.global_model = global_model
        self.ood_datamodule = ood_datamodule
        self.metrics_logger = metrics_logger
        self.device = device
        self.mia_member_loader = mia_member_loader  # small train-sample loader for MIA

    # fit_round

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if failures:
            logger.warning("Round %d: %d client(s) failed during fit.", server_round, len(failures))

        self.metrics_logger.start_round(server_round)

        # Log per-client metadata
        total_upload = 0
        total_download = 0
        max_compute = 0.0

        for proxy, fit_res in results:
            m = fit_res.metrics
            cid = str(m.get("client_id_hash", proxy.cid))
            compute = float(m.get("compute_time_s", 0.0))
            upload = int(m.get("upload_bytes", 0))
            download = int(m.get("download_bytes", 0))
            t_loss = float(m.get("train_loss", float("nan")))

            self.metrics_logger.log_client_stats(
                RoundClientStats(
                    client_id=cid,
                    round_num=server_round,
                    compute_time_s=compute,
                    upload_bytes=upload,
                    download_bytes=download,
                    train_loss=t_loss,
                    num_examples=fit_res.num_examples,
                )
            )
            total_upload += upload
            total_download += download
            max_compute = max(max_compute, compute)

        # Standard FedAvg aggregation
        t_agg_start = time.perf_counter()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        agg_time = time.perf_counter() - t_agg_start

        if aggregated_parameters is not None:
            set_parameters(self.global_model, parameters_to_ndarrays(aggregated_parameters))
            logger.info("Round %d: global model updated (FedAvg, agg_time=%.2fs)", server_round, agg_time)

        return aggregated_parameters, aggregated_metrics

    # evaluate_round

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Weighted average of client metrics
        client_metrics = [(r.num_examples, r.metrics) for _, r in results]
        agg = weighted_average(client_metrics)

        auroc = float(agg.get("auroc_macro", float("nan")))
        f1 = float(agg.get("f1_macro", float("nan")))
        recall = float(agg.get("recall_macro", float("nan")))
        precision = float(agg.get("precision_macro", float("nan")))
        val_loss = float(agg.get("val_loss", float("nan")))

        # Optionally run server-side OOD evaluation
        if self.ood_datamodule is not None:
            ood_metrics = self._server_side_ood_eval()
            logger.info(
                "Round %d | OOD AUROC=%.4f  OOD F1=%.4f",
                server_round,
                ood_metrics.get("auroc_macro", float("nan")),
                ood_metrics.get("f1_macro", float("nan")),
            )
            agg.update({f"ood_{k}": v for k, v in ood_metrics.items()})

        # Calc MIA
        mia_score = self._run_mia(server_round)

        # Log to FLMetricsLogger
        self.metrics_logger.log_aggregated_metrics(
            auroc=auroc, f1=f1, recall=recall, precision=precision,
            val_loss=val_loss,
        )
        self.metrics_logger.log_mia_score(mia_score)
        self.metrics_logger.end_round()
        self.metrics_logger.save()

        logger.info(
            "Round %d | AUROC=%.4f  F1=%.4f  Recall=%.4f  Prec=%.4f  Loss=%.4f  MIA=%.4f",
            server_round, auroc, f1, recall, precision, val_loss, mia_score,
        )

        return val_loss, {**agg, "mia_vulnerability": mia_score}

    # Server-side OOD evaluation

    @torch.no_grad()
    def _server_side_ood_eval(self) -> Dict[str, float]:
        ood_loader = self.ood_datamodule.ood_dataloader()
        if ood_loader is None:
            return {}

        self.global_model.to(self.device)
        self.global_model.eval()

        all_probs: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        for images, targets in ood_loader:
            images = images.to(self.device)
            logits = self.global_model(images)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_targets.append(targets.cpu())

        probs_cat = torch.cat(all_probs, dim=0)
        targets_cat = torch.cat(all_targets, dim=0).int()

        from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
        from torchmetrics.classification import MultilabelRecall, MultilabelPrecision

        n = len(DISEASE_LABELS)
        return {
            "auroc_macro": MultilabelAUROC(num_labels=n, average="macro")(probs_cat, targets_cat).item(),
            "f1_macro": MultilabelF1Score(num_labels=n, average="macro")(probs_cat, targets_cat).item(),
            "recall_macro": MultilabelRecall(num_labels=n, average="macro")(probs_cat, targets_cat).item(),
            "precision_macro": MultilabelPrecision(num_labels=n, average="macro")(probs_cat, targets_cat).item(),
        }
    
    def _run_mia(self, server_round: int) -> float:
        """
        Lightweight loss-based MIA.
        Member set   -> small sample of the OOD data as a proxy non-member set.
        Non-member   -> OOD data (hospital_d) that no client trained on.

        This is kinda an approximiation since we use global data instead of held out silo data
        """
        if self.mia_member_loader is None or self.ood_datamodule is None:
            logger.debug("Skipping MIA – loaders not configured.")
            return float("nan")

        ood_loader = self.ood_datamodule.ood_dataloader()
        if ood_loader is None:
            return float("nan")

        # Use a small OOD sample as non-members
        from baseline.dataset import ChestXrayDataset, build_eval_transforms
        ood_dataset = ChestXrayDataset(
            dataframe=self.ood_datamodule.ood_df,
            transforms=build_eval_transforms(CFG.image_size),
            image_col=CFG.image_col,
            image_path_col=CFG.image_path_col,
            label_col="label",
        )
        non_member_loader = make_small_loader(ood_dataset, n_samples=256, batch_size=64)

        evaluator = MIAEvaluator(
            model=self.global_model,
            device=self.device,
            loss_fn=torch.nn.BCEWithLogitsLoss(reduction="none"),
        )
        return evaluator.evaluate(
            member_loader=self.mia_member_loader,
            non_member_loader=non_member_loader,
        )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flower FL server for Chest X-ray classification")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--num_rounds", type=int, default=20)
    p.add_argument("--min_clients", type=int, default=3,
                   help="Minimum number of clients required per round")
    p.add_argument("--fraction_fit", type=float, default=1.0,
                   help="Fraction of clients used for training each round")
    p.add_argument("--fraction_evaluate", type=float, default=1.0)
    p.add_argument("--local_epochs", type=int, default=3,
                   help="Passed to clients via config_fn (informational only)")
    p.add_argument("--log_dir", type=str, default="fl_logs/server")
    p.add_argument("--seed", type=int, default=CFG.seed)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Server device: %s", device)

    logger.info("Initialising global model …")
    global_model = ChestXrayClassifier(pos_weight=None, cfg=CFG)
    global_model.to(device)
    initial_params = ndarrays_to_parameters(get_parameters(global_model))

    logger.info("Loading server-side datamodule for OOD eval and MIA probing only (no training) …")
    ood_dm: Optional[ChestXrayDataModule] = None
    mia_member_loader = None
    try:
        server_dm = ChestXrayDataModule(cfg=CFG)   # pools A+B+C for MIA member set
        server_dm.setup()

        ood_dm = server_dm
        # Build a small pseudo-member loader from any pool data we have server-side
        # (Approximation: use the val split of the pooled data as "members")
        from baseline.dataset import ChestXrayDataset, build_eval_transforms
        if server_dm.val_df is not None and len(server_dm.val_df) > 0:
                member_dataset = ChestXrayDataset(
                    dataframe=server_dm.val_df,
                    transforms=build_eval_transforms(CFG.image_size),
                    image_col=CFG.image_col,
                    image_path_col=CFG.image_path_col,
                    label_col="label",
                )
                mia_member_loader = make_small_loader(member_dataset, n_samples=256, batch_size=64)
                logger.info("MIA member loader ready (%d samples).", min(256, len(server_dm.val_df)))
                
    except Exception as exc:
        logger.warning("Could not set up server-side OOD datamodule: %s", exc)

    metrics_logger = FLMetricsLogger(log_dir=args.log_dir)

    strategy = ChestXrayFedAvg(
        global_model=global_model,
        ood_datamodule=ood_dm,
        metrics_logger=metrics_logger,
        device=device,
        mia_member_loader=mia_member_loader,
        # FedAvg base params
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    def client_config_fn(server_round: int) -> Dict:
        return {
            "local_epochs": args.local_epochs,
            "server_round": server_round,
        }

    server_config = fl.server.ServerConfig(num_rounds=args.num_rounds)

    logger.info(
        "Starting Flower server on 0.0.0.0:%d  rounds=%d  min_clients=%d",
        args.port, args.num_rounds, args.min_clients,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=server_config,
        strategy=strategy,
    )

    # Final save
    metrics_logger.save("fl_metrics_final.json")
    logger.info("Server finished. Metrics saved to %s", args.log_dir)


if __name__ == "__main__":
    main()
