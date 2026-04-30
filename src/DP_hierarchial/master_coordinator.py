from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    EvaluateRes,
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
from DP_hierarchial.dp_utils import (
    make_opacus_compatible,
    get_parameters,
    set_parameters,
)
from federated.metrics_logger import (
    FLMetricsLogger,
    MIAEvaluator,
    RoundClientStats,
    make_small_loader,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logger = logging.getLogger(__name__)


# def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
#     return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


# def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
#     model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics_list:
        return {}
    total_examples = sum(n for n, _ in metrics_list)
    if total_examples == 0:
        return {}
    keys = metrics_list[0][1].keys()
    return {key: sum(n * m.get(key, 0.0) for n, m in metrics_list) / total_examples for key in keys}


class DPMasterFedAvg(FedAvg):
    def __init__(
        self,
        global_model: ChestXrayClassifier,
        ood_datamodule: Optional[ChestXrayDataModule],
        metrics_logger: FLMetricsLogger,
        device: torch.device,
        mia_member_loader=None,
        save_dir: Optional[Union[str, Path]] = None,
        save_every_round: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.global_model = global_model
        self.ood_datamodule = ood_datamodule
        self.metrics_logger = metrics_logger
        self.device = device
        self.mia_member_loader = mia_member_loader
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.save_every_round = save_every_round
        self.dp_round_metrics: List[Dict[str, float]] = []
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_global_model(self, server_round: int, name: Optional[str] = None) -> None:
        if self.save_dir is None:
            return
        filename = name or f"global_round_{server_round}.pt"
        path = self.save_dir / filename
        payload = {
            "round": server_round,
            "model_state_dict": self.global_model.state_dict(),
            "disease_labels": DISEASE_LABELS,
            "cfg": {"image_size": CFG.image_size, "backbone": CFG.backbone, "pretrained": CFG.pretrained},
        }
        torch.save(payload, path)
        torch.save(payload, self.save_dir / "global_latest.pt")
        logger.info("Saved global model checkpoint: %s", path)

    def _save_dp_metrics(self) -> None:
        dp_path = Path(self.metrics_logger.log_dir) / "dp_privacy_metrics.json"
        dp_path.parent.mkdir(parents=True, exist_ok=True)
        with dp_path.open("w") as f:
            json.dump(self.dp_round_metrics, f, indent=2)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            logger.warning("Round %d: %d coordinator(s) failed during fit.", server_round, len(failures))

        self.metrics_logger.start_round(server_round)

        for proxy, fit_res in results:
            m = fit_res.metrics
            cid = str(m.get("client_id_hash", proxy.cid))
            compute = float(m.get("compute_time_s", 0.0))
            upload = int(m.get("upload_bytes", 0))
            download = int(m.get("download_bytes", 0))
            t_loss = float(m.get("train_loss", float("nan")))

            epsilon = float(m.get("epsilon", float("nan")))
            delta = float(m.get("delta", float("nan")))
            noise_multiplier = float(m.get("noise_multiplier", float("nan")))
            max_grad_norm = float(m.get("max_grad_norm", float("nan")))

            if not np.isnan(epsilon):
                self.dp_round_metrics.append({
                    "round": server_round,
                    "client_id": cid,
                    "epsilon": epsilon,
                    "delta": delta,
                    "noise_multiplier": noise_multiplier,
                    "max_grad_norm": max_grad_norm,
                    "train_loss": t_loss,
                    "num_examples": fit_res.num_examples,
                })
                logger.info(
                    "Round %d | DP client=%s epsilon=%.4f delta=%.1e noise=%.3f clip=%.3f",
                    server_round, cid, epsilon, delta, noise_multiplier, max_grad_norm,
                )

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

        t_agg_start = time.perf_counter()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        agg_time = time.perf_counter() - t_agg_start

        if aggregated_parameters is not None:
            set_parameters(self.global_model, parameters_to_ndarrays(aggregated_parameters))
            logger.info("Round %d: global model updated (FedAvg, agg_time=%.2fs)", server_round, agg_time)
            if self.save_every_round:
                self._save_global_model(server_round)

        if self.dp_round_metrics:
            self._save_dp_metrics()

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        client_metrics = [(r.num_examples, r.metrics) for _, r in results]
        agg = weighted_average(client_metrics)

        auroc = float(agg.get("auroc_macro", float("nan")))
        f1 = float(agg.get("f1_macro", float("nan")))
        recall = float(agg.get("recall_macro", float("nan")))
        precision = float(agg.get("precision_macro", float("nan")))
        val_loss = float(agg.get("val_loss", float("nan")))

        if self.ood_datamodule is not None:
            ood_metrics = self._server_side_ood_eval()
            agg.update({f"ood_{k}": v for k, v in ood_metrics.items()})

        mia_score = self._run_mia(server_round)
        if not np.isnan(mia_score):
            agg["mia_vulnerability"] = mia_score

        self.metrics_logger.log_aggregated_metrics(auroc=auroc, f1=f1, recall=recall, precision=precision, val_loss=val_loss)
        if not np.isnan(mia_score):
            self.metrics_logger.log_mia_score(mia_score)
        self.metrics_logger.end_round()
        self.metrics_logger.save()

        logger.info("Round %d | AUROC=%.4f F1=%.4f Recall=%.4f Prec=%.4f Loss=%.4f", server_round, auroc, f1, recall, precision, val_loss)
        return val_loss, agg

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
        from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelRecall, MultilabelPrecision
        n = len(DISEASE_LABELS)
        return {
            "auroc_macro": MultilabelAUROC(num_labels=n, average="macro")(probs_cat, targets_cat).item(),
            "f1_macro": MultilabelF1Score(num_labels=n, average="macro")(probs_cat, targets_cat).item(),
            "recall_macro": MultilabelRecall(num_labels=n, average="macro")(probs_cat, targets_cat).item(),
            "precision_macro": MultilabelPrecision(num_labels=n, average="macro")(probs_cat, targets_cat).item(),
        }

    def _run_mia(self, server_round: int) -> float:
        if self.mia_member_loader is None or self.ood_datamodule is None:
            return float("nan")
        ood_loader = self.ood_datamodule.ood_dataloader()
        if ood_loader is None:
            return float("nan")
        from baseline.dataset import ChestXrayDataset, build_eval_transforms
        ood_dataset = ChestXrayDataset(
            dataframe=self.ood_datamodule.ood_df,
            transforms=build_eval_transforms(CFG.image_size),
            image_col=CFG.image_col,
            image_path_col=CFG.image_path_col,
            label_col="label",
        )
        non_member_loader = make_small_loader(ood_dataset, n_samples=256, batch_size=64)
        evaluator = MIAEvaluator(model=self.global_model, device=self.device, loss_fn=torch.nn.BCEWithLogitsLoss(reduction="none"))
        return evaluator.evaluate(member_loader=self.mia_member_loader, non_member_loader=non_member_loader)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--num_rounds", type=int, default=20)
    p.add_argument("--min_clients", type=int, default=3)
    p.add_argument("--fraction_fit", type=float, default=1.0)
    p.add_argument("--fraction_evaluate", type=float, default=1.0)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="fl_logs/DP_master")
    p.add_argument("--save_dir", type=str, default="fl_logs/DP_master/checkpoints")
    p.add_argument("--enable_server_eval", action="store_true")
    p.add_argument("--no_save_every_round", action="store_true")
    p.add_argument("--seed", type=int, default=CFG.seed)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DP Master Coordinator device: %s", device)

    global_model = ChestXrayClassifier(pos_weight=None, cfg=CFG)
    global_model = make_opacus_compatible(global_model)
    global_model.to(device)
    initial_params = ndarrays_to_parameters(get_parameters(global_model))

    logger.info("DP Master Flower tensor count: %d", len(get_parameters(global_model)))
    logger.info("DP Master full state_dict tensor count: %d", len(global_model.state_dict()))
    logger.info(
    "DP Master trainable parameter count: %d",
    len([p for _, p in global_model.named_parameters() if p.requires_grad]),
    )

    ood_dm: Optional[ChestXrayDataModule] = None
    mia_member_loader = None
    if args.enable_server_eval:
        try:
            server_dm = ChestXrayDataModule(cfg=CFG)
            server_dm.setup()
            ood_dm = server_dm
            from baseline.dataset import ChestXrayDataset, build_eval_transforms
            if server_dm.train_df is not None and len(server_dm.train_df) > 0:
                member_dataset = ChestXrayDataset(
                    dataframe=server_dm.train_df,
                    transforms=build_eval_transforms(CFG.image_size),
                    image_col=CFG.image_col,
                    image_path_col=CFG.image_path_col,
                    label_col="label",
                )
                mia_member_loader = make_small_loader(member_dataset, n_samples=256, batch_size=64)
        except Exception as exc:
            logger.warning("Could not set up server-side OOD/MIA datamodule: %s", exc)

    metrics_logger = FLMetricsLogger(log_dir=args.log_dir)
    strategy = DPMasterFedAvg(
        global_model=global_model,
        ood_datamodule=ood_dm,
        metrics_logger=metrics_logger,
        device=device,
        mia_member_loader=mia_member_loader,
        save_dir=args.save_dir,
        save_every_round=not args.no_save_every_round,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda server_round: {"local_epochs": args.local_epochs, "server_round": server_round},
        on_evaluate_config_fn=lambda server_round: {"server_round": server_round},
    )

    strategy._save_global_model(0, name="global_initial.pt")
    fl.server.start_server(server_address=f"0.0.0.0:{args.port}", config=fl.server.ServerConfig(num_rounds=args.num_rounds), strategy=strategy)
    strategy._save_global_model(args.num_rounds, name="global_final.pt")
    metrics_logger.save("master_metrics_final.json")
    if strategy.dp_round_metrics:
        strategy._save_dp_metrics()
    logger.info("DP Master Coordinator finished. Metrics saved to %s", args.log_dir)


if __name__ == "__main__":
    main()
