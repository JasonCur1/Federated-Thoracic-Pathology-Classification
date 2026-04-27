"""
Metrics captured
----------------
* Communication overhead  – bytes transferred up (client→server) and down (server→client)
* Local compute time       – wall-clock seconds spent on local training per client per round
* Bandwidth efficiency     – ΔAUROC per MB transferred
* Temporal overhead        – full round latency (max client compute + aggregation time)
* Privacy / MIA            – lightweight membership-inference attack vulnerability score
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# Data classes

@dataclass
class RoundClientStats:
    client_id: str
    round_num: int
    compute_time_s: float = 0.0
    upload_bytes: int = 0    # client -> server (updated weights)
    download_bytes: int = 0  # server -> client (global weights)
    train_loss: float = float("nan")
    num_examples: int = 0


@dataclass
class RoundStats:
    round_num: int
    wall_time_s: float = 0.0          # total round latency
    aggregation_time_s: float = 0.0
    max_client_compute_s: float = 0.0
    total_upload_mb: float = 0.0
    total_download_mb: float = 0.0
    auroc_macro: float = float("nan")
    f1_macro: float = float("nan")
    recall_macro: float = float("nan")
    precision_macro: float = float("nan")
    val_loss: float = float("nan")
    mia_vulnerability: float = float("nan")   # 0 = safe, 1 = fully leaking
    bandwidth_efficiency: float = float("nan")
    client_stats: List[RoundClientStats] = field(default_factory=list)


class FLMetricsLogger:
    def __init__(self, log_dir: str | Path = "fl_logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[RoundStats] = []
        self._current: Optional[RoundStats] = None
        self._round_start: float = 0.0
        self._prev_auroc: float = 0.0

    # Round lifecycle

    def start_round(self, round_num: int) -> None:
        self._current = RoundStats(round_num=round_num)
        self._round_start = time.perf_counter()
        logger.info("[FL] ── Round %d started ──", round_num)

    def end_round(self) -> RoundStats:
        assert self._current is not None, "Call start_round() first."
        self._current.wall_time_s = time.perf_counter() - self._round_start
        self._current.max_client_compute_s = max(
            (c.compute_time_s for c in self._current.client_stats), default=0.0
        )
        self._current.total_upload_mb = sum(
            c.upload_bytes for c in self._current.client_stats
        ) / 1e6
        self._current.total_download_mb = sum(
            c.download_bytes for c in self._current.client_stats
        ) / 1e6

        total_mb = self._current.total_upload_mb + self._current.total_download_mb
        auroc = self._current.auroc_macro
        if total_mb > 0 and not np.isnan(auroc):
            delta = auroc - self._prev_auroc
            self._current.bandwidth_efficiency = delta / total_mb
            self._prev_auroc = auroc

        self.history.append(self._current)
        self._log_round_summary(self._current)
        completed = self._current
        self._current = None
        return completed

    # Per-client logging

    def log_client_stats(self, stats: RoundClientStats) -> None:
        assert self._current is not None, "Call start_round() first."
        self._current.client_stats.append(stats)
        logger.info(
            "[FL] Client %s | compute=%.1fs  upload=%.2fMB  loss=%.4f",
            stats.client_id,
            stats.compute_time_s,
            stats.upload_bytes / 1e6,
            stats.train_loss,
        )

    # Aggregated metrics

    def log_aggregated_metrics(
        self,
        auroc: float,
        f1: float,
        recall: float,
        precision: float,
        val_loss: float,
        aggregation_time_s: float = 0.0,
    ) -> None:
        assert self._current is not None
        self._current.auroc_macro = auroc
        self._current.f1_macro = f1
        self._current.recall_macro = recall
        self._current.precision_macro = precision
        self._current.val_loss = val_loss
        self._current.aggregation_time_s = aggregation_time_s
        logger.info(
            "[FL] Aggregated — AUROC=%.4f  F1=%.4f  Recall=%.4f  Prec=%.4f  Loss=%.4f",
            auroc, f1, recall, precision, val_loss,
        )

    def log_mia_score(self, vulnerability: float) -> None:
        assert self._current is not None
        self._current.mia_vulnerability = vulnerability
        logger.info("[FL] MIA vulnerability = %.4f", vulnerability)

    def save(self, filename: str = "fl_metrics.json") -> Path:
        out = self.log_dir / filename
        data = [asdict(r) for r in self.history]
        out.write_text(json.dumps(data, indent=2))
        logger.info("[FL] Metrics saved → %s", out)
        return out

    def summary_dataframe(self):
        rows = []
        for r in self.history:
            rows.append({
                "round": r.round_num,
                "auroc_macro": r.auroc_macro,
                "f1_macro": r.f1_macro,
                "recall_macro": r.recall_macro,
                "precision_macro": r.precision_macro,
                "val_loss": r.val_loss,
                "mia_vulnerability": r.mia_vulnerability,
                "bandwidth_efficiency": r.bandwidth_efficiency,
                "wall_time_s": r.wall_time_s,
                "total_upload_mb": r.total_upload_mb,
                "total_download_mb": r.total_download_mb,
            })
        return pd.DataFrame(rows)

    def _log_round_summary(self, r: RoundStats) -> None:
        logger.info(
            "[FL] Round %d done | wall=%.1fs  AUROC=%.4f  MIA=%.4f  "
            "BW_eff=%.6f ΔAUROC/MB  upload=%.2f MB  download=%.2f MB",
            r.round_num, r.wall_time_s, r.auroc_macro, r.mia_vulnerability,
            r.bandwidth_efficiency if not np.isnan(r.bandwidth_efficiency) else 0.0,
            r.total_upload_mb, r.total_download_mb,
        )


class MIAEvaluator:
    """
    Label-only / loss-based membership inference attack.

    Method (Yeom et al., 2018 / Salem et al., 2019):
      1. Compute per-sample loss on a held-out member set (train subset) and a non-member set (validation/OOD subset).
      2. Threshold on loss: samples with loss < threshold are classified as members.
      3. Vulnerability = balanced accuracy of the attack classifier.

    A score of 0.5 means the attack is no better than random.
    A score of 1.0 means perfect separation (bad – model has memorised training data).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss(reduction="none")

    @torch.no_grad()
    def _per_sample_loss(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        losses: List[float] = []
        for batch in loader:
            images, targets = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(images)
            # mean over classes, keep batch dim
            loss = self.loss_fn(logits, targets).mean(dim=1)
            losses.extend(loss.cpu().numpy().tolist())
        return np.array(losses)

    def evaluate(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
        n_thresholds: int = 200,
    ) -> float:
        member_losses = self._per_sample_loss(member_loader)
        non_member_losses = self._per_sample_loss(non_member_loader)

        member_labels = np.ones(len(member_losses), dtype=int)
        non_member_labels = np.zeros(len(non_member_losses), dtype=int)

        all_losses = np.concatenate([member_losses, non_member_losses])
        all_labels = np.concatenate([member_labels, non_member_labels])

        # Sweep thresholds. predict member if loss < threshold
        best_balanced_acc = 0.5
        thresholds = np.linspace(all_losses.min(), all_losses.max(), n_thresholds)
        for t in thresholds:
            preds = (all_losses < t).astype(int)
            tp = ((preds == 1) & (all_labels == 1)).sum()
            tn = ((preds == 0) & (all_labels == 0)).sum()
            tpr = tp / max(len(member_labels), 1)
            tnr = tn / max(len(non_member_labels), 1)
            bal_acc = (tpr + tnr) / 2.0
            if bal_acc > best_balanced_acc:
                best_balanced_acc = bal_acc

        logger.info(
            "[MIA] members=%d  non_members=%d  vulnerability=%.4f",
            len(member_losses), len(non_member_losses), best_balanced_acc,
        )
        return float(best_balanced_acc)


def compute_model_size_bytes(parameters: List[np.ndarray]) -> int:
    return sum(p.nbytes for p in parameters)


def make_small_loader(
    dataset,
    n_samples: int = 256,
    batch_size: int = 64,
    seed: int = 42,
) -> DataLoader:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
