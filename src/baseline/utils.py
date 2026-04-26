from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def report_class_distribution(
    dataframe: pd.DataFrame,
    label_cols: List[str],
    split_name: str = "dataset",
) -> Dict[str, Dict]:
    """
    Print a formatted table of per-class statistics and return a summary dict.
    """
    n = len(dataframe)
    summary = {}

    if _RICH_AVAILABLE:
        console = Console()
        table = Table(
            title=f"Class distribution – {split_name} (N={n:,})",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Disease", style="bold")
        table.add_column("Positives", justify="right")
        table.add_column("Prevalence %", justify="right")
        table.add_column("pos_weight", justify="right")

    for label in label_cols:
        n_pos = int(dataframe[label].sum())
        n_neg = n - n_pos
        prev = 100.0 * n_pos / max(n, 1)
        pw = n_neg / max(n_pos, 1)
        summary[label] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": prev,
            "pos_weight": pw,
        }
        if _RICH_AVAILABLE:
            table.add_row(label, f"{n_pos:,}", f"{prev:.2f}", f"{pw:.2f}")
        else:
            logger.info("  %-22s  pos=%6d  prev=%5.2f%%  pw=%.2f", label, n_pos, prev, pw)

    if _RICH_AVAILABLE:
        console.print(table)

    return summary


def report_pos_weights_tensor(pos_weight: torch.Tensor, label_cols: List[str]) -> None:
    """Pretty-print pos_weight tensor alongside label names."""
    if _RICH_AVAILABLE:
        console = Console()
        table = Table(title="pos_weight tensor", header_style="bold magenta")
        table.add_column("Class")
        table.add_column("pos_weight", justify="right")
        for name, w in zip(label_cols, pos_weight.tolist()):
            table.add_row(name, f"{w:.4f}")
        console.print(table)
    else:
        for name, w in zip(label_cols, pos_weight.tolist()):
            logger.info("pos_weight  %-22s  %.4f", name, w)


def validate_parquet_schema(
    df: pd.DataFrame,
    label_cols: List[str],
    image_col: Optional[str] = "image_bytes",
    image_path_col: str = "image_path",
) -> None:

    missing_labels = [l for l in label_cols if l not in df.columns]
    if missing_labels:
        raise ValueError(
            f"Missing label columns: {missing_labels}\n"
            f"Present columns: {df.columns.tolist()}"
        )

    has_bytes = image_col and image_col in df.columns
    has_path = image_path_col in df.columns

    if not has_bytes and not has_path:
        raise ValueError(
            f"DataFrame must contain either '{image_col}' (image bytes) or "
            f"'{image_path_col}' (file path) column.\n"
            f"Present columns: {df.columns.tolist()}"
        )

    for col in label_cols:
        if df[col].dtype not in [np.float32, np.float64, np.int32, np.int64, bool]:
            logger.warning(
                "Label column '%s' has unexpected dtype %s – will be coerced to float.",
                col,
                df[col].dtype,
            )

    logger.info("Parquet schema validation passed.  Columns: %d", len(df.columns))


# Metric Summariser

def summarise_test_results(
    id_metrics: Dict[str, float],
    ood_metrics: Dict[str, float],
    label_cols: List[str],
) -> None:
    """
    Print a comparison table of ID vs OOD test metrics and the generalisation gap.
    """
    if not _RICH_AVAILABLE:
        logger.info("ID test metrics:  %s", id_metrics)
        logger.info("OOD test metrics: %s", ood_metrics)
        return

    console = Console()
    table = Table(title="ID vs OOD Evaluation Summary", header_style="bold green")
    table.add_column("Metric")
    table.add_column("In-Distribution (A/B/C)", justify="right")
    table.add_column("OOD (hospital_d / ICU)", justify="right")
    table.add_column("Gap |ID - OOD|", justify="right")

    macro_keys = ["f1_macro", "recall_macro", "precision_macro", "auroc_macro"]
    for k in macro_keys:
        id_val = id_metrics.get(f"test_id/{k}", float("nan"))
        ood_val = ood_metrics.get(f"test_ood/{k}", float("nan"))
        gap = abs(id_val - ood_val)
        table.add_row(k, f"{id_val:.4f}", f"{ood_val:.4f}", f"{gap:.4f}")

    console.print(table)
