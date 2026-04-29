from __future__ import annotations

import glob
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


# Path setup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline.config import CFG, DISEASE_LABELS, ID_HOSPITALS, OOD_HOSPITAL
from baseline.dataset import ChestXrayDataset, build_eval_transforms
from baseline.model import ChestXrayClassifier

logger = logging.getLogger(__name__)



# Logging / JSON helpers


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return obj


def save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)

    logger.info("Saved results → %s", path)


# Model loading

def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> ChestXrayClassifier:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info("Loaded hierarchical checkpoint. Round=%s", checkpoint.get("round"))
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        logger.info("Loaded Lightning checkpoint.")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        logger.info("Loaded raw state_dict checkpoint.")
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    model = ChestXrayClassifier(cfg=CFG)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning("Missing keys while loading model: %s", missing[:10])
    if unexpected:
        logger.warning("Unexpected keys while loading model: %s", unexpected[:10])

    model.to(device)
    model.eval()

    return model


# Data loading

def load_hospital_parquets(hospital_ids: List[str], split_prefix: str) -> pd.DataFrame:
    frames = []

    for hospital_id in hospital_ids:
        patterns = []

        # Normal hospital layout:
        # data/hospital_a/**/train*.parquet
        # data/hospital_a/**/test*.parquet
        patterns.append(str(CFG.data_root / hospital_id / "**" / f"{split_prefix}*.parquet"))

        # Special OOD hospital_d layout used in this repo:
        # data/test_D-00000-of-00005.parquet
        # data/test_D-00001-of-00005.parquet
        if hospital_id == "hospital_d":
            patterns.append(str(CFG.data_root / f"{split_prefix}_D*.parquet"))

        # Also support possible root-level hospital letter layouts:
        # data/test_A*.parquet, data/train_A*.parquet, etc.
        hospital_suffix_map = {
            "hospital_a": "A",
            "hospital_b": "B",
            "hospital_c": "C",
            "hospital_d": "D",
        }

        if hospital_id in hospital_suffix_map:
            suffix = hospital_suffix_map[hospital_id]
            patterns.append(str(CFG.data_root / f"{split_prefix}_{suffix}*.parquet"))

        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(pattern, recursive=True))

        paths = sorted(set(paths))

        if not paths:
            logger.warning(
                "No parquet files found for hospital=%s split=%s patterns=%s",
                hospital_id,
                split_prefix,
                patterns,
            )
            continue

        logger.info("Loading %d file(s) from %s split=%s", len(paths), hospital_id, split_prefix)

        for path in paths:
            df = pd.read_parquet(path)
            df["_source_hospital"] = hospital_id
            df["_source_file"] = path
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No parquet files found for hospitals={hospital_ids}, split={split_prefix}, data_root={CFG.data_root}"
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded dataframe | hospitals=%s | split=%s | rows=%d",
        hospital_ids,
        split_prefix,
        len(combined),
    )
    return combined


def sample_dataframe(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def make_loader(
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = ChestXrayDataset(
        dataframe=df,
        transforms=build_eval_transforms(CFG.image_size),
        image_col=CFG.image_col,
        image_path_col=CFG.image_path_col,
        label_col="label",
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# Evaluation helpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def get_logits_targets_and_losses(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_logits = []
    all_targets = []
    all_losses = []

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).float()

        logits = model(images)

        per_sample_loss = loss_fn(logits, targets).mean(dim=1)

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_losses.append(per_sample_loss.detach().cpu())

    logits_np = torch.cat(all_logits, dim=0).numpy()
    targets_np = torch.cat(all_targets, dim=0).numpy().astype(int)
    losses_np = torch.cat(all_losses, dim=0).numpy()

    return logits_np, targets_np, losses_np


def compute_multilabel_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    preds = (probs >= threshold).astype(int)

    per_class_auroc = {}
    valid_aurocs = []

    for idx, label in enumerate(DISEASE_LABELS):
        y_true = targets[:, idx]
        y_score = probs[:, idx]

        if len(np.unique(y_true)) < 2:
            per_class_auroc[label] = None
        else:
            auc = float(roc_auc_score(y_true, y_score))
            per_class_auroc[label] = auc
            valid_aurocs.append(auc)

    auroc_macro = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")

    return {
        "num_samples": int(targets.shape[0]),
        "auroc_macro": auroc_macro,
        "f1_macro": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "per_class_auroc": per_class_auroc,
    }


def evaluate_model_on_dataframe(
    model: torch.nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    threshold: float = 0.5,
) -> Dict:
    loader = make_loader(df, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    logits, targets, losses = get_logits_targets_and_losses(model, loader, device)
    probs = sigmoid(logits)

    metrics = compute_multilabel_metrics(probs, targets, threshold=threshold)
    metrics["loss"] = float(np.mean(losses))
    metrics["loss_std"] = float(np.std(losses))

    return metrics


# MIA helper


def run_loss_threshold_mia(
    member_losses: np.ndarray,
    non_member_losses: np.ndarray,
    n_thresholds: int = 200,
) -> Dict:
    all_losses = np.concatenate([member_losses, non_member_losses])

    labels = np.concatenate(
        [
            np.ones(len(member_losses), dtype=int),
            np.zeros(len(non_member_losses), dtype=int),
        ]
    )

    thresholds = np.linspace(float(all_losses.min()), float(all_losses.max()), n_thresholds)

    best_bal_acc = 0.5
    best_threshold = None
    best_tpr = 0.0
    best_tnr = 0.0

    for threshold in thresholds:
        # Lower loss means the model is more likely to have seen the sample.
        preds = (all_losses < threshold).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()

        tpr = tp / max(len(member_losses), 1)
        tnr = tn / max(len(non_member_losses), 1)

        bal_acc = (tpr + tnr) / 2.0

        if bal_acc > best_bal_acc:
            best_bal_acc = float(bal_acc)
            best_threshold = float(threshold)
            best_tpr = float(tpr)
            best_tnr = float(tnr)

    try:
        # Negative loss because low loss indicates membership.
        attack_auc = float(roc_auc_score(labels, -all_losses))
    except ValueError:
        attack_auc = float("nan")

    return {
        "attack_type": "loss_threshold_membership_inference",
        "mia_vulnerability_balanced_accuracy": best_bal_acc,
        "attack_auc": attack_auc,
        "best_loss_threshold": best_threshold,
        "member_tpr": best_tpr,
        "non_member_tnr": best_tnr,
        "member_loss_mean": float(member_losses.mean()),
        "member_loss_std": float(member_losses.std()),
        "non_member_loss_mean": float(non_member_losses.mean()),
        "non_member_loss_std": float(non_member_losses.std()),
        "num_member_samples": int(len(member_losses)),
        "num_non_member_samples": int(len(non_member_losses)),
        "interpretation": "0.5 means random guessing; higher means more membership leakage.",
    }
