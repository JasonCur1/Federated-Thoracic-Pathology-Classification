"""
Responsibilities
1. Glob-search root/data/hospital_{a,b,c}/ for Parquet files, parse them,
   and pool all rows into a single DataFrame.
2. Split the pool into train / val / in-dist-test sets (multilabel-stratified).
3. Load hospital_d as a separate OOD test set.
4. Compute pos_weight per-class from the training split only.
5. Build WeightedRandomSampler for the training DataLoader.
6. Expose train_dataloader, val_dataloader, test_dataloader
   (in-dist), and ood_dataloader (hospital_d).
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from baseline.config import CFG, DISEASE_LABELS, ID_HOSPITALS, OOD_HOSPITAL
from baseline.dataset import ChestXrayDataset, build_eval_transforms, build_train_transforms

logger = logging.getLogger(__name__)


# Weight calculation helpers
def compute_pos_weights(label_matrix: np.ndarray) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.

    Formula: pos_weight[c] = (N - pos_count[c]) / pos_count[c]
    Clipped at 50 to prevent extreme weights on very rare classes.
    """
    n_samples = label_matrix.shape[0]
    pos_count = label_matrix.sum(axis=0).clip(min=1)
    neg_count = n_samples - pos_count
    pos_weight = neg_count / pos_count
    pos_weight = np.clip(pos_weight, a_min=None, a_max=50.0)
    logger.info("pos_weight per class:")
    for name, w in zip(DISEASE_LABELS, pos_weight):
        logger.info("  %-22s %.2f", name, w)
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_sample_weights(
    label_matrix: np.ndarray,
    agg: str = "max",
) -> np.ndarray:
    """
    Derive a scalar weight per sample for WeightedRandomSampler.

    * ``"max"``  – sample weight = highest class weight among its positive labels.
    * ``"sum"``  – sample weight = sum of positive class weights.

    Negative-only samples receive weight = 1.0
    """
    n_samples, n_classes = label_matrix.shape
    pos_count = label_matrix.sum(axis=0).clip(min=1)
    class_weights = n_samples / (n_classes * pos_count)

    weighted = label_matrix * class_weights[np.newaxis, :]

    if agg == "max":
        sample_weights = weighted.max(axis=1)
    elif agg == "sum":
        sample_weights = weighted.sum(axis=1)
    else:
        raise ValueError(f"Unknown agg strategy: {agg!r}")

    sample_weights = np.where(sample_weights == 0, 1.0, sample_weights)
    return sample_weights.astype(np.float64)


# DataModuel

class ChestXrayDataModule(pl.LightningDataModule):
    """
    Centralized DataModule that pools Hospital A, B, C files.
    """

    def __init__(self, cfg=CFG) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)

        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.ood_df: Optional[pd.DataFrame] = None
        self.pos_weight: Optional[torch.Tensor] = None

    # Parquet loading

    def _load_hospital_parquets(self, hospital_ids: List[str]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for h_id in hospital_ids:
            pattern = str(self.cfg.data_root / h_id / "**" / "*.parquet")
            paths = glob.glob(pattern, recursive=True)
            if not paths:
                logger.warning("No Parquet files found for %s at %s", h_id, pattern)
                continue
            logger.info("Loading %d Parquet file(s) from %s …", len(paths), h_id)
            for p in paths:
                df = pd.read_parquet(p)
                df["_source_hospital"] = h_id
                frames.append(df)

        if not frames:
            raise FileNotFoundError(
                "No Parquet files were found. Check that "
                f"{self.cfg.data_root} is correctly populated."
            )
        combined = pd.concat(frames, ignore_index=True)
        logger.info("Total rows loaded: %d", len(combined))
        return combined

    def _validate_columns(self, df: pd.DataFrame) -> None:
        if "label" not in df.columns:
            raise ValueError(f"Parquet missing 'label' column. Found: {df.columns.tolist()}")

        if self.cfg.image_col not in df.columns:
            raise ValueError(f"Parquet missing '{self.cfg.image_col}' column. Found: {df.columns.tolist()}")

        has_bytes = self.cfg.image_col and self.cfg.image_col in df.columns
        has_path = self.cfg.image_path_col in df.columns
        if not has_bytes and not has_path:
            raise ValueError(
                f"Parquet files must contain either '{self.cfg.image_col}' "
                f"(image bytes) or '{self.cfg.image_path_col}' (file path).\n"
                f"Available columns: {df.columns.tolist()}"
            )

    # Splitting utility
    @staticmethod
    def _stratified_split(
        df: pd.DataFrame,
        label_matrix: np.ndarray,
        val_frac: float,
        test_frac: float,
        seed: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Two-stage stratified split -> (train, val, test).
        """
        indices = np.arange(len(df))

        try:
            from skmultilearn.model_selection import iterative_train_test_split

            X = indices.reshape(-1, 1)
            y = label_matrix  # (N, 14)

            X_trainval, y_trainval, X_test, _ = iterative_train_test_split(
                X, y, test_size=test_frac
            )

            relative_val = val_frac / (1.0 - test_frac)
            X_train, _, X_val, _ = iterative_train_test_split(
                X_trainval, y_trainval, test_size=relative_val
            )

            train_idx = X_train.flatten()
            val_idx = X_val.flatten()
            test_idx = X_test.flatten()

            logger.info(
                "Iterative multilabel split – train: %d  val: %d  test: %d",
                len(train_idx), len(val_idx), len(test_idx),
            )

            # Log per-class positive counts
            for c, name in enumerate(DISEASE_LABELS):
                n_tr = label_matrix[train_idx, c].sum()
                n_v = label_matrix[val_idx, c].sum()
                n_te = label_matrix[test_idx, c].sum()
                logger.debug(
                    "  %-22s  train=%d  val=%d  test=%d", name, int(n_tr), int(n_v), int(n_te)
                )

            return (
                df.iloc[train_idx].reset_index(drop=True),
                df.iloc[val_idx].reset_index(drop=True),
                df.iloc[test_idx].reset_index(drop=True),
            )

        except ImportError:
            logger.warning(
                "scikit-multilearn not installed — falling back to binary "
                "stratification on 'any disease' flag. "
                "Install with: pip install scikit-multilearn"
            )

        # Fallback: binary stratification
        df = df.copy()

        def _has_disease(entry) -> int:
            if not isinstance(entry, (list, tuple, np.ndarray)):
                return 0
            return int(any(item in DISEASE_LABELS for item in entry))

        df["_any"] = df["label"].apply(_has_disease)

        trainval, test = train_test_split(
            df,
            test_size=test_frac,
            stratify=df["_any"],
            random_state=seed,
        )

        relative_val = val_frac / (1.0 - test_frac)
        train, val = train_test_split(
            trainval,
            test_size=relative_val,
            stratify=trainval["_any"],
            random_state=seed,
        )

        train = train.drop(columns=["_any"])
        val = val.drop(columns=["_any"])
        test = test.drop(columns=["_any"])

        logger.info(
            "Binary split sizes – train: %d  val: %d  test: %d",
            len(train), len(val), len(test),
        )
        return train, val, test


    def prepare_data(self) -> None:
        """Called only on rank-0 in DDP. Validates paths exist."""
        for h_id in ID_HOSPITALS:
            h_path = self.cfg.data_root / h_id
            if not h_path.exists():
                logger.warning("Hospital directory not found: %s", h_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called on every DDP rank after prepare_data.

        Loads data, decodes label matrix, runs multilabel-stratified split,
        and builds pos_weight + sample weights.
        """
        import albumentations as A

        _noop = A.Compose([A.NoOp()])

        id_df = self._load_hospital_parquets(ID_HOSPITALS)
        self._validate_columns(id_df)

        # Decode once here so we can pass it to the multilabel stratifier.
        _full_ds_tmp = ChestXrayDataset(
            dataframe=id_df,
            transforms=_noop,
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        full_label_matrix = _full_ds_tmp.label_matrix  # (N_total, 14)

        # Multilabel-stratified split
        self.train_df, self.val_df, self.test_df = self._stratified_split(
            id_df,
            label_matrix=full_label_matrix,
            val_frac=self.cfg.val_fraction,
            test_frac=self.cfg.test_fraction,
            seed=self.cfg.seed,
        )

        # Compute class weights from TRAIN split only
        # Extract the training rows from the full label matrix using positional
        # indices recovered from the reset DataFrames.
        # Re-decode only the train split
        _train_ds_tmp = ChestXrayDataset(
            dataframe=self.train_df,
            transforms=_noop,
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        train_label_matrix = _train_ds_tmp.label_matrix  # (N_train, 14)
        self.pos_weight = compute_pos_weights(train_label_matrix)
        self._train_label_matrix = train_label_matrix

        logger.info(
            "Dataset ready — train: %d  val: %d  test: %d",
            len(self.train_df), len(self.val_df), len(self.test_df),
        )

        try:
            self.ood_df = self._load_hospital_parquets([OOD_HOSPITAL])
            self._validate_columns(self.ood_df)
            logger.info("OOD set (hospital_d) rows: %d", len(self.ood_df))
        except FileNotFoundError:
            logger.warning("hospital_d not found – OOD evaluation will be skipped.")
            self.ood_df = None

    # DataLoaders

    def train_dataloader(self) -> DataLoader:
        dataset = ChestXrayDataset(
            dataframe=self.train_df,
            transforms=build_train_transforms(self.cfg.image_size),
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )

        sampler = None
        shuffle = True
        if self.cfg.use_weighted_sampler:
            sample_weights = compute_sample_weights(
                self._train_label_matrix, agg=self.cfg.sampler_agg
            )
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = ChestXrayDataset(
            dataframe=self.val_df,
            transforms=build_eval_transforms(self.cfg.image_size),
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """In-distribution test set (A/B/C)."""
        dataset = ChestXrayDataset(
            dataframe=self.test_df,
            transforms=build_eval_transforms(self.cfg.image_size),
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def ood_dataloader(self) -> Optional[DataLoader]:
        """Out-of-distribution test set (hospital_d). Returns None if unavailable."""
        if self.ood_df is None:
            return None
        dataset = ChestXrayDataset(
            dataframe=self.ood_df,
            transforms=build_eval_transforms(self.cfg.image_size),
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
        )

    @property
    def class_names(self) -> List[str]:
        return DISEASE_LABELS
