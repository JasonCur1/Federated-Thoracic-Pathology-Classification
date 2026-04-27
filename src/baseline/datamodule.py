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
    """pos_weight[c] = (N - pos[c]) / pos[c], clipped at 50."""
    n_samples = label_matrix.shape[0]
    pos_count = label_matrix.sum(axis=0).clip(min=1)
    neg_count = n_samples - pos_count
    pos_weight = np.clip(neg_count / pos_count, a_min=None, a_max=50.0)
    for name, w in zip(DISEASE_LABELS, pos_weight):
        logger.info("  pos_weight  %-22s  %.2f", name, w)
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_sample_weights(label_matrix: np.ndarray, agg: str = "max") -> np.ndarray:
    n_samples, n_classes = label_matrix.shape
    pos_count = label_matrix.sum(axis=0).clip(min=1)
    class_weights = n_samples / (n_classes * pos_count)
    weighted = label_matrix * class_weights[np.newaxis, :]
    if agg == "max":
        sample_weights = weighted.max(axis=1)
    elif agg == "sum":
        sample_weights = weighted.sum(axis=1)
    else:
        raise ValueError(f"Unknown agg: {agg!r}")
    return np.where(sample_weights == 0, 1.0, sample_weights).astype(np.float64)

class ChestXrayDataModule(pl.LightningDataModule):
    """
    Centralized DataModule that pools Hospital A, B, C files.

    For federated setup:
    Pass hospital_id="hospital_a" (or b / c / d) to restrict loading to a
    single hospital.  The train/val/test split then applies only to that
    hospital's rows.
    """

    def __init__(self, cfg=CFG, hospital_id: Optional[str] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.hospital_id = hospital_id
        self.save_hyperparameters(logger=False)

        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.ood_df: Optional[pd.DataFrame] = None
        self.pos_weight: Optional[torch.Tensor] = None
        self._train_label_matrix: Optional[np.ndarray] = None

    # Internal: Parquet loading

    def _load_hospital_parquets(self, hospital_ids: List[str]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for h_id in hospital_ids:
            pattern = str(self.cfg.data_root / h_id / "**" / "*.parquet")
            paths = glob.glob(pattern, recursive=True)
            if not paths:
                logger.warning("No Parquet files for %s at %s", h_id, pattern)
                continue
            logger.info("Loading %d file(s) from %s …", len(paths), h_id)
            for p in paths:
                df = pd.read_parquet(p)
                df["_source_hospital"] = h_id
                frames.append(df)
        if not frames:
            raise FileNotFoundError(
                f"No Parquet files found. data_root={self.cfg.data_root}"
            )
        combined = pd.concat(frames, ignore_index=True)
        logger.info("Total rows: %d", len(combined))
        return combined

    def _validate_columns(self, df: pd.DataFrame) -> None:
        for col in ["label", self.cfg.image_col]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}'. Found: {df.columns.tolist()}")

    # Internal: stratified split

    @staticmethod
    def _stratified_split(
        df: pd.DataFrame,
        label_matrix: np.ndarray,
        val_frac: float,
        test_frac: float,
        seed: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        indices = np.arange(len(df))
        try:
            from skmultilearn.model_selection import iterative_train_test_split
            X = indices.reshape(-1, 1)
            X_tv, y_tv, X_te, _ = iterative_train_test_split(X, label_matrix, test_size=test_frac)
            rel_val = val_frac / (1.0 - test_frac)
            X_tr, _, X_val, _ = iterative_train_test_split(X_tv, y_tv, test_size=rel_val)
            train_idx, val_idx, test_idx = X_tr.flatten(), X_val.flatten(), X_te.flatten()
            logger.info("Iterative split: train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))
            return (
                df.iloc[train_idx].reset_index(drop=True),
                df.iloc[val_idx].reset_index(drop=True),
                df.iloc[test_idx].reset_index(drop=True),
            )
        except ImportError:
            logger.warning("scikit-multilearn not found; falling back to binary stratification.")

        df = df.copy()
        df["_any"] = df["label"].apply(lambda e: int(
            any(item in DISEASE_LABELS for item in e)
            if isinstance(e, (list, tuple, np.ndarray)) else 0
        ))
        trainval, test = train_test_split(df, test_size=test_frac, stratify=df["_any"], random_state=seed)
        train, val = train_test_split(trainval, test_size=val_frac / (1.0 - test_frac),
                                      stratify=trainval["_any"], random_state=seed)
        for frame in [train, val, test]:
            frame.drop(columns=["_any"], inplace=True, errors="ignore")
        logger.info("Binary split: train=%d  val=%d  test=%d", len(train), len(val), len(test))
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    # LightningDataModule hooks

    def prepare_data(self) -> None:
        hospitals = [self.hospital_id] if self.hospital_id else ID_HOSPITALS
        for h_id in hospitals:
            h_path = self.cfg.data_root / h_id
            if not h_path.exists():
                logger.warning("Hospital directory not found: %s", h_path)

    def setup(self, stage: Optional[str] = None) -> None:
        import albumentations as A
        _noop = A.Compose([A.NoOp()])

        # Which hospitals to pool?
        if self.hospital_id:
            source_hospitals = [self.hospital_id]
            logger.info("[Silo] Loading data for hospital: %s", self.hospital_id)
        else:
            source_hospitals = ID_HOSPITALS
            logger.info("[Centralized] Pooling hospitals: %s", source_hospitals)

        id_df = self._load_hospital_parquets(source_hospitals)
        self._validate_columns(id_df)

        # Build temporary dataset to extract the label matrix
        _tmp = ChestXrayDataset(
            dataframe=id_df,
            transforms=_noop,
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        full_label_matrix = _tmp.label_matrix

        self.train_df, self.val_df, self.test_df = self._stratified_split(
            id_df,
            label_matrix=full_label_matrix,
            val_frac=self.cfg.val_fraction,
            test_frac=self.cfg.test_fraction,
            seed=self.cfg.seed,
        )

        _train_tmp = ChestXrayDataset(
            dataframe=self.train_df,
            transforms=_noop,
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        self._train_label_matrix = _train_tmp.label_matrix
        self.pos_weight = compute_pos_weights(self._train_label_matrix)

        logger.info(
            "Ready — train=%d  val=%d  test=%d",
            len(self.train_df), len(self.val_df), len(self.test_df),
        )

        # OOD set: only loaded in centralized mode or if hospital_id == OOD_HOSPITAL (for federated)
        if not self.hospital_id or self.hospital_id == OOD_HOSPITAL:
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
        sampler, shuffle = None, True
        if self.cfg.use_weighted_sampler:
            sw = compute_sample_weights(self._train_label_matrix, agg=self.cfg.sampler_agg)
            sampler = WeightedRandomSampler(
                torch.from_numpy(sw).double(), num_samples=len(dataset), replacement=True
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
        return DataLoader(dataset, batch_size=self.cfg.batch_size * 2, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
                          persistent_workers=self.cfg.num_workers > 0)

    def test_dataloader(self) -> DataLoader:
        dataset = ChestXrayDataset(
            dataframe=self.test_df,
            transforms=build_eval_transforms(self.cfg.image_size),
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        return DataLoader(dataset, batch_size=self.cfg.batch_size * 2, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
                          persistent_workers=self.cfg.num_workers > 0)

    def ood_dataloader(self) -> Optional[DataLoader]:
        if self.ood_df is None:
            return None
        dataset = ChestXrayDataset(
            dataframe=self.ood_df,
            transforms=build_eval_transforms(self.cfg.image_size),
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        return DataLoader(dataset, batch_size=self.cfg.batch_size * 2, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
                          persistent_workers=self.cfg.num_workers > 0)

    @property
    def class_names(self) -> List[str]:
        return DISEASE_LABELS


# Federated silo DataModule

class FederatedDataModule(ChestXrayDataModule):
    HOSPITAL_MAP: Dict[str, str] = {
        "perch":  "hospital_a",
        "turbot": "hospital_b",
        "wahoo":  "hospital_c",
    }

    def __init__(self, cfg=CFG, hospital_id: Optional[str] = None) -> None:
        super().__init__(cfg=cfg, hospital_id=hospital_id)

    @classmethod
    def from_hostname(cls, cfg=CFG, hostname: Optional[str] = None) -> "FederatedDataModule":
        """Resolve hospital_id from the current machine's hostname."""
        import socket
        host = hostname or socket.gethostname().split(".")[0].lower()
        hospital_id = cls.HOSPITAL_MAP.get(host)
        if hospital_id is None:
            raise RuntimeError(
                f"Unknown hostname '{host}'. "
                f"Known nodes: {list(cls.HOSPITAL_MAP.keys())}. "
                "Set HOSPITAL_MAP or pass hospital_id explicitly."
            )
        logger.info("[FL] Node '%s' → hospital '%s'", host, hospital_id)
        return cls(cfg=cfg, hospital_id=hospital_id)

    def member_loader(self, n_samples: int = 256, batch_size: int = 64) -> DataLoader:
        """
        Small DataLoader over a random training subset for MIA evaluation.
        Only called server-side on a representative sample.
        """
        from federated.metrics_logger import make_small_loader
        dataset = ChestXrayDataset(
            dataframe=self.train_df,
            transforms=build_eval_transforms(self.cfg.image_size),
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        return make_small_loader(dataset, n_samples=n_samples, batch_size=batch_size)


    def fl_train_dataloader(self) -> DataLoader:
        dataset = ChestXrayDataset(
            dataframe=self.train_df,
            transforms=build_train_transforms(self.cfg.image_size),
            image_col=self.cfg.image_col,
            image_path_col=self.cfg.image_path_col,
            label_col="label",
        )
        sampler, shuffle = None, True
        if self.cfg.use_weighted_sampler:
            sw = compute_sample_weights(self._train_label_matrix, agg=self.cfg.sampler_agg)
            sampler = WeightedRandomSampler(
                torch.from_numpy(sw).double(), num_samples=len(dataset), replacement=True
            )
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
        )
    
    def fl_val_dataloader(self) -> DataLoader:
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
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )