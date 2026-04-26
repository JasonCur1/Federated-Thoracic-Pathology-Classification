from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from baseline.config import DISEASE_LABELS


# ── Augmentation Pipelines ────────────────────────────────────────────────────

def build_train_transforms(image_size: int = 224) -> A.Compose:
    """
    CLAHE enhances local contrast which is important for subtle findings like nodules.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        # Contrast-Limited Adaptive Histogram Equalization
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],    # pretrained DenseNet weights
        ),
        ToTensorV2(),
    ])


def build_eval_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # always apply
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# Dataset

class ChestXrayDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transforms: A.Compose,
        image_col: Optional[str] = "image_bytes",
        image_path_col: str = "image_path",
        label_col: str = "label",
        labels: List[str] = DISEASE_LABELS,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.labels = labels
        self.label_col = label_col
        self.image_path_col = image_path_col
        self._num_classes = len(labels)

        self._use_bytes = (
            image_col is not None and image_col in self.df.columns
        )
        self._image_col = image_col if self._use_bytes else None

        if not self._use_bytes and image_path_col not in self.df.columns:
            raise ValueError(
                f"DataFrame must contain either '{image_col}' (bytes) or "
                f"'{image_path_col}' (path) column."
            )

        if label_col not in self.df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in DataFrame.\n"
                f"Available columns: {self.df.columns.tolist()}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image as numpy RGB array
        image = self._load_image(row)

        augmented = self.transforms(image=image)
        image_tensor: torch.Tensor = augmented["image"]

        label_tensor = self._parse_label_entry(row[self.label_col])

        return image_tensor, label_tensor

    # Private Helpers

    def _load_image(self, row: pd.Series) -> np.ndarray:
        """Unpacks the dictionary from the Parquet cell."""
        data = row[self._image_col]

        if isinstance(data, dict) and "bytes" in data:
            raw_bytes = data["bytes"]
        else:
            raw_bytes = data

        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.array(pil_image, dtype=np.uint8)

    def _parse_label_entry(self, entry) -> torch.Tensor:
        vec = np.zeros(self._num_classes, dtype=np.float32)

        if entry is None or (isinstance(entry, float) and np.isnan(entry)):
            return torch.from_numpy(vec)

        if isinstance(entry, (list, tuple, np.ndarray)):
            if len(entry) > 0 and any(isinstance(x, str) for x in entry):
                for item in entry:
                    if item in self.labels:
                        vec[self.labels.index(item)] = 1.0
                return torch.from_numpy(vec)

        if isinstance(entry, str):
            entry = entry.strip()
            if not entry or entry == "No Finding":
                return torch.from_numpy(vec)
            if entry in self.labels:
                vec[self.labels.index(entry)] = 1.0
                return torch.from_numpy(vec)

            indices = [int(x) for x in entry.replace(",", " ").split() if x.strip().isdigit()]
            self._fill_indices(vec, indices)
            return torch.from_numpy(vec)

        if isinstance(entry, np.ndarray):
            entry = entry.tolist()

        if isinstance(entry, (list, tuple)):
            if len(entry) == self._num_classes and all(v in (0, 1, 0.0, 1.0, True, False) for v in entry):
                vec = np.array(entry, dtype=np.float32)
            else:
                self._fill_indices(vec, [int(x) for x in entry])
            return torch.from_numpy(vec)

        if isinstance(entry, (int, np.integer)):
            self._fill_indices(vec, [int(entry)])
            return torch.from_numpy(vec)

        return torch.from_numpy(vec)

    def _fill_indices(self, vec: np.ndarray, indices: List[int]) -> None:
        for i in indices:
            if not (0 <= i < self._num_classes):
                raise ValueError(
                    f"Label index {i} is out of range [0, {self._num_classes}). "
                    f"DISEASE_LABELS has {self._num_classes} entries."
                )
            vec[i] = 1.0


    @property
    def label_matrix(self) -> np.ndarray:
        rows = []
        for idx in range(len(self.df)):
            rows.append(self._parse_label_entry(self.df.iloc[idx][self.label_col]).numpy())
        return np.stack(rows, axis=0)   # (N, C)
