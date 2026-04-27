from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DISEASE_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES: int = len(DISEASE_LABELS)

ID_HOSPITALS: List[str] = ["hospital_a", "hospital_b", "hospital_c"]
OOD_HOSPITAL: str = "hospital_d"

@dataclass
class Config:
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def checkpoint_dir(self) -> Path:
        return self.project_root / "checkpoints"

    # Data
    image_col: str = "image"
    image_path_col: str = "image_path"
    label_col_prefix: str = ""
    split_col: str = "split"

    image_size: int = 224
    num_workers: int = min(8, os.cpu_count() or 4)
    pin_memory: bool = True
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    # Model
    backbone: str = "densenet121"
    pretrained: bool = True
    num_classes: int = NUM_CLASSES
    dropout_rate: float = 0.4

    # Training
    batch_size: int = 32
    max_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    lr_patience: int = 5
    lr_factor: float = 0.5
    gradient_clip_val: float = 1.0

    # Sampler
    use_weighted_sampler: bool = False
    sampler_agg: str = "max"

    use_mixup: bool = False
    mixup_alpha: float = 0.4         # Beta(alpha, alpha) concentration parameter

    # Asymmetric loss
    asl_gamma_neg: float = 3.0
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.0

    # Hardware stuff
    accelerator: str = "gpu"
    devices: int = -1
    strategy: str = "ddp"
    precision: str = "16-mixed"

    # Logging
    experiment_name: str = "centralized_baseline"
    log_every_n_steps: int = 50
    save_top_k: int = 3
    monitor_metric: str = "val/auroc_macro"
    monitor_mode: str = "max"

    seed: int = 42


# Singleton instance – import this everywhere
CFG = Config()
