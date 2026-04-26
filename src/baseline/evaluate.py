"""
Standalone evaluation script.  Loads a saved checkpoint and runs inference on in-dist test (A/B/C) and OOD test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from baseline.config import CFG, DISEASE_LABELS
from baseline.datamodule import ChestXrayDataModule
from baseline.dataset import ChestXrayDataset, build_eval_transforms
from baseline.model import ChestXrayClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--ood_only",
        action="store_true",
    )
    parser.add_argument(
        "--custom_data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_size", type=int, default=64
    )
    parser.add_argument("--devices", type=int, default=1)
    return parser.parse_args()


def load_custom_parquets(directory: str) -> pd.DataFrame:
    import glob
    paths = glob.glob(str(Path(directory) / "**" / "*.parquet"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No Parquet files in {directory}")
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()

    datamodule = ChestXrayDataModule(cfg=CFG)
    datamodule.setup()

    logger.info("Loading checkpoint: %s", args.ckpt)
    model = ChestXrayClassifier.load_from_checkpoint(
        args.ckpt,
        pos_weight=datamodule.pos_weight,
        cfg=CFG,
    )
    model.eval()

    loaders = []

    if not args.ood_only and args.custom_data is None:
        loaders.append(datamodule.test_dataloader())

    ood_loader = datamodule.ood_dataloader()
    if ood_loader is not None and args.custom_data is None:
        loaders.append(ood_loader)

    if args.custom_data is not None:
        custom_df = load_custom_parquets(args.custom_data)
        custom_dataset = ChestXrayDataset(
            dataframe=custom_df,
            transforms=build_eval_transforms(CFG.image_size),
            image_col=CFG.image_col,
            image_path_col=CFG.image_path_col,
        )
        loaders.append(
            DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=False)
        )

    if not loaders:
        logger.error("No dataloaders to evaluate.  Exiting.")
        sys.exit(1)

    # Trainer (inference only)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=min(args.devices, max(torch.cuda.device_count(), 1)),
        precision=CFG.precision,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.test(model=model, dataloaders=loaders)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
