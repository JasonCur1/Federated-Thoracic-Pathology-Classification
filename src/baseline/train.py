from __future__ import annotations

import argparse
import logging
import sys
import pathlib
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

if hasattr(torch.serialization, 'add_safe_globals'):
    from baseline.config import Config
    torch.serialization.add_safe_globals([
        Config,
        pathlib.Path,
        pathlib.PosixPath,
        pathlib.WindowsPath
    ])

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from baseline.config import CFG, DISEASE_LABELS
from baseline.datamodule import ChestXrayDataModule
from baseline.model import ChestXrayClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train centralized DenseNet-121 on NIH Chest X-ray data"
    )
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--max_epochs", type=int, default=CFG.max_epochs)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    parser.add_argument(
        "--devices",
        type=int,
        default=CFG.devices,
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
    )
    parser.add_argument(
        "--no_sampler",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--exp_name", type=str, default=CFG.experiment_name)
    parser.add_argument(
        "--precision",
        type=str,
        default=CFG.precision,
        choices=["32", "16-mixed", "bf16-mixed"],
    )

    parser.add_argument(
        "--no_mixup",
        action="store_true",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=CFG.mixup_alpha,
    )

    parser.add_argument(
        "--asl_gamma_neg",
        type=float,
        default=CFG.asl_gamma_neg,
    )
    parser.add_argument(
        "--asl_gamma_pos",
        type=float,
        default=CFG.asl_gamma_pos,
    )
    parser.add_argument(
        "--asl_clip",
        type=float,
        default=CFG.asl_clip,
    )

    return parser.parse_args()


def build_callbacks(cfg=CFG) -> list:
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(cfg.checkpoint_dir / cfg.experiment_name),
        filename="epoch{epoch:02d}-auroc{val/auroc_macro:.4f}",
        monitor=cfg.monitor_metric,
        mode=cfg.monitor_mode,
        save_top_k=cfg.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
    )

    early_stop_cb = EarlyStopping(
        monitor=cfg.monitor_metric,
        mode=cfg.monitor_mode,
        patience=10,
        verbose=True,
        min_delta=1e-4,
    )

    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")
    progress_cb = RichProgressBar(leave=True)

    return [checkpoint_cb, early_stop_cb, lr_monitor_cb, progress_cb]


def run_ood_evaluation(
    trainer: pl.Trainer,
    model: ChestXrayClassifier,
    datamodule: ChestXrayDataModule,
) -> None:

    ood_loader = datamodule.ood_dataloader()
    if ood_loader is None:
        logger.warning("Skipping OOD evaluation – hospital_d data not found.")
        return

    logger.info("=" * 60)
    logger.info("Running OOD evaluation on hospital_d")
    logger.info("=" * 60)

    id_loader = datamodule.test_dataloader()

    trainer.test(
        model=model,
        dataloaders=[id_loader, ood_loader],
        ckpt_path="best",
        verbose=True,
    )


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)

    CFG.batch_size = args.batch_size
    CFG.max_epochs = args.max_epochs
    CFG.learning_rate = args.lr
    CFG.weight_decay = args.weight_decay
    CFG.devices = args.devices
    CFG.pretrained = not args.no_pretrained
    CFG.use_weighted_sampler = not args.no_sampler
    CFG.seed = args.seed
    CFG.experiment_name = args.exp_name
    CFG.precision = args.precision

    CFG.use_mixup = not args.no_mixup
    CFG.mixup_alpha = args.mixup_alpha
    CFG.asl_gamma_neg = args.asl_gamma_neg
    CFG.asl_gamma_pos = args.asl_gamma_pos
    CFG.asl_clip = args.asl_clip

    logger.info(
        "Config — mixup=%s (alpha=%.2f)  ASL(γ-=%.1f, γ+=%.1f, clip=%.2f)",
        CFG.use_mixup, CFG.mixup_alpha,
        CFG.asl_gamma_neg, CFG.asl_gamma_pos, CFG.asl_clip,
    )

    CFG.log_dir.mkdir(parents=True, exist_ok=True)
    CFG.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising DataModule …")
    datamodule = ChestXrayDataModule(cfg=CFG)
    datamodule.prepare_data()
    datamodule.setup()

    logger.info("Label distribution in training set:")
    train_label_matrix = datamodule._train_label_matrix
    n_train = len(datamodule.train_df)
    for col_idx, col in enumerate(DISEASE_LABELS):
        pos = int(train_label_matrix[:, col_idx].sum())
        pct = 100.0 * pos / n_train
        logger.info("  %-22s %6d  (%.2f%%)", col, pos, pct)

    # Model
    logger.info("Initialising model …")
    model = ChestXrayClassifier(
        pos_weight=datamodule.pos_weight,
        cfg=CFG,
    )

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=str(CFG.log_dir),
        name=CFG.experiment_name,
        default_hp_metric=False,
    )

    # Strategy
    n_gpus = torch.cuda.device_count()
    strategy = "ddp" if n_gpus > 1 else "auto"
    if n_gpus <= 1 and CFG.devices == -1:
        effective_devices = max(n_gpus, 1)
    else:
        effective_devices = CFG.devices

    logger.info(
        "Hardware: %d GPU(s) detected – strategy=%s  devices=%s",
        n_gpus, strategy, effective_devices,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator=CFG.accelerator if n_gpus > 0 else "cpu",
        devices=effective_devices,
        strategy=strategy,
        precision=CFG.precision,
        max_epochs=CFG.max_epochs,
        gradient_clip_val=CFG.gradient_clip_val,
        log_every_n_steps=CFG.log_every_n_steps,
        deterministic=False,
        logger=tb_logger,
        callbacks=build_callbacks(CFG),
        sync_batchnorm=True,
        use_distributed_sampler=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
    )

    # Fit
    logger.info("Starting training …")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Training complete.")
    logger.info(
        "Best checkpoint: %s  (val/auroc_macro=%.4f)",
        trainer.checkpoint_callback.best_model_path,
        trainer.checkpoint_callback.best_model_score or float("nan"),
    )

    run_ood_evaluation(trainer, model, datamodule)

    logger.info("All done. TensorBoard logs: %s", CFG.log_dir)


if __name__ == "__main__":
    main()
