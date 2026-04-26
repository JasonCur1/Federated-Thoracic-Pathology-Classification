from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from baseline.config import CFG, DISEASE_LABELS, NUM_CLASSES

logger = logging.getLogger(__name__)



def build_densenet121(
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.0,
) -> nn.Module:
    """
    Instantiate DenseNet-121 with a two-layer classification head.

    No sigmoid is applied here; ASL / BCEWithLogitsLoss handle that internally.
    """
    weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.densenet121(weights=weights)

    in_features: int = model.classifier.in_features  # 1024 for DenseNet-121
    hidden: int = 512

    head = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate if dropout_rate > 0.0 else 0.0),
        nn.Linear(hidden, num_classes),
    )

    model.classifier = head

    logger.info(
        "DenseNet-121 loaded (pretrained=%s), head: %d → %d → %d logits",
        pretrained,
        in_features,
        hidden,
        num_classes,
    )
    return model


def _build_metric_collection(prefix: str, num_labels: int) -> MetricCollection:
    """
    All classification metrics use threshold=0.5 initially; optimal thresholds are injected at test time via on_test_epoch_start.
    """
    return MetricCollection(
        {
            # Global (macro)
            "f1_macro": MultilabelF1Score(
                num_labels=num_labels, average="macro", threshold=0.5
            ),
            "recall_macro": MultilabelRecall(
                num_labels=num_labels, average="macro", threshold=0.5
            ),
            "precision_macro": MultilabelPrecision(
                num_labels=num_labels, average="macro", threshold=0.5
            ),
            "auroc_macro": MultilabelAUROC(
                num_labels=num_labels, average="macro"
            ),
            # Per-class (none)
            "f1_per_class": MultilabelF1Score(
                num_labels=num_labels, average="none", threshold=0.5
            ),
            "recall_per_class": MultilabelRecall(
                num_labels=num_labels, average="none", threshold=0.5
            ),
            "precision_per_class": MultilabelPrecision(
                num_labels=num_labels, average="none", threshold=0.5
            ),
            "auroc_per_class": MultilabelAUROC(
                num_labels=num_labels, average="none"
            ),
        },
        prefix=f"{prefix}/",
    )

class ChestXrayClassifier(pl.LightningModule):
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        cfg=CFG,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["pos_weight"])

        # Model
        self.model = build_densenet121(
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes,
            dropout_rate=cfg.dropout_rate,
        )

        # Threshold buffer
        self.register_buffer(
            "optimal_thresholds",
            torch.full((cfg.num_classes,), 0.5, dtype=torch.float32)
        )

        self._loss_fn: Optional[AsymmetricLoss] = None

        # Metrics
        self.train_metrics = _build_metric_collection("train", NUM_CLASSES)
        self.val_metrics = _build_metric_collection("val", NUM_CLASSES)
        self.test_id_metrics = _build_metric_collection("test_id", NUM_CLASSES)
        self.test_ood_metrics = _build_metric_collection("test_ood", NUM_CLASSES)

        self.train_loss_metric = torchmetrics.MeanMetric()
        self.val_loss_metric = torchmetrics.MeanMetric()

    def _get_loss_fn(self) -> "AsymmetricLoss":
        if self._loss_fn is None:
            self._loss_fn = AsymmetricLoss(
                gamma_neg=getattr(self.cfg, "asl_gamma_neg", 3.0),
                gamma_pos=getattr(self.cfg, "asl_gamma_pos", 0.0),
                clip=getattr(self.cfg, "asl_clip", 0.0),
            )
        return self._loss_fn

    # ── MixUp Helper ─────────────────────────────────────────────────────────

    def _mixup_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        lam = float(np.random.beta(alpha, alpha))
        batch_size = images.size(0)
        idx = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1.0 - lam) * images[idx]
        mixed_targets = lam * targets + (1.0 - lam) * targets[idx]
        return mixed_images, mixed_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (B, num_classes)."""
        return self.model(x)

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        images, targets = batch
        logits = self(images)
        loss = self._get_loss_fn()(logits, targets)
        return loss, logits, targets

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        images, targets = batch

        use_mixup = getattr(self.cfg, "use_mixup", True)
        mixup_alpha = getattr(self.cfg, "mixup_alpha", 0.4)

        if use_mixup and self.training:
            mixed_images, mixed_targets = self._mixup_batch(images, targets, alpha=mixup_alpha)
            logits = self(mixed_images)
            loss = self._get_loss_fn()(logits, mixed_targets)
            with torch.no_grad():
                orig_logits = self(images)
            probs = torch.sigmoid(orig_logits)
        else:
            loss, logits, targets = self._shared_step(batch)
            probs = torch.sigmoid(logits)

        self.train_loss_metric(loss)
        self.train_metrics(probs, targets.int())

        # Log step-level loss for progress bar
        self.log(
            "train/loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.log(
            "train/loss",
            self.train_loss_metric.compute(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_loss_metric.reset()

        results = self.train_metrics.compute()
        self._log_metrics(results, per_class_prefix="train")
        self.train_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_probs = []
        self.val_targets = []

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, logits, targets = self._shared_step(batch)
        probs = torch.sigmoid(logits)

        self.val_loss_metric(loss)
        self.val_metrics(probs, targets.int())

        # Save for threshold tuning — always use un-augmented val data
        self.val_probs.append(probs.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        self.log(
            "val/loss",
            self.val_loss_metric.compute(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.val_loss_metric.reset()

        results = self.val_metrics.compute()
        self._log_metrics(results, per_class_prefix="val")
        self.val_metrics.reset()

        # Optimal Threshold Tuning (F1 Maximization)
        if not (hasattr(self, "val_probs") and len(self.val_probs) > 0):
            return

        local_probs = torch.cat(self.val_probs, dim=0).to(self.device)
        local_targets = torch.cat(self.val_targets, dim=0).to(self.device)

        all_probs = self.all_gather(local_probs).view(-1, self.cfg.num_classes)
        all_targets = self.all_gather(local_targets).view(-1, self.cfg.num_classes)

        thresholds = torch.arange(0.05, 0.96, 0.02).to(self.device)
        best_thresh = torch.full((self.cfg.num_classes,), 0.5, device=self.device)

        for c in range(self.cfg.num_classes):
            c_probs = all_probs[:, c]
            c_targets = all_targets[:, c]

            best_f1 = -1.0
            best_t = 0.5

            for t in thresholds:
                preds = (c_probs >= t).float()
                tp = (preds * c_targets).sum()
                fp = (preds * (1 - c_targets)).sum()
                fn = ((1 - preds) * c_targets).sum()

                denom = 2 * tp + fp + fn
                f1 = float((2 * tp) / denom) if denom > 0 else 0.0

                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t.item()

            best_thresh[c] = best_t

        self.optimal_thresholds.copy_(best_thresh)

        for name, thresh in zip(DISEASE_LABELS, best_thresh.tolist()):
            self.log(f"val/threshold_{name}", thresh, on_epoch=True, sync_dist=True)

        self.val_probs.clear()
        self.val_targets.clear()

    def on_test_epoch_start(self) -> None:
        """
        Inject the optimal thresholds found during validation into test metrics.
        """
        thresholds = self.optimal_thresholds.tolist()
        non_default = sum(1 for t in thresholds if abs(t - 0.5) > 0.01)

        if non_default == 0:
            logger.warning(
                "ALL thresholds are at default 0.5 — threshold tuning may not "
                "have run, or the checkpoint buffer was not restored. "
                "Check that val_probs were populated during validation."
            )
        else:
            logger.info(
                "Injecting optimal thresholds (%d / %d non-default):",
                non_default,
                self.cfg.num_classes,
            )

        for name, thresh in zip(DISEASE_LABELS, thresholds):
            logger.info("  %-22s %.2f", name, thresh)

        # Inject into all test metric objects that use a hard threshold
        for metric_collection in [self.test_id_metrics, self.test_ood_metrics]:
            for metric in metric_collection.values():
                if hasattr(metric, "threshold"):
                    metric.threshold = self.optimal_thresholds

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        dataloader_idx == 0 → in-distribution test (A/B/C)
        dataloader_idx == 1 → OOD test (hospital_d)
        """
        _, logits, targets = self._shared_step(batch)
        probs = torch.sigmoid(logits)

        if dataloader_idx == 0:
            self.test_id_metrics(probs, targets.int())
        else:
            self.test_ood_metrics(probs, targets.int())

    def on_test_epoch_end(self) -> None:
        id_results = self.test_id_metrics.compute()
        self._log_metrics(id_results, per_class_prefix="test_id")
        self.test_id_metrics.reset()

        ood_results = self.test_ood_metrics.compute()
        self._log_metrics(ood_results, per_class_prefix="test_ood")
        self.test_ood_metrics.reset()

        id_auroc = id_results.get("test_id/auroc_macro")
        ood_auroc = ood_results.get("test_ood/auroc_macro")
        if id_auroc is not None and ood_auroc is not None:
            gap = (id_auroc - ood_auroc).abs()
            self.log("test/ood_generalization_gap_auroc", gap, sync_dist=True)
            logger.info(
                "Generalization gap (AUROC) ID=%.4f  OOD=%.4f  Δ=%.4f",
                id_auroc.item(),
                ood_auroc.item(),
                gap.item(),
            )

    def _log_metrics(
        self, results: Dict[str, torch.Tensor], per_class_prefix: str
    ) -> None:
        for key, value in results.items():
            if value.ndim == 0:
                self.log(
                    key,
                    value,
                    on_epoch=True,
                    sync_dist=True,
                    prog_bar=("auroc_macro" in key or "f1_macro" in key),
                )
            else:
                metric_short = key.split("/")[-1].replace("_per_class", "")
                for cls_idx, cls_name in enumerate(DISEASE_LABELS):
                    self.log(
                        f"{per_class_prefix}/{metric_short}_{cls_name}",
                        value[cls_idx],
                        on_epoch=True,
                        sync_dist=True,
                    )

    # Optimizer & Scheduler

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.cfg.monitor_mode,
            factor=self.cfg.lr_factor,
            patience=self.cfg.lr_patience,
            min_lr=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.cfg.monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def set_pos_weight(self, pos_weight: torch.Tensor) -> None:
        """Update pos_weight buffer – must be called before fit()."""
        self.pos_weight = pos_weight.to(self.device if self.device else "cpu")
        self._loss_fn = None


class AsymmetricLoss(nn.Module):
    """
    Paper: https://arxiv.org/abs/2009.14119

    Default hyperparameters changed from (gamma_neg=4, gamma_pos=1,
    clip=0.05) to (gamma_neg=3, gamma_pos=0, clip=0.0).

    These values match the ASL paper's "ASL-S" (soft) variant which is
    recommended when pos_weight or sampler-based reweighting is already in use.
    """

    def __init__(
        self,
        gamma_neg: float = 3.0,
        gamma_pos: float = 0.0,
        clip: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)

        pt_pos = probabilities
        loss_pos = -targets * torch.log(pt_pos.clamp(min=self.eps))
        # gamma_pos=0 → weight_pos = (1-pt)^0 = 1 → no focal scaling
        weight_pos = torch.pow(1.0 - pt_pos, self.gamma_pos)

        pt_neg = torch.clamp(probabilities - self.clip, min=0.0)
        loss_neg = -(1 - targets) * torch.log((1 - pt_neg).clamp(min=self.eps))
        weight_neg = torch.pow(pt_neg, self.gamma_neg)

        # Combine
        loss = (weight_pos * loss_pos) + (weight_neg * loss_neg)
        return loss.mean()
