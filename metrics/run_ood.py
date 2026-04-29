from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from common import (
    ID_HOSPITALS,
    OOD_HOSPITAL,
    PROJECT_ROOT,
    evaluate_model_on_dataframe,
    load_hospital_parquets,
    load_model_from_checkpoint,
    save_json,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ID and OOD evaluation on saved FL global model.")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="fl_logs/master/checkpoints/global_final.pt",
        help="Path to saved .pt model checkpoint.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="metrics/results",
        help="Directory where result JSON files will be saved.",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional max samples per evaluation. 0 means use all.",
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def maybe_sample(df, max_samples: int, seed: int):
    if max_samples and max_samples > 0 and len(df) > max_samples:
        return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def main() -> None:
    setup_logging()
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = load_model_from_checkpoint(checkpoint_path, device)

    logger.info("Loading ID test data from hospitals: %s", ID_HOSPITALS)
    id_df = load_hospital_parquets(ID_HOSPITALS, split_prefix="test")
    id_df = maybe_sample(id_df, args.max_samples, args.seed)

    logger.info("Loading OOD test data from hospital: %s", OOD_HOSPITAL)
    ood_df = load_hospital_parquets([OOD_HOSPITAL], split_prefix="test")
    ood_df = maybe_sample(ood_df, args.max_samples, args.seed + 1)

    logger.info("Evaluating ID test data...")
    id_metrics = evaluate_model_on_dataframe(
        model=model,
        df=id_df,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )

    logger.info("Evaluating OOD test data...")
    ood_metrics = evaluate_model_on_dataframe(
        model=model,
        df=ood_df,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )

    id_auroc = id_metrics.get("auroc_macro")
    ood_auroc = ood_metrics.get("auroc_macro")

    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "id_hospitals": ID_HOSPITALS,
        "ood_hospital": OOD_HOSPITAL,
        "id_test": id_metrics,
        "ood_test": ood_metrics,
        "ood_generalization_gap": {
            "auroc_gap_id_minus_ood": None
            if id_auroc is None or ood_auroc is None
            else id_auroc - ood_auroc,
            "interpretation": "Positive gap means ID performance is higher than OOD performance.",
        },
    }

    save_json(id_metrics, output_dir / "id_eval_results.json")
    save_json(ood_metrics, output_dir / "ood_eval_results.json")
    save_json(summary, output_dir / "ood_summary.json")

    logger.info("ID AUROC:  %s", id_metrics.get("auroc_macro"))
    logger.info("OOD AUROC: %s", ood_metrics.get("auroc_macro"))
    logger.info("Done.")


if __name__ == "__main__":
    main()
