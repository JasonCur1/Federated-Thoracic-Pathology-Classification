from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from common import (
    ID_HOSPITALS,
    OOD_HOSPITAL,
    PROJECT_ROOT,
    get_logits_targets_and_losses,
    load_hospital_parquets,
    load_model_from_checkpoint,
    make_loader,
    run_loss_threshold_mia,
    sample_dataframe,
    save_json,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run loss-based Membership Inference Attack.")

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
        help="Directory where MIA result JSON will be saved.",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--mia_samples",
        type=int,
        default=512,
        help="Number of member and non-member samples to use.",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--non_member_source",
        type=str,
        choices=["ood_test", "id_test"],
        default="ood_test",
        help=(
            "ood_test uses hospital_d test as non-members. "
            "id_test uses hospital_a/b/c test as non-members."
        ),
    )

    return parser.parse_args()


def compute_losses(model, df, device, batch_size, num_workers):
    loader = make_loader(df, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    _, _, losses = get_logits_targets_and_losses(model, loader, device)
    return losses


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

    logger.info("Loading member data: train split from %s", ID_HOSPITALS)
    member_df = load_hospital_parquets(ID_HOSPITALS, split_prefix="train")
    member_df = sample_dataframe(member_df, args.mia_samples, args.seed)

    if args.non_member_source == "ood_test":
        logger.info("Loading non-member data: test split from OOD hospital %s", OOD_HOSPITAL)
        non_member_df = load_hospital_parquets([OOD_HOSPITAL], split_prefix="test")
    else:
        logger.info("Loading non-member data: test split from ID hospitals %s", ID_HOSPITALS)
        non_member_df = load_hospital_parquets(ID_HOSPITALS, split_prefix="test")

    non_member_df = sample_dataframe(non_member_df, args.mia_samples, args.seed + 1)

    logger.info("Computing member losses...")
    member_losses = compute_losses(
        model=model,
        df=member_df,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logger.info("Computing non-member losses...")
    non_member_losses = compute_losses(
        model=model,
        df=non_member_df,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    mia_results = run_loss_threshold_mia(
        member_losses=member_losses,
        non_member_losses=non_member_losses,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "member_source": {
            "hospitals": ID_HOSPITALS,
            "split": "train",
            "samples": int(len(member_df)),
        },
        "non_member_source": {
            "source": args.non_member_source,
            "hospitals": [OOD_HOSPITAL] if args.non_member_source == "ood_test" else ID_HOSPITALS,
            "split": "test",
            "samples": int(len(non_member_df)),
        },
        "mia": mia_results,
    }

    save_json(summary, output_dir / "mia_results.json")

    logger.info("MIA balanced accuracy: %.4f", mia_results["mia_vulnerability_balanced_accuracy"])
    logger.info("Attack AUC: %.4f", mia_results["attack_auc"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
