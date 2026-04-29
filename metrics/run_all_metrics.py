from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from common import PROJECT_ROOT, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOD evaluation and MIA evaluation.")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="fl_logs/master/checkpoints/global_final.pt",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="metrics/results",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mia_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def run_command(cmd):
    logger.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    setup_logging()
    args = parse_args()

    metrics_dir = PROJECT_ROOT / "metrics"

    run_ood = metrics_dir / "run_ood.py"
    run_mia = metrics_dir / "run_mia.py"

    common_args = [
        "--checkpoint",
        args.checkpoint,
        "--output_dir",
        args.output_dir,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
    ]

    run_command([sys.executable, str(run_ood), *common_args])

    run_command([
        sys.executable,
        str(run_mia),
        *common_args,
        "--mia_samples",
        str(args.mia_samples),
        "--seed",
        str(args.seed),
    ])

    logger.info("All metrics completed.")


if __name__ == "__main__":
    main()
