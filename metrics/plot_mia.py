from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def plot_attack_metrics(summary: dict, output_dir: Path):
    mia = summary["mia"]

    labels = ["Balanced Accuracy", "Attack AUC"]
    values = [
        mia.get("mia_vulnerability_balanced_accuracy", 0.0) or 0.0,
        mia.get("attack_auc", 0.0) or 0.0,
    ]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Membership Inference Attack Performance")
    plt.tight_layout()
    plt.savefig(output_dir / "mia_attack_metrics.png", dpi=200)
    plt.close()


def plot_loss_comparison(summary: dict, output_dir: Path):
    mia = summary["mia"]

    means = [
        mia.get("member_loss_mean", 0.0) or 0.0,
        mia.get("non_member_loss_mean", 0.0) or 0.0,
    ]
    stds = [
        mia.get("member_loss_std", 0.0) or 0.0,
        mia.get("non_member_loss_std", 0.0) or 0.0,
    ]

    plt.figure(figsize=(6, 5))
    plt.bar(["Member", "Non-member"], means, yerr=stds, capsize=5)
    plt.ylabel("Mean Loss")
    plt.title("Member vs Non-member Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "mia_member_vs_nonmember_loss.png", dpi=200)
    plt.close()


def plot_membership_rates(summary: dict, output_dir: Path):
    mia = summary["mia"]

    labels = ["Member TPR", "Non-member TNR"]
    values = [
        mia.get("member_tpr", 0.0) or 0.0,
        mia.get("non_member_tnr", 0.0) or 0.0,
    ]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("Membership Attack Operating Characteristics")
    plt.tight_layout()
    plt.savefig(output_dir / "mia_operating_characteristics.png", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MIA evaluation results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="metrics/results",
        help="Directory containing mia_results.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary_path = results_dir / "mia_results.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"MIA results file not found: {summary_path}")

    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_path)

    plot_attack_metrics(summary, output_dir)
    plot_loss_comparison(summary, output_dir)
    plot_membership_rates(summary, output_dir)

    print(f"MIA plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
