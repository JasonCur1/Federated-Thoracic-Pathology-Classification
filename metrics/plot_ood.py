from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def plot_id_vs_ood(summary: dict, output_dir: Path):
    id_metrics = summary["id_test"]
    ood_metrics = summary["ood_test"]

    metric_names = ["auroc_macro", "f1_macro", "recall_macro", "precision_macro"]
    labels = ["AUROC", "F1", "Recall", "Precision"]

    id_values = [id_metrics.get(m, 0.0) or 0.0 for m in metric_names]
    ood_values = [ood_metrics.get(m, 0.0) or 0.0 for m in metric_names]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], id_values, width=width, label="ID")
    plt.bar([i + width / 2 for i in x], ood_values, width=width, label="OOD")

    plt.xticks(list(x), labels)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("ID vs OOD Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ood_id_vs_ood_metrics.png", dpi=200)
    plt.close()


def plot_per_class_auroc(summary: dict, output_dir: Path):
    id_per_class = summary["id_test"].get("per_class_auroc", {})
    ood_per_class = summary["ood_test"].get("per_class_auroc", {})

    labels = []
    id_values = []
    ood_values = []

    for disease, id_val in id_per_class.items():
        ood_val = ood_per_class.get(disease)
        if id_val is None and ood_val is None:
            continue
        labels.append(disease)
        id_values.append(0.0 if id_val is None else id_val)
        ood_values.append(0.0 if ood_val is None else ood_val)

    if not labels:
        return

    x = range(len(labels))
    width = 0.4

    plt.figure(figsize=(14, 6))
    plt.bar([i - width / 2 for i in x], id_values, width=width, label="ID")
    plt.bar([i + width / 2 for i in x], ood_values, width=width, label="OOD")

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("AUROC")
    plt.ylim(0, 1)
    plt.title("Per-Class AUROC: ID vs OOD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ood_per_class_auroc.png", dpi=200)
    plt.close()


def plot_generalization_gap(summary: dict, output_dir: Path):
    id_auroc = summary["id_test"].get("auroc_macro", 0.0) or 0.0
    ood_auroc = summary["ood_test"].get("auroc_macro", 0.0) or 0.0
    gap = summary.get("ood_generalization_gap", {}).get("auroc_gap_id_minus_ood", None)

    plt.figure(figsize=(6, 5))
    plt.bar(["ID AUROC", "OOD AUROC"], [id_auroc, ood_auroc])
    plt.ylim(0, 1)
    plt.ylabel("AUROC")
    title = "OOD Generalization"
    if gap is not None:
        title += f"\nGap (ID - OOD) = {gap:.4f}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / "ood_generalization_gap.png", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot OOD evaluation results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="metrics/results",
        help="Directory containing ood_summary.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary_path = results_dir / "ood_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"OOD summary file not found: {summary_path}")

    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_path)

    plot_id_vs_ood(summary, output_dir)
    plot_per_class_auroc(summary, output_dir)
    plot_generalization_gap(summary, output_dir)

    print(f"OOD plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
