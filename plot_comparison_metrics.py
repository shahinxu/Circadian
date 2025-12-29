#!/usr/bin/env python3
"""Generate scatter plots comparing CYCLOPS and TOAST metrics by dataset group."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "comparison_cyclops_vs_toast.csv"
GROUP_OUTPUTS = {
    "GSE54651": ROOT / "comparison_cyclops_vs_ours_GSE54651.png",
    "GSE54652": ROOT / "comparison_cyclops_vs_ours_GSE54652.png",
    "GTEx": ROOT / "comparison_cyclops_vs_ours_GTEx.png",
}
METRICS = [
    ("circular_r_abs_cyclops", "circular_r_abs_toast", "Circular Correlation |r|"),
    ("spearman_r_cyclops", "spearman_r_toast", "Spearman Correlation"),
]


def group_from_dataset(dataset: str) -> str:
    prefix = dataset.split("/")[0]
    if prefix in GROUP_OUTPUTS:
        return prefix
    if prefix.startswith("GTEx"):
        return "GTEx"
    return "Other"


def plot_group(name: str, group_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
    if len(METRICS) == 1:
        axes = [axes]

    for ax, (cyclops_col, toast_col, metric_label) in zip(axes, METRICS):
        subset = group_df.dropna(subset=[cyclops_col, toast_col])
        if subset.empty:
            ax.axis("off")
            continue

        ax.scatter(
            subset[cyclops_col],
            subset[toast_col],
            s=60,
            color="#1f77b4",
            edgecolors="white",
            linewidth=0.5,
        )
        ax.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"CYCLOPS {metric_label}")
        ax.set_ylabel(f"TOAST {metric_label}")
        ax.set_title(metric_label)

    fig.suptitle(f"CYCLOPS vs TOAST ({name})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(GROUP_OUTPUTS[name], dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["group"] = df["dataset"].apply(group_from_dataset)

    for group_name, output_path in GROUP_OUTPUTS.items():
        subset = df[df["group"] == group_name]
        if subset.empty:
            continue
        plot_group(group_name, subset)
        print(f"Saved plot for {group_name} to {output_path}")


if __name__ == "__main__":
    main()
