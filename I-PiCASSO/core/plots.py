from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def component_expression_scatter(ranks: np.ndarray, values: np.ndarray, title: str, out_png: str, color: str = 'tab:blue'):
    ranks = np.asarray(ranks).astype(float)
    values = np.asarray(values).astype(float)
    if ranks.shape != values.shape:
        raise ValueError("ranks 与 component values 长度不一致")
    order = np.argsort(ranks)
    x_sorted = ranks[order]
    y_sorted = values[order]
    plt.figure(figsize=(7, 4.8))
    plt.scatter(x_sorted, y_sorted, s=18, alpha=0.75, edgecolors='none', color=color)
    plt.xlabel('Predicted Rank')
    plt.ylabel('Expression')
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def phase_scatter(x_shifted: np.ndarray, y_hours: np.ndarray, title: str, out_png: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(x_shifted, y_hours, s=16, alpha=0.8, edgecolors='none')
    plt.xlabel('Predicted Phase (shifted)')
    plt.ylabel('Metadata Time (h)')
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
