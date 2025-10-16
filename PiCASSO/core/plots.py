from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit


def _fit_sine_curve(x_data: np.ndarray, y_data: np.ndarray, max_iter: int = 2000, smooth_points: int = 200) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float, float]]:
    def sine_func(x, amplitude, phase, offset):
        if np.max(x_data) <= 2 * np.pi + 0.1:
            return amplitude * np.sin(x + phase) + offset
        period = len(x_data)
        return amplitude * np.sin(2 * np.pi * x / period + phase) + offset
    amplitude_guess = (np.max(y_data) - np.min(y_data)) / 2
    offset_guess = float(np.mean(y_data))
    phase_guess = 0.0
    try:
        popt, _ = curve_fit(
            sine_func,
            x_data,
            y_data,
            p0=[amplitude_guess, phase_guess, offset_guess],
            maxfev=max_iter,
        )
    except Exception:
        popt = [amplitude_guess, phase_guess, offset_guess]
    amplitude, phase, offset = popt
    x_smooth = np.linspace(x_data[0], x_data[-1], smooth_points)
    y_smooth = sine_func(x_smooth, amplitude, phase, offset)
    y_pred = sine_func(x_data, amplitude, phase, offset)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return x_smooth, y_smooth, r_squared, tuple(popt)


def phase_scatter(x_shift: np.ndarray, y: np.ndarray, title: str, out_png: str):
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(x_shift, y, s=28, alpha=0.85, edgecolors='white', linewidths=0.4, color='tab:blue')
    plt.ylabel('Collection Time')
    plt.xlabel('Predicted Phase')
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def circadian_single_dataset(results: Dict, circadian_genes: List[str], out_dir: str):
    """Create per-gene visualization for single dataset across configs.
    results: { config_name: { 'ranks', 'circadian_expressions', 'metrics' } }
    """
    if not circadian_genes:
        return
    n_genes = len(circadian_genes)
    n_configs = len(results)
    fig, axes = plt.subplots(n_configs, n_genes, figsize=(3 * n_genes, 3 * n_configs))
    if n_configs == 1 and n_genes == 1:
        axes = np.array([[axes]])
    elif n_configs == 1:
        axes = np.array([axes]).reshape(1, n_genes)
    elif n_genes == 1:
        axes = np.array([axes]).reshape(n_configs, 1)
    colors = plt.cm.Set3(np.linspace(0, 1, n_genes))
    for i, (cfg_name, res) in enumerate(results.items()):
        ranks = res['ranks']
        circ = res['circadian_expressions']
        metrics = res['metrics']
        order = np.argsort(ranks.flatten())
        circ_ord = circ[order]
        n_samples = len(circ_ord)
        for j, gname in enumerate(circadian_genes):
            ax = axes[i, j]
            x = np.arange(1, n_samples + 1)
            y = circ_ord[:, j]
            ax.scatter(x, y, color='black', s=25, alpha=0.75, edgecolors='white', linewidth=0.5, zorder=3)
            xs, ys, r2, _ = _fit_sine_curve(x, y)
            ax.plot(xs, ys, '-', color=colors[j], linewidth=2.5, alpha=0.8, zorder=2)
            bal = metrics['balance_score']
            ax.set_title(f'{cfg_name}: {gname}\nBalance={bal:.3f}, R²={r2:.3f}', fontsize=10, fontweight='bold', pad=8)
            ax.set_xlabel('Sample Rank', fontsize=9)
            ax.set_ylabel('Expression', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
            ax.set_xlim(0, n_samples + 1)
            y_min, y_max = float(np.min(y)), float(np.max(y))
            yr = y_max - y_min
            if yr > 0:
                ax.set_ylim(y_min - 0.1 * yr, y_max + 0.1 * yr)
            else:
                ax.set_ylim(-0.1, 0.1)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'circadian_genes_single_dataset.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return out_path
