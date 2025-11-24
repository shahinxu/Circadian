#!/usr/bin/env python3
"""
Plot time-expression curves for a fixed set of clock genes using the
latest `predictions_aligned.csv` under a results folder.

Usage:
  python plot_aligned_expression.py --results_base "D:/.../GREED/results/GTEx/GTEx_prostate"

The script will:
 - find the most recent timestamp subfolder under the given results base
 - load `predictions_aligned.csv` (fallback to `predictions.csv`)
 - map the results folder to the corresponding `data/.../expression.csv`
 - select the requested genes (case-insensitive) and average duplicate probes
 - plot expression vs `Predicted_Phase_Hours_Aligned` for each gene and save a multi-panel figure
 - save per-gene CSVs with columns `Sample_ID, Predicted_Phase_Hours_Aligned, Expression`
"""

import argparse
import os
import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_GENES = [
    "Arntl", "Clock", "Npas2", "Nr1d1", "Bhlhe41", "Nr1d2", "Dbp",
    "Ciart", "Per1", "Per3", "Tef", "Hlf", "Cry2", "Per2", "Cry1",
    "Rorc", "Nfil3"
]


def find_latest_subdir(base_path: Path) -> Path:
    if not base_path.exists():
        raise FileNotFoundError(f"Results base path not found: {base_path}")
    # List subdirectories only
    subs = [p for p in base_path.iterdir() if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No timestamp subdirectories found under: {base_path}")
    # Timestamp folders are ISO-like; lexicographic sort works
    subs_sorted = sorted(subs)
    return subs_sorted[-1]


def map_results_to_data_path(results_base: Path) -> Path:
    """Given a path that includes a 'results' segment, replace 'results' with 'data'
    to construct the corresponding data folder path and expression.csv file.
    """
    parts = list(results_base.resolve().parts)
    # find the index of 'results' (case-insensitive)
    idx = None
    for i, p in enumerate(parts):
        if p.lower() == 'results':
            idx = i
            break
    if idx is None:
        raise ValueError("Given results path does not contain a 'results' segment: %s" % results_base)

    # First attempt: project root is the ancestor before 'GREED' (common layout)
    try:
        gre_idx = next(i for i, p in enumerate(parts) if p == 'GREED')
        project_root = Path(*parts[:gre_idx])
        candidate = project_root / 'data' / Path(*parts[idx+1:])
        expr_file = candidate / 'expression.csv'
        if expr_file.exists():
            return expr_file
    except StopIteration:
        pass

    # Second attempt: replace the 'results' segment with 'data' at the same level
    data_parts = parts[:idx] + ['data'] + parts[idx+1:]
    data_path = Path(*data_parts)
    expr_file = data_path / 'expression.csv'
    if expr_file.exists():
        return expr_file

    # If neither worked, raise with helpful diagnostics
    raise FileNotFoundError(f"Mapped expression.csv not found. Tried: {expr_file}")


def load_expression_for_genes(expr_file: Path, genes_upper: list, sample_order: list):
    df = pd.read_csv(expr_file, low_memory=False)
    if 'Gene_Symbol' not in df.columns:
        raise ValueError(f"expression.csv missing 'Gene_Symbol' column: {expr_file}")
    # Uppercase gene symbols for robust matching
    df['GENE_UP'] = df['Gene_Symbol'].astype(str).str.upper()
    # Select sample columns that appear in sample_order (preserve order)
    sample_cols = [c for c in sample_order if c in df.columns]
    if not sample_cols:
        # fallback: treat all non-Gene_Symbol columns as sample columns
        sample_cols = [c for c in df.columns if c != 'Gene_Symbol' and c != 'GENE_UP']

    # Ensure sample columns are numeric (coerce non-numeric to NaN) then group and average
    try:
        df[sample_cols] = df[sample_cols].apply(lambda col: pd.to_numeric(col, errors='coerce'))
    except Exception:
        # fallback: try column-wise coercion
        for c in sample_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # group by GENE_UP and average duplicates, then reindex to sample_order
    grp = df.groupby('GENE_UP')[sample_cols].mean()

    # For genes not present, create NaNs
    data = {}
    for g in genes_upper:
        if g in grp.index:
            row = grp.loc[g].reindex(sample_order).values
        else:
            row = np.full(len(sample_order), np.nan)
        data[g] = row

    expr_by_gene = pd.DataFrame(data, index=sample_order)
    return expr_by_gene


def plot_genes(expr_by_gene: pd.DataFrame, times: np.ndarray, out_png: Path, genes: list):
    """
    For each gene, fit a cosine model:
      y = a + b*cos(theta) + c*sin(theta)
    where theta = times * 2*pi/24. Plot scatter of points and the fitted cosine curve.
    """
    n = len(genes)
    ncols = 4
    nrows = math.ceil(n / ncols)
    plt.figure(figsize=(4 * ncols, 2.8 * nrows))
    two_pi = 2 * np.pi
    # grid for plotting fitted curve across 0-24 hours
    x_grid = np.linspace(0, 24, 300)
    theta_grid = x_grid * (two_pi / 24.0)

    fit_rows = []
    for i, g in enumerate(genes):
        ax = plt.subplot(nrows, ncols, i + 1)
        y_full = expr_by_gene[g].values.astype(float)
        mask = ~np.isnan(y_full) & np.isfinite(times)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, 'no data', ha='center')
            ax.set_title(g)
            continue
        x = times[mask]
        y = y_full[mask]

        # Convert times (hours) to radians
        theta = x * (two_pi / 24.0)

        # Build design matrix [1, cos(theta), sin(theta)] and solve least squares
        A = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
        try:
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        except Exception:
            coeffs = np.array([np.nan, np.nan, np.nan])

        a, b, c = coeffs
        amp = np.hypot(b, c)
        phase = float(np.arctan2(c, b) % (2 * np.pi)) if np.isfinite(b) and np.isfinite(c) else float('nan')

        # Evaluate fitted curve on grid
        fitted = a + b * np.cos(theta_grid) + c * np.sin(theta_grid)

        # Scatter measured points
        ax.scatter(x, y, s=20, alpha=0.8)
        # Plot fitted cosine curve
        ax.plot(x_grid, fitted, color='C1', linewidth=1.5)
        ax.set_title(f"{g} (amp={amp:.2f}, phase={phase:.2f} rad)")
        ax.set_xlabel('Predicted Phase (hours)')

        fit_rows.append({'Gene': g, 'a': a, 'b': b, 'c': c, 'amplitude': amp, 'phase_rad': phase})

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # return fitted parameters
    return pd.DataFrame(fit_rows)


def plot_acrophase_circle(fit_df: pd.DataFrame, out_png: Path):
    """
    Plot gene acrophases on a circular plot. Expects fit_df with columns:
    'Gene', 'amplitude', 'phase_rad'
    """
    import matplotlib.pyplot as plt

    # Prepare
    genes = fit_df['Gene'].astype(str).tolist()
    phases = fit_df['phase_rad'].astype(float).values
    amps = fit_df['amplitude'].astype(float).values

    # normalize radii for plotting (use amplitude or 1 for visibility)
    # We'll map amplitude to radius between 0.3 and 1.0 for visibility
    amp_norm = np.nan_to_num(amps, nan=0.0)
    if amp_norm.max() > 0:
        radii = 0.3 + 0.7 * (amp_norm / (amp_norm.max()))
    else:
        radii = np.full_like(amp_norm, 0.6)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    # Make 0 at top (12 o'clock) and clockwise increasing
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Plot points
    sc = ax.scatter(phases, radii, c=radii, cmap='viridis', s=60, zorder=3)

    # Annotate genes, offset labels slightly outward
    for (g, th, r) in zip(genes, phases, radii):
        # compute label position
        label_r = r + 0.06
        ax.text(th, label_r, g, ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))

    # Grid and ticks: show 0/24 at top and hours around
    ax.set_rmax(1.05)
    ax.set_rticks([])
    # hour ticks at 0,4,8,12,16,20
    hours = np.array([0, 4, 8, 12, 16, 20])
    two_pi = 2 * np.pi
    ticks = hours * (two_pi / 24.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{h}/24" if h==0 else f"{h}" for h in hours])

    plt.title('Gene Acrophases (hours around circle)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_base', required=True,
                        help='Path to results dataset base, e.g. results/GTEx/GTEx_prostate')
    parser.add_argument('--genes', nargs='*', default=None,
                        help='List of gene symbols to plot (will be uppercased). Defaults to clock genes')
    parser.add_argument('--output', default=None, help='Output PNG path (overrides default)')
    args = parser.parse_args()

    results_base = Path(args.results_base)
    latest = find_latest_subdir(results_base)
    print(f"Using latest results dir: {latest}")

    # Look for predictions_aligned.csv first
    preds_file = latest / 'predictions_aligned.csv'
    if not preds_file.exists():
        preds_file = latest / 'predictions.csv'
        if not preds_file.exists():
            raise FileNotFoundError(f"No predictions_aligned.csv or predictions.csv in {latest}")

    preds = pd.read_csv(preds_file)
    if 'Sample_ID' not in preds.columns:
        raise ValueError('predictions file must contain Sample_ID column')

    if 'Predicted_Phase_Hours_Aligned' in preds.columns:
        times = pd.to_numeric(preds['Predicted_Phase_Hours_Aligned'], errors='coerce').values
    elif 'Predicted_Phase_Hours' in preds.columns:
        times = pd.to_numeric(preds['Predicted_Phase_Hours'], errors='coerce').values
    else:
        raise ValueError('predictions file missing Predicted_Phase_Hours_Aligned or Predicted_Phase_Hours')

    sample_order = preds['Sample_ID'].astype(str).tolist()

    # Map results folder to expression.csv in data
    expr_file = map_results_to_data_path(results_base)
    print(f"Mapped expression file: {expr_file}")

    # genes
    genes = args.genes if args.genes else DEFAULT_GENES
    genes_upper = [g.upper() for g in genes]

    expr_by_gene = load_expression_for_genes(expr_file, genes_upper, sample_order)

    # Save per-gene CSVs
    out_csv_dir = latest / 'aligned_expression_by_gene'
    out_csv_dir.mkdir(exist_ok=True)
    for g in genes_upper:
        df = pd.DataFrame({
            'Sample_ID': sample_order,
            'Predicted_Phase_Hours_Aligned': times,
            'Expression': expr_by_gene[g].values
        })
        df.to_csv(out_csv_dir / f'{g}.csv', index=False)
    print(f"Saved per-gene CSVs to: {out_csv_dir}")

    # Plot time-expression with cosine fits and collect fit parameters
    out_png = Path(args.output) if args.output else (latest / 'aligned_expression_plots.png')
    fit_df = plot_genes(expr_by_gene, times, out_png, genes_upper)
    print(f"Saved plot: {out_png}")

    # Save fit parameters
    fit_params_csv = latest / 'fit_parameters.csv'
    fit_df.to_csv(fit_params_csv, index=False)
    print(f"Saved fit parameters: {fit_params_csv}")

    # Plot circular acrophase positions
    circle_png = latest / 'aligned_acrophases_circle.png'
    plot_acrophase_circle(fit_df, circle_png)
    print(f"Saved acrophase circle plot: {circle_png}")


if __name__ == '__main__':
    main()
