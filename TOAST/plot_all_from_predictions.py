import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_GENES = [
    "ARNTL", "CLOCK", "NPAS2", "NR1D1", "BHLHE41", "NR1D2", "DBP",
    "CIART", "PER1", "PER3", "TEF", "HLF", "CRY2", "PER2", "CRY1",
    "RORC", "NFIL3"
]

# Ideal (mouse) acrophases used in original CYCLOPS.jl (radians),
# ordered to match DEFAULT_GENES above.
CANONICAL_MOUSE_ACROPHASES_RAD = {
    "ARNTL": 0.0,
    "CLOCK": 0.0790637050481884,
    "NPAS2": 0.151440116812406,
    "NR1D1": 2.29555301890004,
    "BHLHE41": 2.90900605826091,
    "NR1D2": 2.98706493493206,
    "DBP": 2.99149022777511,
    "CIART": 3.00769248308471,
    "PER1": 3.1219769314524,
    "PER3": 3.3058682224604,
    "TEF": 3.31357155959037,
    "HLF": 3.42557704861225,
    "CRY2": 3.50078722833753,
    "PER2": 3.88658015146741,
    "CRY1": 4.99480367551318,
    "RORC": 5.04951134876313,
    "NFIL3": 6.00770260397838,
}


def find_latest_subdir(base_path):
    """Find the most recent timestamp subdirectory"""
    if not base_path.exists():
        raise FileNotFoundError(f"Results base path not found: {base_path}")
    subs = [p for p in base_path.iterdir() if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No timestamp subdirectories found under: {base_path}")
    return sorted(subs)[-1]


def map_results_to_data_path(results_base):
    """Map results path to data/expression.csv"""
    parts = list(results_base.resolve().parts)
    idx = None
    for i, p in enumerate(parts):
        if p.lower() == 'results':
            idx = i
            break
    if idx is None:
        raise ValueError(f"Given results path does not contain a 'results' segment: {results_base}")
    
    # Try Circadian parent directory
    try:
        circ_idx = next(i for i, p in enumerate(parts) if p == 'Circadian')
        project_root = Path(*parts[:circ_idx+1])
        dataset_parts = parts[idx+1:]
        candidate = project_root / 'data' / Path(*dataset_parts)
        expr_file = candidate / 'expression.csv'
        if expr_file.exists():
            return expr_file
    except StopIteration:
        pass
    
    # Fallback: replace results with data
    data_parts = parts[:idx] + ['data'] + parts[idx+1:]
    data_path = Path(*data_parts)
    expr_file = data_path / 'expression.csv'
    if expr_file.exists():
        return expr_file
    
    raise FileNotFoundError(f"Mapped expression.csv not found. Tried: {expr_file}")


def load_expression_for_genes(expr_file, genes_upper, sample_order):
    """Load expression data for specified genes"""
    df = pd.read_csv(expr_file, low_memory=False)
    if 'Gene_Symbol' not in df.columns:
        raise ValueError(f"expression.csv missing 'Gene_Symbol' column: {expr_file}")
    
    df['GENE_UP'] = df['Gene_Symbol'].astype(str).str.upper()
    sample_cols = [c for c in sample_order if c in df.columns]
    if not sample_cols:
        sample_cols = [c for c in df.columns if c != 'Gene_Symbol' and c != 'GENE_UP']
    
    for c in sample_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    grp = df.groupby('GENE_UP')[sample_cols].mean()
    
    data = {}
    for g in genes_upper:
        if g in grp.index:
            row = grp.loc[g].reindex(sample_order).values
        else:
            row = np.full(len(sample_order), np.nan)
        data[g] = row
    
    return pd.DataFrame(data, index=sample_order)


def fit_cosine_and_get_acrophase(times, expression):
    """Fit cosine curve and return amplitude and phase"""
    mask = ~np.isnan(expression) & np.isfinite(times)
    if mask.sum() < 3:
        return np.nan, np.nan
    
    x = times[mask]
    y = expression[mask]
    
    theta = x * (2 * np.pi / 24.0)
    A = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    
    try:
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return np.nan, np.nan
    
    a, b, c = coeffs
    amp = np.hypot(b, c)
    phase = float(np.arctan2(c, b) % (2 * np.pi))
    
    return amp, phase


def plot_time_expression(expr_by_gene, times, genes, out_png):
    """Plot time-expression curves with fitted cosine for each gene"""
    n = len(genes)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes
    
    x_grid = np.linspace(0, 24, 300)
    theta_grid = x_grid * (2 * np.pi / 24.0)
    
    fit_results = []
    
    for i, g in enumerate(genes):
        ax = axes[i]
        y_full = expr_by_gene[g].values.astype(float)
        mask = ~np.isnan(y_full) & np.isfinite(times)
        
        if mask.sum() == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', transform=ax.transAxes)
            ax.set_title(g)
            continue
        
        x = times[mask]
        y = y_full[mask]
        
        theta = x * (2 * np.pi / 24.0)
        A = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
        
        try:
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs
            amp = np.hypot(b, c)
            phase = float(np.arctan2(c, b) % (2 * np.pi))
            fitted = a + b * np.cos(theta_grid) + c * np.sin(theta_grid)
            
            ax.scatter(x, y, s=20, alpha=0.8)
            ax.plot(x_grid, fitted, color='C1', linewidth=1.5)
            ax.set_title(f"{g} (amp={amp:.2f}, φ={phase:.2f})")
            
            fit_results.append({'Gene': g, 'amplitude': amp, 'phase_rad': phase})
        except Exception:
            ax.scatter(x, y, s=20, alpha=0.8)
            ax.set_title(f"{g} (fit failed)")
        
        ax.set_xlabel('Phase (hours)')
    
    # Hide empty subplots
    for i in range(len(genes), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")
    
    return pd.DataFrame(fit_results)


def plot_acrophase_circle(fit_df, out_png, title=None):
    """Plot gene acrophases on circular plot"""
    if fit_df.empty:
        print(f"No data to plot for {out_png}")
        return
    
    genes = fit_df['Gene'].astype(str).tolist()
    genes_upper = [g.upper() for g in genes]
    phases = fit_df['phase_rad'].astype(float).values
    amps = fit_df['amplitude'].astype(float).values
    
    # Normalize amplitudes for marker size (outer ring)
    amp_norm = np.nan_to_num(amps, nan=0.0)
    if amp_norm.max() > 0:
        sizes = 80 + 320 * (amp_norm / amp_norm.max())
    else:
        sizes = np.full_like(amp_norm, 120.0)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Match overall style to CYCLOPS Acrophase plots
    ax.set_theta_zero_location('N')  # 0 at top
    ax.set_theta_direction(-1)       # clockwise
    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["", "", ""])
    # Keep radial gridlines (grey circles), hide angular gridlines to match CYCLOPS style
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Outer ring: estimated acrophases from TOAST fits (r=1)
    outer_r = np.ones_like(phases)
    ax.scatter(
        phases,
        outer_r,
        s=sizes,
        c="C0",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
        label="Estimated",
        zorder=3,
    )

    # Inner ring: canonical mouse acrophases from CYCLOPS (if available)
    inner_phases = []
    inner_labels = []
    for g_up in genes_upper:
        if g_up in CANONICAL_MOUSE_ACROPHASES_RAD:
            inner_phases.append(CANONICAL_MOUSE_ACROPHASES_RAD[g_up])
            inner_labels.append(g_up)

    if inner_phases:
        inner_phases = np.array(inner_phases, dtype=float)
        # Place canonical phases exactly on the middle radial gridline (r=0.5)
        inner_r = np.full_like(inner_phases, 0.5)
        ax.scatter(
            inner_phases,
            inner_r,
            s=55,
            c="orange",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
            label="Canonical",
            zorder=3,
        )

        # Label canonical genes outside their points with CYCLOPS-like spacing along the ring
        inner_label_r = float(inner_r[0]) + 0.25
        # Use a dynamic minimum angular separation that grows with gene count
        base_space = np.pi / 35.5
        n_inner = inner_phases.size
        # Target roughly uniform coverage if all labels were on one ring
        dynamic_space = 2 * np.pi / (n_inner + 2) * 0.6
        space_factor = max(base_space, dynamic_space)

        inner_labels = np.array(inner_labels, dtype=str)

        # Choose "middle" gene near circular mean of canonical phases
        # Use complex mean to get circular mean angle
        mean_vec = np.exp(1j * inner_phases).mean()
        mean_angle = float(np.angle(mean_vec) % (2 * np.pi))
        # Find gene whose canonical phase is closest to the mean
        ang_diff = np.arccos(np.cos(inner_phases - mean_angle))
        middle_idx = int(np.argmin(ang_diff))
        middle_phase = inner_phases[middle_idx]

        # Annotate the middle canonical gene straight outwards
        ax.annotate(
            inner_labels[middle_idx],
            xy=(middle_phase, inner_r[middle_idx]),
            xytext=(middle_phase, inner_label_r),
            ha="center",
            va="center",
            fontsize=6,
            arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.6),
        )

        # Split remaining genes to the two sides of the middle gene (by angular distance sign)
        dist_from_middle = middle_phase - inner_phases
        larger_mask = dist_from_middle < 0   # phases larger than middle
        smaller_mask = dist_from_middle > 0  # phases smaller than middle

        # Handle larger-than-middle side: move labels forward to keep at least space_factor separation
        larger_indices = np.where(larger_mask)[0]
        if larger_indices.size > 0:
            larger_phases = inner_phases[larger_indices]
            order = np.argsort(larger_phases)  # ascending
            sorted_phases = larger_phases[order]
            sorted_idx = larger_indices[order]

            prev_angle = middle_phase
            for ph, idx in zip(sorted_phases, sorted_idx):
                label_angle = ph
                if label_angle - prev_angle < space_factor:
                    label_angle = prev_angle + space_factor
                # keep angles in [0, 2π)
                label_angle = float(label_angle % (2 * np.pi))
                ax.annotate(
                    inner_labels[idx],
                    xy=(ph, inner_r[idx]),
                    xytext=(label_angle, inner_label_r),
                    ha="center",
                    va="center",
                    fontsize=6,
                    arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.6),
                )
                prev_angle = label_angle

        # Handle smaller-than-middle side: move labels backward to keep at least space_factor separation
        smaller_indices = np.where(smaller_mask)[0]
        if smaller_indices.size > 0:
            smaller_phases = inner_phases[smaller_indices]
            order = np.argsort(smaller_phases)[::-1]  # descending
            sorted_phases = smaller_phases[order]
            sorted_idx = smaller_indices[order]

            prev_angle = middle_phase
            for ph, idx in zip(sorted_phases, sorted_idx):
                label_angle = ph
                if prev_angle - label_angle < space_factor:
                    label_angle = prev_angle - space_factor
                # keep angles in [0, 2π)
                label_angle = float(label_angle % (2 * np.pi))
                ax.annotate(
                    inner_labels[idx],
                    xy=(ph, inner_r[idx]),
                    xytext=(label_angle, inner_label_r),
                    ha="center",
                    va="center",
                    fontsize=6,
                    arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.6),
                )
                prev_angle = label_angle

    # Label estimated genes just outside outer ring
    label_r = 1.12
    for g, th in zip(genes_upper, phases):
        ax.text(th, label_r, g, ha='center', va='center', fontsize=8)

    # Phase ticks in hours
    hours = np.array([0, 6, 12, 18])
    ticks = hours * (2 * np.pi / 24.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{h}" for h in hours])

    # Legend (only if both rings exist)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05))
    
    if title:
        plt.title(title)
    else:
        plt.title('Gene Acrophases')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")


def plot_acrophase_comparison(all_fit_data, genes, out_png):
    """Plot acrophase comparison across cell types"""
    if not all_fit_data:
        print("No data for comparison plot")
        return
    
    n = len(genes)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             subplot_kw={'projection': 'polar'})
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_fit_data)))
    
    for i, gene in enumerate(genes):
        ax = axes[i]
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        for j, (celltype, fit_df) in enumerate(all_fit_data.items()):
            gene_data = fit_df[fit_df['Gene'].str.upper() == gene.upper()]
            if not gene_data.empty:
                phase = gene_data.iloc[0]['phase_rad']
                amp = gene_data.iloc[0]['amplitude']
                r = 0.3 + 0.7 * min(amp / 2.0, 1.0)
                ax.scatter(phase, r, c=[colors[j]], s=80, label=celltype, zorder=3)
        
        ax.set_title(gene, fontsize=10)
        ax.set_rticks([])
        hours = np.array([0, 6, 12, 18])
        ticks = hours * (2 * np.pi / 24.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{h}" for h in hours], fontsize=8)
    
    # Hide empty subplots
    for i in range(len(genes), len(axes)):
        axes[i].axis('off')
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle('Acrophase Comparison Across Cell Types', fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")


def process_single_celltype(results_base, genes):
    """Process single cell type and generate all plots"""
    results_base = Path(results_base)
    latest = find_latest_subdir(results_base)
    print(f"Processing: {latest}")
    
    # Load predictions - prioritize aligned version
    preds_aligned = latest / 'predictions_aligned.csv'
    preds_file = latest / 'predictions.csv'
    
    if preds_aligned.exists():
        preds = pd.read_csv(preds_aligned)
        print(f"Using aligned predictions from: {preds_aligned.name}")
    elif preds_file.exists():
        preds = pd.read_csv(preds_file)
        print(f"Using unaligned predictions from: {preds_file.name}")
    else:
        raise FileNotFoundError(f"No predictions.csv or predictions_aligned.csv in {latest}")
    
    if 'Sample_ID' not in preds.columns:
        raise ValueError('predictions file must contain Sample_ID column')
    
    # Get predicted phase
    if 'Predicted_Phase_Hours_Aligned' in preds.columns:
        times = pd.to_numeric(preds['Predicted_Phase_Hours_Aligned'], errors='coerce').values
    elif 'Predicted_Phase_Hours' in preds.columns:
        times = pd.to_numeric(preds['Predicted_Phase_Hours'], errors='coerce').values
    else:
        raise ValueError('predictions.csv missing phase columns')
    
    sample_order = preds['Sample_ID'].astype(str).tolist()
    
    # Load expression
    expr_file = map_results_to_data_path(results_base)
    print(f"Loading expression from: {expr_file}")
    
    genes_upper = [g.upper() for g in genes]
    expr_by_gene = load_expression_for_genes(expr_file, genes_upper, sample_order)
    
    # Plot 1: Time-expression curves
    out_png1 = latest / 'expression_vs_phase.png'
    fit_df = plot_time_expression(expr_by_gene, times, genes_upper, out_png1)
    
    # Plot 2: Acrophase circle
    out_png2 = latest / 'acrophase_circle.png'
    celltype_name = results_base.name
    plot_acrophase_circle(fit_df, out_png2, title=f'{celltype_name} Acrophases')
    
    return fit_df


def process_comparison_mode(results_base, genes):
    """Process multiple cell types and generate comparison plots"""
    results_base = Path(results_base)
    
    # Find all subdirectories (cell types)
    celltypes = sorted([d.name for d in results_base.iterdir() if d.is_dir()])
    print(f"Found cell types: {celltypes}")
    
    all_fit_data = {}
    
    for celltype in celltypes:
        celltype_path = results_base / celltype
        try:
            fit_df = process_single_celltype(celltype_path, genes)
            all_fit_data[celltype] = fit_df
        except Exception as e:
            print(f"Warning: Failed to process {celltype}: {e}")
    
    if not all_fit_data:
        print("No cell types processed successfully")
        return
    
    # Plot comparison
    out_dir = results_base.parent / f"{results_base.name}_comparison"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / 'acrophase_comparison.png'
    plot_acrophase_comparison(all_fit_data, genes, out_png)


def main():
    parser = argparse.ArgumentParser(description='Generate all plots from predictions.csv')
    parser.add_argument('--results_base', required=True,
                        help='Path to results base (single cell type or dataset folder)')
    parser.add_argument('--genes', nargs='*', default=None,
                        help='List of genes to plot (default: clock genes)')
    parser.add_argument('--compare_mode', action='store_true',
                        help='Compare multiple cell types (results_base should contain subdirs)')
    args = parser.parse_args()
    
    genes = args.genes if args.genes else DEFAULT_GENES
    
    if args.compare_mode:
        process_comparison_mode(args.results_base, genes)
    else:
        process_single_celltype(args.results_base, genes)
    
    print("\nDone! All plots generated.")


if __name__ == '__main__':
    main()
