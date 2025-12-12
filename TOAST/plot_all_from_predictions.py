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
    phases = fit_df['phase_rad'].astype(float).values
    amps = fit_df['amplitude'].astype(float).values
    
    amp_norm = np.nan_to_num(amps, nan=0.0)
    if amp_norm.max() > 0:
        radii = 0.3 + 0.7 * (amp_norm / amp_norm.max())
    else:
        radii = np.full_like(amp_norm, 0.6)
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    sc = ax.scatter(phases, radii, c=radii, cmap='viridis', s=60, zorder=3)
    
    for (g, th, r) in zip(genes, phases, radii):
        label_r = r + 0.06
        ax.text(th, label_r, g, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
    
    ax.set_rmax(1.05)
    ax.set_rticks([])
    
    hours = np.array([0, 4, 8, 12, 16, 20])
    ticks = hours * (2 * np.pi / 24.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{h}/24" if h == 0 else f"{h}" for h in hours])
    
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
