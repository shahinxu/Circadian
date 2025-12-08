#!/usr/bin/env python3
"""
Plot expression patterns of core clock genes aligned by predicted phases from CYCLOPS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Core clock genes used for alignment (human gene names)
CORE_GENES = ["ARNTL", "CLOCK", "NPAS2", "NR1D1", "BHLHE41", "NR1D2", 
              "DBP", "CIART", "PER1", "PER3", "TEF", "HLF", 
              "CRY2", "PER2", "CRY1", "RORC", "NFIL3"]

def load_data(result_dir):
    """Load CYCLOPS results and expression data"""
    result_path = Path(result_dir)
    
    # Load predicted phases
    phase_file = list(result_path.glob("Fit_Output_*.csv"))[0]
    phases_df = pd.read_csv(phase_file)
    phases_df = phases_df.rename(columns={'ID': 'Sample'})
    phases_df = phases_df[['Sample', 'Phase']].copy()
    phases_df['Phase'] = pd.to_numeric(phases_df['Phase'], errors='coerce')
    
    # Load expression data
    expr_file = Path(result_dir).parent.parent.parent / "data/Zhang_CancerCell_2025_all/expression.csv"
    expr_df = pd.read_csv(expr_file, low_memory=False)
    
    # First row is CellType_D
    celltype_row = expr_df.iloc[0, 1:].to_dict()
    celltype_map = {col: celltype_row[col] for col in expr_df.columns[1:]}
    
    # Remove CellType_D row for expression
    expr_df = expr_df.iloc[1:].copy()
    
    # Set Gene_Symbol as index
    expr_df = expr_df.rename(columns={expr_df.columns[0]: 'Gene_Symbol'})
    expr_df = expr_df.set_index('Gene_Symbol')
    
    # Convert to numeric
    for col in expr_df.columns:
        expr_df[col] = pd.to_numeric(expr_df[col], errors='coerce')
    
    return phases_df, expr_df, celltype_map

def plot_core_genes_expression(phases_df, expr_df, celltype_map, output_dir):
    """Plot aligned expression for core genes, separated by cell type"""
    
    # Filter for core genes present in data
    available_genes = [g for g in CORE_GENES if g in expr_df.index]
    print(f"Found {len(available_genes)}/{len(CORE_GENES)} core genes in data:")
    print(", ".join(available_genes))
    
    if len(available_genes) == 0:
        print("No core genes found in expression data!")
        return
    
    # Prepare data for plotting
    plot_data = []
    for gene in available_genes:
        gene_expr = expr_df.loc[gene]
        for sample in phases_df['Sample']:
            if sample in gene_expr.index:
                phase = phases_df[phases_df['Sample'] == sample]['Phase'].values[0]
                expression = gene_expr[sample]
                celltype = celltype_map.get(sample, 'Unknown')
                if not np.isnan(phase) and not np.isnan(expression):
                    plot_data.append({
                        'Gene': gene,
                        'Phase': phase,
                        'Expression': expression,
                        'Sample': sample,
                        'CellType': celltype
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Get unique cell types
    celltypes = sorted(plot_df['CellType'].unique())
    print(f"\nFound {len(celltypes)} cell types: {', '.join(celltypes)}")
    
    # Plot for each cell type
    for celltype in celltypes:
        print(f"\nPlotting {celltype}...")
        celltype_df = plot_df[plot_df['CellType'] == celltype].copy()
        plot_celltype_genes(celltype_df, available_genes, celltype, output_dir)

def plot_celltype_genes(plot_df, available_genes, celltype, output_dir):
    """Plot expression patterns for one cell type"""
    
    n_samples = len(plot_df['Sample'].unique())
    print(f"  {celltype}: {n_samples} samples")
    
    # Create figure with subplots for each gene
    n_genes = len(available_genes)
    n_cols = 4
    n_rows = (n_genes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_genes > 1 else [axes]
    
    for idx, gene in enumerate(available_genes):
        ax = axes[idx]
        gene_data = plot_df[plot_df['Gene'] == gene].copy()
        
        # Sort by phase
        gene_data = gene_data.sort_values('Phase')
        
        # Plot
        ax.scatter(gene_data['Phase'], gene_data['Expression'], 
                  alpha=0.6, s=30, color='steelblue')
        
        # Fit cosine curve
        phases = gene_data['Phase'].values
        expr = gene_data['Expression'].values
        
        # Create smooth curve
        phase_smooth = np.linspace(phases.min(), phases.max(), 100)
        
        # Fit: y = A*cos(phase - phi) + B
        from scipy.optimize import curve_fit
        def cosine_func(x, A, phi, B):
            return A * np.cos(x - phi) + B
        
        try:
            popt, _ = curve_fit(cosine_func, phases, expr, 
                               p0=[expr.std(), 0, expr.mean()],
                               maxfev=10000)
            expr_smooth = cosine_func(phase_smooth, *popt)
            ax.plot(phase_smooth, expr_smooth, 'r-', linewidth=2, alpha=0.7)
            
            # Add amplitude info
            amplitude = abs(popt[0])
            ax.text(0.05, 0.95, f'Amp: {amplitude:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            pass
        
        ax.set_xlabel('Phase (radians)', fontsize=10)
        ax.set_ylabel('Expression', fontsize=10)
        ax.set_title(f'{gene} (n={len(gene_data)})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 2*np.pi])
        
        # Add phase labels
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Hide unused subplots
    for idx in range(n_genes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Core Clock Genes Expression - {celltype}', 
                 fontsize=16, fontweight='bold', y=1.002)
    plt.tight_layout()
    
    # Safe filename
    safe_celltype = celltype.replace('/', '_').replace(' ', '_')
    output_file = Path(output_dir) / f'core_genes_{safe_celltype}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()
    
    # Also create a combined heatmap for this cell type
    fig, ax = plt.subplots(figsize=(15, len(available_genes)*0.8))
    
    # Create matrix for heatmap
    phase_bins = np.linspace(0, 2*np.pi, 25)
    heatmap_data = []
    
    for gene in available_genes:
        gene_data = plot_df[plot_df['Gene'] == gene].copy()
        if len(gene_data) == 0:
            row = [np.nan] * 24
        else:
            gene_data['Phase_bin'] = pd.cut(gene_data['Phase'], bins=phase_bins, 
                                             labels=False, include_lowest=True)
            
            # Average expression per bin
            binned = gene_data.groupby('Phase_bin')['Expression'].mean()
            
            # Fill missing bins with NaN
            row = [binned.get(i, np.nan) for i in range(len(phase_bins)-1)]
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=available_genes)
    
    # Z-score normalize each row for better visualization
    heatmap_df_zscore = heatmap_df.sub(heatmap_df.mean(axis=1), axis=0).div(heatmap_df.std(axis=1), axis=0)
    
    im = ax.imshow(heatmap_df_zscore, cmap='RdBu_r', aspect='auto', 
                   vmin=-2, vmax=2, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{i*2*np.pi/24:.1f}' for i in range(24)], rotation=45, ha='right')
    ax.set_yticks(range(len(available_genes)))
    ax.set_yticklabels(available_genes)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Z-score Expression')
    
    ax.set_xlabel('Phase (radians)', fontsize=12)
    ax.set_ylabel('Core Clock Genes', fontsize=12)
    ax.set_title(f'Core Clock Genes Heatmap - {celltype} (n={n_samples})', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = Path(output_dir) / f'core_genes_heatmap_{safe_celltype}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        result_dir = "/home/rzh/zhenx/Circadian/CYCLOPS-2.0/results/Zhang_CancerCell_2025_all_20251208_170431"
    
    print(f"Loading data from: {result_dir}")
    phases_df, expr_df, celltype_map = load_data(result_dir)
    
    print(f"\nLoaded {len(phases_df)} samples with predicted phases")
    print(f"Loaded {len(expr_df)} genes")
    
    plot_core_genes_expression(phases_df, expr_df, celltype_map, result_dir)
    
    print("\nDone!")
