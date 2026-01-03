#!/usr/bin/env python3
"""
Plot core gene expression for TUMOR samples only from transfer learning results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Core clock genes
CORE_GENES = ["ARNTL", "CLOCK", "NPAS2", "NR1D1", "BHLHE41", "NR1D2", 
              "DBP", "CIART", "PER1", "PER3", "TEF", "HLF", 
              "CRY2", "PER2", "CRY1", "NFIL3"]

def load_transfer_results(result_dir):
    """Load CYCLOPS transfer learning results and filter for tumor samples"""
    result_path = Path(result_dir).resolve()
    
    # Find Fit_Output CSV
    fit_files = sorted(result_path.glob("Fit_Output_*.csv"))
    if not fit_files:
        raise FileNotFoundError(f"No Fit_Output CSV in {result_dir}")
    
    fit_df = pd.read_csv(fit_files[0])
    print(f"\nLoaded: {fit_files[0].name}")
    print(f"Total samples: {len(fit_df)}")
    
    # Extract tumor samples (marked as "Zhang" in Covariate_D column)
    if 'Covariate_D' in fit_df.columns:
        tumor_samples = fit_df[fit_df['Covariate_D'] == 'Zhang'].copy()
        print(f"Tumor (Zhang) samples: {len(tumor_samples)}")
        print(f"Normal (GTEx) samples: {len(fit_df[fit_df['Covariate_D'] == 'GTEx'])}")
    else:
        print("Warning: No Covariate_D column found, using all samples")
        tumor_samples = fit_df.copy()
    
    # Get sample IDs and phases
    tumor_samples = tumor_samples.rename(columns={'ID': 'Sample'})
    tumor_samples['Phase'] = pd.to_numeric(tumor_samples['Phase'], errors='coerce')
    
    return tumor_samples[['Sample', 'Phase', 'Covariate_D']].copy()

def load_expression_data(result_dir, dataset_name):
    """Load expression data for the tumor dataset"""
    # Get base path by finding CYCLOPS-2.0 directory
    result_path = Path(result_dir).resolve()
    cyclops_dir = result_path
    while cyclops_dir.name != "CYCLOPS-2.0" and cyclops_dir.parent != cyclops_dir:
        cyclops_dir = cyclops_dir.parent
    
    base_path = cyclops_dir.parent / "data"
    
    # For Zhang transfer runs where the result directory is
    # GTEx_Zhang_Transfer_{dataset}_{timestamp}, dataset_name is usually
    # just the tumor subset name (e.g. "Bcell"). These live under
    # data/Zhang_CancerCell_2025_sub/{dataset}/expression.csv.

    # First try Zhang_CancerCell_2025_sub/{dataset_name}
    expr_candidates = []
    expr_candidates.append(base_path / "Zhang_CancerCell_2025_sub" / dataset_name / "expression.csv")

    # If the name looks like it has a timestamp suffix, also try stripping
    # everything after the last underscore.
    if '_' in dataset_name:
        base_name = dataset_name.split('_')[0]
        expr_candidates.append(base_path / "Zhang_CancerCell_2025_sub" / base_name / "expression.csv")

    # As a fallback, support GSE-style datasets directly under data/
    if dataset_name.startswith("GSE"):
        expr_candidates.append(base_path / dataset_name / "expression.csv")

    expr_file = None
    for candidate in expr_candidates:
        if candidate.exists():
            expr_file = candidate
            break

    if expr_file is None:
        raise FileNotFoundError(f"Expression file not found. Tried: " + ", ".join(str(p) for p in expr_candidates))
    
    print(f"\nLoading expression data: {expr_file}")
    expr_df = pd.read_csv(expr_file)
    
    # Check if first row is metadata (TissueType_D)
    if str(expr_df.iloc[0, 0]) == "TissueType_D":
        print("Removing TissueType_D metadata row")
        expr_df = expr_df.iloc[1:].copy()
    
    # Set gene symbol as index
    expr_df = expr_df.rename(columns={expr_df.columns[0]: 'Gene_Symbol'})
    expr_df = expr_df.set_index('Gene_Symbol')
    
    # Convert to numeric
    for col in expr_df.columns:
        expr_df[col] = pd.to_numeric(expr_df[col], errors='coerce')
    
    print(f"Expression data shape: {expr_df.shape}")
    return expr_df

def plot_tumor_expression(phases_df, expr_df, dataset_name, output_dir):
    """Plot phase-aligned expression for tumor samples only"""
    
    # Find available core genes
    available_genes = [g for g in CORE_GENES if g in expr_df.index]
    print(f"\nCore genes found: {len(available_genes)}/{len(CORE_GENES)}")
    print(", ".join(available_genes))
    
    if len(available_genes) == 0:
        print("ERROR: No core genes found!")
        return
    
    # Prepare plotting data
    plot_data = []
    for gene in available_genes:
        gene_expr = expr_df.loc[gene]
        for _, row in phases_df.iterrows():
            sample = row['Sample']
            phase = row['Phase']
            
            if sample in gene_expr.index and not pd.isna(phase):
                expr_val = gene_expr[sample]
                if not pd.isna(expr_val):
                    plot_data.append({
                        'Gene': gene,
                        'Sample': sample,
                        'Phase': phase,
                        'Expression': expr_val
                    })
    
    plot_df = pd.DataFrame(plot_data)
    print(f"\nTotal data points: {len(plot_df)}")
    
    # Create figure with subplots
    n_genes = len(available_genes)
    n_cols = 4
    n_rows = int(np.ceil(n_genes / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_genes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, gene in enumerate(available_genes):
        ax = axes[idx]
        gene_data = plot_df[plot_df['Gene'] == gene].copy()
        
        if len(gene_data) == 0:
            ax.text(0.5, 0.5, f'{gene}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 2*np.pi)
            continue
        
        # Z-score normalization per gene
        gene_data['Expression_Z'] = (gene_data['Expression'] - gene_data['Expression'].mean()) / gene_data['Expression'].std()
        
        # Sort by phase for plotting
        gene_data = gene_data.sort_values('Phase')
        
        # Plot scatter
        ax.scatter(gene_data['Phase'], gene_data['Expression_Z'], 
                  alpha=0.6, s=50, color='red', edgecolors='darkred', linewidth=0.5)
        
        # Fit sinusoid for visualization
        phases = gene_data['Phase'].values
        expr = gene_data['Expression_Z'].values
        
        # Simple cosine fit
        x_fit = np.linspace(0, 2*np.pi, 200)
        
        # Estimate amplitude and acrophase
        mean_expr = np.mean(expr)
        
        # Try different phase shifts to find best fit
        best_r2 = -np.inf
        best_params = None
        
        for phase_shift in np.linspace(0, 2*np.pi, 50):
            pred = mean_expr + np.std(expr) * np.cos(phases - phase_shift)
            r2 = 1 - np.sum((expr - pred)**2) / np.sum((expr - mean_expr)**2)
            if r2 > best_r2:
                best_r2 = r2
                best_params = (np.std(expr), phase_shift, mean_expr)
        
        if best_params:
            amp, acro, baseline = best_params
            y_fit = baseline + amp * np.cos(x_fit - acro)
            ax.plot(x_fit, y_fit, 'b-', linewidth=2, alpha=0.7,
                   label=f'R²={best_r2:.3f}, φ={acro:.2f}')
            ax.legend(fontsize=8)
        
        ax.set_title(f'{gene} (n={len(gene_data)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Phase (radians)', fontsize=10)
        ax.set_ylabel('Z-score Expression', fontsize=10)
        ax.set_xlim(0, 2*np.pi)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add π markers
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Hide unused subplots
    for idx in range(n_genes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Core Gene Expression - Tumor Samples Only\nDataset: {dataset_name} (n={len(phases_df)} samples)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_file = output_path / f'tumor_core_genes_{dataset_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    
    # Also save as PDF
    output_file_pdf = output_path / f'tumor_core_genes_{dataset_name}.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved: {output_file_pdf}")
    
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_tumor_core_genes.py <result_directory> [output_dir]")
        print("\nExample:")
        print("  python plot_tumor_core_genes.py ./results/GTEx_Zhang_Transfer_GSE176078_20251229_142755")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else result_dir
    
    # Extract dataset name from directory
    result_path = Path(result_dir)
    dir_name = result_path.name
    
    # Parse dataset name (format: GTEx_Zhang_Transfer_{dataset}_{timestamp})
    parts = dir_name.split('_')
    if len(parts) >= 4:
        dataset_name = '_'.join(parts[3:-1])  # Everything between "Transfer" and timestamp
    else:
        dataset_name = "Unknown"
    
    print("="*80)
    print(f"Plotting Tumor Core Gene Expression")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Result directory: {result_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    tumor_phases = load_transfer_results(result_dir)
    expr_df = load_expression_data(result_dir, dataset_name)
    
    # Plot
    plot_tumor_expression(tumor_phases, expr_df, dataset_name, output_dir)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)

if __name__ == "__main__":
    main()
