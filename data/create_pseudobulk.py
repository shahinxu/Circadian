#!/usr/bin/env python3
"""
Create pseudo-bulk RNA-seq from single-cell data
Aggregates by Sample and optionally by cell type
"""
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
import argparse
from pathlib import Path

def create_pseudobulk(adata, group_by, min_cells=10):
    """
    Aggregate single cells into pseudo-bulk samples (FAST vectorized version)
    
    Parameters:
    -----------
    adata : AnnData
        Single-cell data
    group_by : str or list
        Column(s) to group by (e.g., 'Sample' or ['Sample', 'Celltype_Nov7_2023'])
    min_cells : int
        Minimum number of cells required for a group
        
    Returns:
    --------
    pd.DataFrame : Pseudo-bulk expression matrix (genes × samples)
    pd.DataFrame : Metadata for pseudo-bulk samples
    """
    
    print(f"\nAggregating by: {group_by}")
    
    # Create grouping key
    if isinstance(group_by, str):
        group_by = [group_by]
    
    # Create combined group labels
    group_labels = adata.obs[group_by[0]].astype(str).values
    for col in group_by[1:]:
        group_labels = group_labels + "_" + adata.obs[col].astype(str).values
    
    # Get unique groups and cell counts
    unique_groups, group_indices, group_counts = np.unique(
        group_labels, return_inverse=True, return_counts=True
    )
    
    print(f"Total groups: {len(unique_groups)}")
    print(f"Cell count range: {group_counts.min()}-{group_counts.max()}")
    
    # Filter groups with too few cells
    valid_mask = group_counts >= min_cells
    valid_groups = unique_groups[valid_mask]
    print(f"Groups with >={min_cells} cells: {len(valid_groups)}")
    
    # Prepare expression matrix
    if sparse.issparse(adata.X):
        X = adata.X
    else:
        X = sparse.csr_matrix(adata.X)
    
    print(f"Expression matrix: {X.shape[0]:,} cells × {X.shape[1]:,} genes")
    
    # FAST VECTORIZED AGGREGATION using sparse matrix multiplication
    print("Building aggregation matrix...")
    # Create indicator matrix: cells × groups (sparse)
    from scipy.sparse import csr_matrix
    
    # Map valid groups to new indices
    group_to_idx = {g: i for i, g in enumerate(valid_groups)}
    
    # Build sparse indicator matrix
    row_indices = []
    col_indices = []
    
    for cell_idx, group_label in enumerate(group_labels):
        if group_label in group_to_idx:
            row_indices.append(cell_idx)
            col_indices.append(group_to_idx[group_label])
    
    data = np.ones(len(row_indices))
    indicator = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(group_labels), len(valid_groups))
    )
    
    print(f"Aggregating {len(valid_groups)} groups using matrix multiplication...")
    # Pseudo-bulk = X.T @ indicator  (genes × groups)
    pseudobulk_expr = X.T @ indicator
    
    # Convert to dense if needed and create DataFrame
    if sparse.issparse(pseudobulk_expr):
        pseudobulk_expr = pseudobulk_expr.toarray()
    
    pseudobulk_expr = pd.DataFrame(
        pseudobulk_expr,
        index=adata.var_names,
        columns=valid_groups
    )
    
    # Collect metadata
    print("Collecting metadata...")
    pseudobulk_metadata = []
    for group in valid_groups:
        mask = group_labels == group
        group_meta = adata.obs[mask].iloc[0].to_dict()
        group_meta['n_cells'] = mask.sum()
        group_meta['group_id'] = group
        pseudobulk_metadata.append(group_meta)
    
    pseudobulk_meta = pd.DataFrame(pseudobulk_metadata)
    pseudobulk_meta.index = valid_groups
    
    print(f"\nPseudo-bulk shape: {pseudobulk_expr.shape}")
    print(f"Genes: {pseudobulk_expr.shape[0]:,}")
    print(f"Samples: {pseudobulk_expr.shape[1]:,}")
    
    return pseudobulk_expr, pseudobulk_meta

def normalize_tpm(expr_df):
    """Normalize to TPM-like values"""
    print("\nNormalizing to TPM...")
    # Sum per sample (column)
    total_counts = expr_df.sum(axis=0)
    # Normalize to 1M
    tpm = expr_df / total_counts * 1e6
    return tpm

def main():
    parser = argparse.ArgumentParser(description='Create pseudo-bulk from single-cell h5ad')
    parser.add_argument('input_h5ad', help='Path to input h5ad file')
    parser.add_argument('--output-dir', default='pseudobulk_output', help='Output directory')
    parser.add_argument('--group-by', nargs='+', default=['Sample'], 
                       help='Columns to group by (e.g., Sample or Sample Celltype_Nov7_2023)')
    parser.add_argument('--min-cells', type=int, default=10, 
                       help='Minimum cells per group')
    parser.add_argument('--normalize', action='store_true', 
                       help='Normalize to TPM')
    parser.add_argument('--filter-genes', type=int, default=0,
                       help='Keep only genes expressed in >= N samples')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Pseudo-bulk aggregation from single-cell data")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nReading h5ad file: {args.input_h5ad}")
    print("This may take a while for large files...")
    adata = sc.read_h5ad(args.input_h5ad)
    
    print(f"\nOriginal data:")
    print(f"  Cells: {adata.n_obs:,}")
    print(f"  Genes: {adata.n_vars:,}")
    
    # Create pseudo-bulk
    pseudobulk_expr, pseudobulk_meta = create_pseudobulk(
        adata, 
        group_by=args.group_by,
        min_cells=args.min_cells
    )
    
    # Normalize if requested
    if args.normalize:
        pseudobulk_expr = normalize_tpm(pseudobulk_expr)
    
    # Filter lowly expressed genes
    if args.filter_genes > 0:
        print(f"\nFiltering genes expressed in < {args.filter_genes} samples...")
        gene_presence = (pseudobulk_expr > 0).sum(axis=1)
        keep_genes = gene_presence >= args.filter_genes
        pseudobulk_expr = pseudobulk_expr[keep_genes]
        print(f"Kept {keep_genes.sum():,} / {len(keep_genes):,} genes")
    
    # Save results with fixed names
    expr_file = output_dir / "expression.csv"
    meta_file = output_dir / "metadata.csv"
    
    print(f"\nSaving results...")
    print(f"  Expression: {expr_file}")
    print(f"  Metadata: {meta_file}")
    
    # Add Gene_Symbol column for CYCLOPS compatibility
    pseudobulk_expr.insert(0, 'Gene_Symbol', pseudobulk_expr.index)
    pseudobulk_expr.to_csv(expr_file, index=False)
    
    pseudobulk_meta.to_csv(meta_file, index=True)
    
    print("\n" + "="*80)
    print("Pseudo-bulk aggregation complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Expression file: {expr_file.name}")
    print(f"Metadata file: {meta_file.name}")
    print(f"\nFinal dimensions: {pseudobulk_expr.shape[0]-1} genes × {pseudobulk_expr.shape[1]-1} samples")
    
    # Print summary stats
    print(f"\nSummary statistics:")
    print(f"  Cells per sample: {pseudobulk_meta['n_cells'].describe()}")

if __name__ == "__main__":
    main()
