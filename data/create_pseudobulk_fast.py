#!/usr/bin/env python3
"""
Fast pseudo-bulk aggregation using h5py (no full file loading)
Processes sparse matrix directly from HDF5 file
"""
import h5py
import pandas as pd
import numpy as np
from scipy import sparse
from collections import defaultdict
import argparse
from pathlib import Path

def read_categorical_column(h5_file, column_name):
    """Read a categorical column from h5ad obs"""
    categories = h5_file[f'obs/{column_name}/categories'][:]
    codes = h5_file[f'obs/{column_name}/codes'][:]
    # Decode bytes to strings if needed
    if categories.dtype == 'O':
        categories = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in categories]
    return [categories[code] if code >= 0 else 'NA' for code in codes]

def aggregate_pseudobulk_fast(h5_path, group_by='Sample', min_cells=10, output_dir='pseudobulk_output'):
    """
    Fast pseudo-bulk aggregation without loading full file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("Fast pseudo-bulk aggregation (h5py-based)")
    print("="*80)
    
    with h5py.File(h5_path, 'r') as f:
        print("\nReading metadata...")
        
        # Get dimensions
        n_cells = f['X/indptr'].shape[0] - 1
        n_genes = len(f['var/_index'])
        print(f"Data dimensions: {n_cells:,} cells × {n_genes:,} genes")
        
        # Read gene names
        gene_names = f['var/_index'][:]
        if gene_names.dtype == 'O':
            gene_names = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in gene_names]
        print(f"Gene names loaded: {len(gene_names):,}")
        
        # Read grouping variable
        print(f"\nReading grouping variable: {group_by}")
        group_labels = read_categorical_column(f, group_by)
        print(f"Unique groups found: {len(set(group_labels)):,}")
        
        # Count cells per group
        from collections import Counter
        group_counts = Counter(group_labels)
        valid_groups = {g for g, count in group_counts.items() if count >= min_cells}
        print(f"Groups with >={min_cells} cells: {len(valid_groups):,}")
        
        # Read sparse matrix components
        print("\nReading sparse matrix structure...")
        print("  - Reading data array...")
        data = f['X/data'][:]
        print(f"    Loaded {len(data):,} non-zero values")
        
        print("  - Reading indices...")
        indices = f['X/indices'][:]
        
        print("  - Reading indptr...")
        indptr = f['X/indptr'][:]
        
        # Construct sparse matrix
        print("\nConstructing sparse matrix...")
        X = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))
        print(f"Sparse matrix shape: {X.shape}")
        print(f"Sparsity: {100 * (1 - X.nnz / (X.shape[0] * X.shape[1])):.2f}% zeros")
        
        # Read all obs metadata
        print("\nReading sample-level metadata...")
        obs_keys = list(f['obs'].keys())
        sample_metadata = {}
        
        # Read all metadata columns
        for key in obs_keys:
            try:
                if key.endswith('categories') or key.endswith('codes'):
                    continue
                    
                if f'obs/{key}/categories' in f:
                    # Categorical variable
                    sample_metadata[key] = read_categorical_column(f, key)
                else:
                    # Numeric variable
                    data = f[f'obs/{key}'][:]
                    sample_metadata[key] = data
            except Exception as e:
                print(f"  Skipping {key}: {e}")
        
        print(f"  Read {len(sample_metadata)} metadata columns")
        
        # Aggregate by group
        print("\nAggregating expression by group...")
        pseudobulk_data = []
        pseudobulk_samples = []
        cell_counts = []
        metadata_records = []
        
        for i, group in enumerate(sorted(valid_groups)):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(valid_groups)} groups...")
            
            # Get cells in this group (convert to numpy array)
            mask = np.array([g == group for g in group_labels])
            n_cells_group = mask.sum()
            
            # Sum expression across cells
            group_expr = np.array(X[mask, :].sum(axis=0)).flatten()
            
            pseudobulk_data.append(group_expr)
            pseudobulk_samples.append(group)
            cell_counts.append(n_cells_group)
            
            # Aggregate metadata (take mode for categorical, mean for numeric)
            meta_record = {'Sample': group, 'n_cells': n_cells_group}
            for key, values in sample_metadata.items():
                if key == 'Sample_Barcode' or key == 'Sample_barcode' or key == 'index':
                    continue  # Skip cell-specific IDs
                    
                group_values = [values[j] for j in range(len(values)) if mask[j]]
                
                # Determine if numeric or categorical
                if isinstance(group_values[0], (int, float, np.integer, np.floating)):
                    # Numeric: use mean, ignore NaN
                    valid_vals = [v for v in group_values if not (isinstance(v, float) and np.isnan(v))]
                    if valid_vals:
                        meta_record[key] = np.mean(valid_vals)
                    else:
                        meta_record[key] = np.nan
                else:
                    # Categorical: use most common
                    from collections import Counter
                    counter = Counter(group_values)
                    meta_record[key] = counter.most_common(1)[0][0]
            
            metadata_records.append(meta_record)
        
        print(f"Completed aggregation for {len(pseudobulk_samples)} groups")
    
    # Create expression DataFrame
    print("\nCreating expression matrix...")
    pseudobulk_expr = pd.DataFrame(
        np.array(pseudobulk_data).T,
        index=gene_names,
        columns=pseudobulk_samples
    )
    
    # Create metadata DataFrame
    print("Creating metadata...")
    pseudobulk_meta = pd.DataFrame(metadata_records)
    pseudobulk_meta.index = pseudobulk_samples
    
    print(f"\nMetadata columns: {list(pseudobulk_meta.columns)}")
    
    return pseudobulk_expr, pseudobulk_meta

def normalize_tpm(expr_df):
    """Normalize to TPM"""
    print("\nNormalizing to TPM...")
    total_counts = expr_df.sum(axis=0)
    tpm = expr_df / total_counts * 1e6
    return tpm

def filter_genes(expr_df, min_samples):
    """Filter genes by presence"""
    print(f"\nFiltering genes (require expression in >={min_samples} samples)...")
    gene_presence = (expr_df > 0).sum(axis=1)
    keep_genes = gene_presence >= min_samples
    filtered = expr_df[keep_genes]
    print(f"Kept {keep_genes.sum():,} / {len(keep_genes):,} genes")
    return filtered

def main():
    parser = argparse.ArgumentParser(description='Fast pseudo-bulk from h5ad')
    parser.add_argument('input_h5ad', help='Input h5ad file')
    parser.add_argument('--output-dir', default='pseudobulk_by_sample', help='Output directory')
    parser.add_argument('--group-by', default='Sample', help='Column to group by')
    parser.add_argument('--min-cells', type=int, default=10, help='Min cells per group')
    parser.add_argument('--normalize', action='store_true', help='Normalize to TPM')
    parser.add_argument('--filter-genes', type=int, default=0, help='Filter genes')
    
    args = parser.parse_args()
    
    # Aggregate
    pseudobulk_expr, pseudobulk_meta = aggregate_pseudobulk_fast(
        args.input_h5ad,
        group_by=args.group_by,
        min_cells=args.min_cells,
        output_dir=args.output_dir
    )
    
    # Normalize
    if args.normalize:
        pseudobulk_expr = normalize_tpm(pseudobulk_expr)
    
    # Filter
    if args.filter_genes > 0:
        pseudobulk_expr = filter_genes(pseudobulk_expr, args.filter_genes)
    
    # Save
    output_dir = Path(args.output_dir)
    expr_file = output_dir / "expression.csv"
    meta_file = output_dir / "metadata.csv"
    
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    # Add Gene_Symbol column for CYCLOPS
    pseudobulk_expr.insert(0, 'Gene_Symbol', pseudobulk_expr.index)
    pseudobulk_expr.to_csv(expr_file, index=False)
    print(f"Expression saved: {expr_file}")
    
    pseudobulk_meta.to_csv(meta_file, index=True)
    print(f"Metadata saved: {meta_file}")
    
    print(f"\nFinal dimensions: {pseudobulk_expr.shape[0]-1:,} genes × {pseudobulk_expr.shape[1]-1:,} samples")
    print(f"\nCell counts per sample:")
    print(pseudobulk_meta['n_cells'].describe())
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
