#!/usr/bin/env python3
"""
Merge all tissue data from GSE54651 into a single dataset
"""
import os
import pandas as pd
from pathlib import Path

def merge_all_tissues():
    base_dir = Path(__file__).parent
    tissues = [d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d)]
    
    print(f"Found {len(tissues)} tissues")
    
    all_expression_dfs = []
    all_metadata_dfs = []
    
    for tissue in sorted(tissues):
        tissue_dir = base_dir / tissue
        expr_file = tissue_dir / "expression.csv"
        meta_file = tissue_dir / "metadata.csv"
        
        if not expr_file.exists():
            print(f"⚠️  Skipping {tissue}: no expression.csv")
            continue
            
        print(f"Processing {tissue}...")
        
        # Load expression
        expr_df = pd.read_csv(expr_file)
        
        # Load metadata if exists
        if meta_file.exists():
            meta_df = pd.read_csv(meta_file)
            
            # Add tissue column
            meta_df['Tissue'] = tissue
            
            # Rename sample columns to avoid conflicts
            sample_cols = [col for col in expr_df.columns if col != 'Gene_Symbol']
            new_sample_cols = [f"{tissue}_{col}" for col in sample_cols]
            
            # Rename in expression
            rename_dict = dict(zip(sample_cols, new_sample_cols))
            expr_df = expr_df.rename(columns=rename_dict)
            
            # Rename in metadata
            if 'Sample' in meta_df.columns:
                meta_df['Sample'] = meta_df['Sample'].apply(lambda x: f"{tissue}_{x}")
            
            all_metadata_dfs.append(meta_df)
        else:
            print(f"  ⚠️  No metadata.csv for {tissue}")
            # Still process expression, create minimal metadata
            sample_cols = [col for col in expr_df.columns if col != 'Gene_Symbol']
            new_sample_cols = [f"{tissue}_{col}" for col in sample_cols]
            rename_dict = dict(zip(sample_cols, new_sample_cols))
            expr_df = expr_df.rename(columns=rename_dict)
            
            # Create minimal metadata
            meta_df = pd.DataFrame({
                'Sample': new_sample_cols,
                'Tissue': tissue
            })
            all_metadata_dfs.append(meta_df)
        
        all_expression_dfs.append(expr_df)
        print(f"  ✓ {len(sample_cols)} samples")
    
    # Merge all expressions
    print("\nMerging expression data...")
    merged_expr = all_expression_dfs[0]
    for expr_df in all_expression_dfs[1:]:
        merged_expr = merged_expr.merge(expr_df, on='Gene_Symbol', how='outer')
    
    # Merge all metadata
    print("Merging metadata...")
    merged_meta = pd.concat(all_metadata_dfs, ignore_index=True)
    
    # Save
    output_dir = base_dir / "all_tissues"
    output_dir.mkdir(exist_ok=True)
    
    expr_output = output_dir / "expression.csv"
    meta_output = output_dir / "metadata.csv"
    
    print(f"\nSaving to {output_dir}/")
    merged_expr.to_csv(expr_output, index=False)
    merged_meta.to_csv(meta_output, index=False)
    
    print(f"✓ Expression: {merged_expr.shape} → {expr_output}")
    print(f"✓ Metadata: {merged_meta.shape} → {meta_output}")
    print(f"\nSummary:")
    print(f"  Genes: {len(merged_expr)}")
    print(f"  Samples: {len(merged_expr.columns) - 1}")
    print(f"  Tissues: {merged_meta['Tissue'].nunique()}")
    print(f"\nTissue distribution:")
    print(merged_meta['Tissue'].value_counts().sort_index())

if __name__ == "__main__":
    merge_all_tissues()
