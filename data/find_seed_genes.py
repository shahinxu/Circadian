#!/usr/bin/env python3
"""
Find rhythmic genes in a dataset by analyzing their expression patterns
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats


def calculate_gene_rhythmicity_with_metadata(expr_file, metadata_file, method='correlation'):
    """Calculate rhythmicity score for each gene based on collection time from metadata"""
    expr_df = pd.read_csv(expr_file, index_col=0)
    meta_df = pd.read_csv(metadata_file)
    
    # Find time column
    time_col = None
    for col in ['time', 'Time', 'time_mod24', 'Time_Hours', 'hour']:
        if col in meta_df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError(f"No time column found in metadata. Available columns: {meta_df.columns.tolist()}")
    
    print(f"Using time column: {time_col}")
    
    # Find sample column
    if 'Sample' in meta_df.columns:
        sample_col = 'Sample'
    elif 'sample' in meta_df.columns:
        sample_col = 'sample'
    elif 'study_sample' in meta_df.columns:
        sample_col = 'study_sample'
    else:
        raise ValueError(f"No sample column found in metadata. Available columns: {meta_df.columns.tolist()}")
    
    # Match samples
    meta_samples = meta_df[sample_col].astype(str).tolist()
    expr_samples = expr_df.columns.tolist()
    common_samples = [s for s in expr_samples if s in meta_samples]
    
    if len(common_samples) == 0:
        print("Warning: No common samples found, trying to match by index...")
        if len(expr_samples) == len(meta_samples):
            common_samples = expr_samples
            meta_df = meta_df.copy()
            meta_df[sample_col] = expr_samples
        else:
            raise ValueError("Cannot match samples between expression and metadata")
    
    print(f"Found {len(common_samples)} common samples")
    
    # Filter and reorder
    expr_filtered = expr_df[common_samples]
    meta_filtered = meta_df[meta_df[sample_col].astype(str).isin(common_samples)]
    meta_filtered = meta_filtered.set_index(meta_df[sample_col].astype(str)).loc[common_samples].reset_index(drop=True)
    
    # Get time values and convert to radians
    times = pd.to_numeric(meta_filtered[time_col], errors='coerce').values
    times = times % 24
    phases = times * (2 * np.pi / 24.0)
    
    print(f"Time range: {times.min():.2f} - {times.max():.2f} hours")
    print(f"Number of valid samples: {np.sum(np.isfinite(phases))}")
    
    # Calculate rhythmicity for each gene
    expr_matrix = expr_filtered.values.T  # (n_samples, n_genes)
    n_samples, n_genes = expr_matrix.shape
    all_genes = expr_df.index.tolist()
    
    print(f"\nCalculating rhythmicity for {n_genes} genes...")
    
    scores = np.zeros(n_genes)
    p_values = np.zeros(n_genes)
    amplitudes = np.zeros(n_genes)
    acrophases = np.zeros(n_genes)
    
    if method == 'correlation':
        sin_phase = np.sin(phases)
        cos_phase = np.cos(phases)
        
        for g in range(n_genes):
            expr = expr_matrix[:, g]
            if np.std(expr) < 1e-8:
                scores[g] = 0
                p_values[g] = 1.0
                continue
            
            r_sin = np.corrcoef(expr, sin_phase)[0, 1]
            r_cos = np.corrcoef(expr, cos_phase)[0, 1]
            
            scores[g] = np.sqrt(r_sin**2 + r_cos**2)
            
            n = len(expr)
            z = 0.5 * np.log((1 + scores[g]) / (1 - scores[g] + 1e-10))
            p_values[g] = 2 * (1 - stats.norm.cdf(abs(z) * np.sqrt(n - 3)))
            
            amplitudes[g] = np.sqrt(r_sin**2 + r_cos**2) * np.std(expr)
            acrophases[g] = (np.arctan2(r_sin, r_cos) % (2 * np.pi)) * 12 / np.pi
    
    scores_df = pd.DataFrame({
        'Gene': all_genes,
        'Rhythmicity_Score': scores,
        'P_Value': p_values,
        'Amplitude': amplitudes,
        'Acrophase_Hours': acrophases,
        'Mean_Expression': expr_filtered.mean(axis=1).values,
        'Std_Expression': expr_filtered.std(axis=1).values
    })
    
    from scipy.stats import false_discovery_control
    scores_df['FDR_Adjusted_P'] = false_discovery_control(p_values, method='bh')
    scores_df['Significant_FDR_0.05'] = scores_df['FDR_Adjusted_P'] < 0.05
    scores_df['Significant_P_0.05'] = scores_df['P_Value'] < 0.05
    
    scores_df = scores_df.sort_values('Rhythmicity_Score', ascending=False)
    
    return scores_df


def main():
    parser = argparse.ArgumentParser(description='Find rhythmic genes in a dataset')
    parser.add_argument('--dataset_path', required=True, help='Dataset path')
    parser.add_argument('--method', default='correlation', choices=['correlation', 'anova'])
    parser.add_argument('--top_n', type=int, default=500)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--min_rhythmicity', type=float, default=0.0)
    parser.add_argument('--max_fdr', type=float, default=1.0)
    parser.add_argument('--min_amplitude', type=float, default=0.0)
    parser.add_argument('--min_mean_expr', type=float, default=0.0)
    
    args = parser.parse_args()
    
    base_data = args.dataset_path
    expr_file = os.path.join(base_data, "expression.csv")
    metadata_file = os.path.join(base_data, "metadata.csv")
    
    if not os.path.exists(expr_file):
        raise FileNotFoundError(f"Expression file not found: {expr_file}")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    output_dir = args.output_dir if args.output_dir else base_data
    
    print(f"\n{'='*60}")
    print(f"Finding Rhythmic Genes")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Method: {args.method}")
    print(f"Filtering criteria:")
    print(f"  - Min rhythmicity score: {args.min_rhythmicity}")
    print(f"  - Max FDR adjusted p-value: {args.max_fdr}")
    print(f"  - Min amplitude: {args.min_amplitude}")
    print(f"  - Min mean expression: {args.min_mean_expr}")
    print(f"{'='*60}\n")
    
    scores_df = calculate_gene_rhythmicity_with_metadata(expr_file, metadata_file, method=args.method)
    
    print(f"\nApplying filtering criteria...")
    filtered_df = scores_df[
        (scores_df['Rhythmicity_Score'] >= args.min_rhythmicity) &
        (scores_df['FDR_Adjusted_P'] <= args.max_fdr) &
        (scores_df['Amplitude'] >= args.min_amplitude) &
        (scores_df['Mean_Expression'] >= args.min_mean_expr)
    ].copy()
    
    print(f"Genes passed filtering: {len(filtered_df)} / {len(scores_df)}")
    
    if len(filtered_df) == 0:
        print("WARNING: No genes passed the filtering criteria!")
        print("Consider relaxing the thresholds.")
        return
    
    top_genes = filtered_df.head(min(args.top_n, len(filtered_df)))
    
    gene_list_path = os.path.join(output_dir, 'seed_genes.txt')
    with open(gene_list_path, 'w') as f:
        for gene in top_genes['Gene']:
            f.write(f"{gene}\n")
    print(f"Seed genes saved to: {gene_list_path}")
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    print(f"Total genes analyzed: {len(scores_df)}")
    print(f"Genes passed all filters: {len(filtered_df)}")
    print(f"Selected as seed genes: {len(top_genes)}")
    print(f"Significant genes (P < 0.05): {scores_df['Significant_P_0.05'].sum()}")
    print(f"Significant genes (FDR < 0.05): {scores_df['Significant_FDR_0.05'].sum()}")
    
    if len(filtered_df) > 0:
        print(f"\nFiltered gene statistics:")
        print(f"  Rhythmicity score range: {filtered_df['Rhythmicity_Score'].min():.4f} - {filtered_df['Rhythmicity_Score'].max():.4f}")
        print(f"  Amplitude range: {filtered_df['Amplitude'].min():.4f} - {filtered_df['Amplitude'].max():.4f}")
        print(f"  Mean expression range: {filtered_df['Mean_Expression'].min():.4f} - {filtered_df['Mean_Expression'].max():.4f}")
        
        print(f"\nTop 10 seed genes:")
        print(filtered_df[['Gene', 'Rhythmicity_Score', 'P_Value', 'FDR_Adjusted_P', 'Amplitude', 'Acrophase_Hours']].head(10).to_string(index=False))
    print(f"{'='*60}\n")
    
    print("Done!")


if __name__ == '__main__':
    main()
