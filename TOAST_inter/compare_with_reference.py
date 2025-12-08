#!/usr/bin/env python3
"""
Compare GREED predictions with reference data from science.add0846_table_s2.csv
Usage: python compare_with_reference.py [--results_dir path/to/results]
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def extract_subject_id(sample_id):
    """Extract SUBJ.ID from Sample_ID (e.g., GTEX-1117F-... -> 1117F)"""
    import re
    match = re.search(r'GTEX-([^-]+)', str(sample_id))
    return match.group(1) if match else None

def compare_predictions(predictions_csv, reference_csv, output_dir):
    """
    Compare predictions with reference data and generate plot.
    
    Args:
        predictions_csv: Path to predictions_aligned.csv or predictions.csv
        reference_csv: Path to science.add0846_table_s2.csv
        output_dir: Directory to save output plot
    """
    # Read data
    pred = pd.read_csv(predictions_csv)
    ref = pd.read_csv(reference_csv)
    
    # Extract subject IDs
    pred['SUBJ_ID'] = pred['Sample_ID'].apply(extract_subject_id)
    
    # Merge on subject ID
    merged = pred.merge(ref, left_on='SUBJ_ID', right_on='SUBJ.ID', how='inner')
    
    if len(merged) == 0:
        print(f"  WARNING: No matching samples found!")
        return None
    
    # Determine which prediction column to use
    if 'Predicted_Phase_Hours_Aligned' in merged.columns:
        pred_col = 'Predicted_Phase_Hours_Aligned'
        plot_title_suffix = '(Aligned)'
    elif 'Predicted_Phase_Hours' in merged.columns:
        pred_col = 'Predicted_Phase_Hours'
        plot_title_suffix = ''
    else:
        print(f"  ERROR: No prediction phase column found!")
        return None
    
    pred_hours = merged[pred_col].values
    ref_hours = merged['hour'].values
    
    # Calculate metrics
    r, p = pearsonr(pred_hours, ref_hours)
    spearman_r, spearman_p = spearmanr(pred_hours, ref_hours)
    errors = pred_hours - ref_hours
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())
    
    # Create plot
    plt.figure(figsize=(8, 8))
    plt.scatter(ref_hours, pred_hours, alpha=0.5, s=30, c='blue', 
                edgecolors='black', linewidth=0.5)
    plt.plot([0, 24], [0, 24], 'r--', label='Perfect prediction', linewidth=2)
    
    plt.xlabel('Reference Phase (hours)', fontsize=14)
    plt.ylabel(f'Predicted Phase (hours) {plot_title_suffix}', fontsize=14)
    
    tissue_name = Path(output_dir).parent.name
    plt.title(f'{tissue_name}\n'
              f'Pearson r = {r:.3f} (p = {p:.2e})\n'
              f'MAE = {mae:.2f}h, RMSE = {rmse:.2f}h\n'
              f'n = {len(merged)} samples', fontsize=12)
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    plt.ylim(0, 24)
    plt.xticks(np.arange(0, 25, 3))
    plt.yticks(np.arange(0, 25, 3))
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'predicted_vs_reference.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Pearson r = {r:.4f} (p = {p:.2e})")
    print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")
    print(f"  MAE = {mae:.2f}h, RMSE = {rmse:.2f}h")
    print(f"  Matched samples: {len(merged)}")
    
    return {
        'tissue': tissue_name,
        'n_samples': len(merged),
        'pearson_r': r,
        'pearson_p': p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mae': mae,
        'rmse': rmse
    }

def find_latest_predictions(tissue_dir):
    """Find the latest predictions file in a tissue directory."""
    if not os.path.isdir(tissue_dir):
        return None
    
    # Find all timestamp subdirectories
    subdirs = [d for d in os.listdir(tissue_dir) 
               if os.path.isdir(os.path.join(tissue_dir, d))]
    
    if not subdirs:
        return None
    
    # Sort by timestamp (assuming format: YYYY-MM-DDTHH_MM_SS)
    subdirs.sort(reverse=True)
    
    # Look for predictions file in the latest subdir
    for subdir in subdirs:
        subdir_path = os.path.join(tissue_dir, subdir)
        
        # Try predictions_aligned.csv first, then predictions.csv
        for filename in ['predictions_aligned.csv', 'predictions.csv']:
            pred_file = os.path.join(subdir_path, filename)
            if os.path.isfile(pred_file):
                return pred_file, subdir_path
    
    return None

def process_all_tissues(results_base, reference_csv):
    """
    Process all tissue directories under results_base.
    
    Args:
        results_base: Base results directory (e.g., GREED/results/GTEx)
        reference_csv: Path to reference CSV file
    """
    if not os.path.isdir(results_base):
        print(f"ERROR: Results directory not found: {results_base}")
        return
    
    if not os.path.isfile(reference_csv):
        print(f"ERROR: Reference file not found: {reference_csv}")
        return
    
    print(f"Scanning tissues in: {results_base}")
    print(f"Using reference: {reference_csv}")
    print("=" * 80)
    
    all_results = []
    
    # Iterate through tissue directories
    tissue_dirs = [d for d in os.listdir(results_base) 
                   if os.path.isdir(os.path.join(results_base, d))]
    tissue_dirs.sort()
    
    for tissue in tissue_dirs:
        tissue_path = os.path.join(results_base, tissue)
        print(f"\nProcessing: {tissue}")
        
        result = find_latest_predictions(tissue_path)
        if result is None:
            print(f"  No predictions file found, skipping...")
            continue
        
        pred_file, output_dir = result
        print(f"  Found: {os.path.basename(pred_file)}")
        
        metrics = compare_predictions(pred_file, reference_csv, output_dir)
        if metrics:
            all_results.append(metrics)
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(results_base, 'comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'=' * 80}")
        print(f"Summary saved to: {summary_path}")
        print(f"\nProcessed {len(all_results)} tissues")
        print(f"\nAverage metrics:")
        print(f"  Pearson r: {summary_df['pearson_r'].mean():.4f} ± {summary_df['pearson_r'].std():.4f}")
        print(f"  MAE: {summary_df['mae'].mean():.2f} ± {summary_df['mae'].std():.2f} hours")
        print(f"  RMSE: {summary_df['rmse'].mean():.2f} ± {summary_df['rmse'].std():.2f} hours")

def main():
    parser = argparse.ArgumentParser(description='Compare GREED predictions with reference data')
    parser.add_argument('--results_dir', default='GREED/results/GTEx',
                        help='Base results directory containing tissue folders')
    parser.add_argument('--reference', default='science.add0846_table_s2.csv',
                        help='Path to reference CSV file')
    parser.add_argument('--tissue', default=None,
                        help='Process specific tissue only (optional)')
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(project_root, results_dir)
    
    reference_csv = args.reference
    if not os.path.isabs(reference_csv):
        reference_csv = os.path.join(project_root, reference_csv)
    
    if args.tissue:
        # Process single tissue
        tissue_path = os.path.join(results_dir, args.tissue)
        result = find_latest_predictions(tissue_path)
        if result:
            pred_file, output_dir = result
            compare_predictions(pred_file, reference_csv, output_dir)
        else:
            print(f"No predictions found for tissue: {args.tissue}")
    else:
        # Process all tissues
        process_all_tissues(results_dir, reference_csv)

if __name__ == '__main__':
    main()
