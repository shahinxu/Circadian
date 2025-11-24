#!/usr/bin/env python3
"""
Automated Zeitzeiger runner with leave-one-out cross-validation.
Automatically detects subdatasets and runs train-on-n test-on-1 strategy.

Usage:
    python run_zeitzeiger_auto.py --dataset GSE54651
    python run_zeitzeiger_auto.py --dataset GSE54652 --sample-col Sample --time-col Time
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


def find_subdatasets(data_dir):
    """Find all subdirectories containing expression.csv and metadata.csv"""
    subdatasets = []
    for item in sorted(os.listdir(data_dir)):
        subdir = os.path.join(data_dir, item)
        if not os.path.isdir(subdir):
            continue
        expr_file = os.path.join(subdir, 'expression.csv')
        meta_file = os.path.join(subdir, 'metadata.csv')
        if os.path.isfile(expr_file) and os.path.isfile(meta_file):
            subdatasets.append(item)
    return subdatasets


def combine_datasets(dataset_list, base_dir, exclude=None):
    """Combine multiple subdatasets into single training set"""
    all_expr = []
    all_meta = []
    
    for subds in dataset_list:
        if exclude and subds == exclude:
            continue
        expr_path = os.path.join(base_dir, subds, 'expression.csv')
        meta_path = os.path.join(base_dir, subds, 'metadata.csv')
        
        expr_df = pd.read_csv(expr_path)
        meta_df = pd.read_csv(meta_path)
        
        # Add tissue/subdataset label if not present
        if 'Tissue' not in meta_df.columns and 'tissue' not in meta_df.columns:
            meta_df['Tissue'] = subds
        
        all_expr.append(expr_df)
        all_meta.append(meta_df)
    
    # Combine expression: assume first column is Gene_Symbol
    gene_col = all_expr[0].columns[0]
    combined_expr = all_expr[0]
    for expr_df in all_expr[1:]:
        combined_expr = combined_expr.merge(expr_df, on=gene_col, how='outer')
    
    # Fill NaN with 0 or mean (simple strategy)
    numeric_cols = combined_expr.select_dtypes(include=['float64', 'int64']).columns
    combined_expr[numeric_cols] = combined_expr[numeric_cols].fillna(0)
    
    # Combine metadata
    combined_meta = pd.concat(all_meta, ignore_index=True)
    
    return combined_expr, combined_meta


def run_zeitzeiger(expr_train_path, meta_train_path, expr_test_path, meta_test_path,
                   out_prefix, sample_col='Sample', time_col='Time_Phase', 
                   time_format='auto', verbose=False):
    """Run zeitzeiger R script"""
    r_script = os.path.join(os.path.dirname(__file__), 'run_zeitzeiger_separate.R')
    
    cmd = [
        'Rscript', r_script,
        '--expr-train', expr_train_path,
        '--meta-train', meta_train_path,
        '--expr-test', expr_test_path,
        '--meta-test', meta_test_path,
        '--sample-col', sample_col,
        '--time-col', time_col,
        '--time-format', time_format,
        '--out-prefix', out_prefix,
        '--save-model'
    ]
    
    if verbose:
        cmd.append('--verbose')
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    
    if verbose:
        print(result.stdout)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Automated Zeitzeiger runner with leave-one-out cross-validation'
    )
    parser.add_argument('--dataset', required=True, 
                       help='Dataset name (e.g., GSE54651, GSE54652)')
    parser.add_argument('--data-root', default='../data',
                       help='Root directory containing datasets')
    parser.add_argument('--output-root', default='./results',
                       help='Root directory for outputs')
    parser.add_argument('--sample-col', default='Sample',
                       help='Sample ID column name in metadata')
    parser.add_argument('--time-col', default='Time_Phase',
                       help='Time column name in metadata')
    parser.add_argument('--time-format', default='auto',
                       choices=['auto', 'hours', 'radians', 'normalized'],
                       help='Time format in metadata')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be done without executing')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = os.path.join(args.data_root, args.dataset)
    if not os.path.isdir(data_dir):
        print(f"ERROR: Dataset directory not found: {data_dir}")
        sys.exit(1)
    
    # Find all subdatasets
    subdatasets = find_subdatasets(data_dir)
    if len(subdatasets) < 2:
        print(f"ERROR: Need at least 2 subdatasets for leave-one-out, found {len(subdatasets)}")
        print(f"Subdatasets: {subdatasets}")
        sys.exit(1)
    
    print(f"Found {len(subdatasets)} subdatasets in {args.dataset}:")
    for sd in subdatasets:
        print(f"  - {sd}")
    print()
    
    # Create output directory
    output_dir = os.path.join(args.output_root, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Leave-one-out cross-validation
    results_summary = []
    
    for test_subds in subdatasets:
        print(f"\n{'='*60}")
        print(f"Test dataset: {test_subds}")
        print(f"Training on: {[s for s in subdatasets if s != test_subds]}")
        print(f"{'='*60}\n")
        
        # Create temporary directory for this fold
        fold_dir = os.path.join(output_dir, f'test_{test_subds}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Combine training datasets
        train_expr_path = os.path.join(fold_dir, 'train_expression.csv')
        train_meta_path = os.path.join(fold_dir, 'train_metadata.csv')
        
        if not args.dry_run:
            print("Combining training datasets...")
            train_expr, train_meta = combine_datasets(
                subdatasets, data_dir, exclude=test_subds
            )
            train_expr.to_csv(train_expr_path, index=False)
            train_meta.to_csv(train_meta_path, index=False)
            print(f"  Training expression shape: {train_expr.shape}")
            print(f"  Training metadata shape: {train_meta.shape}")
        
        # Test dataset paths
        test_expr_path = os.path.join(data_dir, test_subds, 'expression.csv')
        test_meta_path = os.path.join(data_dir, test_subds, 'metadata.csv')
        
        # Output prefix
        out_prefix = os.path.join(fold_dir, 'zeitzeiger')
        
        if args.dry_run:
            print(f"[DRY RUN] Would run Zeitzeiger:")
            print(f"  Train: {train_expr_path}, {train_meta_path}")
            print(f"  Test:  {test_expr_path}, {test_meta_path}")
            print(f"  Output: {out_prefix}")
            continue
        
        # Run Zeitzeiger
        success = run_zeitzeiger(
            train_expr_path, train_meta_path,
            test_expr_path, test_meta_path,
            out_prefix,
            sample_col=args.sample_col,
            time_col=args.time_col,
            time_format=args.time_format,
            verbose=args.verbose
        )
        
        if success:
            print(f"✓ Successfully processed {test_subds}")
            results_summary.append({
                'test_dataset': test_subds,
                'status': 'success',
                'output_dir': fold_dir
            })
        else:
            print(f"✗ Failed to process {test_subds}")
            results_summary.append({
                'test_dataset': test_subds,
                'status': 'failed',
                'output_dir': fold_dir
            })
    
    # Save summary
    if not args.dry_run and results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_path = os.path.join(output_dir, 'run_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
