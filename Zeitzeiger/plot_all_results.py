#!/usr/bin/env python3
"""
Batch-plot predicted phase vs true collection time for all prediction CSVs.

Usage:
  python plot_all_results.py --results-dir ./results --out-dir ./plots
"""
import argparse
import math
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_prediction_files(results_dir, ext='.predictions.csv'):
    # Search recursively in subdirectories
    pattern = os.path.join(results_dir, '**', f'*{ext}')
    return sorted(glob(pattern, recursive=True))


def detect_columns(df):
    pred_col = None
    if 'pred_time_norm' in df.columns:
        pred_col = ('pred_time_norm', 'norm')
    elif 'pred_time_hours' in df.columns:
        pred_col = ('pred_time_hours', 'hours')
    else:
        for c in df.columns:
            if c.startswith('Phases_AG'):
                pred_col = (c, 'radians')
                break
    
    true_col = None
    for cand in ['true_time_hours', 'true_time', 'true_time_norm', 'true_time_radians']:
        if cand in df.columns:
            true_col = cand
            break
    
    return pred_col, true_col


def to_radians(x, from_format):
    x = np.array(x, dtype=float)
    if from_format == 'norm':
        return (x % 1.0) * 2 * math.pi
    elif from_format == 'hours':
        return (x % 24.0) * 2 * math.pi / 24.0
    else:
        return x % (2 * math.pi)


def plot_pair(true_rad, pred_rad, out_path, tissue=None):
    plt.figure(figsize=(8, 7))
    plt.scatter(true_rad, pred_rad, c='black')
    plt.xlabel('Collection Phase', fontsize=24)
    plt.ylabel('Predicted Phase', fontsize=24)
    plt.xlim(0, 2 * math.pi)
    plt.ylim(0, 2 * math.pi)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved: {out_path}')


def process_file(pred_path, out_dir=None):
    df = pd.read_csv(pred_path)
    pred_col_info, true_col = detect_columns(df)
    
    if pred_col_info is None or true_col is None:
        print(f'Skipping {pred_path}: missing required columns', file=sys.stderr)
        return False
    
    pred_col, pred_type = pred_col_info
    pred_rad = to_radians(df[pred_col].values, pred_type)
    
    true_arr = np.array(df[true_col].values, dtype=float)
    if true_arr.max() <= 1.0:
        true_rad = to_radians(true_arr, 'norm')
    elif true_arr.max() <= (2 * math.pi + 0.1):
        true_rad = to_radians(true_arr, 'radians')
    else:
        true_rad = to_radians(true_arr, 'hours')
    
    tissue = os.path.basename(pred_path).replace('.predictions.csv', '').split('_vs_')[-1]
    # If still generic, try to extract from parent directory
    if tissue == 'zeitzeiger':
        parent_dir = os.path.basename(os.path.dirname(pred_path))
        if parent_dir.startswith('test_'):
            tissue = parent_dir.replace('test_', '')
    
    out_dir_final = out_dir or os.path.dirname(pred_path)
    os.makedirs(out_dir_final, exist_ok=True)
    out_path = os.path.join(out_dir_final, f'{tissue}_phase_vs_time.png')
    
    plot_pair(true_rad, pred_rad, out_path, tissue)
    return True


def main():
    p = argparse.ArgumentParser(description='Batch plot phase vs time')
    p.add_argument('--results-dir', required=True, help='Results directory')
    p.add_argument('--out-dir', default=None, help='Output directory')
    p.add_argument('--ext', default='.predictions.csv', help='File extension')
    args = p.parse_args()

    files = find_prediction_files(args.results_dir, args.ext)
    if not files:
        sys.exit(f'No files found in {args.results_dir}')

    for f in files:
        try:
            process_file(f, args.out_dir)
        except Exception as e:
            print(f'Error: {f}: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
