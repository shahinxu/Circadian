#!/usr/bin/env python3
"""
Batch-plot predicted phase vs true collection time for all prediction CSVs in a folder.

Saves one PNG per predictions CSV with both axes fixed to [0, 2*pi].

Usage:
  python Zeitzeiger/plot_all_results.py --results-dir /path/to/results --out-dir /path/to/out --ext .predictions.csv

If --out-dir is omitted the PNGs are written next to the CSVs.
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
    pattern = os.path.join(results_dir, f'*{ext}')
    return sorted(glob(pattern))


def detect_columns(df):
    # sample is not used here, predictions CSV contains true_time and pred columns
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
    # true time
    true_col = None
    for cand in ['true_time_hours', 'true_time', 'true_time_norm', 'true_time_radians', 'true_time_hours']:
        if cand in df.columns:
            true_col = cand
            break
    # fallback: try columns that look like hours
    if true_col is None:
        for c in df.columns:
            if 'true' in c.lower() and 'hour' in c.lower():
                true_col = c
                break
    if true_col is None:
        # last resort: pick the first numeric column that is not the pred column
        for c in df.columns:
            if c == pred_col[0] if pred_col else False:
                continue
            try:
                vals = pd.to_numeric(df[c], errors='coerce')
                if vals.notnull().sum() > 0:
                    true_col = c
                    break
            except Exception:
                continue
    return pred_col, true_col


def to_radians_from_norm(x):
    return (np.array(x, dtype=float) % 1.0) * 2 * math.pi


def to_radians_from_hours(x):
    return (np.array(x, dtype=float) % 24.0) * 2 * math.pi / 24.0


def to_radians_from_radians(x):
    return (np.array(x, dtype=float) % (2 * math.pi))


def plot_pair(true_rad, pred_rad, out_path, tissue=None):
    # Match the style exactly as requested
    plt.figure(figsize=(8, 7))
    plt.scatter(true_rad, pred_rad, c='black', label='Phase vs. Time')
    plt.xlabel('Collection Phase', fontsize=24)
    plt.ylabel('Predicted Phase', fontsize=24)
    plt.xlim(0, 2 * math.pi)
    plt.ylim(0, 2 * math.pi)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    plt.close()
    # print message in Chinese as in the requested style
    if tissue is None:
        tissue = os.path.splitext(os.path.basename(out_path))[0]
    print(f'已保存图片 {tissue}_phase_vs_time.png')


def process_file(pred_path, out_dir=None):
    df = pd.read_csv(pred_path)
    pred_col_info, true_col = detect_columns(df)
    if pred_col_info is None:
        print(f'Could not find a prediction column in {pred_path}, skipping', file=sys.stderr)
        return False
    pred_col, pred_type = pred_col_info

    if true_col is None:
        print(f'Could not find a true-time column in {pred_path}, skipping', file=sys.stderr)
        return False

    # Extract arrays
    pred_vals = df[pred_col].values
    true_vals = df[true_col].values

    # Convert
    if pred_type == 'norm':
        pred_rad = to_radians_from_norm(pred_vals)
    elif pred_type == 'hours':
        pred_rad = to_radians_from_hours(pred_vals)
    else:  # radians
        pred_rad = to_radians_from_radians(pred_vals)

    # true: guess mapping
    true_arr = np.array(true_vals, dtype=float)
    if true_arr.max() <= 1.0:
        true_rad = to_radians_from_norm(true_arr)
    elif true_arr.max() <= (2 * math.pi + 0.1):
        true_rad = to_radians_from_radians(true_arr)
    else:
        true_rad = to_radians_from_hours(true_arr)

    # determine tissue name from filename (expect pattern ..._vs_<tissue>.predictions.csv)
    fname = os.path.basename(pred_path)
    tissue = None
    if '_vs_' in fname:
        # take the part after the last _vs_
        tissue = fname.split('_vs_')[-1]
    else:
        tissue = os.path.splitext(fname)[0]
    # strip known suffixes
    for suf in ['.predictions.csv', '.predictions', '.csv']:
        if tissue.endswith(suf):
            tissue = tissue[: -len(suf)]

    out_dir_final = out_dir if out_dir else os.path.dirname(pred_path)
    os.makedirs(out_dir_final, exist_ok=True)
    out_path = os.path.join(out_dir_final, f'{tissue}_phase_vs_time.png')

    plot_pair(true_rad, pred_rad, out_path, tissue=tissue)
    return True


def main():
    p = argparse.ArgumentParser(description='Batch plot phase vs time for predictions CSVs')
    p.add_argument('--results-dir', required=True, help='Directory containing prediction CSVs')
    p.add_argument('--out-dir', default=None, help='Directory to write PNGs (defaults to same folder as CSVs)')
    p.add_argument('--ext', default='.predictions.csv', help='Filename extension/pattern for predictions files')
    args = p.parse_args()

    files = find_prediction_files(args.results_dir, ext=args.ext)
    if not files:
        print('No prediction files found in', args.results_dir, file=sys.stderr)
        sys.exit(2)

    for f in files:
        try:
            process_file(f, out_dir=args.out_dir)
        except Exception as e:
            print('Error processing', f, e, file=sys.stderr)


if __name__ == '__main__':
    main()
