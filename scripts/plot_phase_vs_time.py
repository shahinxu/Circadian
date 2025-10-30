#!/usr/bin/env python3
"""
Plot true sample collection time vs predicted phase (both in radians 0-2pi).

Usage:
  python scripts/plot_phase_vs_time.py --pred <predictions.csv> --meta <metadata.csv> [--sample-col Sample] [--true-time-col Time_Hours] [--out out.png]

The script will automatically detect prediction columns: prefers columns named 'Phases_AG*', then 'pred_time_norm', then 'pred_time_hours'.
If pred_time_norm is present it will be mapped to radians by *2*pi. If pred_time_hours is present it will be mapped by /24*2*pi.
True times are taken from --true-time-col (default 'Time_Hours') and mapped to radians by /24*2*pi.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_pred_column(df):
    # prefer Phases_AG*, then pred_time_norm, then pred_time_hours, then timePred
    for col in df.columns:
        if col.startswith('Phases_AG'):
            return col, 'phases_ag'
    if 'pred_time_norm' in df.columns:
        return 'pred_time_norm', 'norm'
    if 'pred_time_hours' in df.columns:
        return 'pred_time_hours', 'hours'
    # common alternative names
    for c in df.columns:
        if c.lower().startswith('timepred') or c.lower().startswith('time_pred'):
            return c, 'norm_or_hours'
    return None, None


def to_radians_from_norm(x):
    return (np.array(x, dtype=float) % 1.0) * 2 * np.pi


def to_radians_from_hours(x):
    return (np.array(x, dtype=float) % 24.0) * 2 * np.pi / 24.0


def main():
    p = argparse.ArgumentParser(description='Plot true time vs predicted phase (radians)')
    p.add_argument('--pred', required=True, help='Predictions CSV (from pipeline)')
    p.add_argument('--meta', required=True, help='Metadata CSV')
    p.add_argument('--sample-col', default='sample', help='Sample ID column name in both files (default: sample)')
    p.add_argument('--true-time-col', default='Time_Hours', help='True time column in metadata (default: Time_Hours)')
    p.add_argument('--out', default=None, help='Output PNG path (default: <predbasename>_phase_vs_time.png)')
    args = p.parse_args()

    pred_path = args.pred
    meta_path = args.meta
    if not os.path.exists(pred_path):
        print('Predictions file not found:', pred_path, file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(meta_path):
        print('Metadata file not found:', meta_path, file=sys.stderr)
        sys.exit(2)

    fit_df = pd.read_csv(pred_path)
    meta_df = pd.read_csv(meta_path)

    # sample name column: try given, then common variants
    sample_col = args.sample_col
    if sample_col not in fit_df.columns:
        # try capitalized
        if sample_col.capitalize() in fit_df.columns:
            sample_col = sample_col.capitalize()
    if sample_col not in fit_df.columns:
        # fallback to first col
        sample_col = fit_df.columns[0]

    if sample_col not in meta_df.columns:
        if sample_col.capitalize() in meta_df.columns:
            sample_col = sample_col.capitalize()
    if sample_col not in meta_df.columns:
        # try 'Sample' or first col
        if 'Sample' in meta_df.columns:
            meta_sample_col = 'Sample'
        else:
            meta_sample_col = meta_df.columns[0]
    else:
        meta_sample_col = sample_col

    fit_samples = fit_df[sample_col]
    meta_samples = meta_df[meta_sample_col]

    pred_col, pred_type = find_pred_column(fit_df)
    if pred_col is None:
        raise ValueError('Could not find a prediction column in predictions CSV. Looked for Phases_AG*, pred_time_norm, pred_time_hours, etc.')

    # build maps
    fit_map = dict(zip(fit_samples.astype(str), fit_df[pred_col]))

    # try to find true time in metadata
    true_time_col = args.true_time_col
    if true_time_col not in meta_df.columns:
        # try variants
        if 'Hour_in_24' in meta_df.columns:
            true_time_col = 'Hour_in_24'
        elif 'Time_Hours' in meta_df.columns:
            true_time_col = 'Time_Hours'
        elif 'Time_Phase' in meta_df.columns:
            true_time_col = 'Time_Phase'
        elif 'time' in meta_df.columns:
            true_time_col = 'time'
        else:
            true_time_col = meta_df.columns[0]

    meta_map = dict(zip(meta_samples.astype(str), meta_df[true_time_col]))

    common_samples = sorted(set(fit_map.keys()) & set(meta_map.keys()))
    if len(common_samples) == 0:
        raise ValueError('No intersection between prediction samples and metadata samples')

    # get arrays
    preds = [fit_map[s] for s in common_samples]
    trues = [meta_map[s] for s in common_samples]

    # convert to radians
    if pred_type == 'phases_ag':
        # assume phases_ag already in radians
        pred_rad = np.array(preds, dtype=float) % (2 * np.pi)
    elif pred_type == 'norm':
        pred_rad = to_radians_from_norm(preds)
    elif pred_type == 'hours':
        pred_rad = to_radians_from_hours(preds)
    else:
        # try guess: if all <=1 treat as norm else hours
        arr = np.array(preds, dtype=float)
        if arr.max() <= 1.0:
            pred_rad = to_radians_from_norm(arr)
        else:
            pred_rad = to_radians_from_hours(arr)

    # true times: if values <=1 assume normalized; if >2pi treat as hours? We'll map hours or radians
    true_arr = np.array(trues, dtype=float)
    # prefer hours mapping (common in metadata)
    if true_arr.max() <= 1.0:
        true_rad = (true_arr % 1.0) * 2 * np.pi
    elif true_arr.max() <= (2 * np.pi + 0.1):
        # radians
        true_rad = (true_arr % (2 * np.pi))
    else:
        true_rad = to_radians_from_hours(true_arr)

    # plot
    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(pred_path))[0]
        out_path = os.path.join('.', f'{base}_phase_vs_time.png')

    plt.figure(figsize=(8, 7))
    plt.scatter(true_rad, pred_rad, c='r', label='Phase vs. Time')
    plt.xlabel('Collection Phase (radians)', fontsize=14)
    plt.ylabel('Predicted Phase (radians)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'Saved plot to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
