#!/usr/bin/env python3
"""
Combine expression matrices and metadata from multiple tissue folders into a single training set.
Usage:
  python scripts/combine_train_tissues.py --tissues <t1,t2,...> --base data/GSE54651 --out-dir tmp/train_combined

The script expects each tissue folder to contain expression.csv and metadata.csv.
It will take the intersection of genes across all tissues and write two files:
  <out-dir>/expression.csv  (genes x samples, first column = gene names)
  <out-dir>/metadata.csv    (rows correspond to samples, includes Sample column)

This script does not modify any input CSVs.
"""

import argparse
import os
import pandas as pd
import sys


def read_expr(path):
    # read with first column as index (gene names or probe ids)
    df = pd.read_csv(path, index_col=0, check_default_na=False)
    return df


def read_meta(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tissues', required=True, help='Comma-separated tissue folder names (relative to base)')
    p.add_argument('--base', required=True, help='Base path for tissues (e.g. data/GSE54651)')
    p.add_argument('--out-dir', required=True, help='Output directory for combined files')
    args = p.parse_args()

    tissues = [t.strip() for t in args.tissues.split(',') if t.strip()]
    base = args.base
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    exprs = []
    metas = []
    missing = []
    for t in tissues:
        expr_path = os.path.join(base, t, 'expression.csv')
        meta_path = os.path.join(base, t, 'metadata.csv')
        if not os.path.exists(expr_path) or not os.path.exists(meta_path):
            missing.append(t)
            continue
        expr = read_expr(expr_path)
        meta = read_meta(meta_path)
        # ensure samples in meta are present in expr columns or rows
        exprs.append((t, expr))
        metas.append((t, meta))

    if missing:
        print('Warning: missing expression/metadata for tissues:', missing, file=sys.stderr)

    if not exprs:
        print('No expression files found; aborting', file=sys.stderr)
        sys.exit(2)

    # compute gene intersection (index intersection)
    gene_sets = [e.index.astype(str).tolist() for (_, e) in exprs]
    common = set(gene_sets[0]).intersection(*gene_sets[1:]) if len(gene_sets) > 1 else set(gene_sets[0])
    common = sorted(common)
    if not common:
        print('No intersecting genes found across tissues; aborting', file=sys.stderr)
        sys.exit(3)

    # subset and concatenate expressions (genes x samples -> transpose later if needed)
    expr_subs = []
    for t, e in exprs:
        # subset to common genes
        e2 = e.loc[e.index.astype(str).isin(common), :]
        # keep original column names (samples)
        expr_subs.append(e2)

    combined = pd.concat(expr_subs, axis=1)

    # Build combined metadata: we need a Sample column matching the sample names in combined
    # For each tissue meta, attempt to find sample column; otherwise use first column
    meta_rows = []
    for t, meta in metas:
        # determine sample column name
        sample_col = None
        for candidate in ['Sample', 'sample', 'ID', meta.columns[0]]:
            if candidate in meta.columns:
                sample_col = candidate
                break
        if sample_col is None:
            # fallback: create sample names from expression columns for this tissue
            # find expression for this tissue
            expr_for_t = dict(exprs)[t]
            samples = list(expr_for_t.columns)
            for s in samples:
                # minimal metadata row
                meta_rows.append({'Sample': s})
            continue
        # ensure sample values are strings
        meta = meta.copy()
        meta['Sample'] = meta[sample_col].astype(str)
        # filter metadata to samples that exist in combined columns
        meta_filtered = meta[meta['Sample'].isin(combined.columns)]
        meta_rows.append(meta_filtered)

    if not meta_rows:
        print('No metadata rows found for combined samples; aborting', file=sys.stderr)
        sys.exit(4)

    meta_combined = pd.concat(meta_rows, axis=0, ignore_index=True)
    # ensure ordering matches combined.columns
    # keep only rows for samples in combined
    meta_combined = meta_combined[meta_combined['Sample'].isin(combined.columns)]
    meta_combined = meta_combined.set_index('Sample').loc[combined.columns].reset_index()

    # write outputs: expression with gene as first column
    expr_out = os.path.join(out_dir, 'expression.csv')
    meta_out = os.path.join(out_dir, 'metadata.csv')
    combined_out = combined.copy()
    combined_out.insert(0, 'Gene', combined_out.index)
    combined_out.reset_index(drop=True, inplace=True)
    combined_out.to_csv(expr_out, index=False)
    meta_combined.to_csv(meta_out, index=False)
    print('Wrote combined expression:', expr_out)
    print('Wrote combined metadata:', meta_out)


if __name__ == '__main__':
    main()
