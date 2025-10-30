#!/usr/bin/env python3
"""
Combine multiple tissue expression.csv and metadata.csv pairs into a single
expression and metadata CSV without overwriting originals.

Behavior:
- For each tissue directory provided, reads `expression.csv` (genes x samples,
  first column treated as gene id/index) and `metadata.csv`.
- Drops expression columns that do not appear in the metadata sample column.
- Reorders metadata to match expression column order for that tissue.
- Computes intersection of genes across all tissues and subsets each expression
  matrix to that common gene set (ensures a rectangular matrix for downstream
  tools).
- Concatenates expression matrices (genes x samples) by columns and metadata
  rows by samples. Adds a `tissue_source` column to metadata.
- Writes outputs to `<out_prefix>.expression.csv` and `<out_prefix>.metadata.csv`.

This script is safe: it does not overwrite original files. It emits diagnostics
about dropped samples and resulting dimensions.
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path


def guess_sample_col(df):
    for c in ("Sample", "sample", "SampleID", "sample_id", "sampleId"):
        if c in df.columns:
            return c
    # fallback to the first column
    return df.columns[0]


def read_expr(path):
    # read with first column as index (gene/probe id)
    return pd.read_csv(path, index_col=0)


def read_meta(path):
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Combine GSE54651 tissues into one expression+metadata set")
    parser.add_argument("--tissues", nargs='+', help="List of tissue directories (paths) to combine", required=True)
    parser.add_argument("--out-prefix", default="data/GSE54651/combined", help="Output prefix for combined files (default: data/GSE54651/combined)")
    parser.add_argument("--sample-col", default=None, help="Column name in metadata that contains sample names. If omitted, script will guess.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs; just print diagnostics")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    tissues = [Path(p) for p in args.tissues]
    exprs = []
    metas = []
    tissue_sources = []

    gene_sets = []

    for t in tissues:
        expr_path = t / "expression.csv"
        meta_path = t / "metadata.csv"
        if not expr_path.exists() or not meta_path.exists():
            print(f"Skipping {t}: missing expression.csv or metadata.csv", file=sys.stderr)
            continue
        if args.verbose:
            print(f"Reading expression: {expr_path}")
            print(f"Reading metadata:   {meta_path}")

        expr = read_expr(expr_path)
        meta = read_meta(meta_path)

        sample_col = args.sample_col or guess_sample_col(meta)
        if sample_col not in meta.columns:
            print(f"Warning: guessed sample column '{sample_col}' not in metadata columns for {t}; available columns: {list(meta.columns)}", file=sys.stderr)

        meta_samples = meta[sample_col].astype(str).tolist()
        expr_samples = [str(c) for c in expr.columns]

        # find intersection
        common = [s for s in expr_samples if s in set(meta_samples)]
        dropped = len(expr_samples) - len(common)
        if args.verbose or dropped>0:
            print(f"Tissue {t.name}: expr samples {len(expr_samples)}, metadata samples {len(meta_samples)}, dropping {dropped} expr samples not present in metadata")

        # subset expression to common samples and reorder columns to match metadata order
        expr_sub = expr.loc[:, common]

        # subset metadata to matching samples and reorder
        meta_sub = meta[meta[sample_col].astype(str).isin(set(common))].copy()
        # reorder meta_sub so rows follow expr_sub.columns order
        meta_sub[sample_col] = meta_sub[sample_col].astype(str)
        meta_sub = meta_sub.set_index(sample_col).loc[expr_sub.columns].reset_index()

        exprs.append(expr_sub)
        # add a tissue_source column for traceability
        meta_sub['tissue_source'] = t.name
        metas.append(meta_sub)
        gene_sets.append(set(expr_sub.index))

    if len(exprs) == 0:
        print("No valid tissues found or no expression files read. Exiting.")
        sys.exit(1)

    # intersect genes across tissues
    common_genes = set.intersection(*gene_sets)
    if args.verbose:
        print(f"Common genes across tissues: {len(common_genes)}")

    if len(common_genes) == 0:
        print("No common genes across the provided tissues. Exiting.")
        sys.exit(1)

    common_genes = sorted(common_genes)

    # subset each expression to common genes
    exprs_sub = [e.loc[common_genes] for e in exprs]

    # concatenate columns
    combined_expr = pd.concat(exprs_sub, axis=1)

    # concatenate metadata (they already are filtered to the samples that appear in expr)
    combined_meta = pd.concat(metas, axis=0, ignore_index=True)

    out_expr = Path(f"{args.out_prefix}.expression.csv")
    out_meta = Path(f"{args.out_prefix}.metadata.csv")

    print(f"Combined expression: genes {combined_expr.shape[0]} x samples {combined_expr.shape[1]}")
    print(f"Combined metadata: rows {combined_meta.shape[0]} columns {combined_meta.shape[1]}")

    if args.dry_run:
        print("Dry-run: not writing files")
        return

    # ensure output directory exists
    out_expr.parent.mkdir(parents=True, exist_ok=True)

    # write expression with gene ids as first column (index)
    combined_expr.to_csv(out_expr)
    combined_meta.to_csv(out_meta, index=False)

    print(f"Wrote combined expression to {out_expr}")
    print(f"Wrote combined metadata to {out_meta}")


if __name__ == '__main__':
    main()
