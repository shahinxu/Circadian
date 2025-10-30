#!/usr/bin/env python3
"""
Merge multiple tissue expression matrices (genes x samples) by gene intersection and concatenate samples.
Writes two files: combined expression (genes as first column 'Gene') and combined metadata.

Usage:
  python scripts/merge_train_tissues.py --base data/GSE54651 --tissues cerebellum colon ... --out-prefix scripts/tmp/train_combined

The script creates <out-prefix>.expression.csv and <out-prefix>.metadata.csv
"""
import argparse
import os
import sys
import pandas as pd


def read_expression(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # heuristic: if first column header contains 'gene' or 'probe' treat as gene column
    first_col = df.columns[0]
    if 'gene' in first_col.lower() or 'probe' in first_col.lower() or 'symbol' in first_col.lower():
        df = df.set_index(first_col)
        # convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    else:
        # if many columns are numeric, assume first column is gene names
        try:
            maybe = df.copy()
            maybe_index = maybe.iloc[:,0]
            maybe_vals = maybe.iloc[:,1:]
            # if majority numeric
            num_numeric = (maybe_vals.applymap(lambda x: str(x).replace('.','',1).lstrip('-').isdigit()).sum().sum())
            if num_numeric > 0:
                maybe = maybe.set_index(maybe_index.name)
                maybe = maybe.apply(pd.to_numeric, errors='coerce')
                return maybe
        except Exception:
            pass
        # fallback: try to transpose if rows look like genes
        df2 = df.set_index(df.columns[0])
        df2 = df2.apply(pd.to_numeric, errors='coerce')
        return df2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--tissues', nargs='+', required=True)
    p.add_argument('--out-prefix', default='scripts/tmp/train_combined')
    args = p.parse_args()

    expr_list = []
    meta_list = []
    for t in args.tissues:
        expr_path = os.path.join(args.base, t, 'expression.csv')
        meta_path = os.path.join(args.base, t, 'metadata.csv')
        if not os.path.exists(expr_path) or not os.path.exists(meta_path):
            print('Missing files for tissue', t, file=sys.stderr)
            sys.exit(2)
        expr = read_expression(expr_path)
        # ensure index (genes) and columns (samples)
        expr_list.append(expr)
        meta = pd.read_csv(meta_path, dtype=str)
        meta_list.append(meta)

    # intersect genes
    gene_sets = [set(df.index) for df in expr_list]
    common = sorted(set.intersection(*gene_sets))
    if len(common) == 0:
        print('No common genes across tissues', file=sys.stderr)
        sys.exit(3)

    # subset and concat columns
    expr_sub = [df.loc[common] for df in expr_list]
    combined = pd.concat(expr_sub, axis=1)

    # combined metadata: concat rows
    combined_meta = pd.concat(meta_list, axis=0, ignore_index=True)

    # Ensure Sample column exists
    if 'Sample' not in combined_meta.columns and 'sample' not in combined_meta.columns:
        # try to set from expression columns
        combined_meta['Sample'] = combined.columns.tolist()

    out_expr = args.out_prefix + '.expression.csv'
    out_meta = args.out_prefix + '.metadata.csv'
    os.makedirs(os.path.dirname(out_expr), exist_ok=True)
    # write expression with Gene as first column
    out_df = combined.copy()
    out_df.insert(0, 'Gene', out_df.index)
    out_df.to_csv(out_expr, index=False)
    combined_meta.to_csv(out_meta, index=False)
    print('Wrote', out_expr, 'and', out_meta)


if __name__ == '__main__':
    main()
