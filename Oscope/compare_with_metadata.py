import argparse
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def circ_corr(a, b):
    # a,b in radians
    a_bar = math.atan2(np.sum(np.sin(a)), np.sum(np.cos(a)))
    b_bar = math.atan2(np.sum(np.sin(b)), np.sum(np.cos(b)))
    num = np.sum(np.sin(a - a_bar) * np.sin(b - b_bar))
    den = np.sqrt(np.sum(np.sin(a - a_bar) ** 2) * np.sum(np.sin(b - b_bar) ** 2))
    return float(num / den) if den > 0 else np.nan


def angle_diff_rad(a, b):
    # minimal signed difference a-b in [-pi, pi)
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell_order', type=Path, default=Path('cell_order/consensus_cell_order.csv'))
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--sample_col', type=str, default='Sample')
    ap.add_argument('--time_col', type=str, default='Time_Hours')
    ap.add_argument('--outdir', type=Path, default=Path('cell_order/compare_with_metadata'))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    co = pd.read_csv(args.cell_order)
    # normalize column names
    co_cols = {c.lower(): c for c in co.columns}
    sample_col_co = 'sample' if 'sample' in co_cols else list(co.columns)[0]
    # choose time column from cell_order (prefer 24h if present)
    if 'pseudotime_24h' in co.columns:
        pred_time = co['pseudotime_24h'].astype(float)
    elif 'pseudotime_deg' in co.columns:
        pred_time = (co['pseudotime_deg'].astype(float) / 360.0) * 24.0
    elif 'pseudotime_norm' in co.columns:
        pred_time = co['pseudotime_norm'].astype(float) * 24.0
    else:
        raise ValueError('No pseudotime column found in cell_order CSV')

    co = pd.DataFrame({
        'Sample': co[sample_col_co].astype(str),
        'Pred_24h': (pred_time % 24.0)
    })

    md = pd.read_csv(args.metadata)
    # normalize metadata col names
    if args.sample_col not in md.columns:
        # try common fallbacks
        for c in md.columns:
            if c.lower() == 'sample':
                args.sample_col = c
                break
    if args.time_col not in md.columns:
        for c in md.columns:
            if c.lower() in ('time_hours','time','hour','hours'):
                args.time_col = c
                break
    md = md[[args.sample_col, args.time_col]].rename(columns={args.sample_col: 'Sample', args.time_col: 'Meta_24h'})
    md['Meta_24h'] = md['Meta_24h'].astype(float) % 24.0

    df = pd.merge(co, md, on='Sample', how='inner')
    missed_pred = len(co) - len(df)
    missed_meta = len(md) - len(df)

    if len(df) == 0:
        raise SystemExit('No matching samples between cell_order and metadata')

    pred = df['Pred_24h'].to_numpy()
    meta = df['Meta_24h'].to_numpy()

    # convert to radians
    a = pred / 24.0 * 2 * np.pi
    b = meta / 24.0 * 2 * np.pi

    # circular correlation (rotation-invariant)
    rho_circ = circ_corr(a, b)

    # mean signed phase lead (pred - meta) in radians -> hours
    diffs = angle_diff_rad(a, b)
    mean_diff = math.atan2(np.sum(np.sin(diffs)), np.sum(np.cos(diffs)))
    mean_lead_hours = (mean_diff / (2 * np.pi)) * 24.0  # positive: pred leads meta

    # apply alignment by mean_diff for visualization and compute linear Pearson R
    a_aligned = (a - mean_diff)
    pred_aligned = (a_aligned % (2 * np.pi)) / (2 * np.pi) * 24.0
    # linear Pearson on wrapped hours (approximate)
    R = np.corrcoef(pred_aligned, meta)[0,1]

    # save metrics
    metrics = {
        'n_matches': len(df),
        'missing_in_metadata': int(missed_pred),
        'missing_in_cell_order': int(missed_meta),
        'circular_correlation': float(rho_circ),
        'mean_phase_lead_hours': float(mean_lead_hours),
        'pearson_R_after_align': float(R),
    }
    pd.Series(metrics).to_csv(args.outdir / 'metrics_consensus_vs_metadata.csv')

    # scatter plot (hours scale)
    plt.figure(figsize=(6,5), dpi=140)
    plt.scatter(meta, pred_aligned, s=24, c='black', alpha=0.7)
    plt.xlabel('Metadata time (hours)')
    plt.ylabel('Predicted pseudotime (hours, aligned)')
    plt.title(f"Consensus vs Metadata\nR={R:.3f}, circ_r={rho_circ:.3f}, lead={mean_lead_hours:.2f}h, n={len(df)}")
    plt.xlim(0,24)
    plt.ylim(0,24)
    plt.grid(alpha=0.3)
    # reference diagonal lines with wrap cues
    xs = np.linspace(0,24,200)
    plt.plot(xs, xs, color='steelblue', lw=1, alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.outdir / 'consensus_vs_metadata_scatter.png')
    plt.close()

    # also write the matched table (hours)
    df_out = df.copy()
    df_out['Pred_24h_aligned'] = pred_aligned
    df_out.to_csv(args.outdir / 'matched_consensus_vs_metadata.csv', index=False)

    # radians-scale plot [0, 2π]
    meta_rad = b % (2*np.pi)
    pred_rad_aligned = a_aligned % (2*np.pi)
    # linear correlation on wrapped radians (approximate)
    R_rad = np.corrcoef(pred_rad_aligned, meta_rad)[0,1]
    plt.figure(figsize=(6,5), dpi=140)
    plt.scatter(meta_rad, pred_rad_aligned, s=24, c='black', alpha=0.7)
    plt.xlabel('Metadata angle (radians)')
    plt.ylabel('Predicted angle (radians, aligned)')
    plt.title(f"Consensus vs Metadata (radians)\nR={R_rad:.3f}, circ_r={rho_circ:.3f}, lead={mean_diff:.2f} rad, n={len(df)}")
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 2*np.pi)
    plt.grid(alpha=0.3)
    xsr = np.linspace(0, 2*np.pi, 200)
    plt.plot(xsr, xsr, color='steelblue', lw=1, alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.outdir / 'consensus_vs_metadata_scatter_radians.png')
    plt.close()

    # save metrics (radians as well)
    pd.Series({
        'pearson_R_after_align_radians': float(R_rad),
        'mean_phase_lead_radians': float(mean_diff)
    }).to_csv(args.outdir / 'metrics_consensus_vs_metadata_radians.csv')

    # per-cluster comparisons if available in same folder
    co_dir = args.cell_order.parent
    cluster_files = sorted(co_dir.glob('cluster_*_cell_order.csv'))
    rows = []
    for cf in cluster_files:
        sub = pd.read_csv(cf)
        if 'sample' not in {c.lower() for c in sub.columns}:
            continue
        # reuse detection logic
        if 'pseudotime_24h' in sub.columns:
            pt = sub['pseudotime_24h'].astype(float)
        elif 'pseudotime_deg' in sub.columns:
            pt = (sub['pseudotime_deg'].astype(float) / 360.0) * 24.0
        elif 'pseudotime_norm' in sub.columns:
            pt = sub['pseudotime_norm'].astype(float) * 24.0
        else:
            continue
        sub = pd.DataFrame({'Sample': sub[[c for c in sub.columns if c.lower()=='sample'][0]].astype(str),
                            'Pred_24h': pt % 24.0})
        m = pd.merge(sub, md, on='Sample', how='inner')
        if len(m) == 0:
            continue
        a2 = (m['Pred_24h'].to_numpy() / 24.0) * 2*np.pi
        b2 = (m['Meta_24h'].to_numpy() / 24.0) * 2*np.pi
        rho2 = circ_corr(a2, b2)
        dif2 = angle_diff_rad(a2, b2)
        mean2 = math.atan2(np.sum(np.sin(dif2)), np.sum(np.cos(dif2)))
        lead2 = (mean2 / (2*np.pi)) * 24.0
        a2_align = (a2 - mean2)
        pred2_align = (a2_align % (2*np.pi)) / (2*np.pi) * 24.0
        R2 = np.corrcoef(pred2_align, m['Meta_24h'].to_numpy())[0,1]
        rows.append({'cluster_file': cf.name, 'n_matches': len(m), 'circular_correlation': rho2,
                     'mean_phase_lead_hours': lead2, 'pearson_R_after_align': R2})
    if rows:
        pd.DataFrame(rows).to_csv(args.outdir / 'metrics_per_cluster_vs_metadata.csv', index=False)

    print('Comparison complete. Outputs in', args.outdir)

if __name__ == '__main__':
    main()
