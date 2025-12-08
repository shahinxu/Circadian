import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# List of genes to plot (from user)
GENES = [
    "Arntl", "Clock", "Npas2", "Nr1d1", "Bhlhe41", "Nr1d2",
    "Dbp", "Ciart", "Per1", "Per3", "Tef", "Hlf",
    "Cry2", "Per2", "Cry1", "Rorc", "Nfil3"
]

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results', 'GTEx')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'GTEx_acrophase_by_gene')
os.makedirs(OUT_DIR, exist_ok=True)


def find_latest_fit_csv(tissue_dir):
    # tissue_dir: GREED/results/GTEx/<tissue>
    subdirs = [d for d in glob.glob(os.path.join(tissue_dir, '*')) if os.path.isdir(d)]
    if not subdirs:
        return None
    # timestamps are formatted like YYYY-MM-DDTHH_MM_SS; lexicographic max works
    latest = sorted(subdirs)[-1]
    fit_path = os.path.join(latest, 'fit_parameters.csv')
    if os.path.isfile(fit_path):
        return fit_path
    return None


def read_fit_table(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # try to find gene column name
    gene_col = None
    for c in df.columns:
        if c.lower() in ('gene', 'gene_symbol', 'gene_name'):
            gene_col = c
            break
    if gene_col is None:
        # assume first column is gene-like
        gene_col = df.columns[0]
    df = df.copy()
    df['GENE_UP'] = df[gene_col].astype(str).str.upper()
    # ensure amplitude and phase columns exist (try common names)
    phase_col = None
    amp_col = None
    for c in df.columns:
        if c.lower() in ('phase_rad', 'phase', 'phase_rad_rad', 'phase_radians'):
            phase_col = c
        if c.lower() in ('amplitude', 'amp'):
            amp_col = c
    # fallback guesses
    if phase_col is None:
        # try to infer column containing 'phase' substring
        for c in df.columns:
            if 'phase' in c.lower():
                phase_col = c
                break
    if amp_col is None:
        for c in df.columns:
            if 'ampl' in c.lower() or 'amp' == c.lower():
                amp_col = c
                break
    if phase_col is None:
        return None
    if amp_col is None:
        # if amplitude not present, add default 1.0
        df['amplitude_fill'] = 1.0
        amp_col = 'amplitude_fill'
    df = df[[gene_col, 'GENE_UP', phase_col, amp_col]].rename(columns={phase_col: 'phase_rad', amp_col: 'amplitude'})
    return df


def aggregate_acrophases():
    tissues = sorted([os.path.basename(d) for d in glob.glob(os.path.join(RESULTS_ROOT, '*')) if os.path.isdir(d)])
    all_entries = []
    for tissue in tissues:
        tissue_dir = os.path.join(RESULTS_ROOT, tissue)
        fit_csv = find_latest_fit_csv(tissue_dir)
        if fit_csv is None:
            continue
        df = read_fit_table(fit_csv)
        if df is None:
            continue
        for _, row in df.iterrows():
            all_entries.append({
                'tissue': tissue,
                'GENE_UP': row['GENE_UP'],
                'phase_rad': float(row['phase_rad']),
                'amplitude': float(row['amplitude'])
            })
    if not all_entries:
        print('No fit parameters found in results.')
        return None
    return pd.DataFrame(all_entries)


def plot_for_gene(gene, df_all):
    gene_up = gene.upper()
    df_gene = df_all[df_all['GENE_UP'] == gene_up].copy()
    if df_gene.empty:
        print(f'No data for gene {gene} across tissues; skipping')
        return
    # convert phase to numeric and normalize to [0, 2pi)
    df_gene['phase_rad'] = df_gene['phase_rad'].astype(float) % (2 * pi)
    # radius: amplitude (ensure non-negative)
    df_gene['amplitude'] = np.abs(df_gene['amplitude'].astype(float))

    # Polar plot (no text labels, no CSV output)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='polar')
    angles = df_gene['phase_rad'].values
    radii = df_gene['amplitude'].values
    sc = ax.scatter(angles, radii, c='C0', s=40)
    # no textual annotations or CSV outputs per user request
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    # remove tick labels to avoid any text on the figure
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    out_png = os.path.join(OUT_DIR, f'{gene}_acrophases_circle.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'Saved: {out_png}')


def main():
    df_all = aggregate_acrophases()
    if df_all is None:
        return
    for gene in GENES:
        plot_for_gene(gene, df_all)


if __name__ == '__main__':
    main()
