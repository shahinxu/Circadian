import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_GENES = [
    "ARNTL", "CLOCK", "NPAS2", "NR1D1", "BHLHE41", "NR1D2", "DBP",
    "CIART", "PER1", "PER2", "PER3", "TEF", "HLF", "CRY1", "CRY2",
    "RORC", "NFIL3"
]


def find_fit_file(base: Path) -> Path:
    """Given a CYCLOPS results directory, pick a Fit_Output_*.csv file.

    If base itself is a Fit_Output_*.csv, just return it.
    Otherwise, search under base for files starting with "Fit_Output_" and
    choose the lexicographically latest one (usually the most recent).
    """

    base = base.resolve()
    if base.is_file() and base.name.startswith("Fit_Output_") and base.suffix == ".csv":
        return base

    if not base.is_dir():
        raise FileNotFoundError(f"Results path not found or not a directory: {base}")

    candidates = sorted(
        p for p in base.glob("Fit_Output_*.csv") if p.is_file()
    )
    if not candidates:
        # Allow searching one level down in case user points to higher directory
        candidates = sorted(
            p for p in base.rglob("Fit_Output_*.csv") if p.is_file()
        )
    if not candidates:
        raise FileNotFoundError(f"No Fit_Output_*.csv found under: {base}")
    return candidates[-1]


def map_fit_to_expression(fit_path: Path) -> Path:
    """Map a Fit_Output path to the corresponding expression.csv.

    Assumes the Circadian project layout, with CYCLOPS-2.0/results/... and
    data/... mirroring dataset names, e.g.:

    CYCLOPS-2.0/results/data_AD427/ Fit_Output_...csv
    -> data/AD427/expression.csv
    """

    fit_path = fit_path.resolve()

    # Expect something like .../Circadian/CYCLOPS-2.0/results/<name>/Fit_Output_...
    parts = list(fit_path.parts)
    try:
        circ_idx = next(i for i, p in enumerate(parts) if p == "Circadian")
    except StopIteration:
        raise ValueError(
            "Path does not contain 'Circadian'; cannot infer data directory. "
            "Please pass --expression explicitly."
        )

    project_root = Path(*parts[: circ_idx + 1])

    # results dir name immediately under CYCLOPS-2.0/results
    # e.g. "data_AD427" -> dataset "AD427"
    if "results" not in parts:
        raise ValueError(
            "Fit path does not contain 'results'; cannot infer dataset name. "
            "Please pass --expression explicitly."
        )
    res_idx = parts.index("results")
    if res_idx + 1 >= len(parts):
        raise ValueError("Cannot determine dataset directory from fit path")

    result_dataset = parts[res_idx + 1]

    # Heuristic: if starts with "data_", strip that prefix
    dataset_name = result_dataset
    if dataset_name.lower().startswith("data_"):
        dataset_name = dataset_name[len("data_") :]

    expr_candidate = project_root / "data" / dataset_name / "expression.csv"
    if expr_candidate.exists():
        return expr_candidate

    # Fallback: try interpreting result_dataset directly
    expr_alt = project_root / "data" / result_dataset / "expression.csv"
    if expr_alt.exists():
        return expr_alt

    raise FileNotFoundError(
        "Could not map Fit_Output to expression.csv. Tried: "
        f"{expr_candidate} and {expr_alt}. Use --expression to specify it."
    )


def load_expression_for_genes(expr_file: Path, genes_upper, sample_ids):
    """Load expression values for given genes and samples.

    - expr_file: path to expression.csv
    - genes_upper: list of uppercase gene symbols
    - sample_ids: ordered sample IDs to match the phases from Fit_Output
    Returns a DataFrame with index=sample_ids and columns=genes_upper.
    """

    df = pd.read_csv(expr_file, low_memory=False)
    if "Gene_Symbol" not in df.columns:
        raise ValueError(f"expression.csv missing 'Gene_Symbol' column: {expr_file}")

    df["GENE_UP"] = df["Gene_Symbol"].astype(str).str.upper()

    # Use sample IDs that exist in the expression file
    sample_cols = [sid for sid in sample_ids if sid in df.columns]
    if not sample_cols:
        raise ValueError(
            "None of the Fit_Output sample IDs were found as columns in "
            f"{expr_file}. Check that dataset and results correspond."
        )

    for c in sample_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    grp = df.groupby("GENE_UP")[sample_cols].mean()

    data = {}
    for g in genes_upper:
        if g in grp.index:
            row = grp.loc[g].reindex(sample_ids).values
        else:
            row = np.full(len(sample_ids), np.nan)
        data[g] = row

    return pd.DataFrame(data, index=sample_ids)


def fit_cosine(times_hours, y):
    """Fit a simple 24h cosine: y ~ a + b*cos(w t) + c*sin(w t)."""

    mask = ~np.isnan(y) & np.isfinite(times_hours)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan  # a, amplitude, phase_rad

    x = times_hours[mask]
    v = y[mask]
    theta = x * (2 * np.pi / 24.0)
    A = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    try:
        coeffs, *_ = np.linalg.lstsq(A, v, rcond=None)
    except Exception:
        return np.nan, np.nan, np.nan

    a, b, c = coeffs
    amp = float(np.hypot(b, c))
    phase = float(np.arctan2(c, b) % (2 * np.pi))
    return float(a), amp, phase


def plot_expression_vs_phase(expr_by_gene, phases_hours, genes, out_png):
    """Scatter + cosine fit for each gene across 0–24h phase."""

    genes = list(genes)
    n = len(genes)
    if n == 0:
        print("No genes to plot.")
        return

    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)

    x_grid = np.linspace(0, 24, 300)

    fit_records = []

    for i, g in enumerate(genes):
        ax = axes[i]
        if g not in expr_by_gene.columns:
            ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes)
            ax.set_title(g)
            ax.set_xlabel("Phase (hours)")
            continue

        y = expr_by_gene[g].values.astype(float)
        mask = ~np.isnan(y) & np.isfinite(phases_hours)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
            ax.set_title(g)
            ax.set_xlabel("Phase (hours)")
            continue

        x = phases_hours[mask]
        v = y[mask]

        a, amp, phase_rad = fit_cosine(phases_hours, y)

        ax.scatter(x, v, s=18, alpha=0.8)

        if np.isfinite(amp):
            theta_grid = x_grid * (2 * np.pi / 24.0)
            b = amp * np.cos(phase_rad)
            c = amp * np.sin(phase_rad)
            y_fit = a + b * np.cos(theta_grid) + c * np.sin(theta_grid)
            ax.plot(x_grid, y_fit, color="C1", linewidth=1.5)
            title = f"{g} (amp={amp:.2f}, φ={phase_rad:.2f})"
            fit_records.append({"Gene": g, "amplitude": amp, "phase_rad": phase_rad})
        else:
            title = f"{g} (fit failed)"

        ax.set_title(title)
        ax.set_xlabel("Phase (hours)")

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")

    return pd.DataFrame(fit_records)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot core circadian gene expression vs CYCLOPS-inferred phase "
            "from a Fit_Output_*.csv."
        )
    )
    parser.add_argument(
        "--results_path",
        required=True,
        help=(
            "Path to a CYCLOPS results directory (containing Fit_Output_*.csv) "
            "or to a specific Fit_Output_*.csv file."
        ),
    )
    parser.add_argument(
        "--expression",
        help=(
            "Optional path to expression.csv. If omitted, the script will try "
            "to infer it from the results path using the Circadian layout."
        ),
    )
    parser.add_argument(
        "--genes",
        nargs="*",
        default=None,
        help=(
            "Genes to plot (default = canonical clock genes). Gene symbols "
            "are matched case-insensitively."
        ),
    )

    args = parser.parse_args()

    genes = args.genes if args.genes else DEFAULT_GENES
    genes_upper = [g.upper() for g in genes]

    results_path = Path(args.results_path)
    fit_file = find_fit_file(results_path)
    print(f"Using Fit file: {fit_file}")

    if args.expression:
        expr_file = Path(args.expression)
    else:
        expr_file = map_fit_to_expression(fit_file)
    print(f"Using expression: {expr_file}")

    # Load Fit_Output and get phases and sample IDs
    fit_df = pd.read_csv(fit_file)
    # CYCLOPS Fit_Output uses ID column for sample ID and Phase column in radians
    if "ID" not in fit_df.columns:
        raise ValueError("Fit_Output file must contain an 'ID' column.")
    if "Phase" not in fit_df.columns:
        raise ValueError("Fit_Output file must contain a 'Phase' column.")

    sample_ids = fit_df["ID"].astype(str).tolist()
    phases_rad = pd.to_numeric(fit_df["Phase"], errors="coerce").values
    phases_hours = (phases_rad % (2 * np.pi)) * (24.0 / (2 * np.pi))

    expr_by_gene = load_expression_for_genes(expr_file, genes_upper, sample_ids)

    out_png = fit_file.with_name("expression_vs_phase_cyclops.png")
    plot_expression_vs_phase(expr_by_gene, phases_hours, genes_upper, out_png)


if __name__ == "__main__":
    main()
