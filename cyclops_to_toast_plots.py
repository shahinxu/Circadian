import argparse
import math
from pathlib import Path

import pandas as pd

# We will dynamically import TOAST's plotting module


def make_toast_results_from_cyclops(fit_file: Path, out_root: Path):
    """Convert a CYCLOPS Fit_Output_*.csv into a TOAST-style results folder.

    - fit_file: path to Fit_Output_*.csv
    - out_root: unused placeholder (kept for backward compatibility).

    Returns (dataset_name, run_dir) where run_dir is the directory
    containing Fit_Output_*.csv and predictions_aligned.csv.
    """
    fit_file = fit_file.resolve()

    # Infer dataset name from parent directory (e.g. data_AD427 -> AD427).
    # For aggregated or transfer runs, this may still be a long name such as
    # "GSE54651_adrenal_gland_20251222_184738" or
    # "GTEx_Zhang_Transfer_Bcell_20260101_202925"; we will handle these
    # patterns later when mapping to expression.csv.
    parent_name = fit_file.parent.name
    if parent_name.lower().startswith("data_"):
        dataset_name = parent_name[len("data_"):]
    else:
        dataset_name = parent_name

    # Load CYCLOPS Fit_Output
    df = pd.read_csv(fit_file)
    if "ID" not in df.columns:
        raise ValueError("Fit_Output must contain an 'ID' column")

    # Prefer Phases_MA (mouse-aligned); fallback to Phase if needed
    phase_col = None
    if "Phases_MA" in df.columns:
        phase_col = "Phases_MA"
    elif "Phase" in df.columns:
        phase_col = "Phase"
    else:
        raise ValueError("Fit_Output must contain 'Phases_MA' or 'Phase' column")

    phases_rad = pd.to_numeric(df[phase_col], errors="coerce").values
    # Convert radians to hours (0-24)
    phases_hours = (phases_rad % (2 * math.pi)) * (24.0 / (2 * math.pi))

    preds = pd.DataFrame({
        "Sample_ID": df["ID"].astype(str).values,
        "Predicted_Phase_Hours_Aligned": phases_hours,
    })

    # Save predictions directly next to the original Fit_Output_*.csv
    run_dir = fit_file.parent

    out_file = run_dir / "predictions_aligned.csv"
    preds.to_csv(out_file, index=False)
    print(f"Written TOAST-style predictions: {out_file}")

    return dataset_name, run_dir


def infer_expression_path(project_root: Path, dataset_name: str) -> Path:
    """Infer expression.csv path from a dataset name under this project.

    This uses knowledge of the Circadian layout and specific naming
    conventions used for CYCLOPS runs (GSE54651 tissues, GTEx, Zhang transfer,
    etc.). It is tailored for this project, not a general mapper.
    """

    data_root = project_root / "data"

    # 1) GTEx_Zhang_Transfer_* cases
    if dataset_name.startswith("GTEx_Zhang_Transfer_"):
        tail = dataset_name[len("GTEx_Zhang_Transfer_"):]
        base = tail.split("_")[0]  # e.g. Bcell, CD4Tcell, GSE176078

        # Tumor cell types from Zhang_CancerCell_2025_sub
        zhang_cells = {"Bcell", "CD4Tcell", "CD8Tcell", "Myeloid", "NKcell"}
        if base in zhang_cells:
            expr = data_root / "Zhang_CancerCell_2025_sub" / base / "expression.csv"
            if expr.exists():
                return expr

        # Transfer to a specific GSE dataset (e.g. GSE176078)
        if base.startswith("GSE"):
            expr = data_root / base / "expression.csv"
            if expr.exists():
                return expr

    # 2) Plain tumor cell types (Bcell/CD4Tcell/...) from Zhang_CancerCell_2025_sub
    zhang_cells = {"Bcell", "CD4Tcell", "CD8Tcell", "Myeloid", "NKcell"}
    if dataset_name in zhang_cells:
        expr = data_root / "Zhang_CancerCell_2025_sub" / dataset_name / "expression.csv"
        if expr.exists():
            return expr

    # 3) GSEXXXX_... style where first token is the dataset root, and an
    # optional second token (or more) is tissue; the last token is timestamp.
    if dataset_name.startswith("GSE"):
        parts = dataset_name.split("_")
        base = parts[0]
        base_dir = data_root / base
        if base_dir.exists():
            if len(parts) > 2:
                tissue = "_".join(parts[1:-1])
                expr = base_dir / tissue / "expression.csv"
                if expr.exists():
                    return expr
            # Fallback: dataset without subdirectory
            expr = base_dir / "expression.csv"
            if expr.exists():
                return expr

    # 4) GTEx_GTEx_* style: data/GTEx/GTEx_<tissue>/expression.csv
    if dataset_name.startswith("GTEx_GTEx_"):
        parts = dataset_name.split("_")
        if len(parts) >= 3 and parts[0] == "GTEx" and parts[1] == "GTEx":
            group = parts[0]  # "GTEx"
            tissue = "_".join(parts[1:-1])  # e.g. "GTEx_brain_cortex"
            expr = data_root / group / tissue / "expression.csv"
            if expr.exists():
                return expr

    # 5) Simple dataset name -> data/<name>/expression.csv (e.g. AD427)
    expr = data_root / dataset_name / "expression.csv"
    if expr.exists():
        return expr

    raise FileNotFoundError(f"Cannot infer expression.csv for dataset '{dataset_name}'")


def run_toast_plots_for_run(toast_plot, run_dir: Path, expr_file: Path, genes):
    """Run TOAST plotting functions for a single run directory.

    This mirrors TOAST.process_single_celltype but uses a known expression
    path instead of relying on map_results_to_data_path.
    """

    run_dir = run_dir.resolve()

    preds_aligned = run_dir / "predictions_aligned.csv"
    preds_file = run_dir / "predictions.csv"

    if preds_aligned.exists():
        preds = pd.read_csv(preds_aligned)
        print(f"Using aligned predictions from: {preds_aligned}")
    elif preds_file.exists():
        preds = pd.read_csv(preds_file)
        print(f"Using unaligned predictions from: {preds_file}")
    else:
        raise FileNotFoundError(f"No predictions.csv or predictions_aligned.csv in {run_dir}")

    if "Sample_ID" not in preds.columns:
        raise ValueError("predictions file must contain Sample_ID column")

    # Get predicted phase in hours
    if "Predicted_Phase_Hours_Aligned" in preds.columns:
        times = pd.to_numeric(preds["Predicted_Phase_Hours_Aligned"], errors="coerce").values
    elif "Predicted_Phase_Hours" in preds.columns:
        times = pd.to_numeric(preds["Predicted_Phase_Hours"], errors="coerce").values
    else:
        raise ValueError("predictions.csv missing phase columns")

    sample_order = preds["Sample_ID"].astype(str).tolist()

    print(f"Loading expression from: {expr_file}")
    genes_upper = [g.upper() for g in genes]
    expr_by_gene = toast_plot.load_expression_for_genes(expr_file, genes_upper, sample_order)

    out_png1 = run_dir / "expression_vs_phase.png"
    fit_df = toast_plot.plot_time_expression(expr_by_gene, times, genes_upper, out_png1)

    out_png2 = run_dir / "acrophase_circle.png"
    # Use the parent directory name (dataset name) as title
    celltype_name = run_dir.parent.name
    toast_plot.plot_acrophase_circle(fit_df, out_png2, title=f"{celltype_name} Acrophases")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert CYCLOPS Fit_Output_*.csv to TOAST-style predictions and "
            "reuse TOAST/plot_all_from_predictions.py for plotting."
        )
    )
    parser.add_argument(
        "--cyclops_results",
        required=True,
        help=(
            "Path to a CYCLOPS results directory (containing Fit_Output_*.csv) "
            "or to a specific Fit_Output_*.csv file."
        ),
    )
    parser.add_argument(
        "--out_root",
        default="TOAST/results",
        help=(
            "Base output directory for TOAST-style results (default: "
            "TOAST/results relative to project root; directory name must "
            "contain 'results' to work with TOAST's path mapping)."
        ),
    )
    parser.add_argument(
        "--genes",
        nargs="*",
        default=None,
        help="Optional list of genes to plot (default = TOAST default clock genes)",
    )
    parser.add_argument(
        "--compare_mode",
        action="store_true",
        help=(
            "If set, call TOAST comparison mode on the generated celltype "
            "directory (expects multiple subdirs)."
        ),
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    cyclops_results = Path(args.cyclops_results).resolve()
    out_root = (project_root / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)

    # Find Fit_Output file(s)
    if cyclops_results.is_file():
        fit_files = [cyclops_results]
    else:
        # look for Fit_Output_*.csv under this directory (non-recursive first, then recursive)
        fit_files = sorted(cyclops_results.glob("Fit_Output_*.csv"))
        if not fit_files:
            fit_files = sorted(cyclops_results.rglob("Fit_Output_*.csv"))
        if not fit_files:
            raise FileNotFoundError(f"No Fit_Output_*.csv found under: {cyclops_results}")

    # Convert all found Fit_Output files, track dataset + run_dir
    run_infos = []  # (dataset_name, run_dir)
    for ff in fit_files:
        dataset_name, run_dir = make_toast_results_from_cyclops(ff, out_root)
        run_infos.append((dataset_name, run_dir))

    # Prepare to call TOAST plotting code
    toast_dir = project_root / "TOAST"
    if not toast_dir.exists():
        raise FileNotFoundError(f"Cannot find TOAST directory at: {toast_dir}")

    import sys
    if str(toast_dir) not in sys.path:
        sys.path.insert(0, str(toast_dir))

    try:
        import plot_all_from_predictions as toast_plot
    except ImportError as e:
        raise ImportError(
            f"Failed to import plot_all_from_predictions from {toast_dir}: {e}"
        )

    genes = args.genes if args.genes else toast_plot.DEFAULT_GENES

    # For now, handle runs one by one; comparison mode can be added later
    for dataset_name, run_dir in run_infos:
        try:
            expr_file = infer_expression_path(project_root, dataset_name)
        except FileNotFoundError as e:
            print(f"[WARN] {e}; skipping {run_dir}")
            continue

        print(f"=== TOAST-style plotting for {run_dir} (dataset={dataset_name}) ===")
        run_toast_plots_for_run(toast_plot, run_dir, expr_file, genes)


if __name__ == "__main__":
    main()
