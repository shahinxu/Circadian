import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

def time_to_phase(time_hours, period_hours=24.0):
    return 2 * np.pi * time_hours / period_hours

def circular_correlation_jr(alpha, beta):
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    
    mask = np.isfinite(alpha) & np.isfinite(beta)
    alpha = alpha[mask]
    beta = beta[mask]
    if len(alpha) < 2:
        return np.nan
    alpha_bar = np.arctan2(np.mean(np.sin(alpha)), np.mean(np.cos(alpha)))
    beta_bar = np.arctan2(np.mean(np.sin(beta)), np.mean(np.cos(beta)))
    sin_alpha = np.sin(alpha - alpha_bar)
    sin_beta = np.sin(beta - beta_bar)
    numerator = np.sum(sin_alpha * sin_beta)
    denominator = np.sqrt(np.sum(sin_alpha**2) * np.sum(sin_beta**2))
    if denominator == 0:
        return np.nan
    rho = abs(numerator) / denominator
    return float(rho)

def _run_inference(model, test_loader, device, has_covariates=False):
    all_phase_coords = []
    all_phases = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            expressions = batch['expression'].to(device)
            
            if has_covariates and 'covariates' in batch:
                covariates = batch['covariates'].to(device)
                if covariates.dim() == 2:
                    covariates = covariates.unsqueeze(0)
            else:
                if expressions.dim() == 3:
                    batch_size, n_samples = expressions.shape[0], expressions.shape[1]
                else:
                    batch_size, n_samples = 1, expressions.shape[0]
                covariates = torch.zeros(batch_size, n_samples, 0).to(device)
            
            phase_coords, phase_angles, _ = model(expressions, covariates)

            all_phase_coords.append(phase_coords.cpu().numpy())
            all_phases.append(phase_angles.cpu().numpy())

    phase_coords = np.vstack(all_phase_coords) if all_phase_coords else np.empty((0, 2))
    phases = np.concatenate(all_phases) if all_phases else np.empty((0,))
    return phase_coords, phases


def _assemble_results_df(phase_coords, phases, preprocessing_info, sample_names=None):
    if sample_names and len(sample_names) == len(phases):
        sample_identifiers = sample_names
    else:
        sample_identifiers = [f"Sample_{i}" for i in range(len(phases))]
        logger.warning("Could not get sample names; using generated indices")

    results_data = {
        'Sample_ID': sample_identifiers,
        'Phase_X': phase_coords[:, 0],
        'Phase_Y': phase_coords[:, 1],
        'Predicted_Phase_Radians': phases,
        'Predicted_Phase_Degrees': phases * 180 / np.pi,
        'Predicted_Phase_Hours': phases * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)
    }
    return pd.DataFrame(results_data)


def _remove_legacy_prediction_files(save_dir):
    for fname in ('phase_predictions.csv', 'phase_predictions_simple.csv'):
        fpath = os.path.join(save_dir, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
                logger.debug("Removed legacy file: %s", fpath)
            except Exception:
                logger.exception("Failed to remove legacy file: %s", fpath)


def predict_and_save_phases(
        model, 
        test_loader, 
        preprocessing_info, 
        device='cuda', 
        save_dir='./results'
    ) -> pd.DataFrame:
    logger.info("Predicting test-set phases")

    sample_names = preprocessing_info.get('test_sample_columns', [])
    has_covariates = preprocessing_info.get('has_covariates', False)

    phase_coords, phases = _run_inference(model, test_loader, device, has_covariates=has_covariates)

    os.makedirs(save_dir, exist_ok=True)

    results_df = _assemble_results_df(phase_coords, phases, preprocessing_info, sample_names)

    _remove_legacy_prediction_files(save_dir)
    # Always save predictions to CSV so downstream steps (alignment/plotting)
    # can run even when metadata is unavailable.
    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, 'predictions.csv')
    try:
        results_df.to_csv(out_csv, index=False)
        logger.info("Predicted %d samples; results written to %s", len(phases), out_csv)
    except Exception:
        logger.exception("Failed to write predictions CSV to %s", out_csv)

    logger.debug("create_prediction_plots was removed; no additional plots saved")

    return results_df


def _compute_gene_phase_from_predictions(pred_phases_rad: np.ndarray, expr_values: np.ndarray):
    """
    Estimate the gene acrophase by regressing expression values onto cos/sin
    of predicted phases. Returns estimated phase in radians and amplitude.
    Model: expr = c0 + c1*cos(theta) + c2*sin(theta) + noise
    Gene phase phi = atan2(c2, c1)
    """
    # Build design matrix
    theta = np.asarray(pred_phases_rad, dtype=float)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    # Solve least squares (including intercept)
    A = np.column_stack([np.ones(len(theta)), X])
    try:
        coeffs, *_ = np.linalg.lstsq(A, expr_values, rcond=None)
    except Exception:
        return float('nan'), 0.0
    c1, c2 = coeffs[1], coeffs[2]
    amp = np.hypot(c1, c2)
    phase = float(np.arctan2(c2, c1) % (2 * np.pi))
    return phase, amp


def align_predictions_to_gene_acrophases(
    results_df: pd.DataFrame,
    test_expr_file: str,
    gene_symbols: list,
    ref_acrophases_rad: list,
    sample_id_col: str = 'Sample_ID'
):
    """
    Align predicted sample phases to reference gene acrophases.
    - results_df: DataFrame with `sample_id_col` and `Predicted_Phase_Hours`
    - test_expr_file: path to expression.csv (with 'Gene_Symbol' column)
    - gene_symbols: list of gene symbols to use for alignment (strings)
    - ref_acrophases_rad: list of reference acrophases (radians), same length

    Returns: (aligned_df, alignment_shift_rad, per_gene_df)
    """
    # Load expression file
    expr_df = pd.read_csv(test_expr_file, low_memory=False)
    # Normalize gene symbol case to uppercase for robust matching
    if 'Gene_Symbol' in expr_df.columns:
        expr_df['Gene_Symbol'] = expr_df['Gene_Symbol'].astype(str).str.upper()
    # Determine sample columns order corresponding to results_df
    if sample_id_col in results_df.columns:
        sample_order = results_df[sample_id_col].astype(str).tolist()
    else:
        raise ValueError(f"results_df missing required column {sample_id_col}")

    # Extract predicted phases in radians
    pred_hours = pd.to_numeric(results_df['Predicted_Phase_Hours'], errors='coerce').values
    pred_rad = (pred_hours % 24) * (2 * np.pi / 24.0)

    gene_est_phases = []
    gene_amps = []
    gene_names_found = []

    # Uppercase provided gene list for case-insensitive matching
    gene_symbols_upper = [str(g).upper() for g in gene_symbols]

    # For each gene, find its row(s) in expr_df and aggregate if duplicates
    for g_upper in gene_symbols_upper:
        rows = expr_df[expr_df['Gene_Symbol'] == g_upper]
        if rows.empty:
            gene_est_phases.append(float('nan'))
            gene_amps.append(0.0)
            gene_names_found.append(g_upper)
            continue
        # Determine columns for samples in sample_order
        available_cols = [c for c in sample_order if c in rows.columns]
        if len(available_cols) == 0:
            # try Sample column names like in results_df
            available_cols = [c for c in rows.columns if c in sample_order]
        if len(available_cols) == 0:
            gene_est_phases.append(float('nan'))
            gene_amps.append(0.0)
            gene_names_found.append(g_upper)
            continue
        # If multiple rows (probes), average across rows
        expr_values = rows[available_cols].astype(float).mean(axis=0).reindex(sample_order).values
        # Handle missing values
        expr_values = np.nan_to_num(expr_values, nan=0.0)
        phase, amp = _compute_gene_phase_from_predictions(pred_rad, expr_values)
        gene_est_phases.append(phase)
        gene_amps.append(amp)
        gene_names_found.append(g_upper)

    gene_est_phases = np.array(gene_est_phases, dtype=float)
    gene_amps = np.array(gene_amps, dtype=float)
    ref_acrophases = np.array(ref_acrophases_rad, dtype=float)

    # Compute per-gene shift needed: shift_i = ref - est (mod 2pi)
    diffs = (ref_acrophases - gene_est_phases) % (2 * np.pi)
    # Convert diffs to [-pi, pi]
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi

    # Weight by amplitude for mean shift
    weights = gene_amps.copy()
    # Ignore NaN phase estimates
    valid = np.isfinite(gene_est_phases) & np.isfinite(ref_acrophases) & (weights > 0)
    if not np.any(valid):
        alignment_shift = 0.0
    else:
        w = weights[valid]
        dif = diffs[valid]
        # circular mean of dif with weights
        sin_sum = np.sum(w * np.sin(dif))
        cos_sum = np.sum(w * np.cos(dif))
        alignment_shift = float(np.arctan2(sin_sum, cos_sum))

    # Apply shift to predicted phases
    aligned_rad = (pred_rad + alignment_shift) % (2 * np.pi)
    aligned_hours = aligned_rad * 24.0 / (2 * np.pi)

    aligned_df = results_df.copy()
    aligned_df['Predicted_Phase_Hours_Aligned'] = aligned_hours

    per_gene_df = pd.DataFrame({
        'Gene': gene_names_found,
        'Estimated_Acrophase_Rad': gene_est_phases,
        'Estimated_Amplitude': gene_amps,
        'Reference_Acrophase_Rad': ref_acrophases,
        'Diff_Rad': diffs
    })

    return aligned_df, alignment_shift, per_gene_df


def load_metadata_for_phase_comparison(metadata_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv, low_memory=False)
    cols = set(meta.columns.astype(str).str.strip())

    def to_float_series(s):
        return pd.to_numeric(s, errors='coerce')

    if {'study', 'sample', 'time'}.issubset(cols):
        m = meta.copy()
        m['study'] = m['study'].astype(str)
        m['sample'] = m['sample'].astype(str)
        m['study_sample'] = m['study'] + '_' + m['sample']
        m['time_mod24'] = to_float_series(m['time']) % 24
        return m[['study_sample', 'time_mod24']]

    if {'study_sample', 'time_mod24'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['study_sample'].astype(str)
        m['time_mod24'] = to_float_series(m['time_mod24']) % 24
        return m[['study_sample', 'time_mod24']]

    if {'study_sample', 'time'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['study_sample'].astype(str)
        m['time_mod24'] = to_float_series(m['time']) % 24
        return m[['study_sample', 'time_mod24']]

    if {'Sample', 'Time_Hours'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['Sample'].astype(str)
        m['time_mod24'] = to_float_series(m['Time_Hours']) % 24
        return m[['study_sample', 'time_mod24']]

    if {'Sample', 'time'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['Sample'].astype(str)
        m['time_mod24'] = to_float_series(m['time']) % 24
        return m[['study_sample', 'time_mod24']]

    raise ValueError("Unsupported metadata format. Expected one of: {'study','sample','time'}; {'study_sample','time_mod24'}; {'study_sample','time'}; {'Sample','Time_Hours'}; {'Sample','time'}")


def best_align_phase_for_comparison(
    x_rad: np.ndarray,
    y_rad: np.ndarray,
    step: float = 0.1,
) -> tuple[np.ndarray, float, float, float, float, bool]:
    from scipy.stats import pearsonr, spearmanr

    x_arr: np.ndarray = np.asarray(x_rad, dtype=float)
    y_arr: np.ndarray = np.asarray(y_rad, dtype=float)
    two_pi = 2 * np.pi

    # Validate inputs: expect radians in [0, 2*pi)
    eps = 1e-8
    if np.nanmax(x_arr) > two_pi + eps or np.nanmin(x_arr) < -eps:
        raise ValueError("best_align_phase_for_comparison: x_rad appears not to be in radians in [0, 2*pi)")
    if np.nanmax(y_arr) > two_pi + eps or np.nanmin(y_arr) < -eps:
        raise ValueError("best_align_phase_for_comparison: y_rad appears not to be in radians in [0, 2*pi)")

    # reduce to canonical [0, 2*pi)
    x_arr = x_arr % two_pi
    y_arr = y_arr % two_pi
    shifts: np.ndarray = np.arange(0.0, two_pi, step, dtype=float)

    best_r = -np.inf
    best = {
        'aligned': x_arr,
        'r': float('nan'),
        'r2': float('nan'),
        'spearman_R': float('nan'),
        'shift': 0.0,
        'flipped': False,
    }

    for flipped in (False, True):
        x0: np.ndarray = (two_pi - x_arr) % two_pi if flipped else x_arr
        for s in shifts:
            xs = (x0 + s) % two_pi
            xs_np = np.asarray(xs, dtype=float)
            y_np = y_arr
            try:
                r = float(pearsonr(xs_np, y_np)[0])
            except Exception:
                r = np.nan
            if not np.isfinite(r):
                continue
            if r > best_r:
                best_r = r
                try:
                    spearman_R = float(spearmanr(xs_np, y_np)[0])
                except Exception:
                    spearman_R = float('nan')
                best.update(
                    aligned=xs,
                    r=r,
                    r2=r*r,
                    spearman_R=spearman_R,
                    shift=float(s),
                    flipped=flipped
                )

    return (
        best['aligned'],
        float(best['r']),
        float(best['r2']),
        float(best['spearman_R']),
        float(best['shift']),
        bool(best['flipped'])
    )


def plot_comparsion(results_df: pd.DataFrame, metadata_csv: str, save_dir: str):
    # Load full metadata with tissue info
    meta_full = pd.read_csv(metadata_csv, low_memory=False)
    
    out_dir = os.path.join(save_dir, 'phase_vs_metadata')
    os.makedirs(out_dir, exist_ok=True)

    if 'Sample_ID' not in results_df.columns or 'Predicted_Phase_Hours' not in results_df.columns:
        raise ValueError('results_df must contain Sample_ID and Predicted_Phase_Hours columns')

    # Prepare predictions
    preds = results_df[['Sample_ID', 'Predicted_Phase_Hours']].copy()
    preds = preds.rename(columns={'Sample_ID': 'Sample', 'Predicted_Phase_Hours': 'pred_phase'})
    preds['Sample'] = preds['Sample'].astype(str)

    # Prepare metadata with tissue info
    if 'Sample' not in meta_full.columns or 'Time_Hours' not in meta_full.columns:
        raise ValueError('metadata must contain Sample and Time_Hours columns')
    
    meta_view = meta_full[['Sample', 'Time_Hours']].copy()
    if 'Tissue' in meta_full.columns:
        meta_view['Tissue'] = meta_full['Tissue'].astype(str)
    meta_view['Sample'] = meta_view['Sample'].astype(str)
    meta_view['time_mod24'] = pd.to_numeric(meta_view['Time_Hours'], errors='coerce') % 24

    # Join predictions with metadata
    joined = preds.merge(meta_view, on='Sample', how='left').dropna(subset=['pred_phase', 'time_mod24'])
    if joined.empty:
        print(f"[WARN] No matches between predictions and metadata")
        return None

    # Summary statistics for all results
    all_results = []
    
    # 1. Generate overall plot for all samples
    phase_hours = np.asarray(joined['pred_phase'], dtype=float)
    metadata_hours = np.asarray(joined['time_mod24'], dtype=float)
    phase_rad = time_to_phase(phase_hours, period_hours=24.0)
    metadata_rad = time_to_phase(metadata_hours, period_hours=24.0)
    
    aligned_rad = best_align_phase_for_comparison(phase_rad, metadata_rad, step=0.1)[0]
    
    r = float(pearsonr(aligned_rad, metadata_rad)[0])
    spearman_R = float(spearmanr(aligned_rad, metadata_rad)[0])
    r2 = r * r if np.isfinite(r) else float('nan')
    circ_r = circular_correlation_jr(phase_rad, metadata_rad)
    
    # Plot all samples
    plt.figure(figsize=(8, 7))
    plt.grid(True, linestyle='-')
    plt.scatter(metadata_rad, aligned_rad, c='r', s=100, alpha=0.6)
    
    two_pi = 2 * np.pi
    plt.xlim(0, two_pi)
    plt.ylim(0, two_pi)
    plt.xlabel('Collection Phase', fontsize=24)
    plt.ylabel('Predicted Phase', fontsize=24)
    plt.title(f"All Tissues (N={len(phase_rad)}), JR cor: {circ_r:.2f}", fontsize=24)
    plt.tight_layout()
    
    out_path_all = os.path.join(out_dir, 'comparison_all_tissues.png')
    plt.savefig(out_path_all, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== All Tissues ===")
    print(f"  N={len(phase_rad)}, Pearson R={r:.3f}, Spearman ρ={spearman_R:.3f}, Circular R={circ_r:.3f}")
    print(f"  Plot: {out_path_all}")
    
    all_results.append({
        'Tissue': 'All',
        'N': len(phase_rad),
        'Pearson_R': r,
        'R2': r2,
        'Spearman_R': spearman_R,
        'Circular_R': circ_r,
        'Plot': out_path_all
    })
    
    # 2. Generate per-tissue plots
    if 'Tissue' in joined.columns:
        tissues = sorted(joined['Tissue'].unique())
        print(f"\nGenerating comparison plots for {len(tissues)} tissues...")
        
        for tissue in tissues:
            tissue_data = joined[joined['Tissue'] == tissue]
            if len(tissue_data) < 3:  # Skip tissues with too few samples
                continue
            
            t_phase_hours = np.asarray(tissue_data['pred_phase'], dtype=float)
            t_metadata_hours = np.asarray(tissue_data['time_mod24'], dtype=float)
            t_phase_rad = time_to_phase(t_phase_hours, period_hours=24.0)
            t_metadata_rad = time_to_phase(t_metadata_hours, period_hours=24.0)
            
            t_aligned_rad = best_align_phase_for_comparison(t_phase_rad, t_metadata_rad, step=0.1)[0]
            
            t_r = float(pearsonr(t_aligned_rad, t_metadata_rad)[0])
            t_spearman_R = float(spearmanr(t_aligned_rad, t_metadata_rad)[0])
            t_r2 = t_r * t_r if np.isfinite(t_r) else float('nan')
            t_circ_r = circular_correlation_jr(t_phase_rad, t_metadata_rad)
            
            # Plot this tissue
            plt.figure(figsize=(8, 7))
            plt.grid(True, linestyle='-')
            plt.scatter(t_metadata_rad, t_aligned_rad, c='b', s=100, alpha=0.7)
            
            plt.xlim(0, two_pi)
            plt.ylim(0, two_pi)
            plt.xlabel('Collection Phase', fontsize=24)
            plt.ylabel('Predicted Phase', fontsize=24)
            plt.title(f"{tissue} (N={len(t_phase_rad)}), JR cor: {t_circ_r:.2f}", fontsize=20)
            plt.tight_layout()
            
            tissue_safe = tissue.replace(' ', '_').replace('/', '_')
            out_path_tissue = os.path.join(out_dir, f'comparison_{tissue_safe}.png')
            plt.savefig(out_path_tissue, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  [{tissue}] N={len(t_phase_rad)}, Pearson R={t_r:.3f}, Spearman ρ={t_spearman_R:.3f}, Circular R={t_circ_r:.3f}")
            
            all_results.append({
                'Tissue': tissue,
                'N': len(t_phase_rad),
                'Pearson_R': t_r,
                'R2': t_r2,
                'Spearman_R': t_spearman_R,
                'Circular_R': t_circ_r,
                'Plot': out_path_tissue
            })
    
    # Save summary statistics
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(out_dir, 'comparison_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to: {summary_path}")
    
    return summary_df


# rank_loss and its helper were removed per new training logic (align loss no longer used)