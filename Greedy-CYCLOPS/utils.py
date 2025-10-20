import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

logger = logging.getLogger(__name__)

def time_to_phase(time_hours, period_hours=24.0):
    return 2 * np.pi * time_hours / period_hours

# Helper: run inference over test_loader and collect numpy results
def _run_inference(model, test_loader, celltype_to_idx, device):
    all_phase_coords = []
    all_phases = []
    all_times = []
    all_celltypes = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            expressions = batch['expression'].to(device)
            times = batch.get('time', None)
            celltypes = batch.get('celltype', None)
            phase_coords, phase_angles, _ = model(expressions)

            all_phase_coords.append(phase_coords.cpu().numpy())
            all_phases.append(phase_angles.cpu().numpy())

            if times is not None:
                all_times.append(times.cpu().numpy())
            if celltypes is not None:
                all_celltypes.extend(celltypes)

    phase_coords = np.vstack(all_phase_coords) if all_phase_coords else np.empty((0, 2))
    phases = np.concatenate(all_phases) if all_phases else np.empty((0,))
    times = np.concatenate(all_times) if all_times else None
    celltypes = np.array(all_celltypes) if all_celltypes else None
    return phase_coords, phases, times, celltypes


def _assemble_results_df(phase_coords, phases, times, celltypes, preprocessing_info, sample_names=None):
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

    if times is not None:
        results_data['True_Time_Hours'] = times
        results_data['True_Phase_Radians'] = time_to_phase(times, preprocessing_info.get('period_hours', 24.0))
        perr = np.abs(phases - results_data['True_Phase_Radians'])
        perr = np.minimum(perr, 2 * np.pi - perr)
        results_data['Phase_Error_Radians'] = perr
        results_data['Phase_Error_Hours'] = perr * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)

    if celltypes is not None:
        results_data['Cell_Type'] = celltypes

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


def predict_and_save_phases(model, test_loader, preprocessing_info, device='cuda', save_dir='./results'):
    """Run model on test_loader and return a standardized results DataFrame.

    This function orchestrates inference, assembly of results, optional cleanup and
    logging. It intentionally delegates subtasks to small helpers for clarity.
    """
    logger.info("Predicting test-set phases")

    celltype_to_idx = preprocessing_info.get('celltype_to_idx', {})
    sample_names = preprocessing_info.get('test_sample_columns', [])

    phase_coords, phases, times, celltypes = _run_inference(model, test_loader, celltype_to_idx, device)

    os.makedirs(save_dir, exist_ok=True)

    results_df = _assemble_results_df(phase_coords, phases, times, celltypes, preprocessing_info, sample_names)

    _remove_legacy_prediction_files(save_dir)

    logger.info("Predicted %d samples; results available in %s", len(phases), save_dir)

    if times is not None:
        mean_error_hours = np.mean(results_df['Phase_Error_Hours'])
        std_error_hours = np.std(results_df['Phase_Error_Hours'])
        logger.info("Mean error: %.2f ± %.2f hours", mean_error_hours, std_error_hours)

    if 'Cell_Type' in results_df.columns:
        celltype_stats = results_df.groupby('Cell_Type').agg({'Predicted_Phase_Hours': ['mean', 'std', 'count']}).round(2)
        logger.info("Per-celltype stats:\n%s", celltype_stats)

    logger.debug("create_prediction_plots was removed; no additional plots saved")

    return results_df

def sanitize_filename(s: str) -> str:
    """Sanitize filename for safe file operations"""
    if s is None:
        return 'ALL'
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(s))


def load_metadata_for_phase_comparison(metadata_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv, low_memory=False)
    cols = set(meta.columns.astype(str).str.strip())

    def to_float_series(s):
        return pd.to_numeric(s, errors='coerce')

    # Case 1: study + sample + time
    if {'study', 'sample', 'time'}.issubset(cols):
        m = meta.copy()
        m['study'] = m['study'].astype(str)
        m['sample'] = m['sample'].astype(str)
        m['study_sample'] = m['study'] + '_' + m['sample']
        m['time_mod24'] = to_float_series(m['time']) % 24
        return m[['study_sample', 'time_mod24']]

    # Case 2: study_sample + time_mod24
    if {'study_sample', 'time_mod24'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['study_sample'].astype(str)
        m['time_mod24'] = to_float_series(m['time_mod24']) % 24
        return m[['study_sample', 'time_mod24']]

    # Case 3: study_sample + time
    if {'study_sample', 'time'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['study_sample'].astype(str)
        m['time_mod24'] = to_float_series(m['time']) % 24
        return m[['study_sample', 'time_mod24']]

    # Case 4: Sample + Time_Hours
    if {'Sample', 'Time_Hours'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['Sample'].astype(str)
        m['time_mod24'] = to_float_series(m['Time_Hours']) % 24
        return m[['study_sample', 'time_mod24']]

    # Case 5: Sample + time
    if {'Sample', 'time'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['Sample'].astype(str)
        m['time_mod24'] = to_float_series(m['time']) % 24
        return m[['study_sample', 'time_mod24']]

    raise ValueError("Unsupported metadata format. Expected one of: {'study','sample','time'}; {'study_sample','time_mod24'}; {'study_sample','time'}; {'Sample','Time_Hours'}; {'Sample','time'}")


def load_predictions_for_comparison(pred_csv: str) -> pd.DataFrame:
    """Load predictions CSV and standardize column names"""
    df = pd.read_csv(pred_csv)
    # Infer study_sample
    if 'study_sample' in df.columns:
        df['study_sample'] = df['study_sample'].astype(str)
    elif {'study', 'sample'}.issubset(df.columns):
        df['study'] = df['study'].astype(str)
        df['sample'] = df['sample'].astype(str)
        df['study_sample'] = df['study'] + '_' + df['sample']
    elif 'Sample' in df.columns:
        df['study_sample'] = df['Sample'].astype(str)
    elif 'Sample_ID' in df.columns:
        # my_cyclops phase_predictions_simple.csv uses Sample_ID
        df['study_sample'] = df['Sample_ID'].astype(str)
    elif 'sample' in df.columns:
        df['study_sample'] = df['sample'].astype(str)
    else:
        raise ValueError(f'{pred_csv}: cannot infer study_sample (expected study+sample, Sample, or study_sample)')

    # Infer predicted phase
    if 'Predicted_Phase_Hours' in df.columns:
        pred_hours = pd.to_numeric(df['Predicted_Phase_Hours'], errors='coerce')
    elif 'Predicted_Phase_Radians' in df.columns:
        rad = pd.to_numeric(df['Predicted_Phase_Radians'], errors='coerce')
        pred_hours = rad * 24.0 / (2 * np.pi)
    elif 'Predicted_Phase_Degrees' in df.columns:
        deg = pd.to_numeric(df['Predicted_Phase_Degrees'], errors='coerce')
        pred_hours = deg * (24.0 / 360.0)
    else:
        phase_candidates = [
            'phase', 'Phase', 'phase_hours', 'pred_phase', 'predicted_phase',
            'theta', 'Theta', 'predicted_time', 'pred_time', 'phase_prediction',
        ]
        phase_col = None
        for c in phase_candidates:
            if c in df.columns:
                phase_col = c
                break
        if phase_col is None:
            num_cols = [c for c in df.columns if np.issubdtype(np.asarray(df[c]).dtype, np.number)]
            if num_cols:
                phase_col = num_cols[-1]
            else:
                raise ValueError(f'{pred_csv}: cannot locate predicted phase column')
        pred_hours = pd.to_numeric(df[phase_col], errors='coerce')

    df['pred_phase'] = pred_hours % 24
    return df[['study_sample', 'pred_phase']]




def best_align_phase_for_comparison(
    x_hours: np.ndarray,
    y_hours: np.ndarray,
    step: float = 0.1,
    period: float = 24.0,
) -> tuple[np.ndarray, float, float, float, float, bool]:
    from scipy.stats import pearsonr, spearmanr

    x_arr: np.ndarray = np.asarray(x_hours, dtype=float)
    y_arr: np.ndarray = np.asarray(y_hours, dtype=float)
    shifts: np.ndarray = np.arange(0.0, period, step, dtype=float)

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
        x0: np.ndarray = (period - x_arr) % period if flipped else x_arr
        for s in shifts:
            xs = (x0 + s) % 24.0
            xs_np = np.asarray(xs, dtype=float)
            y_np = y_arr
            try:
                r = float(pearsonr(xs_np, y_np)[0])
            except Exception:
                r = np.nan
            if not np.isfinite(r):
                continue
            # Maximize positive correlation after allowing flip
            if r > best_r:
                best_r = r
                # Calculate Spearman correlation for the best alignment
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


def plot_phase_vs_metadata_comparison(pred_csv: str, celltype: str, meta: pd.DataFrame, out_dir: str):
    try:
        preds = load_predictions_for_comparison(pred_csv)
        base_columns = [c for c in ('study_sample', 'time_mod24') if c in meta.columns]
        if {'study_sample', 'time_mod24'}.issubset(base_columns):
            meta_view = meta[['study_sample', 'time_mod24']].copy()
        elif 'sample' in meta.columns and 'time_mod24' in meta.columns:
            meta_view = meta[['sample', 'time_mod24']].copy()
            meta_view = meta_view.rename(columns={'sample': 'study_sample'})
        else:
            raise ValueError('metadata must contain columns for sample identifiers and time_mod24')

        meta_view['study_sample'] = meta_view['study_sample'].astype(str)
        preds['study_sample'] = preds['study_sample'].astype(str)

        joined = preds.merge(meta_view, on='study_sample', how='left').dropna(subset=['pred_phase', 'time_mod24'])
        if joined.empty:
            print(f"[WARN] No matches for {celltype} ({pred_csv})")
            return None

        # joined contains hours (0..24) for both prediction and metadata
        phase_hours = np.asarray(joined['pred_phase'], dtype=float)
        metadata_hours = np.asarray(joined['time_mod24'], dtype=float)

        # Convert hours -> radians (map [0, period_hours) -> [0, 2*pi))
        phase_rad = time_to_phase(phase_hours, period_hours=24.0)
        metadata_rad = time_to_phase(metadata_hours, period_hours=24.0)

        # Align in radians over a period of 2*pi
        aligned, r, r2, spearman_R, best_shift, flipped = best_align_phase_for_comparison(
            phase_rad, metadata_rad, step=0.1, period=2 * np.pi
        )

        plt.figure(figsize=(8, 7))
        plt.scatter(aligned, metadata_rad, s=12, alpha=0.8)
        # Minimal plot: no legend, no title, no extra subtitle
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        safe_ct = sanitize_filename(celltype)
        out_path = os.path.join(out_dir, f'phase_vs_time_{safe_ct}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_path}')
        return out_path, r, r2, spearman_R

    except Exception as e:
        print(f"[ERROR] Failed to plot phase vs metadata for {celltype}: {e}")
        return None


def plot_comparsion(results_df: pd.DataFrame, metadata_csv: str, save_dir: str):
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Results DataFrame columns: {list(results_df.columns)}")
    meta = load_metadata_for_phase_comparison(metadata_csv)

    out_dir = os.path.join(save_dir, 'phase_vs_metadata')
    os.makedirs(out_dir, exist_ok=True)

    # Prepare a temporary CSV with the required columns
    tmp_all = os.path.join(out_dir, 'preds_ALL.tmp.csv')
    sub = results_df[['Sample_ID', 'Predicted_Phase_Hours']].copy()
    sub.to_csv(tmp_all, index=False)
    print(f"生成临时预测文件: {tmp_all}")

    result = plot_phase_vs_metadata_comparison(tmp_all, 'ALL', meta, out_dir)
    if result is not None:
        _, r, r2, spearman_R = result
        print(f"整体 phase-vs-metadata 图已生成: R={r:.3f}, R²={r2:.3f}, Spearman ρ={spearman_R:.3f}")
    else:
        print("[WARN] 无法生成 phase-vs-metadata 图（可能没有匹配）")

    if os.path.isfile(tmp_all):
        os.remove(tmp_all)

    print(f"Phase-vs-metadata 输出保存在: {out_dir}")


import math
def _angle_diff(a, b):
    two_pi = 2 * math.pi
    diff = torch.remainder(a - b + math.pi, two_pi) - math.pi
    return diff

def rank_loss_sliding_window(pred, ranks, window=5, eps=0.1):
    device = pred.device
    n = len(ranks)
    order = torch.tensor(np.argsort(ranks), dtype=torch.long, device=device)  # shape (n,)
    # center indices in dataset order
    centers = order.unsqueeze(1).repeat(1, 2 * window)  # (n, 2w)
    # build neighbor indices by circularly rolling the order
    neighs = []
    for k in range(1, window + 1):
        neighs.append(torch.roll(order, -k))  # next k
        neighs.append(torch.roll(order, k))   # prev k
    neighs = torch.stack(neighs, dim=1)  # (n, 2w)
    centers_flat = centers.reshape(-1)
    neighs_flat = neighs.reshape(-1)

    p_cent = pred[centers_flat]       # (n*2w,)
    p_nei = pred[neighs_flat]         # (n*2w,)
    diffs = _angle_diff(p_nei, p_cent)
    loss = torch.mean(1.0 - torch.cos(diffs - eps))
    return loss