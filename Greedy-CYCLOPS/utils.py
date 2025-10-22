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

def _run_inference(model, test_loader, device):
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


def predict_and_save_phases(
        model, 
        test_loader, 
        preprocessing_info, 
        device='cuda', 
        save_dir='./results'
    ) -> pd.DataFrame:
    logger.info("Predicting test-set phases")

    sample_names = preprocessing_info.get('test_sample_columns', [])

    phase_coords, phases, times, celltypes = _run_inference(model, test_loader, device)

    os.makedirs(save_dir, exist_ok=True)

    results_df = _assemble_results_df(phase_coords, phases, times, celltypes, preprocessing_info, sample_names)

    _remove_legacy_prediction_files(save_dir)

    logger.info("Predicted %d samples; results available in %s", len(phases), save_dir)

    if times is not None:
        mean_error_hours = np.mean(results_df['Phase_Error_Hours'])
        std_error_hours = np.std(results_df['Phase_Error_Hours'])
        logger.info("Mean error: %.2f ± %.2f hours", mean_error_hours, std_error_hours)

    logger.debug("create_prediction_plots was removed; no additional plots saved")

    return results_df

def sanitize_filename(s: str) -> str:
    if s is None:
        return 'ALL'
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(s))


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


def load_predictions_for_comparison(pred_csv: str) -> pd.DataFrame:
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
        df['study_sample'] = df['Sample_ID'].astype(str)
    elif 'sample' in df.columns:
        df['study_sample'] = df['sample'].astype(str)
    else:
        raise ValueError(f'{pred_csv}: cannot infer study_sample (expected study+sample, Sample, or study_sample)')

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
    meta = load_metadata_for_phase_comparison(metadata_csv)

    out_dir = os.path.join(save_dir, 'phase_vs_metadata')
    os.makedirs(out_dir, exist_ok=True)

    if 'Sample_ID' not in results_df.columns or 'Predicted_Phase_Hours' not in results_df.columns:
        raise ValueError('results_df must contain Sample_ID and Predicted_Phase_Hours columns')

    preds = results_df[['Sample_ID', 'Predicted_Phase_Hours']].copy()
    preds = preds.rename(columns={'Sample_ID': 'study_sample', 'Predicted_Phase_Hours': 'pred_phase'})
    preds['study_sample'] = preds['study_sample'].astype(str)

    meta_view = meta.copy()
    meta_view['study_sample'] = meta_view['study_sample'].astype(str)

    joined = preds.merge(meta_view, on='study_sample', how='left').dropna(subset=['pred_phase', 'time_mod24'])
    if joined.empty:
        print(f"[WARN] No matches between predictions and metadata")
        return None

    phase_hours = np.asarray(joined['pred_phase'], dtype=float)
    metadata_hours = np.asarray(joined['time_mod24'], dtype=float)

    phase_rad = time_to_phase(phase_hours, period_hours=24.0)
    metadata_rad = time_to_phase(metadata_hours, period_hours=24.0)

    aligned_rad = best_align_phase_for_comparison(
        phase_rad, metadata_rad, step=0.1
    )[0]

    r = float(pearsonr(aligned_rad, metadata_rad)[0])
    spearman_R = float(spearmanr(aligned_rad, metadata_rad)[0])
    r2 = r * r if np.isfinite(r) else float('nan')

    plt.figure(figsize=(8, 7))
    plt.grid(True, linestyle='-')
    plt.scatter(aligned_rad, metadata_rad, c='b')

    two_pi = 2 * np.pi
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, two_pi]
    tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    plt.xlim(0, two_pi)
    plt.ylim(0, two_pi)
    plt.xlabel('Collection Phase', fontsize=24)
    plt.ylabel('Predicted Phase', fontsize=24)

    plt.tight_layout()

    out_path = os.path.join(out_dir, f'comparsion.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Pearson R={r:.2f}, Spearman ρ={spearman_R:.2f}")
    print(f"plot saved in: {out_path}")

    return out_path, r, r2, spearman_R


import math
def _angle_diff(a, b):
    two_pi = 2 * math.pi
    diff = torch.remainder(a - b + math.pi, two_pi) - math.pi
    return diff

def rank_loss(pred, ranks, window=3):
    device = pred.device
    n = len(ranks)
    order = torch.tensor(np.argsort(ranks), dtype=torch.long, device=device)
    base_eps = (2 * math.pi) / n
    
    total_loss = 0.0
    num_pairs = 0

    for k in range(1, window + 1):
        centers = order
        neighs = torch.roll(order, -k)

        p_cent = pred[centers]
        p_nei = pred[neighs]

        diffs = _angle_diff(p_nei, p_cent)
        
        target_diff = k * base_eps
        
        loss_k = 1.0 - torch.cos(diffs - target_diff)
        
        total_loss += torch.sum(loss_k)
        num_pairs += n

    if num_pairs == 0:
        return torch.tensor(0.0, device=device)
        
    final_loss = total_loss / num_pairs
    return final_loss