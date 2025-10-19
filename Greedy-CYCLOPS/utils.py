import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

def time_to_phase(time_hours, period_hours=24.0):
    return 2 * np.pi * time_hours / period_hours

def create_prediction_plots(results_df, save_dir):
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_df['Predicted_Phase_Hours'], bins=24, alpha=0.7, edgecolor='black')
    plt.xlabel('Predicted Phase (Hours)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Phases')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['Phase_X'], results_df['Phase_Y'], alpha=0.6)
    plt.xlabel('Phase X')
    plt.ylabel('Phase Y')
    plt.title('Phase Distribution in Unit Circle')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    if 'True_Time_Hours' in results_df.columns:
        plt.subplot(2, 2, 3)
        plt.scatter(results_df['True_Time_Hours'], results_df['Predicted_Phase_Hours'], alpha=0.6)
        plt.plot([0, 24], [0, 24], 'r--', label='Perfect Prediction')
        plt.xlabel('True Time (Hours)')
        plt.ylabel('Predicted Phase (Hours)')
        plt.title('True Time vs Predicted Phase')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.hist(results_df['Phase_Error_Hours'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (Hours)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True, alpha=0.3)
    else:
        if 'Cell_Type' in results_df.columns:
            plt.subplot(2, 2, 3)
            unique_celltypes = results_df['Cell_Type'].unique()
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, len(unique_celltypes)))
            
            for i, celltype in enumerate(unique_celltypes):
                mask = results_df['Cell_Type'] == celltype
                plt.scatter(results_df.loc[mask, 'Phase_X'], 
                          results_df.loc[mask, 'Phase_Y'], 
                          c=[colors[i]], label=celltype, alpha=0.6)
            
            plt.xlabel('Phase X')
            plt.ylabel('Phase Y')
            plt.title('Phase Distribution by Cell Type')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"预测分析图表保存到: {os.path.join(save_dir, 'prediction_analysis.png')}")


def predict_and_save_phases(model, test_loader, preprocessing_info, device='cuda', save_dir='./results'):
    print("\n=== 预测测试集相位 ===")
    model.eval()
    
    celltype_to_idx = preprocessing_info.get('celltype_to_idx', {})
    sample_names = preprocessing_info.get('test_sample_columns', [])
    
    all_phase_coords = []
    all_phases = []
    all_times = []
    all_celltypes = []
    
    batch_start_idx = 0
    
    with torch.no_grad():
        for batch in test_loader:
            expressions = batch['expression'].to(device)
            times = batch.get('time', None)
            celltypes = batch.get('celltype', None)
            
            celltype_indices = None
            if celltypes is not None and celltype_to_idx:
                batch_celltype_indices = []
                for ct in celltypes:
                    batch_celltype_indices.append(celltype_to_idx.get(ct, 0))
                celltype_indices = torch.tensor(batch_celltype_indices, device=device)
            
            phase_coords, phase_angles, _ = model(expressions, celltype_indices)
            
            all_phase_coords.append(phase_coords.cpu().numpy())
            all_phases.append(phase_angles.cpu().numpy())
            
            batch_size = expressions.shape[0]
            batch_start_idx += batch_size
            
            if times is not None:
                all_times.append(times.cpu().numpy())
            if celltypes is not None:
                all_celltypes.extend(celltypes)
    
    phase_coords = np.vstack(all_phase_coords)
    phases = np.concatenate(all_phases)
    
    if all_times:
        times = np.concatenate(all_times)
    else:
        times = None
        
    if all_celltypes:
        celltypes = np.array(all_celltypes)
    else:
        celltypes = None
    
    os.makedirs(save_dir, exist_ok=True)
    
    if sample_names and len(sample_names) == len(phases):
        sample_identifiers = sample_names
    else:
        sample_identifiers = [f"Sample_{i}" for i in range(len(phases))]
        print("警告: 无法获取样本名称，使用生成的索引")
    
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
        results_data['Phase_Error_Radians'] = np.abs(phases - results_data['True_Phase_Radians'])
        results_data['Phase_Error_Radians'] = np.minimum(
            results_data['Phase_Error_Radians'], 
            2*np.pi - results_data['Phase_Error_Radians']
        )
        results_data['Phase_Error_Hours'] = results_data['Phase_Error_Radians'] * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)
    
    if celltypes is not None:
        results_data['Cell_Type'] = celltypes
    
    results_df = pd.DataFrame(results_data)
    
    # 不保存任何预测CSV，确保目录内不遗留历史预测文件
    for fname in ('phase_predictions.csv', 'phase_predictions_simple.csv'):
        fpath = os.path.join(save_dir, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
                print(f"已删除遗留文件: {fpath}")
            except Exception as e:
                print(f"[WARN] 无法删除遗留文件 {fpath}: {e}")
    
    print(f"\n=== 预测统计 ===")
    print(f"预测样本数量: {len(phases)}")
    print(f"预测相位范围: {phases.min():.3f} - {phases.max():.3f} 弧度")
    print(f"预测相位范围: {(phases * 180 / np.pi).min():.1f} - {(phases * 180 / np.pi).max():.1f} 度")
    print(f"预测时间范围: {results_data['Predicted_Phase_Hours'].min():.2f} - {results_data['Predicted_Phase_Hours'].max():.2f} 小时")
    
    if times is not None:
        mean_error_hours = np.mean(results_data['Phase_Error_Hours'])
        std_error_hours = np.std(results_data['Phase_Error_Hours'])
        print(f"平均预测误差: {mean_error_hours:.2f} ± {std_error_hours:.2f} 小时")
        
        for threshold in [1, 2, 3, 6]:
            accuracy = np.mean(results_data['Phase_Error_Hours'] <= threshold) * 100
            print(f"误差 ≤ {threshold}小时的样本比例: {accuracy:.1f}%")
    
    if celltypes is not None:
        print(f"\n按细胞类型统计:")
        celltype_stats = results_df.groupby('Cell_Type').agg({
            'Predicted_Phase_Hours': ['mean', 'std', 'count']
        }).round(2)
        print(celltype_stats)
    
    create_prediction_plots(results_df, save_dir)
    
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
) -> tuple[np.ndarray, float, float, float, float, bool]:
    from scipy.stats import pearsonr, spearmanr

    x_arr: np.ndarray = np.asarray(x_hours, dtype=float)
    y_arr: np.ndarray = np.asarray(y_hours, dtype=float)
    shifts: np.ndarray = np.arange(0.0, 24.0, step, dtype=float)

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
        x0: np.ndarray = (24.0 - x_arr) % 24.0 if flipped else x_arr
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

        phase_hours = np.asarray(joined['pred_phase'], dtype=float)
        metadata_hours = np.asarray(joined['time_mod24'], dtype=float)

        aligned, r, r2, spearman_R, best_shift, flipped = best_align_phase_for_comparison(phase_hours, metadata_hours, step=0.1)

        plt.figure(figsize=(8, 7))
        plt.scatter(aligned, metadata_hours)
        plt.xlabel('Predicted Time', fontsize=24)
        plt.ylabel('Collection Time', fontsize=24)

        subtitle = f'Shift={best_shift:.2f}h'
        if flipped:
            subtitle += ' (flipped)'

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