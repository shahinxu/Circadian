import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 11})
import os
from core.optimizer import MultiScaleOptimizer
from config import Config
from core.data import (
    load_and_process_expression_data,
    load_metadata_generic,
    get_circadian_gene_expressions,
)
from sklearn.preprocessing import StandardScaler
from core.features import eigengenes as core_eigengenes
from core.plots import phase_scatter, circadian_single_dataset
from core.align import to_24, best_shift
from core.utils import trim_weights

from typing import Optional

def main(
    expression_file: Optional[str] = None,
    metadata_file: Optional[str] = None,
    weights: Optional[str] = None,
    device: str = 'cuda',
    method: str = 'greedy',
):
    if expression_file is None:
        expression_file = Config.DEFAULT_EXPRESSION_FILE
    if metadata_file is None:
        metadata_file = Config.DEFAULT_METADATA_FILE
    n_components = Config.N_COMPONENTS

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"result_{timestamp}_components{n_components}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")
    print(f"Device: {device} (note: current pipeline runs on CPU; this flag is for consistency)")
    print(f"Optimization method: {method} (choices: 'greedy' or 'neural')")

    circadian_genes = ['PER1', 'PER2', 'CRY1', 'CRY2', 'CLOCK', 'ARNTL', 'NR1D1', 'NR1D2', 'DBP']

    # Use a single Balanced configuration
    selected_config = {'smoothness_factor': 0.7, 'local_variation_factor': 0.3, 'name': 'Balanced', 'method': method}
    # Default: no weights unless provided via --weights
    weights_to_use = None
    if weights:
        try:
            parts = [p.strip() for p in weights.split(',') if p.strip()]
            arr = [float(p) for p in parts]
            if len(arr) > 0:
                weights_to_use = arr
                print(f"Using inline weights list of length {len(arr)}")
        except Exception as e:
            print(f"Could not parse --weights '{weights}': {e}")

    print("=== Eigengene-Based Multi-Scale Optimization ===")

    data_info = load_and_process_expression_data(expression_file, n_components)

    circadian_expressions, found_circadian_genes = get_circadian_gene_expressions(
        data_info['original_expression'],
        data_info['gene_names'],
        circadian_genes,
    )
    if found_circadian_genes is None:
        found_circadian_genes = []

    if circadian_expressions is None:
        # Proceed without circadian gene visualization; keep the rest of the pipeline running
        print("[WARN] No circadian genes found; proceeding without circadian gene visualization.")
        found_circadian_genes = []
        circadian_expressions = None

    actual_output_dir = f"{output_dir}_genes{len(found_circadian_genes)}_single_dataset"
    if output_dir != actual_output_dir:
        os.rename(output_dir, actual_output_dir)
        output_dir = actual_output_dir
        print(f"Updated output directory to: {os.path.abspath(output_dir)}")

    # Always process as single dataset (no cell type separation)
    print("\nProcessing all samples as single dataset...")

    results = {}
    # Compute PCA for the entire dataset once (single-dataset mode)
    original_expression = data_info['original_expression']
    n_samples, n_features = original_expression.shape
    print(f"Total samples: {n_samples}, features: {n_features}")

    # Choose valid number of PCA components based on data shape
    n_components_sd = min(n_components, n_samples, n_features)
    if n_components_sd < n_components:
        print(f"[Info] Reducing PCA components from {n_components} to {n_components_sd} based on data shape")

    scaler_sd = StandardScaler()
    scaled_sd = scaler_sd.fit_transform(original_expression)

    eigengenes_sd, pca_model_sd, explained_variance_sd = core_eigengenes(scaled_sd, n_components=n_components_sd)

    # Trim weights for single-dataset PCA
    weights_for_sd = trim_weights(weights_to_use, eigengenes_sd.shape[1])
    if weights_for_sd is not None and not isinstance(weights_for_sd, np.ndarray):
        weights_for_sd = np.array(weights_for_sd, dtype=float)

    optimizer = MultiScaleOptimizer(
        smoothness_factor=selected_config['smoothness_factor'],
        local_variation_factor=selected_config['local_variation_factor'],
        window_size=Config.DEFAULT_WINDOW_SIZE,
        max_iterations_ratio=Config.MAX_ITERATIONS_RATIO,
        variation_tolerance_ratio=Config.VARIATION_TOLERANCE_RATIO,
        method=method,
        device=device
    )

    ranks = optimizer.optimize(eigengenes_sd, weights_for_sd)
    
    metrics = optimizer.analyze_metrics(eigengenes_sd, ranks)

    results[selected_config['name']] = {
        'ranks': ranks,
        'metrics': metrics,
        'config': selected_config,
        'eigengenes': eigengenes_sd,
        'circadian_expressions': circadian_expressions,
        'pca_model': pca_model_sd,
        'explained_variance': explained_variance_sd
    }

    print(f"  Balance Score: {metrics['balance_score']:.4f}")

    # Only plot circadian gene visualization if we have any
    if found_circadian_genes:
        circadian_single_dataset(
            results,
            found_circadian_genes,
            output_dir,
        )
    else:
        print("[Info] Skipping circadian gene visualization (0 genes found).")

    if metadata_file:
        if not os.path.isfile(metadata_file):
            print(f"Provided metadata file not found: {metadata_file}. Skipping rank-time plots.")
            return

        ranks_root = os.path.abspath(output_dir)
        meta = load_metadata_generic(metadata_file)
        if meta is None:
            print("Skipping rank-vs-time plotting due to incompatible metadata format.")
            return

        # Use core.align + core.plots for单数据集对齐与出图
        sample_names = np.array(data_info['sample_columns'])
        for config_name, resdict in results.items():
            ranks_arr = resdict['ranks'].flatten()
            df = pd.DataFrame({'study_sample': sample_names, 'Rank': ranks_arr})
            joined = df.merge(meta[['study_sample', 'time_mod24']], on='study_sample', how='left')
            df_clean = joined[['Rank', 'time_mod24']].copy()
            df_clean['Rank'] = pd.to_numeric(df_clean['Rank'], errors='coerce')
            df_clean['time_mod24'] = pd.to_numeric(df_clean['time_mod24'], errors='coerce')
            df_clean = df_clean.dropna(subset=['Rank', 'time_mod24'])
            x0 = to_24(df_clean['Rank'].to_numpy(dtype=float))
            y = df_clean['time_mod24'].to_numpy(dtype=float)
            res = best_shift(x0, y, step=0.05)
            x_use = x0 if res.orientation == 'normal' else (24.0 - x0) % 24.0
            x_shift = (x_use + (res.shift if res.shift is not None else 0.0)) % 24.0

            out_dir = os.path.join(ranks_root, 'rank_vs_time')
            os.makedirs(out_dir, exist_ok=True)
            phase_scatter(x_shift, y, f"r={res.corr:.3f}, shift={res.shift:.2f}h, {res.orientation}",
                          os.path.join(out_dir, 'rank_vs_time.png'))
            # Save summary
            pd.DataFrame([{
                'n': int(len(y)),
                'pearson_r': float(res.corr) if res.corr is not None else np.nan,
                'r_square': float(res.r2) if res.r2 is not None else np.nan,
                'slope': float(res.slope) if res.slope is not None else np.nan,
                'shift': float(res.shift) if res.shift is not None else np.nan,
                'orientation': res.orientation,
                'config': config_name
            }]).to_csv(os.path.join(out_dir, 'compare_summary.csv'), index=False)
        return

    # If no metadata provided, just finish after visualization
    return

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="PiCASSO single-dataset eigengene optimization and visualization")
    parser.add_argument('--expression', type=str, default=None, help='Path to expression.csv (samples x genes)')
    parser.add_argument('--metadata', type=str, default=None, help='Optional metadata.csv with study_sample/time_mod24')
    parser.add_argument('--weights', type=str, default=None, help='Inline comma-separated weights, e.g., 1,0.8,0.6')
    parser.add_argument('--device', type=str, default='cuda', help='Device hint (cuda or cpu). For compatibility; computations use CPU.')
    parser.add_argument('--method', type=str, default='greedy', choices=['greedy', 'neural'], help="Optimization method: 'greedy' (default) or 'neural'.")
    return parser

if __name__ == '__main__':
    # Parse CLI args and run
    _parser = _build_arg_parser()
    args = _parser.parse_args()
    main(
        expression_file=args.expression,
        metadata_file=args.metadata,
        weights=args.weights,
        device=args.device,
        method=args.method,
    )
