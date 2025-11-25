import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from PCA import create_eigengenes
import gc


class ExpressionDataset(Dataset):
    def __init__(self, expressions):
        self.expressions = torch.FloatTensor(expressions)

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        return {'expression': self.expressions[idx]}


def blunt_percentile(data, percent=0.975):
    n = data.shape[0]
    nfloor_idx = max(0, int(np.floor((1 - percent) * n)))
    nceiling_idx = min(n - 1, int(np.ceil(percent * n)) - 1)
    if nfloor_idx >= nceiling_idx or n <= 1:
        return data
    sorted_data = np.sort(data, axis=0)
    row_min = sorted_data[nfloor_idx, :]
    row_max = sorted_data[nceiling_idx, :]
    data = np.clip(data, row_min, row_max)
    return data


def load_and_preprocess_train_data(
        train_file,
        n_components=100,
        blunt_percent=0.975,
        min_cv=0.14,
        max_cv=0.9,
        min_mean_rank=10000,
        use_oscope_filter=True,
        min_pair_corr=0.2,
        max_pair_corr=0.9,
        min_gene_pairs=5
    ):
    """
    Load and preprocess training data with PCA to intermediate dimension.
    PCA reduces genes to n_components (e.g., 100), then Transformer compresses further.
    """
    print(f"=== Loading training data with PCA (n_components={n_components}) ===")
    df = pd.read_csv(train_file, low_memory=False)
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    gene_df = df[~df['Gene_Symbol'].isin(['time_C'])].copy()
    initial_gene_symbols = gene_df['Gene_Symbol'].values
    sample_df = gene_df[sample_columns].copy()
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')
    if sample_df.isna().any().any():
        n_nan = int(sample_df.isna().sum().sum())
        print(f"Warning: {n_nan} non-numeric values found in sample columns; coercing to NaN and imputing 0.")
        try:
            example_nan = sample_df.columns[sample_df.isna().any()].tolist()[:5]
            print(f"Columns with NaNs (examples): {example_nan}")
        except Exception:
            pass
    expression_data = sample_df.values.T

    print(f"Initial data shape: {expression_data.shape}")

    expression_data = blunt_percentile(expression_data, percent=blunt_percent)
    print(f"Shape after blunt_percentile: {expression_data.shape}")

    gene_means = np.mean(expression_data, axis=0)
    gene_stds = np.std(expression_data, axis=0)
    gene_cvs = gene_stds / (gene_means + 1e-8)
    mean_rank = np.argsort(np.argsort(-gene_means))
    initial_keep = (gene_cvs > min_cv) & (gene_cvs < max_cv) & (mean_rank < min_mean_rank)

    expression_data_initial_filtered = expression_data[:, initial_keep]
    initial_filtered_gene_symbols = initial_gene_symbols[initial_keep]
    print(f"Shape after initial CV/Mean filtering: {expression_data_initial_filtered.shape}")

    if expression_data_initial_filtered.shape[1] == 0:
        print("Warning: Initial CV/Mean filtering removed all genes. Falling back to keep all genes.")
        final_keep_mask = np.ones_like(initial_keep, dtype=bool)
        final_gene_symbols = initial_gene_symbols
        expression_data_final_filtered = expression_data[:, final_keep_mask]
        fallback_initial_filter = True
    else:
        fallback_initial_filter = False

    if not fallback_initial_filter:
        final_keep_mask = initial_keep
        final_gene_symbols = initial_filtered_gene_symbols
        expression_data_final_filtered = expression_data_initial_filtered

    if use_oscope_filter:
        print("Applying Oscope-inspired pair filtering...")
        if expression_data_initial_filtered.shape[1] <= 1:
             print("Skipping Oscope filter: Not enough genes after initial filtering.")
        else:
            scaler_filter = StandardScaler()
            expression_scaled_filter = scaler_filter.fit_transform(expression_data_initial_filtered)

            print("Calculating correlation matrix...")
            try:
                corr_matrix = np.corrcoef(expression_scaled_filter.astype(np.float64), rowvar=False)
            except MemoryError:
                 print("MemoryError calculating full correlation matrix. Consider reducing genes or using incremental methods.")
                 raise
            corr_matrix = np.nan_to_num(corr_matrix)
            n_filtered_genes = corr_matrix.shape[0]
            print(f"Correlation matrix shape: {corr_matrix.shape}")

            print("Identifying good pairs and counting gene occurrences...")
            gene_pair_counts = np.zeros(n_filtered_genes, dtype=int)
            abs_corr = np.abs(corr_matrix)
            triu_mask = np.triu(np.ones_like(abs_corr, dtype=bool), k=1)
            corr_range_mask = (abs_corr > min_pair_corr) & (abs_corr < max_pair_corr)
            valid_pair_mask = triu_mask & corr_range_mask
            row_indices, col_indices = np.where(valid_pair_mask)
            np.add.at(gene_pair_counts, row_indices, 1)
            np.add.at(gene_pair_counts, col_indices, 1)
            n_good_pairs = len(row_indices)
            print(f"Found {n_good_pairs} pairs with abs(corr) between {min_pair_corr} and {max_pair_corr}.")
            oscope_keep_mask_relative = gene_pair_counts >= min_gene_pairs
            n_oscope_genes = np.sum(oscope_keep_mask_relative)
            print(f"Found {n_oscope_genes} genes participating in at least {min_gene_pairs} good pairs.")

            if n_oscope_genes == 0:
                print("Warning: Oscope filter removed all initially filtered genes. Reverting to initial CV/Mean filter.")
            else:
                # Use oscope filtered genes regardless of count
                print(f"Using {n_oscope_genes} genes after Oscope filtering.")
                initial_indices = np.where(initial_keep)[0]
                oscope_kept_indices = initial_indices[oscope_keep_mask_relative]
                oscope_keep_mask_final = np.zeros_like(initial_keep, dtype=bool)
                oscope_keep_mask_final[oscope_kept_indices] = True
                final_keep_mask = oscope_keep_mask_final

                final_gene_symbols = initial_gene_symbols[final_keep_mask]
                expression_data_final_filtered = expression_data[:, final_keep_mask]
                print(f"Shape after Oscope filtering: {expression_data_final_filtered.shape}")

            del expression_scaled_filter, corr_matrix, abs_corr, triu_mask, corr_range_mask, valid_pair_mask
            gc.collect()

    if expression_data_final_filtered.shape[1] == 0:
        raise ValueError("All genes were removed after all filtering steps.")

    final_scaler = StandardScaler()
    expression_scaled = final_scaler.fit_transform(expression_data_final_filtered)

    print(f"Performing PCA to {n_components} components...")
    actual_n_components = min(n_components, expression_scaled.shape[1], expression_scaled.shape[0])
    if actual_n_components < n_components:
        print(f"Warning: Reducing n_components from {n_components} to {actual_n_components} due to limited features ({expression_scaled.shape[1]}) or samples ({expression_scaled.shape[0]}).")
    if actual_n_components <= 0:
        raise ValueError("Cannot perform PCA with 0 components.")

    try:
        pca_components, pca_model, explained_variance_ratios = create_eigengenes(expression_scaled, actual_n_components)
        print(f"PCA output shape: {pca_components.shape}")
        if isinstance(explained_variance_ratios, np.ndarray):
            print(f"Explained variance by {actual_n_components} components: {np.sum(explained_variance_ratios):.4f}")
        else:
            print(f"Explained variance by {actual_n_components} components: {explained_variance_ratios:.4f}")
    except Exception as e:
        print(f"Error during PCA: {e}")
        raise

    # Create dataset from PCA components
    train_dataset = ExpressionDataset(pca_components)

    preprocessing_info = {
        'scaler': final_scaler,
        'pca_model': pca_model,
        'sample_columns': sample_columns,
        'n_components': actual_n_components,
        'gene_keep_mask': final_keep_mask,
        'final_gene_symbols': final_gene_symbols,
        'blunt_percent': blunt_percent,
        'use_oscope_filter': use_oscope_filter,
        'min_pair_corr': min_pair_corr,
        'max_pair_corr': max_pair_corr,
        'min_gene_pairs': min_gene_pairs,
        'fallback_initial_filter': fallback_initial_filter
    }
    return train_dataset, preprocessing_info


def load_and_preprocess_test_data(test_file, preprocessing_info):
    """
    Load and preprocess test data with PCA.
    """
    print("\n=== Loading test data with PCA ===")
    df = pd.read_csv(test_file, low_memory=False)
    sample_columns = preprocessing_info['sample_columns']
    available_sample_columns = [col for col in sample_columns if col in df.columns]
    gene_df = df[~df['Gene_Symbol'].isin(['time_C'])].copy()
    final_gene_symbols = preprocessing_info['final_gene_symbols']
    if gene_df['Gene_Symbol'].duplicated().any():
        # If the test file contains duplicate Gene_Symbol rows (e.g., multiple
        # probes mapping to the same gene), aggregate them by mean across the
        # available sample columns. Ensure aggregation columns are numeric to
        # avoid dtype/object aggregation errors.
        agg_cols = [c for c in available_sample_columns if c in gene_df.columns]
        if len(agg_cols) == 0:
            # nothing to aggregate; leave gene_df as-is
            pass
        else:
            # coerce aggregation columns to numeric, non-parsable -> NaN
            try:
                gene_df[agg_cols] = gene_df[agg_cols].apply(pd.to_numeric, errors='coerce')
            except Exception:
                print("Warning: failed to coerce test aggregation columns to numeric; proceeding with original types")
            grouped = gene_df.groupby('Gene_Symbol', as_index=False)[agg_cols].mean()
            gene_df = grouped

    gene_df_indexed = gene_df.set_index('Gene_Symbol').reindex(final_gene_symbols)
    test_expression_data = gene_df_indexed[available_sample_columns].values.T

    if np.isnan(test_expression_data).any():
        test_expression_data = np.nan_to_num(test_expression_data, nan=0.0)

    test_expression_data = blunt_percentile(test_expression_data, percent=preprocessing_info['blunt_percent'])

    scaler = preprocessing_info['scaler']
    test_expression_scaled = scaler.transform(test_expression_data)
    
    pca_model = preprocessing_info['pca_model']
    test_pca_components = pca_model.transform(test_expression_scaled)

    print(f"Test data after PCA shape: {test_pca_components.shape}")

    test_dataset = ExpressionDataset(test_pca_components)

    test_preprocessing_info = preprocessing_info.copy()
    test_preprocessing_info.update({
        'test_sample_columns': available_sample_columns,
    })
    return test_dataset, test_preprocessing_info