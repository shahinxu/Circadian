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
        n_components=50,
        blunt_percent=0.975
    ):
    print("=== Loading training data ===")
    df = pd.read_csv(train_file, low_memory=False)
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    gene_df = df[~df['Gene_Symbol'].isin(['time_C'])].copy()
    
    import os
    train_dir = os.path.dirname(train_file)
    seed_genes_path = os.path.join(train_dir, 'seed_genes.txt')
    
    if os.path.exists(seed_genes_path):
        print(f"Loading seed genes from {seed_genes_path}")
        with open(seed_genes_path, 'r') as f:
            seed_genes = set(line.strip().upper() for line in f if line.strip())
        print(f"Found {len(seed_genes)} seed genes")
        
        gene_df['Gene_Symbol_Upper'] = gene_df['Gene_Symbol'].str.upper()
        
        # Filter to keep only genes in seed_genes
        before_filter = len(gene_df)
        gene_df = gene_df[gene_df['Gene_Symbol_Upper'].isin(seed_genes)].copy()
        after_filter = len(gene_df)
        
        print(f"Filtered genes: {before_filter} -> {after_filter} (kept {after_filter} seed genes)")
        
        # Remove the temporary uppercase column
        gene_df = gene_df.drop(columns=['Gene_Symbol_Upper'])
        
        if len(gene_df) == 0:
            raise ValueError("No genes remain after filtering with seed_genes.txt")
    else:
        print("No seed_genes.txt found, processing all genes")
    
    initial_gene_symbols = gene_df['Gene_Symbol'].values
    sample_df = gene_df[sample_columns].copy()
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')
    if sample_df.isna().any().any():
        n_nan = int(sample_df.isna().sum().sum())
        print(f"Warning: {n_nan} non-numeric values found in sample columns; coercing to NaN and imputing 0.")
        sample_df = sample_df.fillna(0.0)
    expression_data = sample_df.values.T

    print(f"Initial data shape: {expression_data.shape}")

    expression_data = blunt_percentile(expression_data, percent=blunt_percent)
    print(f"Shape after blunt_percentile: {expression_data.shape}")
    
    if np.isnan(expression_data).any():
        print(f"Warning: NaN values detected after blunt_percentile, filling with 0.")
        expression_data = np.nan_to_num(expression_data, nan=0.0)

    final_gene_symbols = initial_gene_symbols
    expression_data_final_filtered = expression_data

    if expression_data_final_filtered.shape[1] == 0:
        raise ValueError("All genes were removed after all filtering steps.")

    final_scaler = StandardScaler()
    expression_scaled = final_scaler.fit_transform(expression_data_final_filtered)

    print("Performing PCA...")
    actual_n_components = min(n_components, expression_scaled.shape[1], expression_scaled.shape[0])
    if actual_n_components < n_components:
        print(f"Warning: Reducing n_components from {n_components} to {actual_n_components} due to limited features ({expression_scaled.shape[1]}) or samples ({expression_scaled.shape[0]}) after filtering.")
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
         print(f"Data shape fed to PCA: {expression_scaled.shape}")
         raise

    train_dataset = ExpressionDataset(pca_components)

    preprocessing_info = {
        'scaler': final_scaler,
        'pca_model': pca_model,
        'sample_columns': sample_columns,
        'n_components': actual_n_components,
        'final_gene_symbols': final_gene_symbols,
        'blunt_percent': blunt_percent
    }
    return train_dataset, preprocessing_info


def load_and_preprocess_test_data(test_file, preprocessing_info):
    print("\n=== Loading test data ===")
    df = pd.read_csv(test_file, low_memory=False)
    sample_columns = preprocessing_info['sample_columns']
    available_sample_columns = [col for col in sample_columns if col in df.columns]
    gene_df = df[~df['Gene_Symbol'].isin(['time_C'])].copy()
    final_gene_symbols = preprocessing_info['final_gene_symbols']
    if gene_df['Gene_Symbol'].duplicated().any():
        agg_cols = [c for c in available_sample_columns if c in gene_df.columns]
        if len(agg_cols) == 0:
            pass
        else:
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
    actual_n_components = preprocessing_info['n_components']
    test_pca_components = pca_model.transform(test_expression_scaled)

    print(f"Test data after PCA shape: {test_pca_components.shape}")
    print(f"Expected n_components: {actual_n_components}")

    test_dataset = ExpressionDataset(test_pca_components)

    test_preprocessing_info = preprocessing_info.copy()
    test_preprocessing_info.update({
        'test_sample_columns': available_sample_columns,
        'test_pca_components': test_pca_components
    })
    return test_dataset, test_preprocessing_info