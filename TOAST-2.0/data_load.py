import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gc


class ExpressionDataset(Dataset):
    def __init__(self, expressions, covariates=None):
        self.expressions = torch.FloatTensor(expressions)
        self.covariates = torch.FloatTensor(covariates) if covariates is not None else None

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        item = {'expression': self.expressions[idx]}
        if self.covariates is not None:
            item['covariates'] = self.covariates[idx]
        return item


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
        blunt_percent=0.975,
        pathway_csv=None
    ):
    print("=== Loading training data ===")
    df = pd.read_csv(train_file, low_memory=False)
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    gene_df = df[~df['Gene_Symbol'].isin(['time_C'])].copy()
    
    import os
    train_dir = os.path.dirname(train_file)
    seed_genes_path = os.path.join(train_dir, 'seed_genes.txt')
    
    # Skip seed genes filtering - use all genes
    print(f"Using all {len(gene_df)} genes (seed_genes.txt ignored if present)")
    
    initial_gene_symbols = gene_df['Gene_Symbol'].values
    sample_df = gene_df[sample_columns].copy()
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')
    if sample_df.isna().any().any():
        n_nan = int(sample_df.isna().sum().sum())
        print(f"Warning: {n_nan} non-numeric values found in sample columns; coercing to NaN and imputing 0.")
    expression_data = sample_df.values.T

    print(f"Initial data shape: {expression_data.shape}")

    expression_data = blunt_percentile(expression_data, percent=blunt_percent)
    print(f"Shape after blunt_percentile: {expression_data.shape}")

    final_gene_symbols = initial_gene_symbols
    expression_data_final_filtered = expression_data

    if expression_data_final_filtered.shape[1] == 0:
        raise ValueError("All genes were removed after all filtering steps.")

    final_scaler = StandardScaler()
    expression_scaled = final_scaler.fit_transform(expression_data_final_filtered)

    print(f"Scaled expression data shape: {expression_scaled.shape}")

    train_dataset = ExpressionDataset(expression_scaled, covariates=None)

    # Load pathway information if provided
    pathway_info = None
    if pathway_csv and os.path.exists(pathway_csv):
        print(f"Loading pathway information from {pathway_csv}")
        from pathway_loader import load_pathway_dataset, build_pathway_map, get_pathway_statistics
        
        pathway_df = load_pathway_dataset(pathway_csv)
        pathway_indices, pathway_names, gene_to_idx = build_pathway_map(
            pathway_df, 
            final_gene_symbols.tolist(),
            min_pathway_size=100,
            max_pathway_size=500
        )
        
        pathway_stats = get_pathway_statistics(pathway_indices, pathway_names)
        print(f"Pathway statistics:\n{pathway_stats.head(10)}")
        
        pathway_info = {
            'pathway_indices': pathway_indices,
            'pathway_names': pathway_names,
            'gene_to_idx': gene_to_idx,
            'pathway_stats': pathway_stats
        }
    else:
        print("No pathway information provided, using standard model")

    preprocessing_info = {
        'scaler': final_scaler,
        'sample_columns': sample_columns,
        'final_gene_symbols': final_gene_symbols,
        'blunt_percent': blunt_percent,
        'input_dim': expression_scaled.shape[1],
        'pathway_info': pathway_info
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
        if len(agg_cols) > 0:
            try:
                gene_df[agg_cols] = gene_df[agg_cols].apply(pd.to_numeric, errors='coerce')
            except Exception:
                print("Warning: failed to coerce test aggregation columns to numeric")
            grouped = gene_df.groupby('Gene_Symbol', as_index=False)[agg_cols].mean()
            gene_df = grouped

    gene_df_indexed = gene_df.set_index('Gene_Symbol').reindex(final_gene_symbols)
    test_expression_data = gene_df_indexed[available_sample_columns].values.T

    if np.isnan(test_expression_data).any():
        test_expression_data = np.nan_to_num(test_expression_data, nan=0.0)

    test_expression_data = blunt_percentile(test_expression_data, percent=preprocessing_info['blunt_percent'])

    scaler = preprocessing_info['scaler']
    test_expression_scaled = scaler.transform(test_expression_data)
    print(f"Test data scaled shape: {test_expression_scaled.shape}")

    test_dataset = ExpressionDataset(test_expression_scaled, covariates=None)

    test_preprocessing_info = preprocessing_info.copy()
    test_preprocessing_info['sample_columns'] = available_sample_columns

    return test_dataset, test_preprocessing_info
