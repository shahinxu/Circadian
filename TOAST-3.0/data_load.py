import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import gc


class ExpressionDataset(Dataset):
    def __init__(self, expressions, tissue_labels=None):
        self.expressions = torch.FloatTensor(expressions)
        self.tissue_labels = tissue_labels  # List of tissue names or None

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        item = {'expression': self.expressions[idx]}
        if self.tissue_labels is not None:
            item['tissue'] = self.tissue_labels[idx]
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
        blunt_percent=0.975
    ):
    print("=== Loading training data ===")
    df = pd.read_csv(train_file, low_memory=False)
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    gene_df = df[~df['Gene_Symbol'].isin(['time_C'])].copy()
    
    # Load metadata for tissue labels
    import os
    train_dir = os.path.dirname(train_file)
    metadata_path = os.path.join(train_dir, 'metadata.csv')
    tissue_labels = None
    tissue_to_idx = None
    tissue_indices = None
    
    if os.path.exists(metadata_path):
        print(f"Loading metadata from {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)
        if 'Sample' in metadata_df.columns and 'Tissue' in metadata_df.columns:
            # Create a mapping from sample name to tissue
            sample_to_tissue = dict(zip(metadata_df['Sample'], metadata_df['Tissue']))
            # Get tissue labels for each sample in sample_columns
            tissue_labels = [sample_to_tissue.get(col, 'unknown') for col in sample_columns]
            # Create tissue to index mapping
            unique_tissues = sorted(set(tissue_labels))
            tissue_to_idx = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
            tissue_indices = [tissue_to_idx[t] for t in tissue_labels]
            print(f"Found {len(unique_tissues)} unique tissues: {unique_tissues}")
            print(f"Tissue distribution: {pd.Series(tissue_labels).value_counts().to_dict()}")
        else:
            print("Warning: metadata.csv does not contain 'Sample' and 'Tissue' columns")
    else:
        print("No metadata.csv found, tissue labels will not be available")
    
    # Load seed genes if available
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

    print(f"Final expression data shape: {expression_scaled.shape}")
    print(f"Number of genes: {expression_scaled.shape[1]}")
    print(f"Number of samples: {expression_scaled.shape[0]}")

    train_dataset = ExpressionDataset(expression_scaled, tissue_labels=tissue_labels)

    preprocessing_info = {
        'scaler': final_scaler,
        'sample_columns': sample_columns,
        'input_dim': expression_scaled.shape[1],
        'final_gene_symbols': final_gene_symbols,
        'blunt_percent': blunt_percent,
        'tissue_labels': tissue_labels,
        'tissue_to_idx': tissue_to_idx,
        'tissue_indices': tissue_indices if tissue_labels else None,
        'num_tissues': len(tissue_to_idx) if tissue_to_idx else 0
    }
    return train_dataset, preprocessing_info


def load_and_preprocess_test_data(test_file, preprocessing_info):
    print("\n=== Loading test data ===")
    df = pd.read_csv(test_file, low_memory=False)
    sample_columns = preprocessing_info['sample_columns']
    available_sample_columns = [col for col in sample_columns if col in df.columns]
    
    # Load test metadata for tissue labels
    import os
    test_dir = os.path.dirname(test_file)
    metadata_path = os.path.join(test_dir, 'metadata.csv')
    test_tissue_labels = None
    
    if os.path.exists(metadata_path) and preprocessing_info.get('tissue_to_idx'):
        print(f"Loading test metadata from {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)
        if 'Sample' in metadata_df.columns and 'Tissue' in metadata_df.columns:
            sample_to_tissue = dict(zip(metadata_df['Sample'], metadata_df['Tissue']))
            test_tissue_labels = [sample_to_tissue.get(col, 'unknown') for col in available_sample_columns]
            print(f"Test tissue distribution: {pd.Series(test_tissue_labels).value_counts().to_dict()}")
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

    print(f"Test data shape: {test_expression_scaled.shape}")
    print(f"Number of genes: {test_expression_scaled.shape[1]}")
    print(f"Number of samples: {test_expression_scaled.shape[0]}")

    test_dataset = ExpressionDataset(test_expression_scaled, tissue_labels=test_tissue_labels)

    test_preprocessing_info = preprocessing_info.copy()
    test_preprocessing_info.update({
        'test_sample_columns': available_sample_columns,
        'test_expression_scaled': test_expression_scaled,
        'test_tissue_labels': test_tissue_labels
    })
    return test_dataset, test_preprocessing_info