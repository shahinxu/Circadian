import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import gc
import matplotlib.pyplot as plt
import os


class ExpressionDataset(Dataset):
    def __init__(self, expressions):
        self.expressions = torch.FloatTensor(expressions)

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        return {'expression': self.expressions[idx]}


def plot_expression_over_time(expression_scaled, metadata_file, gene_symbols, save_dir, n_genes=20):
    """
    Plot normalized expression curves over time for selected genes.
    
    Parameters:
    -----------
    expression_scaled : np.ndarray
        Normalized expression data with shape (n_samples, n_genes)
    metadata_file : str
        Path to metadata.csv containing Time_Hours column
    gene_symbols : list or np.ndarray
        Gene symbols corresponding to columns in expression_scaled
    save_dir : str
        Directory to save the plot
    n_genes : int
        Number of genes to plot (default: 20, randomly selected)
    """
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return
    
    # Read metadata
    metadata = pd.read_csv(metadata_file)
    if 'Time_Hours' not in metadata.columns:
        print("Time_Hours column not found in metadata")
        return
    
    # Check dimensions match
    if len(metadata) != expression_scaled.shape[0]:
        print(f"Warning: Metadata samples ({len(metadata)}) != Expression samples ({expression_scaled.shape[0]})")
        return
    
    # Create dataframe
    df = pd.DataFrame(expression_scaled, columns=gene_symbols)
    df['Time_Hours'] = metadata['Time_Hours'].values
    
    # Select genes to plot
    n_genes = min(n_genes, len(gene_symbols))
    np.random.seed(42)
    selected_indices = np.random.choice(len(gene_symbols), n_genes, replace=False)
    selected_genes = [gene_symbols[i] for i in selected_indices]
    
    time_points = sorted(df['Time_Hours'].unique())
    
    # Create plot with subplots
    n_cols = 4
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_genes > 1 else [axes]
    
    for idx, gene in enumerate(selected_genes):
        ax = axes[idx]
        means = []
        stds = []
        
        for t in time_points:
            values = df[df['Time_Hours'] == t][gene]
            means.append(values.mean())
            stds.append(values.std())
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot with error bars
        ax.errorbar(time_points, means, yerr=stds, 
                   marker='o', linewidth=2, capsize=5,
                   label=f'{gene}')
        ax.set_xlabel('Time (Hours)', fontsize=10)
        ax.set_ylabel('Normalized Expression', fontsize=10)
        ax.set_title(f'{gene}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(time_points)
    
    # Hide unused subplots
    for idx in range(n_genes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'expression_scaled_over_time.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nExpression time course plot saved to: {plot_path}")
    print(f"Plotted {n_genes} genes across {len(time_points)} time points")


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

    # Plot expression over time before feeding to Transformer
    metadata_path = os.path.join(train_dir, 'metadata.csv')
    if os.path.exists(metadata_path):
        plot_save_dir = os.path.join(train_dir, 'expression_analysis')
        plot_expression_over_time(expression_scaled, metadata_path, final_gene_symbols, 
                                 plot_save_dir, n_genes=20)

    print("Using normalized expression directly (no PCA)...")
    print(f"Expression scaled shape: {expression_scaled.shape}")
    print(f"Number of features (genes): {expression_scaled.shape[1]}")
    print(f"Number of samples: {expression_scaled.shape[0]}")

    train_dataset = ExpressionDataset(expression_scaled)

    preprocessing_info = {
        'scaler': final_scaler,
        'sample_columns': sample_columns,
        'n_features': expression_scaled.shape[1],
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

    print(f"Test data after normalization shape: {test_expression_scaled.shape}")
    print(f"Number of features: {test_expression_scaled.shape[1]}")

    test_dataset = ExpressionDataset(test_expression_scaled)

    test_preprocessing_info = preprocessing_info.copy()
    test_preprocessing_info.update({
        'test_sample_columns': available_sample_columns,
        'test_expression_scaled': test_expression_scaled
    })
    return test_dataset, test_preprocessing_info