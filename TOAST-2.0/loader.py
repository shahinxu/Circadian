#!/usr/bin/env python3
"""
Data loading and preprocessing for TOAST-2.0
Includes pathway loading and expression data processing
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional


# ==================== Pathway Loader ====================

def load_pathway_dataset(pathway_csv: str) -> pd.DataFrame:
    """Load pathway dataset from CSV"""
    return pd.read_csv(pathway_csv)


def build_pathway_map(
    pathway_df: pd.DataFrame,
    gene_symbols: List[str],
    min_pathway_size: int = 100,
    max_pathway_size: int = 500
) -> Tuple[List[List[int]], List[str], Dict[str, int]]:
    """Build pathway to gene index mapping"""
    gene_to_idx = {gene.upper(): idx for idx, gene in enumerate(gene_symbols)}
    
    pathway_groups = pathway_df.groupby('pathway_name')
    pathway_indices = []
    pathway_names = []
    
    for pathway_name, group in pathway_groups:
        genes_in_pathway = group['gene_symbol'].str.upper().unique()
        indices = [gene_to_idx[g] for g in genes_in_pathway if g in gene_to_idx]
        
        if min_pathway_size <= len(indices) <= max_pathway_size:
            pathway_indices.append(indices)
            pathway_names.append(pathway_name)
    
    print(f"Built pathway map: {len(pathway_names)} pathways covering {len(gene_symbols)} genes")
    if pathway_indices:
        print(f"  Pathway size range: [{min([len(p) for p in pathway_indices])}, {max([len(p) for p in pathway_indices])}]")
    
    return pathway_indices, pathway_names, gene_to_idx


def get_pathway_statistics(pathway_indices: List[List[int]], pathway_names: List[str]) -> pd.DataFrame:
    """Get statistics about pathways"""
    stats = pd.DataFrame({
        'pathway_name': pathway_names,
        'num_genes': [len(idx) for idx in pathway_indices]
    })
    return stats.sort_values('num_genes', ascending=False)


# ==================== Expression Dataset ====================

class ExpressionDataset(Dataset):
    """PyTorch dataset for gene expression data"""
    def __init__(self, expressions, covariates=None, tissue_indices=None):
        self.expressions = torch.FloatTensor(expressions)
        self.covariates = torch.FloatTensor(covariates) if covariates is not None else None
        self.tissue_indices = torch.LongTensor(tissue_indices) if tissue_indices is not None else None

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        item = {'expression': self.expressions[idx]}
        if self.covariates is not None:
            item['covariates'] = self.covariates[idx]
        if self.tissue_indices is not None:
            item['tissue_idx'] = self.tissue_indices[idx]
        return item


# ==================== Data Processing ====================

def blunt_percentile(data, percent=0.975):
    """Clip data to percentile range"""
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
        metadata_file=None,
        blunt_percent=0.975,
        pathway_csv=None
    ):
    """
    Load and preprocess training data
    
    Args:
        train_file: path to expression.csv
        metadata_file: path to metadata.csv (optional, for tissue info)
        blunt_percent: percentile for outlier clipping
        pathway_csv: path to pathway data (optional)
    
    Returns:
        train_dataset: ExpressionDataset
        preprocessing_info: dict with preprocessing parameters
    """
    print("=== Loading training data ===")
    df = pd.read_csv(train_file, low_memory=False)
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    gene_df = df[~df['Gene_Symbol'].isin(['time_C'])].copy()
    
    print(f"Using all {len(gene_df)} genes")
    
    initial_gene_symbols = gene_df['Gene_Symbol'].values
    sample_df = gene_df[sample_columns].copy()
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')
    
    if sample_df.isna().any().any():
        n_nan = int(sample_df.isna().sum().sum())
        print(f"Warning: {n_nan} non-numeric values found; coercing to NaN and imputing 0.")
        sample_df = sample_df.fillna(0)
    
    expression_data = sample_df.values.T
    print(f"Initial data shape: {expression_data.shape}")

    expression_data = blunt_percentile(expression_data, percent=blunt_percent)
    print(f"Shape after blunt_percentile: {expression_data.shape}")

    final_gene_symbols = initial_gene_symbols

    if expression_data.shape[1] == 0:
        raise ValueError("All genes were removed after filtering.")

    final_scaler = StandardScaler()
    expression_scaled = final_scaler.fit_transform(expression_data)
    print(f"Scaled expression data shape: {expression_scaled.shape}")

    # Load tissue information from metadata if provided
    tissue_indices = None
    tissue_to_idx = None
    if metadata_file and os.path.exists(metadata_file):
        print(f"Loading tissue information from {metadata_file}")
        meta_df = pd.read_csv(metadata_file, low_memory=False)
        if 'Sample' in meta_df.columns and 'Tissue' in meta_df.columns:
            unique_tissues = sorted(meta_df['Tissue'].unique())
            tissue_to_idx = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
            print(f"Found {len(tissue_to_idx)} unique tissues: {list(tissue_to_idx.keys())}")
            
            sample_to_tissue = dict(zip(meta_df['Sample'].astype(str), meta_df['Tissue'].astype(str)))
            tissue_indices = [tissue_to_idx.get(sample_to_tissue.get(col, ''), 0) for col in sample_columns]
            tissue_indices = np.array(tissue_indices)
            print(f"Tissue indices shape: {tissue_indices.shape}")
        else:
            print("Warning: metadata file missing 'Sample' or 'Tissue' columns")
    else:
        print("No metadata file provided, all samples will use tissue_idx=0")

    train_dataset = ExpressionDataset(expression_scaled, covariates=None, tissue_indices=tissue_indices)

    # Load pathway information if provided
    pathway_info = None
    if pathway_csv and os.path.exists(pathway_csv):
        print(f"Loading pathway information from {pathway_csv}")
        
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
        print("No pathway information provided")

    preprocessing_info = {
        'scaler': final_scaler,
        'sample_columns': sample_columns,
        'final_gene_symbols': final_gene_symbols,
        'blunt_percent': blunt_percent,
        'input_dim': expression_scaled.shape[1],
        'pathway_info': pathway_info,
        'tissue_to_idx': tissue_to_idx,
        'num_tissues': len(tissue_to_idx) if tissue_to_idx else 1
    }
    return train_dataset, preprocessing_info


def load_and_preprocess_test_data(test_file, preprocessing_info):
    """
    Load and preprocess test data using training preprocessing parameters
    
    Args:
        test_file: path to test expression.csv
        preprocessing_info: preprocessing info from training
    
    Returns:
        test_dataset: ExpressionDataset
        test_preprocessing_info: updated preprocessing info
    """
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
