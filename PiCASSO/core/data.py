from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class Dataset:
    X: np.ndarray
    gene_names: np.ndarray
    samples: List[str]


def load_expression(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path, low_memory=False)
    if 'Gene_Symbol' not in df.columns:
        raise ValueError("expression.csv must contain 'Gene_Symbol'")
    gene_df = df[df['Gene_Symbol'] != 'time_C'].copy()
    genes = gene_df['Gene_Symbol'].astype(str).values
    samples = [c for c in gene_df.columns if c != 'Gene_Symbol']
    X = gene_df[samples].apply(pd.to_numeric, errors='coerce').fillna(0.0).T.values.astype(float)
    return X, genes, samples


def build_dataset(expression_path: str) -> Dataset:
    X, genes, samples = load_expression(expression_path)
    return Dataset(X=X, gene_names=genes, samples=samples)


def load_metadata_generic(metadata_file: str) -> Optional[pd.DataFrame]:
    try:
        meta = pd.read_csv(metadata_file, low_memory=False)
    except Exception as e:
        print(f"Failed to read metadata: {e}")
        return None
    cols = set(meta.columns.str.strip())
    if {'study_sample', 'time_mod24'}.issubset(cols):
        return meta[['study_sample', 'time_mod24']].copy()
    if {'Sample', 'Time_Hours'}.issubset(cols):
        out = pd.DataFrame({
            'study_sample': meta['Sample'].astype(str).values,
            'time_mod24': (pd.to_numeric(meta['Time_Hours'], errors='coerce').fillna(0.0) % 24).values,
        })
        return out
    print("[Warning] Unsupported metadata columns; expected ('Sample','Time_Hours') or ('study_sample','time_mod24').")
    return None


def get_circadian_gene_expressions(original_expression: np.ndarray, gene_names: np.ndarray, circadian_genes: List[str]):
    name_to_idx = {name: i for i, name in enumerate(gene_names)}
    found_genes = [g for g in circadian_genes if g in name_to_idx]
    for g in circadian_genes:
        if g not in name_to_idx:
            print(f"Warning: Gene {g} not found in data")
    if not found_genes:
        print("Error: No circadian genes found in data!")
        return None, None
    idx = [name_to_idx[g] for g in found_genes]
    gene_expressions = original_expression[:, idx]
    print(f"Found {len(found_genes)} circadian genes: {found_genes}")
    print(f"Circadian expression data shape: {gene_expressions.shape}")
    print(f"Circadian expression data type: {gene_expressions.dtype}")
    return gene_expressions, found_genes


def load_and_process_expression_data(expression_file: str, n_components: int = 5):
    print("=== Loading Expression Data ===")
    print(f"File: {expression_file}")
    df = pd.read_csv(expression_file, low_memory=False)
    print(f"Data shape: {df.shape}")
    time_row = df[df['Gene_Symbol'] == 'time_C']
    has_time = not time_row.empty
    print(f"Contains time info: {has_time}")
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    times = None
    if has_time:
        times = pd.to_numeric(time_row.iloc[0][sample_columns], errors='coerce').values.astype(float)
        print(f"Time range: {np.nanmin(times):.2f} - {np.nanmax(times):.2f} hours")
    gene_df = df[~df['Gene_Symbol'].isin(['time_C'])].copy()
    gene_names = gene_df['Gene_Symbol'].values
    print("Converting expression data to numeric (vectorized)...")
    expr_numeric = gene_df[sample_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    expression_data = expr_numeric.T.values.astype(float)
    print(f"Number of genes: {len(gene_names)}")
    print(f"Expression data shape: {expression_data.shape}")
    print(f"Expression data range: {np.nanmin(expression_data):.4f} to {np.nanmax(expression_data):.4f}")
    return {
        'original_expression': expression_data,
        'gene_names': gene_names,
        'sample_columns': sample_columns,
        'celltypes': None,
        'times': times,
        'pca_model': None,
        'scaler': None,
        'explained_variance': None,
        'n_components': n_components,
    }
