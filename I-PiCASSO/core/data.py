from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Dataset:
    expression: pd.DataFrame
    samples: List[str]
    gene_names: np.ndarray
    X: np.ndarray  # samples x genes (float)
    y_hours: np.ndarray  # length = samples
    meta_joined: pd.DataFrame  # with study_sample + time_mod24 aligned to samples
    idx_valid: np.ndarray  # indices of valid samples in original sample list


def read_expression(path: str) -> tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, low_memory=False)
    if 'Gene_Symbol' not in df.columns:
        raise ValueError("expression.csv must contain a 'Gene_Symbol' column")
    sample_cols = [c for c in df.columns if c != 'Gene_Symbol']
    return df, sample_cols


def load_metadata(path: str) -> pd.DataFrame:
    from .utils import load_metadata_generic
    meta = load_metadata_generic(path)
    if meta is None:
        raise ValueError("Unsupported metadata: need Sample/Time_Hours or study_sample/time_mod24, or Sample/polar_coord_phi")
    return meta


def align_samples(sample_cols: List[str], meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    meta = meta.copy()
    meta['study_sample'] = meta['study_sample'].astype(str)
    order = pd.DataFrame({'study_sample': sample_cols})
    joined = order.merge(meta, on='study_sample', how='left')
    mask_valid = ~joined['time_mod24'].isna()
    dropped = joined.loc[~mask_valid, 'study_sample'].astype(str).tolist()
    joined_valid = joined.loc[mask_valid].reset_index(drop=True)
    idx_valid = np.where(mask_valid.to_numpy(dtype=bool))[0]
    time_hours = joined_valid['time_mod24'].to_numpy(dtype=float)
    y = np.remainder(time_hours, 24.0)
    return joined_valid, y, idx_valid, dropped


def build_dataset(expression_path: str, metadata_path: str) -> Dataset:
    df, sample_cols = read_expression(expression_path)
    meta = load_metadata(metadata_path)
    meta_joined, y_hours, idx_valid, dropped = align_samples(sample_cols, meta)

    gene_df = df[df['Gene_Symbol'] != 'time_C'].copy()
    gene_names = gene_df['Gene_Symbol'].astype(str).to_numpy()
    expr_full = gene_df[sample_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).T.values.astype(float)
    X = expr_full[idx_valid, :]
    return Dataset(
        expression=df,
        samples=sample_cols,
        gene_names=gene_names,
        X=X,
        y_hours=y_hours,
        meta_joined=meta_joined,
        idx_valid=idx_valid,
    )
