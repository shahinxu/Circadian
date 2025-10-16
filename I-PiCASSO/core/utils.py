from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional


def load_metadata_generic(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'Sample' not in df.columns and 'study_sample' in df.columns:
        df = df.rename(columns={'study_sample': 'Sample'})
    if 'Sample' in df.columns and 'Time_Hours' in df.columns:
        return pd.DataFrame({
            'study_sample': df['Sample'].astype(str),
            'time_mod24': pd.to_numeric(df['Time_Hours'], errors='coerce') % 24.0
        })
    if 'Sample' in df.columns and 'polar_coord_phi' in df.columns:
        phi = pd.to_numeric(df['polar_coord_phi'], errors='coerce')
        hours = (phi % (2*np.pi)) / (2*np.pi) * 24.0
        return pd.DataFrame({
            'study_sample': df['Sample'].astype(str),
            'time_mod24': hours
        })
    if 'study_sample' in df.columns and 'time_mod24' in df.columns:
        return pd.DataFrame({
            'study_sample': df['study_sample'].astype(str),
            'time_mod24': pd.to_numeric(df['time_mod24'], errors='coerce') % 24.0
        })
    return None


def trim_weights(weights, n_cols: int):
    if weights is None:
        return np.ones(n_cols, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.size != n_cols:
        if w.size == 0:
            return np.ones(n_cols, dtype=float)
        reps = int(np.ceil(n_cols / w.size))
        w = np.tile(w, reps)[:n_cols]
    return w
