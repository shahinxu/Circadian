from __future__ import annotations
import numpy as np
import pandas as pd
from collections import namedtuple
from typing import Tuple

AlignResult = namedtuple('AlignResult', ['corr', 'r2', 'slope', 'shift', 'orientation'])


def to_24(x) -> np.ndarray:
    v = x.values if isinstance(x, pd.Series) else np.asarray(x)
    v = v.astype(float)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    den = vmax - vmin
    if not np.isfinite(den) or den < 1e-8:
        return np.zeros_like(v, dtype=float)
    return ((v - vmin) / den * 24.0) % 24.0


def _linfit_no_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = x.astype(float)
    y = y.astype(float)
    if x.size < 2:
        return np.nan, np.nan, np.nan
    xs, ys = np.std(x), np.std(y)
    r = float(np.corrcoef(x, y)[0, 1]) if (xs >= 1e-12 and ys >= 1e-12) else np.nan
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return r, np.nan, np.nan
    slope = float(np.dot(x, y) / denom)
    y_hat = slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r, r2, slope


def best_shift(x_hours, y_hours, step: float = 0.05) -> AlignResult:
    x0 = x_hours.values if isinstance(x_hours, pd.Series) else np.asarray(x_hours)
    y = y_hours.values if isinstance(y_hours, pd.Series) else np.asarray(y_hours)
    x0 = x0.astype(float)
    y = y.astype(float)
    if x0.size == 0 or y.size == 0 or x0.size != y.size:
        return AlignResult(np.nan, np.nan, np.nan, np.nan, 'normal')
    best = None
    best_score = -np.inf
    shifts = np.arange(0.0, 24.0, max(step, 1e-6))
    for orientation in ('normal', 'inverted'):
        x_use = x0 if orientation == 'normal' else (24.0 - x0) % 24.0
        for s in shifts:
            xs = (x_use + s) % 24.0
            r, _, _ = _linfit_no_intercept(xs, y)
            score = -np.inf if not np.isfinite(r) else r
            if score > best_score:
                best_score = score
                best = (orientation, s)
    if best is None:
        return AlignResult(np.nan, np.nan, np.nan, np.nan, 'normal')
    orientation, shift = best
    x_use = x0 if orientation == 'normal' else (24.0 - x0) % 24.0
    xs = (x_use + shift) % 24.0
    r, r2, slope = _linfit_no_intercept(xs, y)
    return AlignResult(r, r2, slope, float(shift), orientation)
