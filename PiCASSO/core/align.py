from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import pearsonr


def to_24(ranks) -> np.ndarray:
    r = ranks.values if isinstance(ranks, pd.Series) else np.asarray(ranks)
    r = r.astype(float)
    rmin, rmax = np.nanmin(r), np.nanmax(r)
    span = max(rmax - rmin, 1e-9)
    return ((r - rmin) / span * 24.0) % 24.0


@dataclass
class ShiftFit:
    slope: float
    shift: float
    r2: float
    corr: float
    n: int
    orientation: str


def best_shift(x_base: np.ndarray, y: np.ndarray, step: float = 0.05) -> ShiftFit:
    def eval_one(x_in: np.ndarray, orient: str) -> ShiftFit:
        best = ShiftFit(np.nan, np.nan, -np.inf, -np.inf, len(y), orient)
        if len(x_in) == 0 or len(y) == 0 or float(np.var(y)) <= 1e-12:
            return best
        for s in np.arange(0.0, 24.0, step):
            x = (x_in + s) % 24.0
            try:
                corr = float(pearsonr(x, y)[0])
            except Exception:
                corr = -np.inf
            if not np.isfinite(corr):
                continue
            if corr > best.corr:
                xx = float(np.dot(x, x))
                if xx <= 1e-12:
                    continue
                a = float(np.dot(x, y) / xx)
                if a <= 0:
                    continue
                y_hat = a * x
                sst0 = float(np.sum(y ** 2))
                sse = float(np.sum((y - y_hat) ** 2))
                r2 = 1.0 - sse / sst0 if sst0 > 1e-12 else -np.inf
                best = ShiftFit(a, s, r2, corr, len(y), orient)
        return best

    normal = eval_one(x_base, 'normal')
    flipped = eval_one((24.0 - x_base) % 24.0, 'flipped')
    if normal.corr > flipped.corr:
        return normal
    if flipped.corr > normal.corr:
        return flipped
    return normal if normal.r2 >= flipped.r2 else flipped
