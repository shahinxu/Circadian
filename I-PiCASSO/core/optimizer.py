from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class MultiScaleOptimizer:
    smoothness_factor: float = 0.7
    local_variation_factor: float = 0.3
    window_size: int = 10
    max_iterations_ratio: int = 50
    variation_tolerance_ratio: float = 0.5
    method: str = 'greedy'
    device: str = 'cpu'

    def optimize(self, eigengenes: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(eigengenes, dtype=float)
        n, d = X.shape
        if n == 0:
            return np.array([], dtype=int)
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd < 1e-9] = 1.0
        Z = (X - mu) / sd
        if d >= 2:
            x, y = Z[:, 0], Z[:, 1]
            theta = np.arctan2(y, x) % (2*np.pi)
            order = np.argsort(theta)
        else:
            order = np.argsort(Z[:, 0])
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        return ranks.reshape(-1, 1)
