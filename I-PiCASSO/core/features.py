from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def compute_pca_components(X: np.ndarray, n_components: int) -> np.ndarray:
    """Compute standardized PCA components using sklearn with safe guards."""
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("输入表达矩阵包含 NaN 或 Inf，无法执行标准 PCA")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_samples, n_genes = X_scaled.shape
    k = max(1, min(n_components, n_samples, n_genes))

    pca = PCA(n_components=k, svd_solver='auto')
    scores = pca.fit_transform(X_scaled)

    if k < n_components:
        pad_width = n_components - k
        scores = np.pad(scores, ((0, 0), (0, pad_width)), mode='constant')

    mean = scores.mean(axis=0, keepdims=True)
    std = scores.std(axis=0, keepdims=True) + 1e-6
    normalized = (scores - mean) / std

    if np.isnan(normalized).any() or np.isinf(normalized).any():
        raise ValueError("PCA 结果出现 NaN/Inf，请检查输入数据或标准化流程")

    return normalized.astype(np.float32)


def periodic_scores(X: np.ndarray, y_hours: np.ndarray) -> np.ndarray:
    n_samples, n_genes = X.shape
    t = y_hours.astype(float)
    omega = 2.0 * np.pi / 24.0
    # [1, sin, cos]
    H = np.stack([np.ones(n_samples), np.sin(omega * t), np.cos(omega * t)], axis=1)
    HtH = H.T @ H
    HtH_inv = np.linalg.pinv(HtH)
    scores = np.zeros(n_genes, dtype=float)
    for g in range(n_genes):
        y = X[:, g]
        if np.nanstd(y) < 1e-9:
            scores[g] = 0.0
            continue
        beta = HtH_inv @ (H.T @ y)
        y_hat = H @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        scores[g] = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return scores


def eigengenes(X_sel: np.ndarray, n_components: int = 5) -> Tuple[np.ndarray, PCA, np.ndarray]:
    scaler = StandardScaler()
    Z = scaler.fit_transform(X_sel)
    n, d = Z.shape
    k = max(1, min(n_components, n, d))
    pca = PCA(n_components=k)
    comps = pca.fit_transform(Z)
    comps = StandardScaler().fit_transform(comps)
    return comps, pca, pca.explained_variance_ratio_
