from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple


def scale(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def eigengenes(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA, np.ndarray]:
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    components_norm = StandardScaler().fit_transform(components)
    return components_norm, pca, explained_variance
