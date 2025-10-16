from __future__ import annotations
import numpy as np

__all__ = ["trim_weights"]

def trim_weights(weights, target_dim: int):
    """Trim or broadcast weights list/array to target_dim; None passes through."""
    if weights is None:
        return None
    try:
        arr = np.array(weights, dtype=float).flatten()
    except Exception:
        return weights
    if arr.size < target_dim:
        # simple repeat padding
        reps = int(np.ceil(target_dim / arr.size))
        arr = np.tile(arr, reps)
    return arr[:target_dim].tolist()
