from __future__ import annotations

from typing import Optional

import numpy as np


def make_intercept_features(X: np.ndarray) -> np.ndarray:
    """Build (N,d,2) features from single feature matrix X=(N,d): [1, x]."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,d), got shape {X.shape}")
    N, d = X.shape
    ones = np.ones((N, d, 1), dtype=float)
    x_col = X[:, :, None]
    return np.concatenate([ones, x_col], axis=2)


def coerce_theta_intercept(theta: Optional[np.ndarray], d: int) -> Optional[np.ndarray]:
    """Coerce theta into shape (d,2)=[alpha,beta] when possible."""
    if theta is None:
        return None
    arr = np.asarray(theta, dtype=float)
    if arr.ndim == 2 and arr.shape == (d, 2):
        return arr
    flat = arr.reshape(-1)
    if flat.shape[0] == d:
        # Backward-compat: treat a length-d vector as the slope only (alpha=0).
        alpha = np.zeros(d, dtype=float)
        beta = flat.astype(float)
        return np.stack([alpha, beta], axis=1)
    if flat.shape[0] == d * 2:
        return flat.reshape(d, 2)
    raise ValueError(
        f"theta has incompatible shape {arr.shape}; expected (d,2) or flat length {d*2} (or length {d} for beta-only)"
    )


__all__ = ["make_intercept_features", "coerce_theta_intercept"]

