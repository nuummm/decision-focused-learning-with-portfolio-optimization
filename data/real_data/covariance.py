from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional dependency
    from sklearn.covariance import LedoitWolf
except ImportError:  # pragma: no cover
    LedoitWolf = None


ShrinkMethod = Literal["diag", "ledoit_wolf"]


@dataclass
class CovarianceStats:
    eigen_min: float
    eigen_max: float


def _shrink_to_diag(sample_cov: np.ndarray, shrinkage: float, eps: float) -> np.ndarray:
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    diag = np.diag(np.diag(sample_cov))
    shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * diag
    shrunk = 0.5 * (shrunk + shrunk.T)
    shrunk += eps * np.eye(sample_cov.shape[0])
    return shrunk


def _ledoit_wolf_cov(window_vals: np.ndarray, eps: float) -> np.ndarray:
    if LedoitWolf is None:
        raise RuntimeError("sklearn がインストールされていないため LedoitWolf を使用できません。")
    lw = LedoitWolf(store_precision=False, assume_centered=False)
    lw.fit(window_vals)
    cov = lw.covariance_
    cov = 0.5 * (cov + cov.T)
    cov += eps * np.eye(cov.shape[0])
    return cov


def estimate_shrinkage_covariances(
    returns: pd.DataFrame,
    window: int,
    *,
    method: ShrinkMethod = "diag",
    shrinkage: float = 0.94,
    eps: float = 1e-6,
) -> Tuple[List[np.ndarray], List[pd.Timestamp], List[CovarianceStats]]:
    if window < 2:
        raise ValueError("window は 2 以上を指定してください。")

    covs: List[np.ndarray] = []
    times: List[pd.Timestamp] = []
    stats: List[CovarianceStats] = []

    values = returns.to_numpy(dtype=float)
    idx_list: Sequence[pd.Timestamp] = list(returns.index)
    d = values.shape[1]

    for idx in range(window, len(values) + 1):
        window_vals = values[idx - window : idx]
        if np.isnan(window_vals).any():
            continue
        sample_cov = np.cov(window_vals, rowvar=False)
        if method == "ledoit_wolf":
            cov = _ledoit_wolf_cov(window_vals, eps)
        else:
            cov = _shrink_to_diag(sample_cov, shrinkage=shrinkage, eps=eps)
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = float(eigvals.min())
        max_eig = float(eigvals.max())
        if min_eig <= 0:
            cov += (abs(min_eig) + eps) * np.eye(d)
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = float(eigvals.min())
            max_eig = float(eigvals.max())
        covs.append(cov)
        times.append(idx_list[idx - 1])
        stats.append(CovarianceStats(eigen_min=min_eig, eigen_max=max_eig))

    if not covs:
        raise ValueError("共分散を推定できるウィンドウがありません。")

    return covs, times, stats


__all__ = [
    "CovarianceStats",
    "estimate_shrinkage_covariances",
]
