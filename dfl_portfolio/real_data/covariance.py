"""共分散推定ユーティリティ
================================

実データ実験で使用する共分散行列 (V) を推定するための関数群。
以下の手法をサポートしており、どれも ``estimate_shrinkage_covariances`` から
選択・利用できる。

1. ``diag``  
   標本共分散を対角方向にシュリンク (λ) し、`eps` を足して正定値化。
2. ``oas``  
   scikit-learn の Oracle Approximating Shrinkage 推定器。
3. ``robust_lw``  
   リターンを Huber 化してから Ledoit-Wolf を適用するロバスト共分散。
4. ``mini_factor``  
   PCA による低ランク因子近似 + 残差の対角シュリンク。

使い方の例::

    covs, times, stats = estimate_shrinkage_covariances(
        returns_df,
        window=10,
        method="robust_lw",
        robust_huber_k=1.5,
    )

戻り値 ``covs`` はローリング共分散行列のリスト、``times`` は対応する
タイムスタンプ、``stats`` は ``CovarianceStats`` (最小/最大固有値)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional dependency
    from sklearn.covariance import LedoitWolf, OAS
except ImportError:  # pragma: no cover
    LedoitWolf = None
    OAS = None


ShrinkMethod = Literal[
    "diag",
    "oas",
    "robust_lw",
    "mini_factor",
]


@dataclass
class CovarianceStats:
    eigen_min: float
    eigen_max: float
    # OAS × EWMA で推定したときの shrinkage 係数 δ_t。
    # それ以外の手法では None のまま。
    oas_delta: float | None = None


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


def _oas_cov(window_vals: np.ndarray, eps: float) -> np.ndarray:
    if OAS is None:
        raise RuntimeError("sklearn がインストールされていないため OAS を使用できません。")
    oas = OAS(store_precision=False, assume_centered=False)
    oas.fit(window_vals)
    cov = oas.covariance_
    cov = 0.5 * (cov + cov.T)
    cov += eps * np.eye(cov.shape[0])
    return cov


def _apply_huber(window_vals: np.ndarray, k: float) -> np.ndarray:
    median = np.median(window_vals, axis=0)
    mad = np.median(np.abs(window_vals - median), axis=0)
    mad = np.where(mad < 1e-6, 1e-6, mad)
    normalized = (window_vals - median) / mad
    clipped = np.clip(normalized, -k, k)
    return clipped * mad + median


def _robust_ledoit_wolf_cov(window_vals: np.ndarray, huber_k: float, eps: float) -> np.ndarray:
    huber_vals = _apply_huber(window_vals, huber_k)
    return _ledoit_wolf_cov(huber_vals, eps)


def _mini_factor_cov(
    window_vals: np.ndarray,
    rank: int,
    residual_shrinkage: float,
    eps: float,
) -> np.ndarray:
    d = window_vals.shape[1]
    rank = max(1, min(rank, d))
    centered = window_vals - np.mean(window_vals, axis=0)
    sample_cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(sample_cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    B = eigvecs[:, :rank]
    Sigma_f = np.diag(np.maximum(eigvals[:rank], eps))
    factor_cov = B @ Sigma_f @ B.T
    factor_scores = centered @ B
    residuals = centered - factor_scores @ B.T
    resid_cov = np.cov(residuals, rowvar=False)
    resid_cov = _shrink_to_diag(resid_cov, shrinkage=residual_shrinkage, eps=eps)
    cov = factor_cov + resid_cov
    cov = 0.5 * (cov + cov.T)
    cov += eps * np.eye(d)
    return cov


def estimate_shrinkage_covariances(
    returns: pd.DataFrame,
    window: int,
    *,
    method: ShrinkMethod = "diag",
    shrinkage: float = 0.94,
    eps: float = 1e-6,
    robust_huber_k: float = 1.5,
    factor_rank: int = 1,
    factor_shrinkage: float = 0.5,
    ewma_alpha: float | None = None,
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
        oas_delta_value: float | None = None
        if method == "oas":
            # OAS × EWMA: 0<α<1 のときだけ EWMA 共分散 S_t^{(α)} に OAS を適用。
            # α>=1 や α<=0 のときは従来どおり等重み OAS にフォールバックする。
            if ewma_alpha is not None and 0.0 < float(ewma_alpha) < 1.0:
                alpha = float(ewma_alpha)
                w = window_vals.shape[0]
                d = sample_cov.shape[0]
                # 中心化
                centered = window_vals - np.mean(window_vals, axis=0)
                # EWMA 重み ω_k(α) = (1-α) α^k / (1-α^w), k=0..w-1, k=0 が直近
                ks = np.arange(w - 1, -1, -1, dtype=float)  # 行 0 が最も古く, w-1 が最新
                raw = (1.0 - alpha) * (alpha ** ks)
                denom = 1.0 - alpha ** w
                omega = raw / denom
                # 有効サンプルサイズ n_eff
                n_eff = 1.0 / float(np.sum(omega**2))
                # 時間重み付き共分散 S_t^{(α)}
                S = np.zeros((d, d), dtype=float)
                for i_row in range(w):
                    r = centered[i_row].reshape(-1, 1)
                    S += float(omega[i_row]) * (r @ r.T)
                S = 0.5 * (S + S.T)
                # OAS shrinkage
                tr_S = float(np.trace(S))
                tr_S2 = float(np.sum(S * S))
                if tr_S2 <= 0.0:
                    cov = S + eps * np.eye(d)
                else:
                    num = (1.0 - 2.0 / d) * tr_S2 + tr_S**2
                    den = (n_eff + 1.0 - 2.0 / d) * (tr_S2 - tr_S**2 / d)
                    if den <= 0.0:
                        delta_oas = 0.0
                    else:
                        delta_oas = float(num / den)
                        delta_oas = float(np.clip(delta_oas, 0.0, 1.0))
                    F = (tr_S / d) * np.eye(d, dtype=float)
                    cov = (1.0 - delta_oas) * S + delta_oas * F
                    oas_delta_value = float(delta_oas)
                    cov = 0.5 * (cov + cov.T)
                    cov += eps * np.eye(d)
            else:
                # α が指定されていないか 1 のときは、sklearn.OAS による標準的な OAS 共分散。
                cov = _oas_cov(window_vals, eps)
        elif method == "diag":
            # diag × EWMA: OAS と同様に、0<α<1 のときは
            # 時間減衰付き共分散 S_t^{(α)} を計算し、それに対して
            # 対角シュリンクを適用する。
            if ewma_alpha is not None and 0.0 < float(ewma_alpha) < 1.0:
                alpha = float(ewma_alpha)
                w = window_vals.shape[0]
                centered = window_vals - np.mean(window_vals, axis=0)
                ks = np.arange(w - 1, -1, -1, dtype=float)
                raw = (1.0 - alpha) * (alpha ** ks)
                denom = 1.0 - alpha ** w
                omega = raw / denom
                S = np.zeros((d, d), dtype=float)
                for i_row in range(w):
                    r = centered[i_row].reshape(-1, 1)
                    S += float(omega[i_row]) * (r @ r.T)
                S = 0.5 * (S + S.T)
                cov = _shrink_to_diag(S, shrinkage=shrinkage, eps=eps)
            else:
                cov = _shrink_to_diag(sample_cov, shrinkage=shrinkage, eps=eps)
        elif method == "robust_lw":
            cov = _robust_ledoit_wolf_cov(window_vals, huber_k=robust_huber_k, eps=eps)
        elif method == "mini_factor":
            cov = _mini_factor_cov(
                window_vals,
                rank=factor_rank,
                residual_shrinkage=factor_shrinkage,
                eps=eps,
            )
        else:
            raise ValueError(f"unknown covariance shrinkage method: {method!r}")
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
        stats.append(
            CovarianceStats(
                eigen_min=min_eig,
                eigen_max=max_eig,
                oas_delta=oas_delta_value,
            )
        )

    if not covs:
        raise ValueError("共分散を推定できるウィンドウがありません。")

    return covs, times, stats


__all__ = [
    "CovarianceStats",
    "estimate_shrinkage_covariances",
]
