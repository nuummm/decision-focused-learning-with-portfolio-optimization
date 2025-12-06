from __future__ import annotations

from typing import Tuple

import numpy as np


def train_ols_multi(
    X_feat: np.ndarray,
    Y: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    資産ごとに K 次元特徴を持つ重回帰版 OLS を学習する。

    モデル:
        y_{t,j} = x_{t,j,:}^T theta_{j,:} + noise

    Parameters
    ----------
    X_feat : (N, d, K) ndarray
        説明変数。N=サンプル数, d=資産数, K=特徴量数。
    Y : (N, d) ndarray
        目的変数（リターン）。
    eps : float, default 1e-12
        数値安定化用の微小値（(X^T X) の対角に足す）。

    Returns
    -------
    theta : (d, K) ndarray
        各資産 j ごとの回帰係数ベクトル。
    """
    X_feat = np.asarray(X_feat, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X_feat.ndim != 3:
        raise ValueError(f"X_feat must be 3D (N,d,K), got shape {X_feat.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N,d), got shape {Y.shape}")

    N, d, K = X_feat.shape
    if Y.shape != (N, d):
        raise ValueError(f"Y shape {Y.shape} is incompatible with X_feat shape {X_feat.shape}")

    theta = np.zeros((d, K), dtype=float)
    eye_K = np.eye(K, dtype=float)

    for j in range(d):
        Xj = X_feat[:, j, :]  # (N, K)
        yj = Y[:, j]          # (N,)
        XtX = Xj.T @ Xj + eps * eye_K
        Xty = Xj.T @ yj
        theta[j, :] = np.linalg.solve(XtX, Xty)

    return theta


def predict_yhat_multi(
    X_feat: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    重回帰版の予測リターンを計算する。

    ŷ_{t,j} = x_{t,j,:}^T theta_{j,:}

    Parameters
    ----------
    X_feat : (N, d, K) ndarray
        説明変数。
    theta : (d, K) ndarray
        推定済み係数。

    Returns
    -------
    Y_hat : (N, d) ndarray
        各時点・各資産の予測リターン。
    """
    X_feat = np.asarray(X_feat, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if X_feat.ndim != 3:
        raise ValueError(f"X_feat must be 3D (N,d,K), got shape {X_feat.shape}")
    if theta.ndim != 2:
        raise ValueError(f"theta must be 2D (d,K), got shape {theta.shape}")

    N, d, K = X_feat.shape
    if theta.shape != (d, K):
        raise ValueError(f"theta shape {theta.shape} incompatible with X_feat shape {X_feat.shape}")

    # (N,d,K) ・ (d,K) -> (N,d)
    yhat = np.einsum("tdk,dk->td", X_feat, theta)
    return yhat


__all__ = ["train_ols_multi", "predict_yhat_multi"]

