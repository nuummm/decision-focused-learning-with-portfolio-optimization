# ========================================
# File: models/vanilla.py
# Purpose: OLSモデルの学習と予測
# ========================================

import numpy as np
from typing import Tuple

# ---------- 学習 ----------
def train_ols(X_train: np.ndarray, Y_train: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    OLS (Ordinary Least Squares) により θ を推定する。
    モデル: y_{t,j} = x_{t,j} * θ_j + noise
           （切片なし、資産 j ごとに独立に推定）

    Parameters
    ----------
    X_train : (N_train, d) ndarray
        説明変数（学習データ）。行=時点, 列=資産
    Y_train : (N_train, d) ndarray
        目的変数（学習データ）。行=時点, 列=資産リターン
    eps : float, default=1e-12
        数値安定化用の微小値（分母ゼロ防止）

    Returns
    -------
    theta_hat : (d,) ndarray
        推定された θ ベクトル（資産ごとの回帰係数）
    """
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)

    # 各資産 j ごとの Σ_t x_{t,j} y_{t,j}
    num = (X_train * Y_train).sum(axis=0)
    # 各資産 j ごとの Σ_t x_{t,j}^2 + eps
    den = (X_train ** 2).sum(axis=0) + eps

    return num / den


# ---------- 予測 ----------
def predict_yhat(X: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    """
    予測リターンを計算: \hat y^{(i)} = x^{(i)} ⊙ \hat\theta

    Parameters
    ----------
    X : (N, d) ndarray
        説明変数（各時点 i の特徴量ベクトル）
    theta_hat : (d,) ndarray
        推定済みの係数ベクトル

    Returns
    -------
    Y_hat : (N, d) ndarray
        各時点・各資産の予測リターン
    """
    X = np.asarray(X, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)
    assert X.shape[1] == theta_hat.shape[0], \
        "Xの列数とtheta_hatの長さが一致していません。"

    return X * theta_hat  # ブロードキャストによる列ごとの掛け算

