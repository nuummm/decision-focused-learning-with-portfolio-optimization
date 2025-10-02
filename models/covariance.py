# path: models/covariance.py
# Purpose: ローリング共分散 V_hat^(i) の推定

import numpy as np
from typing import List, Optional, Tuple

def estimate_epscov_rolling(
    Y,
    X,
    theta0,
    tau,
    res: int = 20,
    include_current: bool = False,
    eps: float = 1e-8
):
    """
    Y, X: (N, d)
    想定モデル: Y = diag(X) * theta0 + tau * epsilon
    ここで epsilon ~ N(0, Sigma)
    戻り値:
      Vhats: List[np.ndarray]  各時点 i に対応する (d, d) 共分散推定
      idx:   List[int]         各 Vhat が対応する時点 i（0-index）
    """
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    N, d = Y.shape

    # ε を復元
    E = (Y - X * theta0) / float(tau)

    if res > N:
        return [], []

    Vhats: List[np.ndarray] = []
    idx: List[int] = []

    for i in range(N):
        end = i + 1 if include_current else i
        start = end - int(res)
        if start < 0:
            continue
        W = E[start:end]
        if W.shape[0] < 2:   # ddof=1 のときウィンドウ長が1だと不正
            continue
        V_hat = np.cov(W, rowvar=False, ddof=1)
        # 数値対称化 + εI（固有値の下駄履かせ）
        V_hat = 0.5 * (V_hat + V_hat.T) + eps * np.eye(d)
        Vhats.append(V_hat)
        idx.append(i)

    return Vhats, idx