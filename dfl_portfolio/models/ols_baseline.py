# ~/VScode/GraduationResearch/DFL_Portfolio_Optimization2/models/ols_baseline.py
from __future__ import annotations

from typing import Optional, Sequence, List, Tuple

import numpy as np

from dfl_portfolio.models.ols import train_ols

def fit_ols_baseline(
    X, Y, Vhats, idx,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    delta: float = 1.0,
    theta_init: Optional[np.ndarray] = None,
    tee: bool = False,
):
    """
    統一ランナー用の“ダミーTrainer”。
    - Pyomoは使わず、学習区間[start_index, end_index]のスライスでOLSを学習。
    - ランナーの期待する返り値(θ, Z, MU, LAM, used_idx) 形式に合わせるため、
      Z/MU/LAMは空を返す（ランナー側ではθだけ使う）。
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, d = X.shape

    # 学習範囲（安全にクリップ）
    if start_index is None or end_index is None:
        # 全期間でOLS（フォールバック）
        X_tr, Y_tr = X, Y
        used_idx = list(range(N))
    else:
        s = max(0, int(start_index))
        e = min(N - 1, int(end_index))
        if s > e:
            raise ValueError("fit_ols_baseline: invalid train range")
        X_tr = X[s:e+1]
        Y_tr = Y[s:e+1]
        used_idx = list(range(s, e+1))

    theta_hat = train_ols(X_tr, Y_tr)

    # ランナー互換のダミー出力
    Z  = np.empty((0, d))
    MU = np.empty((0,))
    LAM= np.empty((0, d))

    if tee:
        print(f"[OLS] train range = [{used_idx[0]}, {used_idx[-1]}], theta shape={theta_hat.shape}")

    meta = {
        "solver": "ols",
        "termination_condition": None,
        "termination_condition_str": None,
        "status": "completed",
        "status_str": "completed",
        "solver_time": None,
        "message": None,
    }

    return theta_hat, Z, MU, LAM, used_idx, meta
