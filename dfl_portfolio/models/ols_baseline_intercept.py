from __future__ import annotations

from typing import Optional

import numpy as np

from dfl_portfolio.models.ols_intercept import train_ols_intercept


def fit_ols_baseline_intercept(
    X,
    Y,
    Vhats,
    idx,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    delta: float = 1.0,
    theta_init: Optional[np.ndarray] = None,
    tee: bool = False,
):
    """
    Runner-compatible OLS baseline with intercept.

    This mirrors ``dfl_portfolio.models.ols_baseline.fit_ols_baseline`` but trains
    per-asset intercept OLS:
        y = alpha + beta * x
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, d = X.shape

    if start_index is None or end_index is None:
        X_tr, Y_tr = X, Y
        used_idx = list(range(N))
    else:
        s = max(0, int(start_index))
        e = min(N - 1, int(end_index))
        if s > e:
            raise ValueError("fit_ols_baseline_intercept: invalid train range")
        X_tr = X[s : e + 1]
        Y_tr = Y[s : e + 1]
        used_idx = list(range(s, e + 1))

    theta_hat = train_ols_intercept(X_tr, Y_tr)

    Z = np.empty((0, d))
    MU = np.empty((0,))
    LAM = np.empty((0, d))

    if tee and used_idx:
        print(
            f"[OLS-Intercept] train range=[{used_idx[0]}, {used_idx[-1]}], theta shape={theta_hat.shape}"
        )

    meta = {
        "solver": "ols_intercept",
        "termination_condition": None,
        "termination_condition_str": None,
        "status": "completed",
        "status_str": "completed",
        "solver_time": None,
        "message": None,
    }

    return theta_hat, Z, MU, LAM, used_idx, meta


__all__ = ["fit_ols_baseline_intercept"]

