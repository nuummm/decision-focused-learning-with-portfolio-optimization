from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dfl_portfolio.models._intercept_utils import coerce_theta_intercept, make_intercept_features
from dfl_portfolio.models.ipo_closed_form_multi import fit_ipo_closed_form_multi
from dfl_portfolio.models.ols_baseline_intercept import fit_ols_baseline_intercept


def fit_ipo_closed_form_intercept(
    X: np.ndarray,
    Y: np.ndarray,
    Vhats: Sequence[np.ndarray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    delta: float = 1.0,
    mode: str = "budget",
    psd_eps: float = 1e-12,
    ridge_theta: float = 1e-10,
    tee: bool = False,
    theta_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
    """
    IPO closed-form trainer with intercept using the multi-feature implementation.

    Model:
        y_hat[t,j] = alpha_j + beta_j * x[t,j]
    implemented by stacking features [1, x] and solving for theta[j,:]=[alpha,beta].
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,d), got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N,d), got shape {Y.shape}")
    if X.shape != Y.shape:
        raise ValueError(f"X,Y shape mismatch: {X.shape} vs {Y.shape}")

    # Match registry behavior: IPO closed-form is undefined at delta=0; fall back to OLS baseline.
    if float(delta) == 0.0:
        theta_hat, Z, MU, LAM, used_idx, meta = fit_ols_baseline_intercept(
            X,
            Y,
            Vhats,
            idx,
            start_index=start_index,
            end_index=end_index,
            delta=float(delta),
            theta_init=theta_init,
            tee=tee,
        )
        meta = dict(meta)
        meta["solver"] = "ols_baseline_intercept"
        meta["message"] = "delta=0: used intercept OLS baseline instead of IPO closed-form."
        return theta_hat, Z, MU, LAM, used_idx, meta

    X_feat = make_intercept_features(X)  # (N,d,2)
    d = int(X.shape[1])
    theta_init_mat = coerce_theta_intercept(theta_init, d)

    theta_hat, Z, MU, LAM, used_idx, meta = fit_ipo_closed_form_multi(
        X_feat,
        Y,
        Vhats,
        idx,
        start_index=start_index,
        end_index=end_index,
        delta=float(delta),
        mode=str(mode),
        psd_eps=float(psd_eps),
        ridge_theta=float(ridge_theta),
        tee=tee,
    )
    meta = dict(meta)
    meta["solver"] = "ipo_closed_form_intercept"
    if theta_init_mat is not None:
        meta["theta_init_provided"] = True
    return theta_hat, Z, MU, LAM, used_idx, meta


__all__ = ["fit_ipo_closed_form_intercept"]

