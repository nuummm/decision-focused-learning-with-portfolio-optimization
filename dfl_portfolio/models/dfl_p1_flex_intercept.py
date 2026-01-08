from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dfl_portfolio.models._intercept_utils import coerce_theta_intercept, make_intercept_features
from dfl_portfolio.models.dfl_p1_flex_multi import fit_dfl_p1_flex_multi


def fit_dfl_p1_flex_intercept(
    X: np.ndarray,
    Y: np.ndarray,
    Vhats: Sequence[np.ndarray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    formulation: str = "dual",
    delta: float = 1.0,
    theta_init: Optional[Any] = None,
    w_warmstart: bool = True,
    aux_init_mode: str = "none",
    aux_init_seed: Optional[int] = None,
    aux_init_sigma_w: float = 0.05,
    aux_init_sigma_lam: float = 1e-2,
    aux_init_sigma_mu: float = 1e-2,
    aux_init_keep: bool = False,
    solver: str = "gurobi",
    solver_options: Optional[dict] = None,
    tee: bool = False,
    lambda_theta_anchor: float = 0.0,
    theta_anchor: Optional[Any] = None,
    lambda_theta_anchor_l1: float = 0.0,
    lambda_theta_iso: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], dict]:
    """
    DFL-P1 flex trainer with intercept using the multi-feature implementation.

    Notes
    -----
    The underlying implementation is based on ``fit_dfl_p1_flex_multi`` with features [1, x].
    Warm-start and auxiliary random init options are accepted for CLI compatibility but are
    not currently used in the multi-feature solver.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"fit_dfl_p1_flex_intercept: X must be (N,d), got {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"fit_dfl_p1_flex_intercept: Y must be (N,d), got {Y.shape}")
    if X.shape != Y.shape:
        raise ValueError("fit_dfl_p1_flex_intercept: X,Y shape mismatch")
    d = int(X.shape[1])

    X_feat = make_intercept_features(X)  # (N,d,2)
    theta_init_mat = coerce_theta_intercept(theta_init, d)
    theta_anchor_mat = coerce_theta_intercept(theta_anchor, d)

    return fit_dfl_p1_flex_multi(
        X_feat,
        Y,
        Vhats,
        idx,
        start_index=start_index,
        end_index=end_index,
        formulation=str(formulation),
        delta=float(delta),
        theta_init=theta_init_mat,
        solver=str(solver),
        solver_options=solver_options,
        tee=tee,
        lambda_theta_anchor=float(lambda_theta_anchor),
        theta_anchor=theta_anchor_mat,
        lambda_theta_anchor_l1=float(lambda_theta_anchor_l1),
        lambda_theta_iso=float(lambda_theta_iso),
    )


__all__ = ["fit_dfl_p1_flex_intercept"]

