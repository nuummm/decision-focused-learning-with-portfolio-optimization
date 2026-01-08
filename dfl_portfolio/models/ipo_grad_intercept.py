from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dfl_portfolio.models._intercept_utils import coerce_theta_intercept, make_intercept_features
from dfl_portfolio.models.ipo_grad_multi import fit_ipo_grad_multi


def fit_ipo_grad_intercept(
    X: Any,
    Y: Any,
    Vhats: Sequence[Any],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    delta: float = 1.0,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 0,
    qp_max_iter: int = 5000,
    qp_tol: float = 1e-6,
    seed: Optional[int] = None,
    theta_init: Optional[Any] = None,
    lambda_anchor: float = 0.0,
    theta_anchor: Optional[Any] = None,
    tee: bool = False,
    debug_kkt: bool = False,
) -> Tuple[Any, Any, Any, Any, List[int], Dict[str, Any]]:
    """IPO-GRAD trainer with intercept using the multi-feature implementation."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"fit_ipo_grad_intercept: X must be (N,d), got {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"fit_ipo_grad_intercept: Y must be (N,d), got {Y.shape}")
    if X.shape != Y.shape:
        raise ValueError("fit_ipo_grad_intercept: X,Y shape mismatch")
    d = int(X.shape[1])

    X_feat = make_intercept_features(X)
    theta_init_mat = coerce_theta_intercept(theta_init, d)
    theta_anchor_mat = coerce_theta_intercept(theta_anchor, d)

    return fit_ipo_grad_multi(
        X_feat,
        Y,
        Vhats,
        idx,
        start_index=start_index,
        end_index=end_index,
        delta=float(delta),
        epochs=int(epochs),
        lr=float(lr),
        batch_size=int(batch_size),
        qp_max_iter=int(qp_max_iter),
        qp_tol=float(qp_tol),
        seed=seed,
        theta_init=theta_init_mat,
        lambda_anchor=float(lambda_anchor),
        theta_anchor=theta_anchor_mat,
        tee=tee,
        debug_kkt=bool(debug_kkt),
    )


__all__ = ["fit_ipo_grad_intercept"]

