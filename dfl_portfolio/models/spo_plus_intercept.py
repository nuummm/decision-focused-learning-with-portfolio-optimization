from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dfl_portfolio.models._intercept_utils import coerce_theta_intercept, make_intercept_features
from dfl_portfolio.models.spo_plus_multi import fit_spo_plus_multi


def fit_spo_plus_intercept(
    X: Any,
    Y: Any,
    Vhats: Sequence[Any],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    delta: float = 1.0,  # interface compatibility; not used by SPO+
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 0,
    lambda_reg: float = 0.0,
    lambda_anchor: float = 0.0,
    theta_anchor: Optional[np.ndarray] = None,
    risk_constraint: bool = True,
    risk_mult: float = 2.0,
    psd_eps: float = 1e-9,
    tee: bool = False,
    theta_init: Optional[np.ndarray] = None,
) -> Tuple[Any, Any, Any, Any, List[int], Dict[str, Any]]:
    """SPO+ trainer with intercept using the multi-feature implementation."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"fit_spo_plus_intercept: X must be (N,d), got {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"fit_spo_plus_intercept: Y must be (N,d), got {Y.shape}")
    if X.shape != Y.shape:
        raise ValueError("fit_spo_plus_intercept: X,Y shape mismatch")
    d = int(X.shape[1])

    X_feat = make_intercept_features(X)
    theta_init_mat = coerce_theta_intercept(theta_init, d)
    theta_anchor_mat = coerce_theta_intercept(theta_anchor, d)

    return fit_spo_plus_multi(
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
        lambda_reg=float(lambda_reg),
        lambda_anchor=float(lambda_anchor),
        theta_anchor=theta_anchor_mat,
        risk_constraint=bool(risk_constraint),
        risk_mult=float(risk_mult),
        psd_eps=float(psd_eps),
        tee=tee,
        theta_init=theta_init_mat,
    )


__all__ = ["fit_spo_plus_intercept"]

