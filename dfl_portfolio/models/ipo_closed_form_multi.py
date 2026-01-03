from __future__ import annotations

"""
IPO closed-form trainer (multi-feature version).

This is a multi-variate extension of `dfl_portfolio.models.ipo_closed_form.fit_ipo_closed_form`
for the per-asset multi-feature regression model:

    y_hat[t, j] = x[t, j, :]^T theta[j, :]

where X_feat has shape (N, d, K) and theta has shape (d, K).

The implementation follows the same derivation as the univariate version, replacing the
diagonal feature map y_hat = diag(x) theta with the block-diagonal linear map y_hat = B(x) θ_vec,
where θ_vec = vec(theta) ∈ R^{dK}.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _make_psd(V: np.ndarray, psd_eps: float) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    V = 0.5 * (V + V.T)
    return V + float(psd_eps) * np.eye(V.shape[0], dtype=float)


def _build_affine_decision(V: np.ndarray, delta: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    d = int(V.shape[0])
    M = np.linalg.inv(V)
    if mode == "unconstrained":
        H = M
        c = np.zeros(d, dtype=float)
    elif mode == "budget":
        one = np.ones(d, dtype=float)
        a = M @ one
        b = float(one @ a)
        H = (np.eye(d, dtype=float) - np.outer(a, np.ones(d, dtype=float)) / b) @ M
        c = a / b
    else:
        raise ValueError('mode must be "budget" or "unconstrained"')
    return H, c


def _bt_vec(X_i: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute B(x)^T v for block-diagonal B induced by X_i=(d,K), returning (d,K)."""
    v = np.asarray(v, dtype=float).reshape(-1)
    X_i = np.asarray(X_i, dtype=float)
    if X_i.ndim != 2:
        raise ValueError(f"X_i must be 2D (d,K), got shape {X_i.shape}")
    d, _K = X_i.shape
    if v.shape[0] != d:
        raise ValueError(f"v length mismatch: got {v.shape[0]}, expected {d}")
    return v[:, None] * X_i


def _bt_mat_b(X_i: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Compute B(x)^T S B(x) as a (d,K,d,K) tensor."""
    X_i = np.asarray(X_i, dtype=float)
    S = np.asarray(S, dtype=float)
    if X_i.ndim != 2:
        raise ValueError(f"X_i must be 2D (d,K), got shape {X_i.shape}")
    d, _K = X_i.shape
    if S.shape != (d, d):
        raise ValueError(f"S shape mismatch: got {S.shape}, expected {(d, d)}")
    # (d,d) and (d,K) -> (d,K,d,K), where block(j,l)=S[j,l]*outer(x_j, x_l)
    return np.einsum("jl,ja,lb->jalb", S, X_i, X_i)


def fit_ipo_closed_form_multi(
    X_feat: np.ndarray,
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
    """
    Multi-feature IPO closed-form trainer.

    Returns
    -------
    theta_hat : (d, K)
    Z, MU, LAM : empty arrays (runner compatibility)
    used_idx : list[int]
    meta : dict
    """
    X_feat = np.asarray(X_feat, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X_feat.ndim != 3:
        raise ValueError(f"X_feat must be 3D (N,d,K), got shape {X_feat.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N,d), got shape {Y.shape}")
    N, d, K = X_feat.shape
    if Y.shape != (N, d):
        raise ValueError(f"X_feat,Y shape mismatch: {X_feat.shape} vs {Y.shape}")

    s = 0 if start_index is None else max(0, int(start_index))
    e = (N - 1) if end_index is None else min(N - 1, int(end_index))
    if s > e:
        raise ValueError("fit_ipo_closed_form_multi: invalid train range")

    pairs = [(int(i), np.asarray(V, float)) for i, V in zip(idx, Vhats) if s <= int(i) <= e]
    if len(pairs) == 0:
        theta_hat = np.zeros((d, K), dtype=float)
        Z = np.empty((0, d), dtype=float)
        MU = np.empty((0,), dtype=float)
        LAM = np.empty((0, d), dtype=float)
        meta = {"solver": "ipo_closed_form_multi", "status": "skipped", "message": "no pairs in range"}
        return theta_hat, Z, MU, LAM, [], meta

    alpha = 1.0 - float(delta)
    if not (0.0 < float(delta) < 1.0) or alpha <= 0.0:
        raise ValueError(f"delta must be in (0,1) for IPO closed-form; got {delta}")
    delta_eff = float(delta) / alpha

    p = int(d * K)
    A = np.zeros((p, p), dtype=float)
    b = np.zeros(p, dtype=float)
    coef_quad = (alpha**2) / float(delta)

    for (i, V_i) in pairs:
        V_i = _make_psd(V_i, psd_eps)
        if V_i.shape != (d, d):
            raise ValueError(f"V shape mismatch: got {V_i.shape}, expected {(d, d)}")
        H, c = _build_affine_decision(V_i, delta_eff, mode)
        S = H.T @ V_i @ H
        t = H.T @ (V_i @ c)

        X_i = X_feat[i, :, :]  # (d,K)
        y_i = Y[i, :]  # (d,)

        # A_i = coef_quad * B^T S B
        A_i = _bt_mat_b(X_i, S).reshape(p, p)
        A += coef_quad * A_i

        # b_i = coef_quad * B^T H^T y - alpha * B^T t
        u = H.T @ y_i
        bt_u = _bt_vec(X_i, u).reshape(p)
        bt_t = _bt_vec(X_i, t).reshape(p)
        b += coef_quad * bt_u - alpha * bt_t

    if float(ridge_theta) > 0.0:
        A = 0.5 * (A + A.T) + float(ridge_theta) * np.eye(p, dtype=float)
    else:
        A = 0.5 * (A + A.T)

    try:
        theta_vec = np.linalg.solve(A, b)
        status = "completed"
        msg = None
    except np.linalg.LinAlgError:
        theta_vec = np.linalg.lstsq(A, b, rcond=None)[0]
        status = "completed_lstsq"
        msg = "A was singular; used lstsq."

    theta_hat = theta_vec.reshape(d, K)
    if tee:
        print(
            f"[IPO Closed-Form Multi] train range=[{s},{e}] mode={mode} theta_shape={theta_hat.shape}"
        )

    Z = np.empty((0, d), dtype=float)
    MU = np.empty((0,), dtype=float)
    LAM = np.empty((0, d), dtype=float)
    used_idx = [i for (i, _) in pairs]
    meta: Dict[str, Any] = {
        "solver": "ipo_closed_form_multi",
        "status": status,
        "termination_condition": None,
        "termination_condition_str": None,
        "message": msg,
        "mode": mode,
        "delta": float(delta),
        "psd_eps": float(psd_eps),
        "ridge_theta": float(ridge_theta),
        "n_samples": int(len(pairs)),
        "d": int(d),
        "k": int(K),
    }
    return theta_hat, Z, MU, LAM, used_idx, meta


__all__ = ["fit_ipo_closed_form_multi"]

