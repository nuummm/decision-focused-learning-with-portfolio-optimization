from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from dfl_portfolio.experiments.real_data_common import (
    mvo_cost,
    ScheduleItem,
    build_rebalance_schedule,
    build_flex_dual_kkt_ensemble,
    resolve_trading_cost_rates,
)
from dfl_portfolio.models.ols_intercept import train_ols_intercept
from dfl_portfolio.models.ipo_closed_form_intercept import fit_ipo_closed_form_intercept


def prepare_flex_training_args(
    bundle,
    train_start: int,
    train_end: int,
    delta: float,
    tee: bool,
    flex_options: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """Intercept-aware variant of real_data_common.prepare_flex_training_args."""
    flex_kwargs: Dict[str, Any] = dict(flex_options or {})

    theta_init_mode = str(flex_kwargs.pop("theta_init_mode", flex_kwargs.pop("flex_theta_init_mode", "none")) or "none").lower()
    theta_anchor_mode = str(
        flex_kwargs.pop("theta_anchor_mode", flex_kwargs.pop("flex_theta_anchor_mode", "none")) or "none"
    ).lower()
    if theta_anchor_mode == "zero":
        theta_anchor_mode = "zero"
    theta_init_seed = flex_kwargs.pop("theta_init_seed", None)
    theta_init_sigma = float(flex_kwargs.pop("theta_init_sigma", 0.0) or 0.0)
    theta_init_clip = float(flex_kwargs.pop("theta_init_clip", 0.0) or 0.0)
    theta_init_base_mode = str(flex_kwargs.pop("theta_init_base_mode", "none") or "none").lower()
    flex_kwargs["formulation"] = str(flex_kwargs.get("formulation", "dual") or "dual").lower()

    X = np.asarray(bundle.dataset.X, float)
    Y = np.asarray(bundle.dataset.Y, float)
    idx_list = bundle.cov_indices.tolist()
    n_samples, n_assets = X.shape

    start_idx = max(0, int(train_start))
    end_idx = min(n_samples - 1, int(train_end))
    if end_idx < start_idx:
        raise ValueError(f"Invalid train window [{train_start}, {train_end}]")
    train_slice = slice(start_idx, end_idx + 1)
    X_train = X[train_slice]
    Y_train = Y[train_slice]

    theta_sources: Dict[str, np.ndarray] = {}

    def ensure_theta_source(mode: str, *, fallback_to_ols: bool, context: str) -> Optional[np.ndarray]:
        m = (mode or "none").lower()
        if m in {"", "none"}:
            return None
        if m == "zero":
            return np.zeros((n_assets, 2), dtype=float)
        if m == "ols":
            if "ols" not in theta_sources:
                theta_sources["ols"] = np.asarray(train_ols_intercept(X_train, Y_train), dtype=float)
            return theta_sources["ols"].copy()
        if m == "ipo":
            if float(delta) == 0.0:
                if "ipo" not in theta_sources:
                    theta_sources["ipo"] = np.asarray(train_ols_intercept(X_train, Y_train), dtype=float)
                return theta_sources["ipo"].copy()
            if "ipo" in theta_sources:
                return theta_sources["ipo"].copy()
            try:
                theta_ipo, *_ = fit_ipo_closed_form_intercept(
                    X,
                    Y,
                    bundle.covariances,
                    idx_list,
                    start_index=train_start,
                    end_index=train_end,
                    delta=float(delta),
                    mode="budget",
                    tee=tee,
                )
                theta_sources["ipo"] = np.asarray(theta_ipo, dtype=float)
                return theta_sources["ipo"].copy()
            except Exception as exc:
                if fallback_to_ols:
                    print(f"[WARN] IPO anchor construction failed ({exc}); using OLS for {context}")
                    if "ols" not in theta_sources:
                        theta_sources["ols"] = np.asarray(train_ols_intercept(X_train, Y_train), dtype=float)
                    return theta_sources["ols"].copy()
                raise
        raise ValueError(f"Unsupported theta source '{mode}' for {context}")

    theta_init: Optional[np.ndarray] = None
    theta_init_meta: Dict[str, Any] = {
        "theta_init_mode": str(theta_init_mode),
        "theta_init_seed": int(theta_init_seed) if theta_init_seed is not None else None,
        "theta_init_sigma": float(theta_init_sigma),
        "theta_init_clip": float(theta_init_clip),
        "theta_init_base_mode": str(theta_init_base_mode),
        "theta_init_eta_l2": float("nan"),
        "theta_init_l2": float("nan"),
        "theta_base_l2": float("nan"),
    }

    def _flat_norm(arr: Optional[np.ndarray]) -> float:
        if arr is None:
            return float("nan")
        vec = np.asarray(arr, dtype=float).reshape(-1)
        return float(np.linalg.norm(vec))

    if theta_init_mode in {"rand_zero", "randn_zero", "random_zero"}:
        rng = np.random.default_rng(int(theta_init_seed) if theta_init_seed is not None else None)
        sigma = float(theta_init_sigma) if theta_init_sigma > 0.0 else 0.1
        theta = rng.normal(size=(n_assets, 2)) * sigma
        if theta_init_clip > 0.0:
            theta = np.clip(theta, -theta_init_clip, theta_init_clip)
        theta_init = np.asarray(theta, dtype=float)
        theta_init_meta["theta_init_eta_l2"] = _flat_norm(theta_init)
        theta_init_meta["theta_init_l2"] = _flat_norm(theta_init)
    elif theta_init_mode in {"rand_local", "randn_local", "random_local"}:
        base_theta = ensure_theta_source(theta_init_base_mode, fallback_to_ols=True, context="theta_init_base")
        if base_theta is None:
            base_theta = np.zeros((n_assets, 2), dtype=float)
        rng = np.random.default_rng(int(theta_init_seed) if theta_init_seed is not None else None)
        sigma = float(theta_init_sigma) if theta_init_sigma > 0.0 else 0.1
        theta = np.asarray(base_theta, dtype=float).copy()
        theta = theta + rng.normal(size=(n_assets, 2)) * sigma
        if theta_init_clip > 0.0:
            theta = np.clip(theta, -theta_init_clip, theta_init_clip)
        theta_init = np.asarray(theta, dtype=float)
        theta_init_meta["theta_base_l2"] = _flat_norm(base_theta)
        theta_init_meta["theta_init_eta_l2"] = _flat_norm(theta_init - np.asarray(base_theta, dtype=float))
        theta_init_meta["theta_init_l2"] = _flat_norm(theta_init)
    elif theta_init_mode not in {"", "none"}:
        theta_init = ensure_theta_source(theta_init_mode, fallback_to_ols=True, context="theta_init")
        theta_init_meta["theta_init_eta_l2"] = 0.0
        theta_init_meta["theta_init_l2"] = _flat_norm(theta_init)
    else:
        theta_init_meta["theta_init_eta_l2"] = 0.0

    lam_theta_anchor = float(flex_kwargs.get("lambda_theta_anchor", 0.0) or 0.0)
    lam_theta_anchor_l1 = float(flex_kwargs.get("lambda_theta_anchor_l1", 0.0) or 0.0)
    if (
        "theta_anchor" not in flex_kwargs
        and (lam_theta_anchor > 0.0 or lam_theta_anchor_l1 > 0.0 or theta_anchor_mode not in {"", "none"})
    ):
        theta_anchor = ensure_theta_source(theta_anchor_mode, fallback_to_ols=False, context="theta_anchor")
        if theta_anchor is None:
            raise ValueError("theta_anchor_mode requires a valid reference but none was available.")
        flex_kwargs["theta_anchor"] = theta_anchor

    flex_kwargs["lambda_theta_anchor"] = lam_theta_anchor
    flex_kwargs["lambda_theta_anchor_l1"] = float(flex_kwargs.get("lambda_theta_anchor_l1", 0.0) or 0.0)
    flex_kwargs["lambda_theta_iso"] = float(flex_kwargs.get("lambda_theta_iso", 0.0) or 0.0)
    return theta_init, flex_kwargs, theta_init_meta


__all__ = [
    "mvo_cost",
    "ScheduleItem",
    "build_rebalance_schedule",
    "prepare_flex_training_args",
    "build_flex_dual_kkt_ensemble",
    "resolve_trading_cost_rates",
]

