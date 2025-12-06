"""
============================================================
DFL-P1 flex model with learnable delta (experimental)
============================================================

現行の `dfl_p1_flex` をベースに、下位 MVO の
リスク回避パラメータ δ を Pyomo の変数として学習する実験版。

- 目的関数:  -(1-δ) r^T w + (δ/2) w^T V w
- KKT / dual 条件は元の実装と同じ形だが、δ が Var になる

他の部分（θ アンカー正則化など）は `dfl_p1_flex` と同じ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pyomo.environ as pyo

from dfl_portfolio.optimization.solvers import make_pyomo_solver
from dfl_portfolio.models.ipo_closed_form import fit_ipo_closed_form
from dfl_portfolio.models.ols import train_ols, predict_yhat
from dfl_portfolio.models.ols_gurobi import solve_series_mvo_gurobi


@dataclass
class SolverMeta:
    solver: str
    termination_condition: object | None
    termination_condition_str: str | None
    status: object | None
    status_str: str | None
    solver_time: float | None
    message: str | None


def _solver_metadata(opt, res, solver_name: str) -> dict:
    """Extract minimal, run.py-compatible metadata from a Pyomo result object."""
    solver_info = getattr(res, "solver", None)
    term = getattr(solver_info, "termination_condition", None) if solver_info else None
    status = getattr(solver_info, "status", None) if solver_info else None
    solver_time = getattr(solver_info, "time", None) if solver_info else None
    message = getattr(solver_info, "message", None) if solver_info else None
    meta = {
        "solver": solver_name,
        "termination_condition": term,
        "termination_condition_str": str(term) if term is not None else None,
        "status": status,
        "status_str": str(status) if status is not None else None,
        "solver_time": solver_time,
        "message": message,
    }
    try:
        if solver_name and str(solver_name).lower() == "gurobi":
            solver_model = getattr(opt, "_solver_model", None)
            if solver_model is not None:
                meta["gurobi_obj_val"] = getattr(solver_model, "ObjVal", None)
                meta["gurobi_obj_bound"] = getattr(solver_model, "ObjBound", None)
                meta["gurobi_mip_gap"] = getattr(solver_model, "MIPGap", None)
                meta["gurobi_runtime"] = getattr(solver_model, "Runtime", None)
    except Exception:
        pass
    return meta


def _prepare_pairs(
    idx: Sequence[int],
    Vhats: Sequence[np.ndarray],
    start_index: Optional[int],
    end_index: Optional[int],
) -> Tuple[List[int], List[np.ndarray]]:
    pairs = [
        (i, V) for i, V in zip(idx, Vhats)
        if (start_index is None or i >= start_index)
        and (end_index is None or i <= end_index)
    ]
    if not pairs:
        raise ValueError("no training indices in the specified range")
    used_idx = [i for i, _ in pairs]
    V_list = [0.5 * (np.asarray(V, float) + np.asarray(V, float).T) for _, V in pairs]
    return used_idx, V_list


def fit_dfl_p1_flex_delta(
    X: np.ndarray,
    Y: np.ndarray,
    Vhats: Sequence[np.ndarray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    formulation: str = "dual",   # "dual" or "kkt"
    delta: float = 0.5,
    theta_init: Optional[Sequence[float]] = None,
    solver: str = "gurobi",
    solver_options: Optional[dict] = None,
    tee: bool = False,
    # regularisation knobs
    lambda_theta_anchor: float = 0.0,
    theta_anchor: Optional[Sequence[float]] = None,
    lambda_theta_anchor_l1: float = 0.0,
    lambda_theta_iso: float = 0.0,
    delta_min: float = 0.01,
    delta_max: float = 0.99,
    lambda_delta_center: float = 0.0,
    lambda_delta_smooth: float = 0.0,
    delta_prev: Optional[float] = None,
):
    """
    DFL-P1 flex (dual/KKT) with learnable delta (experimental).

    他の引数は `fit_dfl_p1_flex` と同様だが、delta は
    - Var `m.delta` の初期値として使用
    - bounds は [0.01, 0.99]
    """

    formulation = formulation.lower()
    if formulation not in {"dual", "kkt"}:
        raise ValueError("formulation must be 'dual' or 'kkt'")

    delta_min = float(delta_min)
    delta_max = float(delta_max)
    if not (0.0 <= delta_min < delta_max <= 1.0):
        raise ValueError("delta_min/max must satisfy 0 <= delta_min < delta_max <= 1")

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    d = X.shape[1]

    used_idx, V_list = _prepare_pairs(idx, Vhats, start_index, end_index)
    T_used = len(used_idx)
    used_idx_array = np.array(used_idx, dtype=int)
    X_used = X[used_idx_array, :]
    Y_used = Y[used_idx_array, :]

    # --- θアンカー正則化の準備 ---
    lam_theta_l2 = float(lambda_theta_anchor)
    lam_theta_l1 = float(lambda_theta_anchor_l1)
    lam_theta_iso = float(lambda_theta_iso)
    lambda_delta_center = float(lambda_delta_center)
    lambda_delta_smooth = float(lambda_delta_smooth)
    # 事前の中心値は 0.5 に固定
    delta0_const = 0.5
    delta_prev_val = float(delta_prev) if delta_prev is not None else delta0_const
    theta_anchor_vec = None
    if lam_theta_l2 > 0.0 or lam_theta_l1 > 0.0:
        if theta_anchor is None:
            raise ValueError("theta_anchor required when lambda_theta_anchor or lambda_theta_anchor_l1 > 0")
        theta_anchor_vec = np.asarray(theta_anchor, float).reshape(-1)
        if theta_anchor_vec.shape[0] != d:
            raise ValueError("theta_anchor must have length d")

    def _resolve_theta_anchor(mode: str) -> Optional[np.ndarray]:
        m = (mode or "none").lower()
        if m == "ols":
            return np.asarray(train_ols(X_used, Y_used), float)
        if m == "ipo":
            theta_ipo, *_ = fit_ipo_closed_form(
                X,
                Y,
                Vhats,
                idx,
                start_index=start_index,
                end_index=end_index,
                delta=delta,
                mode="budget",
                tee=tee,
            )
            return np.asarray(theta_ipo, float)
        return None

    def _resolve_w_anchor(mode: str) -> Optional[np.ndarray]:
        m = (mode or "none").lower()
        if m == "none":
            return None
        if m == "ipo":
            theta_ref = _resolve_theta_anchor("ipo")
            if theta_ref is None:
                return None
        else:
            theta_ref = np.asarray(train_ols(X_used, Y_used), float)
        try:
            yhat_all = predict_yhat(X, theta_ref)
            return solve_series_mvo_gurobi(
                Yhat_all=yhat_all,
                Vhats=V_list,
                idx=used_idx,
                delta=delta,
                psd_eps=1e-12,
                output=False,
            )
        except Exception:
            return None

    # ---------------------------------------------------------
    # Build Pyomo model
    # ---------------------------------------------------------
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T_used - 1)
    m.J = pyo.RangeSet(0, d - 1)

    def x_ini(m, t, j): return float(X[used_idx[int(t)], int(j)])
    def y_ini(m, t, j): return float(Y[used_idx[int(t)], int(j)])
    def V_ini(m, t, j, k): return float(V_list[int(t)][int(j), int(k)])

    # learnable delta in [delta_min, delta_max]
    delta_init = float(np.clip(delta, delta_min, delta_max))
    m.delta = pyo.Var(bounds=(delta_min, delta_max), initialize=delta_init)
    m.x = pyo.Param(m.T, m.J, initialize=x_ini)
    m.y = pyo.Param(m.T, m.J, initialize=y_ini)
    m.V = pyo.Param(m.T, m.J, m.J, initialize=V_ini)
    m.theta = pyo.Var(m.J)
    m.w = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
    m.lam = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
    if lam_theta_l1 > 0.0 and theta_anchor_vec is not None:
        m.theta_dev = pyo.Var(m.J, domain=pyo.NonNegativeReals)
    m.mu = pyo.Var(m.T)

    theta_init_vec: Optional[np.ndarray] = None
    if theta_init is not None:
        if isinstance(theta_init, str):
            key = theta_init.lower().strip()
            if key == "ols":
                theta_init_vec = train_ols(X, Y)
            else:
                raise ValueError(f"Unsupported theta_init option: {theta_init}")
        else:
            theta_init_vec = np.asarray(theta_init, float).reshape(-1)

        if theta_init_vec.shape[0] != d:
            raise ValueError("theta_init must have length d")
        for j in m.J:
            m.theta[j].value = float(theta_init_vec[int(j)])
    else:
        for j in m.J:
            m.theta[j].value = 0.0

    w_init_mat: Optional[np.ndarray] = None
    if theta_init_vec is not None:
        try:
            yhat_all = predict_yhat(X, theta_init_vec)
            w_init_candidate = solve_series_mvo_gurobi(
                Yhat_all=yhat_all,
                Vhats=V_list,
                idx=used_idx,
                delta=delta,
                psd_eps=1e-12,
                output=False,
            )
            if w_init_candidate.shape == (T_used, d) and np.all(np.isfinite(w_init_candidate)):
                w_init_mat = w_init_candidate
        except Exception:
            w_init_mat = None

    for t in m.T:
        for j in m.J:
            if w_init_mat is not None:
                m.w[t, j].value = float(w_init_mat[int(t), int(j)])
            else:
                m.w[t, j].value = 1.0 / d
            m.lam[t, j].value = 0.0
        m.mu[t].value = 0.0

    def budget(m, t):
        return sum(m.w[t, j] for j in m.J) == 1.0
    m.budget = pyo.Constraint(m.T, rule=budget)

    def dual_link(m, t, j):
        yhat = m.x[t, j] * m.theta[j]
        quad_j = sum(m.V[t, j, k] * m.w[t, k] for k in m.J)
        return m.mu[t] + m.lam[t, j] == m.delta * quad_j - (1.0 - m.delta) * yhat

    def strong_duality(m, t):
        quad = sum(m.w[t, j] * sum(m.V[t, j, k] * m.w[t, k] for k in m.J) for j in m.J)
        yhat_dot_w = sum(m.x[t, j] * m.theta[j] * m.w[t, j] for j in m.J)
        return m.delta * quad - (1.0 - m.delta) * yhat_dot_w == m.mu[t]

    def stationarity(m, t, j):
        yhat = m.x[t, j] * m.theta[j]
        quad_j = sum(m.V[t, j, k] * m.w[t, k] for k in m.J)
        return m.delta * quad_j - (1.0 - m.delta) * yhat - m.mu[t] - m.lam[t, j] == 0.0

    def complementarity(m, t, j):
        return m.w[t, j] * m.lam[t, j] == 0.0

    if formulation == "dual":
        m.dual_link = pyo.Constraint(m.T, m.J, rule=dual_link)
        m.strong = pyo.Constraint(m.T, rule=strong_duality)
    else:  # kkt
        m.stationarity = pyo.Constraint(m.T, m.J, rule=stationarity)
        m.comp = pyo.Constraint(m.T, m.J, rule=complementarity)

    # Objective
    def obj_rule(m):
        total = 0.0
        for t in m.T:
            lin = sum(m.y[t, j] * m.w[t, j] for j in m.J)
            quad = sum(m.w[t, j] * sum(m.V[t, j, k] * m.w[t, k] for k in m.J) for j in m.J)
            total += -(1.0 - m.delta) * lin + 0.5 * m.delta * quad
        total = total / float(T_used)

        if lam_theta_l2 > 0.0 and theta_anchor_vec is not None:
            total += 0.5 * lam_theta_l2 * sum((m.theta[j] - float(theta_anchor_vec[int(j)])) ** 2 for j in m.J)
        if lam_theta_l1 > 0.0 and theta_anchor_vec is not None:
            total += lam_theta_l1 * sum(m.theta_dev[j] for j in m.J)

        if lam_theta_iso > 0.0:
            total += 0.5 * lam_theta_iso * sum(m.theta[j] ** 2 for j in m.J)

        # delta に対する正則化項
        if lambda_delta_center > 0.0:
            total += 0.5 * lambda_delta_center * (m.delta - delta0_const) ** 2
        if lambda_delta_smooth > 0.0:
            total += 0.5 * lambda_delta_smooth * (m.delta - delta_prev_val) ** 2

        return total

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    if lam_theta_l1 > 0.0 and theta_anchor_vec is not None:
        def theta_dev_pos(m, j):
            return m.theta_dev[j] >= m.theta[j] - float(theta_anchor_vec[int(j)])

        def theta_dev_neg(m, j):
            return m.theta_dev[j] >= -(m.theta[j] - float(theta_anchor_vec[int(j)]))

        m.theta_dev_pos = pyo.Constraint(m.J, rule=theta_dev_pos)
        m.theta_dev_neg = pyo.Constraint(m.J, rule=theta_dev_neg)

    opt = make_pyomo_solver(m, solver=solver, tee=tee, options=solver_options)
    res = opt.solve(m, tee=tee)
    meta = _solver_metadata(opt, res, solver)

    theta_hat = np.array([pyo.value(m.theta[j]) for j in m.J], dtype=float)
    W = np.array([[pyo.value(m.w[t, j]) for j in m.J] for t in m.T], dtype=float)
    MU = np.array([pyo.value(m.mu[t]) for t in m.T], dtype=float)
    LAM = np.array([[pyo.value(m.lam[t, j]) for j in m.J] for t in m.T], dtype=float)
    delta_hat = float(pyo.value(m.delta))
    meta["delta_hat"] = delta_hat

    return theta_hat, W, MU, LAM, used_idx, meta


__all__ = ["fit_dfl_p1_flex_delta", "SolverMeta"]
