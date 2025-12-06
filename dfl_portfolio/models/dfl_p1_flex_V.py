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


def _prepare_pairs_V(
    idx: Sequence[int],
    V_sample_list: Sequence[np.ndarray],
    V_diag_list: Sequence[np.ndarray],
    start_index: Optional[int],
    end_index: Optional[int],
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """Align indices and covariance matrices for the given train window."""
    pairs = [
        (i, Vs, Vd)
        for i, Vs, Vd in zip(idx, V_sample_list, V_diag_list)
        if (start_index is None or i >= start_index)
        and (end_index is None or i <= end_index)
    ]
    if not pairs:
        raise ValueError("no training indices in the specified range")
    used_idx = [i for i, _, _ in pairs]
    Vs_used = [0.5 * (np.asarray(Vs, float) + np.asarray(Vs, float).T) for _, Vs, _ in pairs]
    Vd_used = [0.5 * (np.asarray(Vd, float) + np.asarray(Vd, float).T) for _, _, Vd in pairs]
    return used_idx, Vs_used, Vd_used


def fit_dfl_p1_flex_V(
    X: np.ndarray,
    Y: np.ndarray,
    V_sample_list: Sequence[np.ndarray],
    V_diag_list: Sequence[np.ndarray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    formulation: str = "dual",   # "dual" or "kkt"
    delta: float = 1.0,
    theta_init: Optional[Sequence[float]] = None,
    solver_options: Optional[dict] = None,
    tee: bool = False,
    # regularisation knobs
    lambda_theta_anchor: float = 0.0,
    theta_anchor: Optional[Sequence[float]] = None,
    lambda_theta_anchor_l1: float = 0.0,
    lambda_theta_iso: float = 0.0,
):
    """
    DFL-P1 flex モデルの「共分散縮小係数 φ 学習版」。

    - 下位 MVO の共分散は
        V_eff(t) = (1-φ) * V_sample(t) + φ * diag(V_sample(t))
      として表現し、φ を [0,1] の変数として学習する。
    - θ の正則化や dual/KKT の切り替えは dfl_p1_flex と同じ。
    - ソルバは Knitro 固定（非凸連続最適化）。
    """

    formulation = formulation.lower()
    if formulation not in {"dual", "kkt"}:
        raise ValueError("formulation must be 'dual' or 'kkt'")

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    d = X.shape[1]

    used_idx, Vs_list, Vd_list = _prepare_pairs_V(idx, V_sample_list, V_diag_list, start_index, end_index)
    T_used = len(used_idx)
    used_idx_array = np.array(used_idx, dtype=int)
    X_used = X[used_idx_array, :]
    Y_used = Y[used_idx_array, :]

    lam_theta_l2 = float(lambda_theta_anchor)
    lam_theta_l1 = float(lambda_theta_anchor_l1)
    lam_theta_iso = float(lambda_theta_iso)
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
                V_sample_list,
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
            # 初期化用の V は φ=0.94 の縮小版を仮に用いる
            phi0 = 0.94
            V_list_init = [
                (1.0 - phi0) * Vs + phi0 * Vd for Vs, Vd in zip(Vs_list, Vd_list)
            ]
            return solve_series_mvo_gurobi(
                Yhat_all=yhat_all,
                Vhats=V_list_init,
                idx=used_idx,
                delta=delta,
                psd_eps=1e-12,
                output=False,
            )
        except Exception:
            return None

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T_used - 1)
    m.J = pyo.RangeSet(0, d - 1)

    def x_ini(m, t, j): return float(X[used_idx[int(t)], int(j)])

    def y_ini(m, t, j): return float(Y[used_idx[int(t)], int(j)])

    def Vs_ini(m, t, j, k): return float(Vs_list[int(t)][int(j), int(k)])

    def Vd_ini(m, t, j, k): return float(Vd_list[int(t)][int(j), int(k)])

    m.delta = pyo.Param(initialize=float(delta))
    m.x = pyo.Param(m.T, m.J, initialize=x_ini)
    m.y = pyo.Param(m.T, m.J, initialize=y_ini)
    m.Vs = pyo.Param(m.T, m.J, m.J, initialize=Vs_ini)
    m.Vd = pyo.Param(m.T, m.J, m.J, initialize=Vd_ini)

    # 縮小係数 φ（全時点共通）を Var として導入
    m.phi = pyo.Var(bounds=(0.0, 1.0), initialize=0.94)

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
            phi0 = 0.94
            V_list_init = [
                (1.0 - phi0) * Vs + phi0 * Vd for Vs, Vd in zip(Vs_list, Vd_list)
            ]
            w_init_candidate = solve_series_mvo_gurobi(
                Yhat_all=yhat_all,
                Vhats=V_list_init,
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

    def V_eff(m, t, j, k):
        return (1.0 - m.phi) * m.Vs[t, j, k] + m.phi * m.Vd[t, j, k]

    def dual_link(m, t, j):
        yhat = m.x[t, j] * m.theta[j]
        quad_j = sum(V_eff(m, t, j, k) * m.w[t, k] for k in m.J)
        return m.mu[t] + m.lam[t, j] == m.delta * quad_j - (1.0 - m.delta) * yhat

    def strong_duality(m, t):
        quad = sum(
            m.w[t, j] * sum(V_eff(m, t, j, k) * m.w[t, k] for k in m.J)
            for j in m.J
        )
        yhat_dot_w = sum(m.x[t, j] * m.theta[j] * m.w[t, j] for j in m.J)
        return m.delta * quad - (1.0 - m.delta) * yhat_dot_w == m.mu[t]

    def stationarity(m, t, j):
        yhat = m.x[t, j] * m.theta[j]
        quad_j = sum(V_eff(m, t, j, k) * m.w[t, k] for k in m.J)
        return m.delta * quad_j - (1.0 - m.delta) * yhat - m.mu[t] - m.lam[t, j] == 0.0

    def complementarity(m, t, j):
        return m.w[t, j] * m.lam[t, j] == 0.0

    if formulation == "dual":
        m.dual_link = pyo.Constraint(m.T, m.J, rule=dual_link)
        m.strong = pyo.Constraint(m.T, rule=strong_duality)
    else:
        m.stationarity = pyo.Constraint(m.T, m.J, rule=stationarity)
        m.comp = pyo.Constraint(m.T, m.J, rule=complementarity)

    def obj_rule(m):
        total = 0.0
        for t in m.T:
            lin = sum(m.y[t, j] * m.w[t, j] for j in m.J)
            quad = sum(
                m.w[t, j] * sum(V_eff(m, t, j, k) * m.w[t, k] for k in m.J)
                for j in m.J
            )
            total += -(1.0 - m.delta) * lin + 0.5 * m.delta * quad
        total = total / float(T_used)

        if lam_theta_l2 > 0.0 and theta_anchor_vec is not None:
            total += 0.5 * lam_theta_l2 * sum(
                (m.theta[j] - float(theta_anchor_vec[int(j)])) ** 2 for j in m.J
            )
        if lam_theta_l1 > 0.0 and theta_anchor_vec is not None:
            total += lam_theta_l1 * sum(m.theta_dev[j] for j in m.J)
        if lam_theta_iso > 0.0:
            total += 0.5 * lam_theta_iso * sum(m.theta[j] ** 2 for j in m.J)
        return total

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    if lam_theta_l1 > 0.0 and theta_anchor_vec is not None:
        def theta_dev_pos(m, j):
            return m.theta_dev[j] >= m.theta[j] - float(theta_anchor_vec[int(j)])

        def theta_dev_neg(m, j):
            return m.theta_dev[j] >= -(m.theta[j] - float(theta_anchor_vec[int(j)]))

        m.theta_dev_pos = pyo.Constraint(m.J, rule=theta_dev_pos)
        m.theta_dev_neg = pyo.Constraint(m.J, rule=theta_dev_neg)

    # Knitro 固定で解く
    opt = make_pyomo_solver(m, solver="knitro", tee=tee, options=solver_options)
    res = opt.solve(m, tee=tee)
    meta = _solver_metadata(opt, res, "knitro")
    meta["phi_hat"] = float(pyo.value(m.phi))

    theta_hat = np.array([pyo.value(m.theta[j]) for j in m.J], dtype=float)
    W = np.array([[pyo.value(m.w[t, j]) for j in m.J] for t in m.T], dtype=float)
    MU = np.array([pyo.value(m.mu[t]) for t in m.T], dtype=float)
    LAM = np.array([[pyo.value(m.lam[t, j]) for j in m.J] for t in m.T], dtype=float)

    return theta_hat, W, MU, LAM, used_idx, meta

