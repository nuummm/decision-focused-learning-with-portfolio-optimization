"""
============================================================
DFL-P1 flex model (multi-feature version, experimental)
============================================================

各資産 j ごとに K 次元特徴 x_{t,j} ∈ R^K を持つ重回帰モデルを前提とした
DFL-P1 (dual / KKT) 実装。

予測モデル:
    \hat r_j(x_t) = x_{t,j,:}^T theta_{j,:}

下位 MVO 目的:
    min_w  (delta / 2) w^T V w - (1 - delta) \hat r(x)^T w

使い方は dfl_p1_flex.fit_dfl_p1_flex とほぼ同じだが、
入力 X_feat の形が (N, d, K)、theta_hat が (d, K) になる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyomo.environ as pyo

from dfl_portfolio.models.dfl_p1_flex import _prepare_pairs, _solver_metadata, SolverMeta
from dfl_portfolio.models.ols_multi import train_ols_multi, predict_yhat_multi
from dfl_portfolio.optimization.solvers import make_pyomo_solver


def fit_dfl_p1_flex_multi(
    X_feat: np.ndarray,              # (N, d, K)
    Y: np.ndarray,                   # (N, d)
    Vhats: Sequence[np.ndarray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    formulation: str = "dual",       # "dual" or "kkt"
    delta: float = 1.0,
    theta_init: Optional[np.ndarray] = None,  # (d, K) or "ols"
    solver: str = "gurobi",
    solver_options: Optional[dict] = None,
    tee: bool = False,
    # regularisation knobs
    lambda_theta_anchor: float = 0.0,
    theta_anchor: Optional[np.ndarray] = None,  # (d, K)
    lambda_theta_anchor_l1: float = 0.0,
    lambda_theta_iso: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], dict]:
    """
    Multi-feature DFL-P1 flex solver.

    Parameters
    ----------
    X_feat : (N, d, K)
        説明変数。
    Y : (N, d)
        目的変数（リターン）。
    Vhats, idx, start_index, end_index :
        共分散系列とローリング範囲指定（dfl_p1_flex と同じ）。
    formulation : {"dual","kkt"}
        dual  -> 強双対形式
        kkt   -> KKT 条件形式
    delta : float
        リスク回避パラメータ。
    theta_init :
        初期値。None / "ols" / (d,K) ndarray のいずれか。

    戻り値は dfl_p1_flex と揃えて:
        theta_hat, W, MU, LAM, used_idx, meta
    """

    formulation = (formulation or "dual").lower()
    if formulation not in {"dual", "kkt"}:
        raise ValueError("formulation must be 'dual' or 'kkt'")

    X_feat = np.asarray(X_feat, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X_feat.ndim != 3:
        raise ValueError(f"X_feat must be 3D (N,d,K), got shape {X_feat.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N,d), got shape {Y.shape}")

    N, d, K = X_feat.shape
    if Y.shape != (N, d):
        raise ValueError(f"Y shape {Y.shape} incompatible with X_feat shape {X_feat.shape}")

    used_idx, V_list = _prepare_pairs(idx, Vhats, start_index, end_index)
    T_used = len(used_idx)
    used_idx_array = np.array(used_idx, dtype=int)
    X_used = X_feat[used_idx_array, :, :]  # (T_used, d, K)
    Y_used = Y[used_idx_array, :]         # (T_used, d)

    lam_theta_l2 = float(lambda_theta_anchor)
    lam_theta_l1 = float(lambda_theta_anchor_l1)
    lam_theta_iso = float(lambda_theta_iso)
    theta_anchor_mat: Optional[np.ndarray] = None
    if lam_theta_l2 > 0.0 or lam_theta_l1 > 0.0:
        if theta_anchor is None:
            raise ValueError("theta_anchor required when lambda_theta_anchor or lambda_theta_anchor_l1 > 0")
        theta_anchor_mat = np.asarray(theta_anchor, dtype=float)
        if theta_anchor_mat.shape != (d, K):
            raise ValueError(f"theta_anchor must have shape (d,K)={d,K}, got {theta_anchor_mat.shape}")

    # ---------------------------------------------------------
    # Pyomo model
    # ---------------------------------------------------------
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T_used - 1)
    m.J = pyo.RangeSet(0, d - 1)
    m.F = pyo.RangeSet(0, K - 1)

    def x_ini(m, t, j, f):
        return float(X_feat[used_idx[int(t)], int(j), int(f)])

    def y_ini(m, t, j):
        return float(Y[used_idx[int(t)], int(j)])

    def V_ini(m, t, j, k):
        return float(V_list[int(t)][int(j), int(k)])

    m.delta = pyo.Param(initialize=float(delta))
    m.x = pyo.Param(m.T, m.J, m.F, initialize=x_ini)
    m.y = pyo.Param(m.T, m.J, initialize=y_ini)
    m.V = pyo.Param(m.T, m.J, m.J, initialize=V_ini)

    m.theta = pyo.Var(m.J, m.F)
    m.w = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
    m.lam = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
    if lam_theta_l1 > 0.0 and theta_anchor_mat is not None:
        m.theta_dev = pyo.Var(m.J, m.F, domain=pyo.NonNegativeReals)
    m.mu = pyo.Var(m.T)

    # theta 初期値
    theta_init_mat: Optional[np.ndarray] = None
    if isinstance(theta_init, str):
        if theta_init.lower().strip() == "ols":
            theta_init_mat = train_ols_multi(X_feat, Y)
        else:
            raise ValueError(f"Unsupported theta_init option: {theta_init}")
    elif theta_init is not None:
        theta_init_mat = np.asarray(theta_init, dtype=float)

    if theta_init_mat is not None:
        if theta_init_mat.shape != (d, K):
            raise ValueError(f"theta_init must have shape (d,K)={d,K}, got {theta_init_mat.shape}")
        for j in m.J:
            for f in m.F:
                m.theta[j, f].value = float(theta_init_mat[int(j), int(f)])
    else:
        for j in m.J:
            for f in m.F:
                m.theta[j, f].value = 0.0

    # w 初期値は等分散ポートフォリオ
    for t in m.T:
        for j in m.J:
            m.w[t, j].value = 1.0 / d
            m.lam[t, j].value = 0.0
        m.mu[t].value = 0.0

    # budget constraint
    def budget(m, t):
        return sum(m.w[t, j] for j in m.J) == 1.0

    m.budget = pyo.Constraint(m.T, rule=budget)

    # yhat(t,j) = sum_f x(t,j,f) * theta(j,f)
    def _yhat_expr(m, t, j):
        return sum(m.x[t, j, f] * m.theta[j, f] for f in m.F)

    # dual / KKT constraints
    def dual_link(m, t, j):
        yhat = _yhat_expr(m, t, j)
        quad_j = sum(m.V[t, j, k] * m.w[t, k] for k in m.J)
        return m.mu[t] + m.lam[t, j] == m.delta * quad_j - (1.0 - m.delta) * yhat

    def strong_duality(m, t):
        quad = sum(
            m.w[t, j] * sum(m.V[t, j, k] * m.w[t, k] for k in m.J)
            for j in m.J
        )
        yhat_dot_w = sum(
            _yhat_expr(m, t, j) * m.w[t, j]
            for j in m.J
        )
        return m.delta * quad - (1.0 - m.delta) * yhat_dot_w == m.mu[t]

    def stationarity(m, t, j):
        yhat = _yhat_expr(m, t, j)
        quad_j = sum(m.V[t, j, k] * m.w[t, k] for k in m.J)
        return m.delta * quad_j - (1.0 - m.delta) * yhat - m.mu[t] - m.lam[t, j] == 0.0

    def complementarity(m, t, j):
        return m.w[t, j] * m.lam[t, j] == 0.0

    if formulation == "dual":
        m.dual_link = pyo.Constraint(m.T, m.J, rule=dual_link)
        m.strong = pyo.Constraint(m.T, rule=strong_duality)
    else:
        m.stationarity = pyo.Constraint(m.T, m.J, rule=stationarity)
        m.comp = pyo.Constraint(m.T, m.J, rule=complementarity)

    # Objective
    def obj_rule(m):
        total = 0.0
        for t in m.T:
            lin = sum(_yhat_expr(m, t, j) * m.w[t, j] for j in m.J)
            quad = sum(
                m.w[t, j] * sum(m.V[t, j, k] * m.w[t, k] for k in m.J)
                for j in m.J
            )
            total += -(1.0 - m.delta) * lin + 0.5 * m.delta * quad
        total = total / float(T_used)

        if lam_theta_l2 > 0.0 and theta_anchor_mat is not None:
            total += 0.5 * lam_theta_l2 * sum(
                (m.theta[j, f] - float(theta_anchor_mat[int(j), int(f)])) ** 2
                for j in m.J for f in m.F
            )
        if lam_theta_l1 > 0.0 and theta_anchor_mat is not None:
            total += lam_theta_l1 * sum(m.theta_dev[j, f] for j in m.J for f in m.F)

        if lam_theta_iso > 0.0:
            total += 0.5 * lam_theta_iso * sum(
                m.theta[j, f] ** 2 for j in m.J for f in m.F
            )

        return total

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    if lam_theta_l1 > 0.0 and theta_anchor_mat is not None:
        def theta_dev_pos(m, j, f):
            return m.theta_dev[j, f] >= m.theta[j, f] - float(theta_anchor_mat[int(j), int(f)])

        def theta_dev_neg(m, j, f):
            return m.theta_dev[j, f] >= -(m.theta[j, f] - float(theta_anchor_mat[int(j), int(f)]))

        m.theta_dev_pos = pyo.Constraint(m.J, m.F, rule=theta_dev_pos)
        m.theta_dev_neg = pyo.Constraint(m.J, m.F, rule=theta_dev_neg)

    opt = make_pyomo_solver(m, solver=solver, tee=tee, options=solver_options)
    res = opt.solve(m, tee=tee)
    meta = _solver_metadata(opt, res, solver)

    theta_hat = np.zeros((d, K), dtype=float)
    for j in m.J:
        for f in m.F:
            theta_hat[int(j), int(f)] = float(pyo.value(m.theta[j, f]))

    W = np.array(
        [[float(pyo.value(m.w[t, j])) for j in m.J] for t in m.T],
        dtype=float,
    )
    MU = np.array([float(pyo.value(m.mu[t])) for t in m.T], dtype=float)
    LAM = np.array(
        [[float(pyo.value(m.lam[t, j])) for j in m.J] for t in m.T],
        dtype=float,
    )

    return theta_hat, W, MU, LAM, used_idx, meta


__all__ = ["fit_dfl_p1_flex_multi", "SolverMeta"]

