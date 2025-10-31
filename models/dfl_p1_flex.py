"""
============================================================
Unified DFL-P1 Solver (dual / KKT 切り替え + 正則化/DRO オプション)
============================================================

用途
----
DFL-P1（ベースラインの dual 版と KKT 版）に対して、
以下の拡張を単一の関数 `fit_dfl_p1_flex` で切り替えられるようにした:

* θ のアンカー正則化:          例) θ を OLS 解に寄せる  (lambda_theta_anchor)
* w のアンカー正則化:          例) w を OLS の w に寄せる (lambda_w_anchor)
* w の等方 L2^2 正則化:        例) ||w||_2^2 で安定化     (lambda_w_iso)
各 λ(ラグランジュ係数) や ρ を 0 に設定すれば、その機能は無効化される。

備考
----
* 既存コード (dfl_p1_dual.py, dfl_p1_dual_reg.py,
  dfl_p1_kkt.py) のロジックを統合したもの。
* `formulation='dual'` を選べば強双対形式、`'kkt'` を選べば KKT 制約(補完条件)を使用。
* 返り値は既存関数と揃えて `(theta_hat, W, MU, LAM, used_idx, meta)`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pyomo.environ as pyo

from optimization.solvers import make_pyomo_solver
from models.ols import train_ols, predict_yhat
from models.ols_gurobi import solve_series_mvo_gurobi


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


def _align_matrix(
    anchor: Optional[np.ndarray],
    d: int,
    used_idx: Sequence[int],
    full_idx: Sequence[int],
) -> Optional[np.ndarray]:
    if anchor is None:
        return None
    mat = np.asarray(anchor, float)
    if mat.ndim != 2 or mat.shape[1] != d:
        raise ValueError(f"anchor matrix must have shape (T, d); got {mat.shape}.")
    if mat.shape[0] == len(used_idx):
        return mat
    if mat.shape[0] == len(full_idx):
        lookup = {ii: k for k, ii in enumerate(full_idx)}
        rows = [lookup[i] for i in used_idx]
        return mat[rows, :]
    raise ValueError(
        "w_anchor must have rows equal to T_used or len(idx); "
        f"got shape {mat.shape} with T_used={len(used_idx)}"
    )


def fit_dfl_p1_flex(
    X: np.ndarray,
    Y: np.ndarray,
    Vhats: Sequence[np.ndarray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    formulation: str = "dual",   # "dual" or "kkt"
    delta: float = 1.0,
    theta_init: Optional[Sequence[float]] = None,
    solver: str = "gurobi",
    solver_options: Optional[dict] = None,
    tee: bool = False,
    # regularisation knobs
    lambda_theta_anchor: float = 0.0,
    theta_anchor: Optional[Sequence[float]] = None,
    lambda_theta_anchor_l1: float = 0.0,
    lambda_theta_iso: float = 0.0,
    lambda_w_anchor: float = 0.0,
    lambda_w_anchor_l1: float = 0.0,
    w_anchor: Optional[np.ndarray] = None,
    lambda_w_iso: float = 0.0,
):
    """
    DFL-P1 の柔軟な学習ルーチン。

    Parameters
    ----------
    formulation : {"dual","kkt"}
        dual  -> 強双対 + 非負制約 (従来の dual 実装に近い)
        kkt   -> 停留条件 + 相補条件 (従来の KKT 実装に近い)
    delta : float
        リスク回避パラメータ (MVO の δ)
    theta_init : array-like or None
        θ の初期値 (OLS 解など)。None の場合は 0 初期化。
    solver, solver_options, tee :
        Pyomo のソルバー指定。既存の make_pyomo_solver に委譲。

    lambda_theta_anchor, theta_anchor :
        θ をアンカーに寄せる L2^2 正則化。
        0 なら無効。θ_anchor (d 次元) が必要。
    lambda_theta_anchor_l1 :
        θ とアンカーの L1 正則化（非ゼロで補助変数を導入）。
    lambda_theta_iso :
        θ の等方 L2^2 正則化 (||θ||_2^2)。
    lambda_w_anchor, lambda_w_anchor_l1, w_anchor :
        w をアンカーに寄せる L2^2 / L1 正則化。
        w_anchor は (T_used×d) もしくは全 idx の長さに対応。
    lambda_w_iso :
        w の等方的 L2^2 正則化 (||w||_2^2)。安定化目的。

    Returns
    -------
    theta_hat : (d,)
    W         : (T_used, d)
    MU        : (T_used,)
    LAM       : (T_used, d)
    used_idx  : list[int]
    meta      : SolverMeta
    """

    formulation = formulation.lower()
    if formulation not in {"dual", "kkt"}:
        raise ValueError("formulation must be 'dual' or 'kkt'")

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    d = X.shape[1]

    used_idx, V_list = _prepare_pairs(idx, Vhats, start_index, end_index)
    T_used = len(used_idx)

    # --- θアンカー正則化の準備 ---
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

    # --- wアンカー正則化の準備 ---
    lam_w_l2 = float(lambda_w_anchor)
    lam_w_l1 = float(lambda_w_anchor_l1)
    w_anchor_mat = None
    if lam_w_l2 > 0.0 or lam_w_l1 > 0.0:
        w_anchor_mat = _align_matrix(w_anchor, d, used_idx, idx)

    # --- w 等方正則化 / DRO の準備 ---
    lam_w_iso = float(lambda_w_iso)
    # ---------------------------------------------------------
    # Build Pyomo model
    # ---------------------------------------------------------
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T_used - 1)
    m.J = pyo.RangeSet(0, d - 1)

    def x_ini(m, t, j): return float(X[used_idx[int(t)], int(j)])
    def y_ini(m, t, j): return float(Y[used_idx[int(t)], int(j)])
    def V_ini(m, t, j, k): return float(V_list[int(t)][int(j), int(k)])

    m.delta = pyo.Param(initialize=float(delta))
    m.x = pyo.Param(m.T, m.J, initialize=x_ini)
    m.y = pyo.Param(m.T, m.J, initialize=y_ini)
    m.V = pyo.Param(m.T, m.J, m.J, initialize=V_ini)
    m.theta = pyo.Var(m.J)
    m.w = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
    m.lam = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
    if lam_theta_l1 > 0.0 and theta_anchor_vec is not None:
        m.theta_dev = pyo.Var(m.J, domain=pyo.NonNegativeReals)
    if lam_w_l1 > 0.0 and w_anchor_mat is not None:
        m.w_dev = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
    m.mu = pyo.Var(m.T)

    theta_init_vec: Optional[np.ndarray] = None
    theta_init_source = "none"
    if theta_init is not None:
        if isinstance(theta_init, str):
            key = theta_init.lower().strip()
            if key == "ols":
                theta_init_vec = train_ols(X, Y)
                theta_init_source = "ols"
            else:
                raise ValueError(f"Unsupported theta_init option: {theta_init}")
        else:
            theta_init_vec = np.asarray(theta_init, float).reshape(-1)
            theta_init_source = "provided"

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
        return m.mu[t] + m.lam[t, j] == m.delta * quad_j - yhat

    def strong_duality(m, t):
        quad = sum(m.w[t, j] * sum(m.V[t, j, k] * m.w[t, k] for k in m.J) for j in m.J)
        yhat_dot_w = sum(m.x[t, j] * m.theta[j] * m.w[t, j] for j in m.J)
        return m.delta * quad - yhat_dot_w == m.mu[t]

    def stationarity(m, t, j):
        yhat = m.x[t, j] * m.theta[j]
        quad_j = sum(m.V[t, j, k] * m.w[t, k] for k in m.J)
        return m.delta * quad_j - yhat - m.mu[t] - m.lam[t, j] == 0.0

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
            total += -lin + 0.5 * m.delta * quad
        total = total / float(T_used)

        if lam_theta_l2 > 0.0 and theta_anchor_vec is not None:
            total += 0.5 * lam_theta_l2 * sum((m.theta[j] - float(theta_anchor_vec[int(j)])) ** 2 for j in m.J)
        if lam_theta_l1 > 0.0 and theta_anchor_vec is not None:
            total += lam_theta_l1 * sum(m.theta_dev[j] for j in m.J)

        if lam_theta_iso > 0.0:
            total += 0.5 * lam_theta_iso * sum(m.theta[j] ** 2 for j in m.J)

        if lam_w_l2 > 0.0 and w_anchor_mat is not None:
            total += (lam_w_l2 / (2.0 * float(T_used))) * sum(
                sum((m.w[t, j] - float(w_anchor_mat[int(t), int(j)])) ** 2 for j in m.J)
                for t in m.T
            )
        if lam_w_l1 > 0.0 and w_anchor_mat is not None:
            total += (lam_w_l1 / float(T_used)) * sum(
                sum(m.w_dev[t, j] for j in m.J)
                for t in m.T
            )

        if lam_w_iso > 0.0:
            total += (lam_w_iso / float(T_used)) * sum(
                sum(m.w[t, j] ** 2 for j in m.J) for t in m.T
            )

        return total

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    if lam_theta_l1 > 0.0 and theta_anchor_vec is not None:
        def theta_dev_pos(m, j):
            return m.theta_dev[j] >= m.theta[j] - float(theta_anchor_vec[int(j)])

        def theta_dev_neg(m, j):
            return m.theta_dev[j] >= -(m.theta[j] - float(theta_anchor_vec[int(j)]))

        m.theta_dev_pos = pyo.Constraint(m.J, rule=theta_dev_pos)
        m.theta_dev_neg = pyo.Constraint(m.J, rule=theta_dev_neg)

    if lam_w_l1 > 0.0 and w_anchor_mat is not None:
        def w_dev_pos(m, t, j):
            return m.w_dev[t, j] >= m.w[t, j] - float(w_anchor_mat[int(t), int(j)])

        def w_dev_neg(m, t, j):
            return m.w_dev[t, j] >= -(m.w[t, j] - float(w_anchor_mat[int(t), int(j)]))

        m.w_dev_pos = pyo.Constraint(m.T, m.J, rule=w_dev_pos)
        m.w_dev_neg = pyo.Constraint(m.T, m.J, rule=w_dev_neg)

    opt = make_pyomo_solver(m, solver=solver, tee=tee, options=solver_options)
    res = opt.solve(m, tee=tee)
    meta = _solver_metadata(opt, res, solver)

    theta_hat = np.array([pyo.value(m.theta[j]) for j in m.J], dtype=float)
    W = np.array([[pyo.value(m.w[t, j]) for j in m.J] for t in m.T], dtype=float)
    MU = np.array([pyo.value(m.mu[t]) for t in m.T], dtype=float)
    LAM = np.array([[pyo.value(m.lam[t, j]) for j in m.J] for t in m.T], dtype=float)

    return theta_hat, W, MU, LAM, used_idx, meta
