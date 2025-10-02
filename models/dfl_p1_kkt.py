# ~/VScode/GraduationResearch/DFL_Portfolio_Optimization2/models/dfl_p1_kkt.py

import numpy as np
import pyomo.environ as pyo
from pyomo.mpec import Complementarity, complements  # ← 使わないが互換のため残す
from typing import Sequence, Optional
from pyomo.environ import *

# ★ 追加：共通ソルバ工場
from optimization.solvers import make_pyomo_solver


def _solver_metadata(res, solver_name: str) -> dict:
    solver_info = getattr(res, "solver", None)
    term = getattr(solver_info, "termination_condition", None) if solver_info else None
    status = getattr(solver_info, "status", None) if solver_info else None
    solver_time = getattr(solver_info, "time", None) if solver_info else None
    message = getattr(solver_info, "message", None) if solver_info else None
    return {
        "solver": solver_name,
        "termination_condition": term,
        "termination_condition_str": str(term) if term is not None else None,
        "status": status,
        "status_str": str(status) if status is not None else None,
        "solver_time": solver_time,
        "message": message,
    }

def fit_dfl_p1_pyomo_ipopt(
    X, Y, Vhats, idx,
    start_index=None, end_index=None,
    delta=1.0,
    theta_init=None, tee=False,
    # ★ 追加：外部からソルバとオプションを渡せるように
    solver: str = "gurobi",
    solver_options: dict | None = None,
    reg_theta_l2: float = 0.0,
):
    """
    ============================================================
    DFL-P1-KKT（相補性条件ベース）を QCQP として解く実装（Pyomo）
    ============================================================

    ■ 問題の全体像
    - 下位（投資家の意思決定）QP：
        min_w  [- w^T yhat(θ) + (δ/2) w^T V w]
        s.t.    1^T w = 1,   w >= 0
      ここで yhat_i(θ) = diag(x_i) θ（＝資産 j の予測は x_{ij} * θ_j）

    - KKT：
        予算： 1^T w = 1
        非負： w >= 0, λ >= 0
        相補： w ⊙ λ = 0
        停留： δ V w - yhat(θ) - μ 1 - λ = 0

    - 上位目的（学習の損失）：
        R(θ) = 平均_t [ - y_true_t^T w_t + (δ/2) w_t^T V_t w_t ]
    """

    # ---------- 0) 前処理 ----------
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    d = X.shape[1]

    # ---------- 1) 学習対象インデックス抽出 ----------
    pairs = [(i, V) for i, V in zip(idx, Vhats)
             if (start_index is None or i >= start_index)
             and (end_index   is None or i <= end_index)]
    if not pairs:
        raise ValueError("no training indices in the specified range")

    used_idx = [i for i, _ in pairs]
    V_list = [0.5*(np.asarray(V, float) + np.asarray(V, float).T) for _, V in pairs]
    T = len(used_idx)

    # ---------- 2) Pyomo モデル ----------
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T-1)
    m.J = pyo.RangeSet(0, d-1)

    # --- 定数パラメータ ---
    def x_ini(m, t, j): return float(X[used_idx[t], j])
    def y_ini(m, t, j): return float(Y[used_idx[t], j])
    def V_ini(m, t, j, k): return float(V_list[t][j, k])

    m.delta = pyo.Param(initialize=float(delta))
    m.x     = pyo.Param(m.T, m.J, initialize=x_ini)
    m.y     = pyo.Param(m.T, m.J, initialize=y_ini)
    m.V     = pyo.Param(m.T, m.J, m.J, initialize=V_ini)

    # --- 変数 ---
    m.theta = pyo.Var(m.J, bounds=(-10, 10))
    m.z     = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
    m.lam   = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
    m.mu    = pyo.Var(m.T)

    # --- 初期値 ---
    if theta_init is not None:
        for j in m.J:
            m.theta[j].value = float(theta_init[int(j)])
    else:
        for j in m.J:
            m.theta[j].value = 0.0
    for t in m.T:
        for j in m.J:
            m.z[t, j].value   = 1.0 / d
            m.lam[t, j].value = 0.0
        m.mu[t].value = 0.0

    # ---------- 3) 制約 ----------
    # (1) 予算： sum_j z[t,j] = 1
    def budget_rule(m, t):
        return sum(m.z[t, j] for j in m.J) == 1.0
    m.budget = pyo.Constraint(m.T, rule=budget_rule)

    # (2) 停留条件： δ*(V_t z_t)_j - yhat_{t,j}(θ) - μ_t - λ_{t,j} = 0
    def kkt_stationarity(m, t, j):
        yhat_tj = m.x[t, j] * m.theta[j]                     # diag(x_t)θ → x_{t,j} θ_j
        quad_j  = sum(m.V[t, j, k] * m.z[t, k] for k in m.J) # (V z)_j
        return m.delta * quad_j - yhat_tj - m.mu[t] - m.lam[t, j] == 0
    m.kkt = pyo.Constraint(m.T, m.J, rule=kkt_stationarity)

    # (3) 相補性： z_{t,j} * lam_{t,j} = 0（非凸QCQP）
    def comp_rule(m, t, j):
        return m.z[t, j] * m.lam[t, j] == 0
    m.comp = pyo.Constraint(m.T, m.J, rule=comp_rule)

    # ---------- 4) 目的関数 ----------
    def obj_rule(m):
        cost = 0.0
        for t in m.T:
            lin  = sum(m.y[t, j] * m.z[t, j] for j in m.J)
            quad = sum(m.z[t, j] * sum(m.V[t, j, k] * m.z[t, k] for k in m.J) for j in m.J)
            cost += -lin + 0.5 * m.delta * quad
        reg = 0.5 * float(reg_theta_l2) * sum(m.theta[j]**2 for j in m.J)  # ★追加
        return cost / float(T) + reg
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ---------- 5) ソルバ（共通ファクトリ経由） ----------
    opt = make_pyomo_solver(m, solver=solver, tee=tee, options=solver_options)
    res = opt.solve(m, tee=tee)
    meta = _solver_metadata(res, solver)

    # ---------- 6) 解の取り出し ----------
    theta_hat = np.array([pyo.value(m.theta[j]) for j in m.J])
    Z  = np.array([[pyo.value(m.z[t, j])   for j in m.J] for t in m.T])
    MU = np.array([pyo.value(m.mu[t])      for t in m.T])
    LAM= np.array([[pyo.value(m.lam[t, j]) for j in m.J] for t in m.T])

    return theta_hat, Z, MU, LAM, used_idx, meta
