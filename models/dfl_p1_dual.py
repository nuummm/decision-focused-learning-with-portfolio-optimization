# # ~/VScode/GraduationResearch/DFL_Portfolio_Optimization2/models/dfl_p1_dual.py

# import numpy as np
# import pyomo.environ as pyo
# from typing import Sequence, Optional

# # ★ 追加：共通ソルバ工場
# from optimization.solvers import make_pyomo_solver

# def fit_dfl_p1_dual_pyomo(
#     X, Y, Vhats, idx,
#     start_index=None, end_index=None,
#     delta=1.0,
#     theta_init=None, tee=False,
#     # ★ 追加：外部からソルバとオプションを渡せるように
#     solver: str = "gurobi",
#     solver_options: dict | None = None,
#     reg_theta_l2: float = 0.0,
# ):
#     """
#     ============================================================
#     DFL-P1-DUAL（強双対条件ベース）の実装（Pyomo）
#     ============================================================

#     - 下位QP：
#         min_w  [- w^T rhat(θ) + (δ/2) w^T V w]
#         s.t.    1^T w = 1,   w >= 0

#     - DUAL条件：
#         μ 1 + λ = δ V w - rhat(θ)
#         w >= 0, λ >= 0
#         強双対： δ w^T V w - w^T rhat(θ) ≤ μ
#     """

#     # ---------- 0) 前処理 ----------
#     X = np.asarray(X, float)
#     Y = np.asarray(Y, float)
#     d = X.shape[1]

#     # ---------- 1) 学習対象 ----------
#     pairs = [(i, V) for i, V in zip(idx, Vhats)
#              if (start_index is None or i >= start_index)
#              and (end_index   is None or i <= end_index)]
#     if not pairs:
#         raise ValueError("no training indices in the specified range")

#     used_idx = [i for i, _ in pairs]
#     V_list = [0.5*(np.asarray(V, float) + np.asarray(V, float).T) for _, V in pairs]
#     T = len(used_idx)

#     # ---------- 2) モデル ----------
#     m = pyo.ConcreteModel()
#     m.T = pyo.RangeSet(0, T-1)
#     m.J = pyo.RangeSet(0, d-1)

#     # --- 定数 ---
#     def x_ini(m, t, j): return float(X[used_idx[t], j])
#     def y_ini(m, t, j): return float(Y[used_idx[t], j])
#     def V_ini(m, t, j, k): return float(V_list[t][j, k])

#     m.delta = pyo.Param(initialize=float(delta))
#     m.x     = pyo.Param(m.T, m.J, initialize=x_ini)
#     m.y     = pyo.Param(m.T, m.J, initialize=y_ini)
#     m.V     = pyo.Param(m.T, m.J, m.J, initialize=V_ini)

#     # --- 変数 ---
#     m.theta = pyo.Var(m.J, bounds=(-10, 10))
#     m.z     = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
#     m.lam   = pyo.Var(m.T, m.J, domain=pyo.NonNegativeReals)
#     m.mu    = pyo.Var(m.T)

#     # --- 初期値 ---
#     if theta_init is not None:
#         for j in m.J:
#             m.theta[j].value = float(theta_init[int(j)])
#     else:
#         for j in m.J:
#             m.theta[j].value = 0.0
#     for t in m.T:
#         for j in m.J:
#             m.z[t, j].value   = 1.0 / d
#             m.lam[t, j].value = 0.0
#         m.mu[t].value = 0.0

#     # ---------- 3) 制約 ----------
#     # (1) 予算
#     def budget_rule(m, t):
#         return sum(m.z[t, j] for j in m.J) == 1.0
#     m.budget = pyo.Constraint(m.T, rule=budget_rule)

#     # (2) 双対リンク： μ 1 + λ = δ V w - rhat(θ)
#     def dual_link(m, t, j):
#         yhat_tj = m.x[t, j] * m.theta[j]
#         quad_j  = sum(m.V[t, j, k] * m.z[t, k] for k in m.J)
#         return m.mu[t] + m.lam[t, j] == m.delta * quad_j - yhat_tj
#     m.dual_link = pyo.Constraint(m.T, m.J, rule=dual_link)

#     # (3) 強双対： δ w^T V w - w^T rhat(θ) ≤ μ
#     def strong_duality(m, t):
#         quad = sum(m.z[t, j] * sum(m.V[t, j, k] * m.z[t, k] for k in m.J) for j in m.J)
#         yhat_dot_z = sum(m.x[t, j] * m.theta[j] * m.z[t, j] for j in m.J)
#         return m.delta * quad - yhat_dot_z <= m.mu[t]
#     m.strong_duality = pyo.Constraint(m.T, rule=strong_duality)

#     # ---------- 4) 目的関数 ----------
#     def obj_rule(m):
#         cost = 0.0
#         for t in m.T:
#             lin  = sum(m.y[t, j] * m.z[t, j] for j in m.J)
#             quad = sum(m.z[t, j] * sum(m.V[t, j, k] * m.z[t, k] for k in m.J) for j in m.J)
#             cost += -lin + 0.5 * m.delta * quad
#         # ★ θのL2正則化（時間平均には割らない）
#         # reg = 0.5 * float(reg_theta_l2) * (sum(m.theta[j]**2 for j in m.J)/ float(T))
#         # z の L2 正則化（時間平均スケールに合わせて 1/T を掛ける）
#         reg = 0.5 * float(reg_theta_l2) * ( 
#             sum(sum(m.z[t, j]**2 for j in m.J) for t in m.T) / float(T)
#         )
#         return cost / float(T) + reg
#     m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

#     # ---------- 5) ソルバ（共通ファクトリ経由） ----------
#     opt = make_pyomo_solver(m, solver=solver, tee=tee, options=solver_options)
#     res = opt.solve(m, tee=tee)

#     # ---------- 6) 解の取り出し ----------
#     theta_hat = np.array([pyo.value(m.theta[j]) for j in m.J])
#     Z  = np.array([[pyo.value(m.z[t, j])   for j in m.J] for t in m.T])
#     MU = np.array([pyo.value(m.mu[t])      for t in m.T])
#     LAM= np.array([[pyo.value(m.lam[t, j]) for j in m.J] for t in m.T])

#     return theta_hat, Z, MU, LAM, used_idx

#下位問題の L2^2 正則化を w に対して適用するバージョン
import numpy as np
import pyomo.environ as pyo 
from typing import Sequence, Optional

# ★ 追加：共通ソルバ工場
from optimization.solvers import make_pyomo_solver


def _solver_metadata(res, solver_name: str) -> dict:
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
    return meta

def fit_dfl_p1_dual_pyomo(
    X, Y, Vhats, idx,
    start_index=None, end_index=None,
    delta=1.0,
    theta_init=None, tee=False,
    # ★ 追加：外部からソルバとオプションを渡せるように
    solver: str = "gurobi",
    solver_options: dict | None = None,
    reg_theta_l2: float = 0.0,  # ← 現状は w の L2^2 罰則として利用
):
    """
    ============================================================
    DFL-P1-DUAL（強双対条件ベース）の実装（Pyomo）
    ============================================================

    - 下位QP：
        min_w  [- w^T rhat(θ) + (δ/2) w^T V w]
        s.t.    1^T w = 1,   w >= 0

      ※ reg_theta_l2 > 0 のときは V に (2*reg_theta_l2/δ)I を加えて
         w の L2^2 罰則 (λ_w ||w||_2^2) を実現する。
    """

    # ---------- 0) 前処理 ----------
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    d = X.shape[1]

    # ---------- 1) 学習対象 ----------
    pairs = [(i, V) for i, V in zip(idx, Vhats)
             if (start_index is None or i >= start_index)
             and (end_index   is None or i <= end_index)]
    if not pairs:
        raise ValueError("no training indices in the specified range")

    used_idx = [i for i, _ in pairs]
    V_list = [0.5 * (np.asarray(V, float) + np.asarray(V, float).T) for _, V in pairs]
    T = len(used_idx)

    # --- w の L2^2 正則化（V に等方的ダンピングを付与） ---
    lambda_w = float(reg_theta_l2)
    if lambda_w < 0:
        raise ValueError("reg_theta_l2 must be non-negative when used as w regularizer.")
    if lambda_w > 0.0:
        if delta == 0.0:
            raise ValueError("delta must be non-zero to apply w regularization.")
        damping = (2.0 * lambda_w / float(delta))
        eye = np.eye(d)
        V_list = [V + damping * eye for V in V_list]

    # ---------- 2) モデル ----------
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T-1)
    m.J = pyo.RangeSet(0, d-1)

    # --- 定数 ---
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
    def budget_rule(m, t):
        return sum(m.z[t, j] for j in m.J) == 1.0
    m.budget = pyo.Constraint(m.T, rule=budget_rule)

    def dual_link(m, t, j):
        yhat_tj = m.x[t, j] * m.theta[j]
        quad_j  = sum(m.V[t, j, k] * m.z[t, k] for k in m.J)
        return m.mu[t] + m.lam[t, j] == m.delta * quad_j - yhat_tj
    m.dual_link = pyo.Constraint(m.T, m.J, rule=dual_link)

    def strong_duality(m, t):
        quad = sum(m.z[t, j] * sum(m.V[t, j, k] * m.z[t, k] for k in m.J) for j in m.J)
        yhat_dot_z = sum(m.x[t, j] * m.theta[j] * m.z[t, j] for j in m.J)
        return m.delta * quad - yhat_dot_z <= m.mu[t]
    m.strong_duality = pyo.Constraint(m.T, rule=strong_duality)

    # ---------- 4) 目的関数 ----------
    def obj_rule(m):
        cost = 0.0
        for t in m.T:
            lin  = sum(m.y[t, j] * m.z[t, j] for j in m.J)
            quad = sum(m.z[t, j] * sum(m.V[t, j, k] * m.z[t, k] for k in m.J) for j in m.J)
            cost += -lin + 0.5 * m.delta * quad
        # ★ θや z の L2 正則化は一旦オフ
        # reg_theta = 0.5 * float(reg_theta_l2) * (sum(m.theta[j]**2 for j in m.J) / float(T))
        # reg_z     = 0.5 * float(reg_theta_l2) * (
        #     sum(sum(m.z[t, j]**2 for j in m.J) for t in m.T) / float(T)
        # )
        return cost / float(T)
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
