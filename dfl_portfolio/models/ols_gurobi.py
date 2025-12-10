# ========================================
# File: models/mvo_gurobi.py
# Purpose: 非負制約つき MVO を Gurobi で解く（単時点/複数時点） OLS用を想定
# ========================================

from __future__ import annotations

from typing import List, Sequence, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_mvo_gurobi(
    y_hat: np.ndarray,
    V_hat: np.ndarray,
    delta: float = 1.0,
    psd_eps: float = 1e-12,
    output: bool = False,
) -> Optional[np.ndarray]:
    """
    Solve:
        min_z  - (1-δ) y_hat^T z + (δ/2) z^T V_hat z
        s.t.   1^T z = 1 (optional)
               z >= 0  (optional)

    Parameters
    ----------
    y_hat : (d,) ndarray
        時点 i における予測リターンベクトル
    V_hat : (d,d) ndarray
        推定共分散行列（できれば対称・（半）正定値）
    delta : float
        リスク回避パラメータ（論文では 1 が多い）
    psd_eps : float
        数値安定化のために V_hat の対角へ加える微小リッジ
    output : bool
        Gurobiのログ出力（デフォルトFalseで非表示）

    Returns
    -------
    z_opt : (d,) ndarray or None
        最適解。最適化に失敗したら None
    """
    # --------- 入力整形 & チェック ---------
    y_hat = np.asarray(y_hat, dtype=float).ravel()
    V_hat = np.asarray(V_hat, dtype=float)
    d = y_hat.shape[0]
    assert V_hat.shape == (d, d), "V_hat shape must be (d,d)"

    # δ=0 のときは目的が線形
    #   min_z  -y_hat^T z  s.t. 1^T z = 1, z>=0
    # に退化し、最適解は「期待リターン最大の資産にフルベット」になる。
    # この場合は Gurobi を呼ばず、解析解をそのまま返す。
    if delta == 0.0:
        if d == 0:
            return np.zeros(0, dtype=float)
        j = int(np.nanargmax(y_hat))
        z_star = np.zeros(d, dtype=float)
        if 0 <= j < d:
            z_star[j] = 1.0
        return z_star

    # 対称化 + PSD化（微小リッジ）
    V_hat = 0.5 * (V_hat + V_hat.T) + psd_eps * np.eye(d)

    # --------- モデル構築 ---------
    m = gp.Model()
    m.Params.OutputFlag = 1 if output else 0

    z = m.addMVar(d, lb=0, name="z")

    # 目的関数: -(1-δ) y^T z + (δ/2) z^T V z
    m.setObjective(0.5 * delta * (z @ V_hat @ z) - (1.0 - delta) * (y_hat @ z), GRB.MINIMIZE)

    m.addConstr(z.sum() == 1.0, name="budget")

    # 最適化
    m.optimize()

    if m.status == GRB.OPTIMAL:
        return z.X  # numpy配列
    else:
        # 必要なら m.status を見て分岐（INFEASIBLE, INF_OR_UNBD など）
        print(f"Optimization failed with status {m.status}")
        return None


def solve_series_mvo_gurobi(
    Yhat_all: np.ndarray,
    Vhats: Sequence[np.ndarray],
    idx: Sequence[int],
    delta: float = 1.0,
    psd_eps: float = 1e-12,
    output: bool = False,
    start_index: int | None = None,
) -> np.ndarray:
    """
    複数時点のMVOをまとめて解く。
    - idx[k] に対応する時点 i で y_hat = Yhat_all[i], V_hat = Vhats[k] を使う
    - start_index があれば、i >= start_index の時点のみ解く
    - 失敗があれば ValueError を投げる
    - 一件も解かない場合は shape (0, d) の空配列を返す
    """
    # --- 入力をnumpy配列に整形 ---
    Yhat_all = np.asarray(Yhat_all, dtype=float)
    N, d = Yhat_all.shape

    # --- (idx, Vhats) をペアで扱う（対応ずれ防止） ---
    pairs = list(zip(idx, Vhats))  # 例: [(i1, V1), (i2, V2), ...]
    if start_index is not None:
        pairs = [(i, V) for (i, V) in pairs if i >= start_index]

    # フィルタ後に何も残らなければ空行列を返す
    if len(pairs) == 0:
        return np.empty((0, d))

    # --- 1件ずつMVOを解く ---
    Z_list = []
    for i, V_i in pairs:
        y_i = Yhat_all[i]  # shape: (d,)
        z_i = solve_mvo_gurobi(
            y_hat=y_i,
            V_hat=np.asarray(V_i, dtype=float),
            delta=delta,
            psd_eps=psd_eps,
            output=output,
        )
        if z_i is None:
            print(f"Failed to solve MVO for index {i}. Returning NaN.")
            Z_list.append(np.full(d, np.nan))
        else:
            Z_list.append(z_i)

    return np.vstack(Z_list)
