# ========================================
# File: models/ipo_closed_form.py
# Purpose: 解析解（等式のみ or 無制約）の下位MVOを用いて
#          IPO目的の二次式を閉形式で最小化し、θを一発推定
# ========================================

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict, Any, List

import numpy as np

# ---- 小道具: 対称化 + PSD化（微小リッジ） ----
def _make_psd(V: np.ndarray, psd_eps: float) -> np.ndarray:
    V = 0.5 * (V + V.T)
    # 最小固有値で補正しても良いが、簡潔に対角へ微小値
    return V + psd_eps * np.eye(V.shape[0])

def _build_affine_decision(V: np.ndarray, delta: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    下位MVOの解析解 z*(y_hat) = (1/δ) H y_hat + c を返す。
    mode ∈ {"budget", "unconstrained"}:
      - "budget": min (δ/2) zᵀ V z - y_hatᵀ z  s.t. 1ᵀ z = 1
      - "unconstrained": 同目的で制約なし
    戻り値:
      H: (d,d), c: (d,)
    """
    d = V.shape[0]
    M = np.linalg.inv(V)  # Vは PSD + リッジ済みを期待
    if mode == "unconstrained":
        # z* = (1/δ) M y_hat
        H = M
        c = np.zeros(d)
    elif mode == "budget":
        # z* = (1/δ) H y_hat + c
        one = np.ones(d)
        a = M @ one                     # (d,)
        b = float(one @ a)              # 1ᵀ M 1
        # 導出：
        # z = (1/δ) M y_hat - (1/δ) a * (1ᵀ M y_hat - δ)/b
        #   = (1/δ)[ (I - a 1ᵀ / b) M y_hat ] + a / b
        H = (np.eye(d) - np.outer(a, np.ones(d)) / b) @ M
        c = a / b
    else:
        raise ValueError('mode must be "budget" or "unconstrained"')
    return H, c

def fit_ipo_closed_form(
    X: np.ndarray,
    Y: np.ndarray,
    Vhats: Sequence[np.ndarray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    delta: float = 1.0,
    mode: str = "budget",        # "budget"（1ᵀz=1）or "unconstrained"
    psd_eps: float = 1e-12,      # Vのリッジ
    ridge_theta: float = 1e-10,  # A への微小リッジ（θ安定化）
    tee: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
    """
    解析解ベースで θ を推定して返す“Trainer”。

    入力:
      X: (N, d) 説明変数
      Y: (N, d) 真のリターン
      Vhats: 各評価時点に対応する (d,d) 推定共分散の列
      idx: Vhats と対応する時点インデックスの列（例: [i1, i2, ...]）
      start_index, end_index: 学習区間（含む）
      delta, mode, psd_eps, ridge_theta: 上記説明
    出力（ランナー互換）:
      theta_hat, Z(empty), MU(empty), LAM(empty), used_idx, meta
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, d = X.shape
    assert Y.shape == (N, d), "X,Y のshape不一致"

    # ---- 学習範囲を決定 ----
    if start_index is None:
        s = 0
    else:
        s = max(0, int(start_index))
    if end_index is None:
        e = N - 1
    else:
        e = min(N - 1, int(end_index))
    if s > e:
        raise ValueError("fit_ipo_closed_form: invalid train range")

    # idx と Vhats は “評価対象時点”の組みで与えられる想定。
    # 学習対象は i ∈ [s, e] に絞る。
    pairs = [(int(i), np.asarray(V, float)) for i, V in zip(idx, Vhats) if s <= int(i) <= e]
    if len(pairs) == 0:
        # 何も学習しない場合はゼロ返し
        theta_hat = np.zeros(d)
        Z = np.empty((0, d))
        MU = np.empty((0,))
        LAM = np.empty((0, d))
        meta = {"solver": "ipo_closed_form", "status": "skipped", "message": "no pairs in range"}
        return theta_hat, Z, MU, LAM, [], meta

    # IPO 目的関数は mvo_cost(z,y,V; delta) = -(1-δ) zᵀ y + (δ/2) zᵀ V z を前提とする。
    # 一方、下位 MVO の解析解 _build_affine_decision は
    #   min_z  (δ_eff/2) zᵀ V z - y_hatᵀ z
    # に対する解を返す設計になっているため、ここで δ_eff = δ / (1-δ) に変換する。
    alpha = 1.0 - float(delta)
    if delta <= 0.0 or alpha <= 0.0:
        raise ValueError(f"delta must be in (0,1) for IPO closed-form; got {delta}")
    delta_eff = float(delta) / alpha

    # ---- A θ = b を構成（凸二次最小化の正規方程式）----
    A = np.zeros((d, d), dtype=float)
    b = np.zeros(d, dtype=float)

    # 新しい目的関数: -(1-δ) r^T z + (δ/2) z^T V z
    # で導出した係数のための係数 α = 1 - δ
    # alpha = 1 - δ は mvo_cost におけるリターン項の重み。
    # 上位の二次目的 L(θ) の係数は、この alpha と δ の組み合わせから導出される。

    for (i, V_i) in pairs:
        # V を安定化
        V_i = _make_psd(V_i, psd_eps)

        # 下位MVO: z*(y_hat) = (1/δ) H y_hat + c
        # ただし、mvo_cost に対しては δ_eff = δ / (1-δ) を使った Markowitz 目的と等価なので、
        # 解析解レイヤには δ_eff を渡す。
        H, c = _build_affine_decision(V_i, delta_eff, mode)

        # 便利な中間量
        # S = Hᵀ V H（対称, PSD）
        S = H.T @ V_i @ H
        # t = Hᵀ V c
        t = H.T @ (V_i @ c)

        # サンプル i の特徴/真値
        x_i = X[i]        # (d,)
        y_i = Y[i]        # (d,)
        D_i = np.diag(x_i)

        # 新しい目的関数に基づく係数:
        #   A_i = (α^2 / δ) D_i S D_i
        #   b_i = (α^2 / δ) D_i Hᵀ y_i - α D_i t
        coef_quad = (alpha ** 2) / float(delta)
        A += coef_quad * (D_i @ S @ D_i)
        b += coef_quad * (D_i @ (H.T @ y_i)) - alpha * (D_i @ t)

    # 微小リッジで可逆化
    if ridge_theta > 0:
        A = 0.5 * (A + A.T) + ridge_theta * np.eye(d)

    # 解く
    try:
        theta_hat = np.linalg.solve(A, b)
        status = "completed"
        msg = None
    except np.linalg.LinAlgError:
        theta_hat = np.linalg.lstsq(A, b, rcond=None)[0]
        status = "completed_lstsq"
        msg = "A was singular; used lstsq."

    if tee:
        print(f"[IPO Closed-Form] train range = [{s}, {e}], mode={mode}, θ shape={theta_hat.shape}")

    # ランナー互換のダミー出力
    Z   = np.empty((0, d))
    MU  = np.empty((0,))
    LAM = np.empty((0, d))
    used_idx = [i for (i, _) in pairs]
    meta = {
        "solver": "ipo_closed_form",
        "status": status,
        "status_str": status,
        "termination_condition": None,
        "termination_condition_str": None,
        "solver_time": None,
        "message": msg,
        "mode": mode,
        "delta": delta,
        "psd_eps": psd_eps,
        "ridge_theta": ridge_theta,
        "n_samples": len(pairs),
    }
    return theta_hat, Z, MU, LAM, used_idx, meta
