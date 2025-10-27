# -*- coding: utf-8 -*-
"""
models/ipo_nn_qp.py

厳密NN比較手法 (IPO-NN-QP):
- 下位QPを微分可能に解き、IPO損失で学習
- 等式制約: 1^T z = 1（投資比率の総和1）
- 不等式制約: z >= 0（空売り禁止）
- 共分散は呼び出し側から与える（人工データなら真のVをそのまま渡せる）

※ 既存ランナー互換のI/Oに合わせた関数 `fit_ipo_nn_qp(...)` を提供します。
  入出力シグネチャはユーザー提示のコードと同じで、内部実装は
  先に作成したNN+QPレイヤ方式（cvxpylayers）を用いています。
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


# ============================================================
# QP layer (cvxpylayers):
#   min_z  0.5*||L z||^2 - yhat^T z   s.t.  sum(z)=1,  z>=0 (box拡張可)
#   ここで L は delta*V のCholesky等の因子（PSD安定化は呼び出し側で実施）
# ============================================================
def build_qp_layer(d: int, box: Tuple[float, float] | None = (0.0, None)) -> CvxpyLayer:
    """Create a differentiable QP layer using a factor param L and linear term yhat.

    Objective: 0.5 * ||L z||^2 - yhat^T z
    Constraints: sum(z) == 1, and (optional) box bounds lb <= z <= ub
    """
    z = cp.Variable(d)
    L = cp.Parameter((d, d))  # factor such that (L^T L) ≈ delta * V
    yh = cp.Parameter(d)

    obj = 0.5 * cp.sum_squares(L @ z) - yh @ z
    cons = [cp.sum(z) == 1]
    if box is not None:
        lb, ub = box
        if lb is not None:
            cons += [z >= lb]
        if ub is not None:
            cons += [z <= ub]

    prob = cp.Problem(cp.Minimize(obj), cons)
    return CvxpyLayer(prob, parameters=[L, yh], variables=[z])


# ============================================================
# Predictor: x -> yhat（論文想定の線形モデル）
# ============================================================
class IPOPredictor(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        # 論文の一般ケース最小構成に合わせ、線形・バイアス無し
        self.net = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# メイン学習関数（既存ランナー互換I/O）
# ============================================================
def fit_ipo_nn_qp(
    X: np.ndarray,
    Y: np.ndarray,
    Vhats: Sequence[np.ndarray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    delta: float = 1.0,
    box: Tuple[float, float] | None = (0.0, None),  # 非負なら (0, None)
    psd_eps: float = 1e-8,
    theta_init: np.ndarray | str | None = "identity",
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 2000,
    batch_size: int = 64,
    seed: Optional[int] = 42,
    device: str = "cpu",
    tee: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], Dict[str, Any], np.ndarray]:
    """
    厳密NN比較手法（IPO-NN-QP）。
    - 予測器: 線形 nn.Linear(d,d,bias=False)
    - 下位: cvxpylayers で微分可能QP (sum(z)=1, z>=0, box拡張可)
    - 損失: 真の (Y_i, V_i) で MVO コストを評価

    Parameters
    ----------
    X : (N,d) 特徴
    Y : (N,d) 真の（将来）リターン
    Vhats : len(idx) 個の推定共分散（人工データでは真のVを与えて良い）
    idx : 学習に使うサンプルのインデックス列
    start_index, end_index : 学習範囲（閉区間）
    delta : MVOの二次項係数
    box : 下限・上限制約（Noneで無制限）。非負制約は(0,None)
    psd_eps : 対称化・正定値化のための微小ダイアゴナル
    theta_init : 初期化（"ols" | "identity"/"none" | np.ndarray）
    lr, weight_decay, max_epochs, batch_size, seed, device, tee : 学習ハイパラ

    Returns (runner互換)
    -------
    theta_hat : (d,) 既存予測器インターフェース互換のダミーθ（post-hoc 射影）
    Z, MU, LAM : 互換用ダミー（空配列）
    used_idx : 学習に実際用いたインデックス
    meta : 学習メタ情報
    Yhat_all : (N,d) 学習後の yhat 予測全体
    """
    # ---- 入力整形 ----
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, d = X.shape
    assert Y.shape == (N, d)

    s = 0 if start_index is None else max(0, int(start_index))
    e = N - 1 if end_index is None else min(N - 1, int(end_index))
    if s > e:
        theta_hat = np.zeros(d)
        Z = np.empty((0, d)); MU = np.empty((0,)); LAM = np.empty((0, d))
        meta = {"solver": "ipo_nn_qp", "status": "skipped", "message": "invalid train range"}
        return theta_hat, Z, MU, LAM, [], meta, np.zeros_like(X)

    # 学習対象 (i, V_i)
    pairs_np: List[Tuple[int, np.ndarray]] = []
    for i, V in zip(idx, Vhats):
        i = int(i)
        if s <= i <= e:
            V = np.asarray(V, float)
            V = 0.5 * (V + V.T) + psd_eps * np.eye(d)  # 対称化 + リッジ
            pairs_np.append((i, V))
    used_idx = [i for (i, _) in pairs_np]
    if len(pairs_np) == 0:
        theta_hat = np.zeros(d)
        Z = np.empty((0, d)); MU = np.empty((0,)); LAM = np.empty((0, d))
        meta = {"solver": "ipo_nn_qp", "status": "skipped", "message": "no pairs in range"}
        return theta_hat, Z, MU, LAM, [], meta, np.zeros_like(X)

    # ---- Torch 準備 ----
    torch.manual_seed(seed if seed is not None else 0)
    dev = torch.device(device)
    dtype = torch.float64

    X_t = torch.from_numpy(X).to(dev, dtype=dtype)
    Y_t = torch.from_numpy(Y).to(dev, dtype=dtype)

    # 各サンプル i の因子 L_i を用意（delta*V のCholesky）
    V_dict: Dict[int, torch.Tensor] = {}
    L_dict: Dict[int, torch.Tensor] = {}
    for i, V in pairs_np:
        V_psd = 0.5 * (V + V.T) + psd_eps * np.eye(d)
        L_i = np.linalg.cholesky(delta * V_psd)
        V_dict[i] = torch.from_numpy(V_psd).to(dev, dtype=dtype)
        L_dict[i] = torch.from_numpy(L_i).to(dev, dtype=dtype)
    train_ids = torch.tensor(used_idx, device=dev, dtype=torch.long)

    # QPレイヤを構築（d固定なので1回だけ）
    qp_layer = build_qp_layer(d=d, box=box)

    # 予測器
    model = IPOPredictor(d=d).to(dev).to(dtype)

    # 初期化（論文では特に初期解の指定なし → 恒等行列で十分）
    theta_init_np: Optional[np.ndarray] = None
    theta_init_meta = "identity"
    if theta_init is not None:
        if isinstance(theta_init, str):
            key = theta_init.lower()
            if key in {"identity", "eye", "none", "ols"}:  # 'ols' は使わず恒等にフォールバック
                theta_init_np = None
                theta_init_meta = "identity" if key != "ols" else "ignored_ols_fallback_identity"
            else:
                raise ValueError(f"Unknown theta_init option: {theta_init}")
        else:
            theta_init_np = np.asarray(theta_init, dtype=float)
            if theta_init_np.shape not in {(d,), (d, d)}:
                raise ValueError(f"theta_init must have shape ({d},) or ({d}, {d}); got {theta_init_np.shape}")
            theta_init_meta = "provided"

    with torch.no_grad():
        W = model.net.weight  # (d,d)
        W.zero_()
        if theta_init_np is not None:
            if theta_init_np.shape == (d,):
                diag = torch.from_numpy(theta_init_np).to(W)
                W.copy_(torch.diag(diag))
            else:  # (d,d)
                W.copy_(torch.from_numpy(theta_init_np).to(W))
        else:
            W.copy_(torch.eye(d, dtype=W.dtype, device=W.device))

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- 学習ループ ----
    model.train()
    n = train_ids.shape[0]
    for epoch in range(max_epochs):
        perm = train_ids[torch.randperm(n, device=dev)]
        for b in range(0, n, batch_size):
            batch = perm[b:b+batch_size]
            xb = X_t[batch]   # (B, d)
            yb = Y_t[batch]   # (B, d)
            yhat_b = model(xb)  # (B, d)

            # QPを各サンプルに対して解く（cvxpylayersはバッチ引数OK）
            batch_idx = [int(i) for i in batch.tolist()]
            L_batch = torch.stack([L_dict[i] for i in batch_idx], dim=0)  # (B,d,d)
            (z_b,) = qp_layer(L_batch, yhat_b)  # (B,d)

            # 損失: 真の(Y, V)で MVO コスト
            V_batch = torch.stack([V_dict[i] for i in batch_idx], dim=0)  # (B,d,d)
            Vz = torch.matmul(V_batch, z_b.unsqueeze(-1)).squeeze(-1)     # (B,d)
            quad = 0.5 * torch.sum(z_b * Vz, dim=1) * delta               # (B,)
            lin = torch.sum(yb * z_b, dim=1)                              # (B,)
            loss = (quad - lin).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()

        if tee and (epoch % max(1, (max_epochs // 10)) == 0 or epoch == max_epochs - 1):
            print(f"[IPO-NN-QP] epoch {epoch+1}/{max_epochs} loss={loss.item():.6e}")

    # ---- 予測出力・互換θの作成 ----
    model.eval()
    with torch.no_grad():
        Yhat_all = model(X_t).cpu().numpy()  # (N,d)

    # ランナー互換の θ：既存 predict_yhat を流用する都合の post-hoc 射影
    eps = 1e-12
    num = (X * Yhat_all).sum(axis=0)
    den = (X**2).sum(axis=0) + eps
    theta_hat = num / den  # (d,)

    # ランナー互換のダミー出力
    Z = np.empty((0, d)); MU = np.empty((0,)); LAM = np.empty((0, d))
    meta: Dict[str, Any] = {
        "solver": "ipo_nn_qp",
        "status": "completed",
        "status_str": "completed",
        "delta": delta,
        "box": box,
        "psd_eps": psd_eps,
        "model_arch": "linear_no_bias",
        "theta_init": theta_init_meta,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(dev),
        "n_samples": len(used_idx),
        "returns_yhat": True,
    }
    return theta_hat, Z, MU, LAM, used_idx, meta, Yhat_all
