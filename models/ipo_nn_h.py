# ========================================
# File: models/ipo_nn.py  (linear predictor version)
# ========================================

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Tuple, Dict, Any, List

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.optim as optim


# ---------- 数値安定の小道具 ----------
def _make_psd(V: np.ndarray, psd_eps: float) -> np.ndarray:
    V = 0.5 * (V + V.T)
    return V + psd_eps * np.eye(V.shape[0])

def _build_affine_decision_torch(V: torch.Tensor, delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    下位MVO（等式のみ）の閉形式:
      z*(yhat) = (1/δ) H yhat + c
    ここでは torch.Tensor で返す（バックプロパゲーション可）。
    """
    d = V.shape[0]
    M = torch.linalg.inv(V)                # V は PSD + リッジ済みを期待
    one = torch.ones(d, dtype=V.dtype, device=V.device)
    a = M @ one                            # (d,)
    b = torch.dot(one, a)                  # 1^T M 1 (scalar)
    H = (torch.eye(d, dtype=V.dtype, device=V.device) - torch.outer(a, one) / b) @ M
    c = a / b
    return H, c


# ---------- 予測器（線形・バイアスなし） ----------
class IPOPredictor(nn.Module):
    """
    線形予測器: yhat = W x
    - bias=False 固定
    - 恒等初期化（W=I）
    - hidden / num_layers は互換のため受け取るが未使用
    """
    def __init__(self, d: int, hidden: int = 64, num_layers: int = 2):
        super().__init__()
        self.net = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.net.weight.zero_()
            self.net.weight.copy_(torch.eye(d, dtype=self.net.weight.dtype, device=self.net.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d) -> yhat: (batch, d)
        return self.net(x)


# ---------- メイン：IPO-NN の学習 ----------
def fit_ipo_nn(
    X: np.ndarray,
    Y: np.ndarray,
    Vhats: Sequence[np.ndarray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    delta: float = 1.0,
    psd_eps: float = 1e-12,
    hidden: int = 32,          # ← 互換用（無視されます）
    num_layers: int = 2,       # ← 互換用（無視されます）
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 2000,
    batch_size: int = 64,
    seed: Optional[int] = 42,
    device: str = "cpu",
    tee: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], Dict[str, Any], np.ndarray]:
    """
    IPO-NN（線形予測器版）:
      - 予測器: yhat = W x（線形、biasなし、恒等初期化）
      - 下位問題: 1^T z = 1 の等式のみ MVO を閉形式で解き、IPO損失
                  L_i = (δ/2) z^T V_i z - y_i^T z を最小化
    返り値はランナー互換（θのみ使用想定）。θは学習後の yhat を OLS で射影して導出。
    """
    # ---- 準備（データ/範囲/対応） ----
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, d = X.shape
    assert Y.shape == (N, d), "X,Y shape mismatch"

    if start_index is None:
        s = 0
    else:
        s = max(0, int(start_index))
    if end_index is None:
        e = N - 1
    else:
        e = min(N - 1, int(end_index))
    if s > e:
        raise ValueError("fit_ipo_nn: invalid train range")

    pairs_np = [(int(i), _make_psd(np.asarray(V, float), psd_eps)) for i, V in zip(idx, Vhats) if s <= int(i) <= e]
    used_idx = [i for (i, _) in pairs_np]
    if len(pairs_np) == 0:
        # 何も学習しない
        theta_hat = np.zeros(d)
        Z = np.empty((0, d)); MU = np.empty((0,)); LAM = np.empty((0, d))
        meta = {"solver": "ipo_nn", "status": "skipped", "message": "no pairs in range"}
        return theta_hat, Z, MU, LAM, [], meta, np.zeros_like(X)

    # ---- Torch 変換 ----
    torch.manual_seed(seed if seed is not None else 0)
    device = torch.device(device)
    X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    Y_t = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

    # Vは時点ごとに異なるので辞書管理（インデックス -> Tensor）
    V_dict: Dict[int, torch.Tensor] = {i: torch.from_numpy(V).to(device=device, dtype=torch.float32) for (i, V) in pairs_np}

    # 学習サンプルのインデックス配列（シャッフル用）
    train_ids = torch.tensor(used_idx, device=device, dtype=torch.long)

    # ---- モデル/最適化器 ----
    model = IPOPredictor(d=d, hidden=hidden, num_layers=num_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- 学習ループ ----
    model.train()
    n_train = train_ids.shape[0]

    for epoch in range(max_epochs):
        # シャッフル
        perm = train_ids[torch.randperm(n_train, device=device)]
        # ミニバッチ
        for b in range(0, n_train, batch_size):
            batch_idx = perm[b:b+batch_size]
            xb = X_t[batch_idx]   # (B, d)
            yb = Y_t[batch_idx]   # (B, d)

            # 線形予測
            yhat_b = model(xb)    # (B, d)

            # 下位MVOの閉形式（等式のみ）をサンプルごとに解いて IPO 損失
            # L = mean_i [ (δ/2) z^T V z - y^T z ]
            loss_terms = []
            for k in range(batch_idx.shape[0]):
                i = int(batch_idx[k].item())
                V_i = V_dict[i]                 # (d,d)
                H, c = _build_affine_decision_torch(V_i, delta=delta)
                z_i = (1.0/delta) * (H @ yhat_b[k]) + c  # (d,)

                # IPO損失
                quad = 0.5 * delta * (z_i @ (V_i @ z_i))
                lin  = torch.dot(yb[k], z_i)
                loss_terms.append(quad - lin)

            loss = torch.stack(loss_terms).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if tee and (epoch % max(1, (max_epochs // 10)) == 0 or epoch == max_epochs - 1):
            print(f"[IPO-NN(linear)] epoch {epoch+1}/{max_epochs}  loss={loss.item():.6e}")

    # ---- 学習後：予測を OLS で θ に射影して返す（既存 predict_yhat と整合） ----
    model.eval()
    with torch.no_grad():
        Yhat_all_t = model(X_t)          # (N, d)
    Yhat_all = Yhat_all_t.cpu().numpy()

    eps = 1e-12
    num = (X * Yhat_all).sum(axis=0)
    den = (X ** 2).sum(axis=0) + eps
    theta_hat = num / den

    if tee:
        approx = X * theta_hat
        rmse = np.sqrt(np.mean((approx[s:e+1] - Yhat_all[s:e+1])**2))
        print(f"[IPO-NN(linear)] post-hoc θ projection RMSE (train window) = {rmse:.6e}")

    # ---- 返り値（ランナー互換） ----
    Z   = np.empty((0, d))
    MU  = np.empty((0,))
    LAM = np.empty((0, d))
    meta = {
        "solver": "ipo_nn_linear",
        "status": "completed",
        "status_str": "completed",
        "termination_condition": None,
        "termination_condition_str": None,
        "solver_time": None,
        "message": None,
        "delta": delta,
        "psd_eps": psd_eps,
        "hidden": hidden,          # 互換のため残す（未使用）
        "num_layers": num_layers,  # 互換のため残す（未使用）
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "n_samples": len(used_idx),
        "returns_yhat": True,
        "predictor": "linear_no_bias_identity_init",
    }
    return theta_hat, Z, MU, LAM, used_idx, meta, Yhat_all