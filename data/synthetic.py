# ========================================
# File: data/synthetic.py
# Purpose: IPO論文 (2021) Simulation 1 準拠の人工データ生成
# ========================================

from __future__ import annotations
import numpy as np

def generate_simulation1_dataset(
    n_samples: int = 2000,
    n_assets: int = 10,
    snr: float = 0.5,
    rho: float = 0.5,
    sigma: float = 0.0125,   # 各資産の周辺標準偏差（例: 日次1.25% ≒ 年率20%）
    seed: int | None = 42,
    rng: np.random.Generator | None = None,
    jitter: float = 1e-12,   # V の数値安定化用の微小対角加算
) -> tuple:
    """
    IPO (2021) Simulation 1 に準拠して、以下を生成する:
      X:  (N, d)  各行が説明変数 x^{(i)}
      Y:  (N, d)  各行が目的変数 y^{(i)}
      V_true: (d, d) 真の共分散行列 (Toeplitz: V_{jk} = σ^2 ρ^{|j-k|})
      theta_0: (d,) 真の係数ベクトル
      delta:  float   評価で使うリスクアバージョン（ここでは生成に未使用）
      (option) tau:  float   SNR から決めたスケール

    生成式（論文通り）:
      y^{(i)} = diag(x^{(i)}) θ_0 + τ ε^{(i)},   ε^{(i)} ~ N(0, V_true)
               = x^{(i)} ⊙ θ_0 + τ ε^{(i)}   （⊙ は要素ごとの積）

    引数:
      - snr:  Var(signal)/Var(noise) の目標値。
              signal = x ⊙ θ_0, noise = τ ε として、
              τ = sqrt( E[θ_0^2] / (snr * σ^2) ) でスケールを合わせる。
      - rho:  相関係数。±1ちょうどは特異になるため、内部で ±0.999 にクリップ。
      - sigma: 周辺標準偏差。σ=1 でもOK（その場合は τ を再計算するため SNR は保たれる）。
      - seed / rng: 再現性制御。rng を渡す場合は seed は無視。

    戻り値:
      return_tau=False: (X, Y, V_true, theta_0, delta)
      return_tau=True : (X, Y, V_true, theta_0, delta, tau)
    """
    # --- RNG 準備 ---
    if rng is None:
        rng = np.random.default_rng(seed)

    # --- 次元 ---
    N = int(n_samples)
    d = int(n_assets)

    # --- 真の係数 θ_0 と説明変数 X ---
    theta_0 = rng.standard_normal(d)      # shape: (d,)
    X = rng.standard_normal((N, d))       # shape: (N, d)

    # --- 共分散 V_true (Toeplitz: V_{jk} = σ^2 ρ^{|j-k|}) ---
    rho = float(np.clip(rho, -0.999, 0.999))
    idx = np.arange(d)
    lags = np.abs(idx[:, None] - idx[None, :])
    V_true = (sigma**2) * (rho ** lags)

    # 数値安定化（正定値性の担保：微小な対角加算）
    if jitter and jitter > 0:
        V_true = V_true + jitter * np.eye(d)

    # --- τ を SNR から決める ---
    # Var(signal) ≈ E[θ_0^2]（x ~ N(0,1) のとき列平均で近似）
    tau = np.sqrt(np.mean(theta_0**2) / (snr * sigma**2))

    # --- ノイズ ε ~ N(0, V_true) を N 本生成 ---
    # multivariate_normal でもOK。数値安定の観点では Cholesky が明示的。
    try:
        L = np.linalg.cholesky(V_true)          # V = L L^T
        Z = rng.standard_normal((N, d))         # Z ~ N(0, I)
        eps = Z @ L.T                           # eps ~ N(0, V_true)
    except np.linalg.LinAlgError:
        # 万一失敗したら np.random.Generator.multivariate_normal にフォールバック
        eps = rng.multivariate_normal(mean=np.zeros(d), cov=V_true, size=N)

    # --- 目的変数 Y を生成: Y = X ⊙ θ_0 + τ ε ---
    # Y = X * theta_0
    Y = X * theta_0 + tau * eps                # shape: (N, d)

    return X, Y, V_true, theta_0, tau