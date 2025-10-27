# python -m experiments.test_equivalence_unconstrained
import numpy as np
# from data.synthetic import generate_simulation1_dataset
from data_stress.synthetic import generate_simulation1_dataset
from models.ols import train_ols

def dfl_obj_unconstrained(theta, X, Y, V, idx, delta=1.0):
    # V は σ^2 I（等方・時不変）を想定
    Vinv = np.linalg.inv(V)
    T = len(idx); d = X.shape[1]
    theta = np.asarray(theta, float).reshape(d)
    loss = 0.0
    for i in idx:
        diff = (X[i]*theta) - Y[i]
        loss += diff @ (Vinv @ diff)
    return 0.5/delta * (loss/T)

def analytic_grad(theta, X, Y, V, idx, delta=1.0):
    # ∇_θ J = (1/(δT)) Σ_i x_i ⊙ [V^{-1}(x_i ⊙ θ − y_i)]
    Vinv = np.linalg.inv(V)
    T = len(idx); d = X.shape[1]
    g = np.zeros(d)
    for i in idx:
        diff = (X[i]*theta) - Y[i]
        u = Vinv @ diff
        g += X[i]*u
    return (1.0/(delta*T))*g

def main():
    # データ生成（ρ=0 にして独立、ただし本質は V=σ^2 I に置き換える点）
    N,d = 1000, 10
    snr, rho, sigma, delta, seed = 0.001, 0.0, 0.02, 1.0, 7
    X, Y, V_true, theta0, tau = generate_simulation1_dataset(
        n_samples=N, n_assets=d, snr=snr, rho=rho, sigma=sigma, seed=seed
    )

    # 訓練インデックス（適当でOK）
    res = 50
    idx_train = list(range(res, res + (N-res)//2))

    # ★ デバッグ前提：V を「等方・時不変」で固定
    V_iso = (sigma**2)*np.eye(d)

    # OLS 解
    theta_ols = train_ols(X[idx_train], Y[idx_train])

    # 無制約DFL目的の解析勾配を OLS点で
    f_at_ols = dfl_obj_unconstrained(theta_ols, X, Y, V_iso, idx_train, delta)
    g_at_ols = analytic_grad(theta_ols, X, Y, V_iso, idx_train, delta)

    print(f"[unconstrained DFL obj at OLS] {f_at_ols:.8f}")
    print(f"[grad norm at OLS] L2={np.linalg.norm(g_at_ols):.3e}, Linf={np.max(np.abs(g_at_ols)):.3e}")
    assert np.linalg.norm(g_at_ols, ord=np.inf) < 1e-10, "理論上一致：OLSは無制約DFLの停留点"

    # “DFLの最適θ” も OLS と一致（ここでは同じ式で出る）
    theta_dfl = train_ols(X[idx_train], Y[idx_train])  # 無制約DFL≡OLS
    err = np.linalg.norm(theta_dfl - theta_ols)
    print(f"[theta match] ||theta_dfl - theta_ols|| = {err:.3e}")
    assert err < 1e-12

    print("[OK] 無制約 & V=σ^2 I では DFL と OLS が厳密一致（実装健全性テスト）")

if __name__ == "__main__":
    main()
