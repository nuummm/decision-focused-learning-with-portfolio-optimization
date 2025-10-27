"""
IPO-NN-QP の健康診断用ミニテスト。

既存の人工データ生成スクリプト（data/synthetic.py）を利用して
X, Y, 共分散行列 V を用意し、fit_ipo_nn_qp の基本的な挙動を確認します。
実行方法:

    cd GraduationResearch/DFL_Portfolio_Optimization2
    python experiments/sanity_ipo_nn_qp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cvxpy as cp

# リポジトリ直下で直接実行した際に models/ や data/ を import できるようにする
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic import generate_simulation1_dataset
from models.ipo_closed_form import fit_ipo_closed_form
from models.ipo_nn_qp import fit_ipo_nn_qp


def _load_reference_dataset():
    """
    data/synthetic.py の generate_simulation1_dataset を使い、
    小さめのサイズで X, Y, V を生成する。
    """
    X, Y, V_true, _, _ = generate_simulation1_dataset(
        n_samples=100,
        n_assets=5,
        snr=0.1,
        rho=0.5,
        sigma=0.025,
        seed=25,
    )

    # fit_ipo_nn_qp はサンプルごとの Vhat を要求するため、ここでは
    # 真の共分散 V_true をそのまま複製して各時点に割り当てる。
    Vhats = [V_true.copy() for _ in range(X.shape[0])]
    idx = list(range(X.shape[0]))
    return X, Y, V_true, Vhats, idx


def run_sanity():
    # --- データ生成 ---
    X, Y, V_true, Vhats, idx = _load_reference_dataset()
    n = X.shape[0]
    train_cut = n // 2
    train_ids = idx[:train_cut]
    test_ids = idx[train_cut:]

    # --- IPO-NN-QP を学習 ---
    theta_hat, Z, MU, LAM, used_idx, meta, yhat_all = fit_ipo_nn_qp(
        X=X,
        Y=Y,
        Vhats=Vhats,
        idx=idx,
        delta=1,
        psd_eps=1e-7,
        theta_init="identity",
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=1000,
        batch_size=64,
        seed=123,
        device="cpu",
        tee=True,
    )

    print("=== IPO-NN-QP サニティ実験 ===")
    print(f"使用サンプル数        : {len(used_idx)}")
    print(f"theta_hat の形状       : {theta_hat.shape}")
    print(f"Yhat_all の形状        : {yhat_all.shape}")
    print(f"メタ情報 status        : {meta.get('status')}")
    print(f"theta 初期化モード     : {meta.get('theta_init')}")
    print(f"|yhat| の平均          : {np.abs(yhat_all).mean():.4f}")

    def solve_qp_decision(yhat: np.ndarray, V: np.ndarray, delta: float, box=(0.0, 1.0)):
        """min δ/2 z^T V z - yhat^T z, s.t. 1^T z = 1, box 制約下の最適 z を求める。"""
        d = yhat.shape[0]
        z = cp.Variable(d)
        objective = 0.5 * delta * cp.quad_form(z, V) - yhat @ z
        constraints = [cp.sum(z) == 1]
        if box is not None:
            lb, ub = box
            if lb is not None:
                constraints.append(z >= lb)
            if ub is not None:
                constraints.append(z <= ub)
        prob = cp.Problem(cp.Minimize(objective), constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.error.SolverError:
            prob.solve(solver=cp.SCS, verbose=False)
        if z.value is None:
            raise RuntimeError("QP の解が得られませんでした")
        return np.array(z.value, dtype=float).ravel()

    def objective_value(z, y_true, V, delta):
        return 0.5 * delta * z @ (V @ z) - y_true @ z

    # 真の y を用いた「理想的な」意思決定を事前に計算しておく
    z_true_all = [solve_qp_decision(y_true, V_true, delta=1) for y_true in Y]

    def evaluate_strategy(tag, yhat_matrix):
        """学習済みモデルの意思決定誤差と目的値を train/test に分けて表示。"""
        for split_name, ids in [("train", train_ids), ("test", test_ids)]:
            decision_errors = []
            objectives = []
            for i in ids:
                z_pred = solve_qp_decision(yhat_matrix[i], V_true, delta=1)
                z_true = z_true_all[i]
                decision_errors.append(np.linalg.norm(z_pred - z_true, ord=1))
                objectives.append(objective_value(z_pred, Y[i], V_true, delta=1))
            decision_errors = np.array(decision_errors)
            objectives = np.array(objectives)
            print(
                f"【{tag} - {split_name}】平均意思決定誤差(L1): "
                f"{decision_errors.mean():.4f} (std {decision_errors.std():.4f})"
            )
            print(
                f"【{tag} - {split_name}】平均目的関数値(V_true): "
                f"{objectives.mean():.4f} (std {objectives.std():.4f})"
            )

    # IPO-NN-QP の評価
    evaluate_strategy("IPO-NN-QP", yhat_all)

    # 同一条件で OLS (解析解) の予測も比較
    theta_ols, _, _, _, _, _ = fit_ipo_closed_form(
        X=X,
        Y=Y,
        Vhats=Vhats,
        idx=idx,
        delta=1,
        mode="budget",
        psd_eps=1e-7,
        tee=True,
    )
    yhat_ols = X * theta_ols
    evaluate_strategy("OLS", yhat_ols)

    # ---- 最低限のチェック ----
    assert yhat_all.shape == X.shape, "予測行列の形状が入力と一致していません"
    assert theta_hat.shape == (X.shape[1],), "theta_hat の次元が特徴量数と一致しません"
    assert np.isfinite(theta_hat).all(), "theta_hat に無限大/NaN が含まれています"
    assert meta["status"] == "completed", "学習が正常終了しませんでした"

    print("→ サニティチェックに合格しました。")


if __name__ == "__main__":
    run_sanity()
