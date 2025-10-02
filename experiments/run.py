# DFL_Portfolio_Optimization2/experiments/run.py

from __future__ import annotations
import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import yaml  # ← 追加

# ルートを import パスに追加（直接実行対応）
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.registry import get_trainer, SolverSpec, available_models
from data.synthetic import generate_simulation1_dataset
from models.covariance import estimate_epscov_rolling
from models.ols import train_ols, predict_yhat
from models.ols_gurobi import solve_series_mvo_gurobi
from viz.plots import save_summary_plots  # ← 追加


# ===== 共通: MVOコスト =====
def mvo_cost(z: np.ndarray, y: np.ndarray, V: np.ndarray, delta: float = 1.0) -> float:
    return float(- z @ y + 0.5 * delta * (z @ V @ z))


def summarize(arr: np.ndarray) -> Tuple[float, float, Tuple[float, float], int]:
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, (np.nan, np.nan), 0
    mean = arr.mean()
    std = arr.std(ddof=1) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    ci95 = (mean - 1.96 * se, mean + 1.96 * se)
    return float(mean), float(se), (float(ci95[0]), float(ci95[1])), int(n)


def run_once(
    model_key: str,
    solver_spec: SolverSpec,
    seed: int,
    N: int, d: int, snr: float, rho: float, sigma: float,
    res: int, delta: float,
    tee: bool = True,
    reg_theta_l2: float = 0.0,
    use_true_cov: bool = False,
) -> Tuple[float, float, int, float, float, float, float, float, float, float, float, float, float, float, float]:

    # ----- データ生成 -----
    X, Y, V_true, theta_0, tau = generate_simulation1_dataset(
        n_samples=N, n_assets=d, snr=snr, rho=rho, sigma=sigma, seed=seed
    )

    # ----- burn-in と分割（res=ウィンドウ幅をそのまま使用） -----
    burn_in = int(res)
    if burn_in >= N:
        raise ValueError("burn-in (res) >= N")

    n_eff = N - burn_in
    n_tr = n_eff // 2
    n_te = n_eff - n_tr

    X_tr, Y_tr = X[burn_in:burn_in+n_tr], Y[burn_in:burn_in+n_tr]
    X_te, Y_te = X[burn_in+n_tr:],        Y[burn_in+n_tr:]

    # ----- ローリング共分散 -----
    if use_true_cov:
        idx = list(range(burn_in, N))
        Vhats = [V_true.copy() for _ in idx]   # 必要なら copy()
    else:
        Vhats, idx = estimate_epscov_rolling(
            Y, X, theta_0, tau, res=res, include_current=False
    )
    if len(Vhats) == 0:
        return (
            np.nan, np.nan, 0, 1.0,
            np.nan, np.nan, np.nan, 0.0,
            np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan,
        )

    # ----- 学習区間（train: burn_in .. burn_in+n_tr-1） -----
    train_pairs = [(i, V) for i, V in zip(idx, Vhats)
                   if burn_in <= i < burn_in + n_tr]
    if not train_pairs:
        return (
            np.nan, np.nan, 0, 1.0,
            np.nan, np.nan, np.nan, 0.0,
            np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan,
        )
    start_train, end_train = train_pairs[0][0], train_pairs[-1][0]

    # ----- OLS 初期値（train 全体で学習）-----
    theta_init = train_ols(X_tr, Y_tr)


    # ----- 学習（KKT / DUAL） -----
    trainer = get_trainer(model_key, solver_spec)
    t0 = time.perf_counter()
    trainer_ret = trainer(
        X, Y, Vhats, idx,
        start_index=start_train, end_index=end_train,
        delta=delta, theta_init=theta_init, tee=tee,
        reg_theta_l2=reg_theta_l2,
    )
    elapsed = time.perf_counter() - t0

    # 統一：(theta_hat, Z_tr, MU_tr, LAM_tr, used_train_idx, info)
    if not (isinstance(trainer_ret, (list, tuple)) and len(trainer_ret) >= 5):
        raise RuntimeError("Trainer returned unexpected result format")
    theta_hat, Z_tr, MU_tr, LAM_tr, used_train_idx = trainer_ret[:5]
    info = trainer_ret[5] if len(trainer_ret) >= 6 else {}

    # ---- 早期打ち切り（max_iter / max_cpu_time / timeout）検知 → スキップ ----
    tc  = str((info or {}).get("termination_condition", "")).lower()
    st  = str((info or {}).get("status", "")).lower()
    msg = str((info or {}).get("message", "") or "").lower()

    bad_tc = {
        "maxiterations", "maxtimelimit", "max_time_limit",
        "max_time", "maxcputime", "error", "infeasible"
    }
    if (tc in bad_tc) or ("maximum cpu time exceeded" in msg) or (st in {"max_iter","max_cpu_time","timeout"}):
        raise RuntimeError(f"solver_stopped:{tc or st or 'time_limit'}")

    # フォールバック：経過時間が上限に十分近い
    try:
        max_cpu_time_opt = solver_spec.options.get("max_cpu_time", None)
        if max_cpu_time_opt is not None:
            max_cpu_time = float(max_cpu_time_opt)
            if max_cpu_time > 0 and elapsed >= 0.95 * max_cpu_time:
                raise RuntimeError("solver_stopped:approx_max_cpu_time")
    except Exception:
        pass

    # ----- 学習区間の MVO コスト（train で評価） -----
    train_cost_true_mean = np.nan
    train_cost_vhat_mean = np.nan
    train_best_cost_true_mean = np.nan
    decision_error_train = np.nan

    if train_pairs:
        used_train_idx = [i for i, _ in train_pairs]
        Vhats_train = [V for _, V in train_pairs]

        Y_hat_all = predict_yhat(X, theta_hat)
        Z_train = solve_series_mvo_gurobi(
            Yhat_all=Y_hat_all, Vhats=Vhats_train, idx=used_train_idx,
            delta=delta, psd_eps=1e-12, output=False, start_index=None
        )
        Z_train_opt = solve_series_mvo_gurobi(
            Yhat_all=Y, Vhats=Vhats_train, idx=used_train_idx,
            delta=delta, psd_eps=1e-12, output=False, start_index=None
        )

        Y_train_used = Y[used_train_idx]
        mask_tr_pred = ~np.isnan(Z_train).any(axis=1)
        mask_tr_opt = ~np.isnan(Z_train_opt).any(axis=1)
        mask_tr = mask_tr_pred & mask_tr_opt
        Z_tr_eval = Z_train[mask_tr]
        Y_tr_eval = Y_train_used[mask_tr]
        Vhats_tr_eval = [Vhats_train[k] for k, m in enumerate(mask_tr) if m]
        Z_tr_opt_eval = Z_train_opt[mask_tr]

        if len(Z_tr_eval) > 0:
            costs_true = [mvo_cost(Z_tr_eval[t], Y_tr_eval[t], V_true, delta)
                          for t in range(len(Z_tr_eval))]
            train_cost_true_mean = float(np.mean(costs_true))

            costs_vhat = [mvo_cost(Z_tr_eval[t], Y_tr_eval[t], Vhats_tr_eval[t], delta)
                          for t in range(len(Z_tr_eval))]
            train_cost_vhat_mean = float(np.mean(costs_vhat))

            train_best_costs_true = [mvo_cost(Z_tr_opt_eval[t], Y_tr_eval[t], V_true, delta)
                                     for t in range(len(Z_tr_opt_eval))]
            if train_best_costs_true:
                train_best_cost_true_mean = float(np.mean(train_best_costs_true))
            if np.isfinite(train_best_cost_true_mean) and abs(train_best_cost_true_mean) > 1e-12:
                decision_error_train = float((train_best_cost_true_mean - train_cost_true_mean) / train_best_cost_true_mean)

    best_cost_true_mean = np.nan
    decision_error_test = np.nan

    # ----- テスト区間 -----
    test_pairs = [(i, V) for i, V in zip(idx, Vhats)
                  if burn_in + n_tr <= i < N]
    if not test_pairs:
        return (
            np.nan, np.nan, 0, 1.0,
            train_cost_true_mean, train_cost_vhat_mean, np.nan, elapsed,
            np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan,
        )

    used_test_idx = [i for i, _ in test_pairs]
    Vhats_test = [V for _, V in test_pairs]

    # 全期間の予測 -> テスト抽出
    Y_hat_all = predict_yhat(X, theta_hat)
    Z_test = solve_series_mvo_gurobi(
        Yhat_all=Y_hat_all, Vhats=Vhats_test, idx=used_test_idx,
        delta=delta, psd_eps=1e-12, output=False, start_index=None
    )
    Z_test_opt = solve_series_mvo_gurobi(
        Yhat_all=Y, Vhats=Vhats_test, idx=used_test_idx,
        delta=delta, psd_eps=1e-12, output=False, start_index=None
    )

    Y_used = Y[used_test_idx]
    Yhat_used = Y_hat_all[used_test_idx]
    mask_pred = ~np.isnan(Z_test).any(axis=1)
    mask_opt = ~np.isnan(Z_test_opt).any(axis=1)
    mask = mask_pred & mask_opt
    fail_rate = 1.0 - (np.sum(mask_pred) / len(Z_test)) if len(Z_test) > 0 else 1.0
    Z_eval = Z_test[mask]
    Y_eval = Y_used[mask]
    Yhat_eval = Yhat_used[mask]
    Vhats_eval = [Vhats_test[k] for k, m in enumerate(mask) if m]
    Z_opt_eval = Z_test_opt[mask]

    # ----- 指標（MVOコスト / 予測指標） -----
    costs_true = [mvo_cost(Z_eval[t], Y_eval[t], V_true, delta) for t in range(len(Z_eval))]
    mean_cost_true = float(np.mean(costs_true))

    costs_vhat = [mvo_cost(Z_eval[t], Y_eval[t], Vhats_eval[t], delta) for t in range(len(Z_eval))]
    mean_cost_vhat = float(np.mean(costs_vhat))

    best_cost_true_mean = np.nan
    if len(Z_opt_eval) > 0:
        best_costs_true = [mvo_cost(Z_opt_eval[t], Y_eval[t], V_true, delta)
                           for t in range(len(Z_opt_eval))]
        best_cost_true_mean = float(np.mean(best_costs_true))

    decision_error_test = np.nan
    if np.isfinite(best_cost_true_mean) and abs(best_cost_true_mean) > 1e-12:
        decision_error_test = float((best_cost_true_mean - mean_cost_true) / best_cost_true_mean)

    # --- corr^2（既存）
    corr_cols = []
    for j in range(d):
        yj, yhj = Y_eval[:, j], Yhat_eval[:, j]
        if np.std(yj) < 1e-12 or np.std(yhj) < 1e-12:
            corr_cols.append(0.0)
        else:
            corr_cols.append(np.corrcoef(yj, yhj)[0, 1]**2)
    mean_r2_corrsq = float(np.nanmean(corr_cols))

    # --- 通常の R^2（scikit-learn）
    r2_cols = []
    for j in range(d):
        yj, yhj = Y_eval[:, j], Yhat_eval[:, j]
        # y が定数だと r2_score が定義できないので 0 扱い
        if np.std(yj) < 1e-12:
            r2_cols.append(0.0)
        else:
            r2_cols.append(r2_score(yj, yhj))
    mean_r2_sklearn = float(np.nanmean(r2_cols))

    # --- MSE（全資産・全時点平均）
    mse_mean = float(np.mean((Y_eval - Yhat_eval)**2))

    # --- PVE（論文定義：分散は不偏ではなく母分散扱い＝ddof=0）
    pve_cols = []
    for j in range(d):
        yj, yhj = Y_eval[:, j], Yhat_eval[:, j]
        var_y = np.var(yj)  # ddof=0
        if var_y < 1e-18:
            pve_cols.append(0.0)
        else:
            pve_cols.append(1.0 - np.mean((yj - yhj)**2) / var_y)
    mean_pve = float(np.nanmean(pve_cols))

    # fail_rate は mask_pred 基準で算出済み

    # ----- 学習区間の指標 -----
    train_r2_corrsq = np.nan
    train_r2_sklearn = np.nan
    train_mse_mean = np.nan
    train_pve_mean = np.nan

    if len(Z_tr_eval) > 0:
        Yhat_tr_all = predict_yhat(X, theta_hat)[used_train_idx]
        Yhat_tr_eval = Yhat_tr_all[mask_tr]

        # corr^2
        corr_cols_tr = []
        for j in range(d):
            yj, yhj = Y_tr_eval[:, j], Yhat_tr_eval[:, j]
            if np.std(yj) < 1e-12 or np.std(yhj) < 1e-12:
                corr_cols_tr.append(0.0)
            else:
                corr_cols_tr.append(np.corrcoef(yj, yhj)[0, 1]**2)
        train_r2_corrsq = float(np.nanmean(corr_cols_tr))

        # R^2
        r2_cols_tr = []
        for j in range(d):
            yj, yhj = Y_tr_eval[:, j], Yhat_tr_eval[:, j]
            if np.std(yj) < 1e-12:
                r2_cols_tr.append(0.0)
            else:
                r2_cols_tr.append(r2_score(yj, yhj))
        train_r2_sklearn = float(np.nanmean(r2_cols_tr))

        # MSE
        train_mse_mean = float(np.mean((Y_tr_eval - Yhat_tr_eval)**2))

        # PVE
        pve_cols_tr = []
        for j in range(d):
            yj, yhj = Y_tr_eval[:, j], Yhat_tr_eval[:, j]
            var_y = np.var(yj)
            if var_y < 1e-18:
                pve_cols_tr.append(0.0)
            else:
                pve_cols_tr.append(1.0 - np.mean((yj - yhj)**2) / var_y)
        train_pve_mean = float(np.nanmean(pve_cols_tr))

    return (
        mean_cost_true,           # 1
        mean_r2_corrsq,           # 2 (test corr^2)
        len(Z_eval),              # 3
        fail_rate,                # 4
        train_cost_true_mean,     # 5
        train_cost_vhat_mean,     # 6
        mean_cost_vhat,           # 7
        elapsed,                  # 8
        train_r2_corrsq,          # 9 (train corr^2)
        mean_r2_sklearn,          # 10 (test R^2)
        train_r2_sklearn,         # 11 (train R^2)
        mse_mean,                 # 12 (test MSE)
        train_mse_mean,           # 13 (train MSE)
        mean_pve,                 # 14 (test PVE)
        train_pve_mean,           # 15 (train PVE)
        best_cost_true_mean,      # 16 (test optimal cost)
        train_best_cost_true_mean,# 17 (train optimal cost)
        decision_error_test,      # 18 (test decision error)
        decision_error_train      # 19 (train decision error)
    )
def main():
    p = argparse.ArgumentParser(description="Unified runner for DFL experiments (KKT/DUAL) with pluggable solvers")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file with defaults")
    p.add_argument("--model", type=str, default="kkt",
                   help=f"Which model to run. Options: {list(available_models().keys())}")
    p.add_argument("--solver", type=str, default="knitro", choices=["gurobi", "ipopt", "knitro"])
    p.add_argument("--tee", action="store_true", help="Show solver logs")

    # data / exp params
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--snr", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--res", type=int, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--seed0", type=int, default=None)
    p.add_argument("--lambda-theta", dest="lambda_theta", type=float, default=None,
               help="L2 regularization strength for theta")

    # output
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--no-plots", action="store_true", help="Do not save plots")
    # argparse に CLI 上書きフラグもあると便利
    p.add_argument("--ipopt-max-cpu-time", type=float, default=None)
    p.add_argument("--ipopt-max-iter", type=int, default=None)
    p.add_argument("--use-true-cov", action="store_true", help="共分散を推定せず V_true をそのまま使用する")

    args = p.parse_args()

    # ===== YAML 読み込み & マージ =====
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if "data" in cfg:
            for k, v in cfg["data"].items():
                setattr(args, k, v)
        for k in ["model", "solver", "tee", "outdir", "log_every"]:
            if k in cfg:
                setattr(args, k, cfg[k])
        reg_cfg = cfg.get("regularization", {})
        lambda_theta_from_yaml = float(reg_cfg.get("theta_l2", 0.0))
    else:
        cfg = {}
        lambda_theta_from_yaml = 0.0

    lambda_theta = float(args.lambda_theta) if getattr(args, "lambda_theta", None) is not None else lambda_theta_from_yaml

    def pick(key, default=None):
        return getattr(args, key) if getattr(args, key) is not None else cfg.get(key, default)

    data_cfg = cfg.get("data", {})
    def pick_data(key, default=None):
        cli_val = getattr(args, key, None)
        if cli_val is not None: return cli_val
        if key in data_cfg: return data_cfg[key]
        return default

    # ここで model/solver を確定
    model_key = pick("model", "kkt")
    solver_name = pick("solver", "knitro")
    tee = bool(pick("tee", False))
    outdir = Path(pick("outdir", "results/exp_unified"))
    log_every = int(pick("log_every", 1))

    N     = int(pick_data("N", 2000))
    d     = int(pick_data("d", 10))
    snr   = float(pick_data("snr", 0.01))
    rho   = float(pick_data("rho", 0.5))
    sigma = float(pick_data("sigma", 0.0125))
    res   = int(pick_data("res", 50))
    delta = float(pick_data("delta", 1.0))
    runs  = int(pick_data("runs", 10))
    seed0 = int(pick_data("seed0", 100))
    use_true_cov = bool(getattr(args, "use_true_cov", False))  # 共分散行列V_trueをそのまま使用するかどうか

    # ここで solver_options を一回だけ作る
    solver_options_all = cfg.get("solver_options", {})
    solver_options = dict(solver_options_all.get(solver_name, {}))  # 独立コピー

    if solver_name == "ipopt":
        # デフォルト（未設定キーだけ）
        solver_options.setdefault("tol", "1e-6")
        solver_options.setdefault("acceptable_tol", "1e-5")
        solver_options.setdefault("max_iter", 100000)
        solver_options.setdefault("linear_solver", "mumps")
        solver_options.setdefault("hessian_approximation", "limited-memory")
        solver_options.setdefault("limited_memory_max_history", 20)
        solver_options.setdefault("mu_strategy", "adaptive")
        solver_options.setdefault("watchdog_shortened_iter_trigger", 10)
        solver_options.setdefault("max_cpu_time", 200)

        # CLI 上書き
        if args.ipopt_max_cpu_time is not None:
            solver_options["max_cpu_time"] = float(args.ipopt_max_cpu_time)
        if args.ipopt_max_iter is not None:
            solver_options["max_iter"] = int(args.ipopt_max_iter)

    # これ以降、solver_spec は一度だけ作る
    solver_spec = SolverSpec(name=solver_name, options=solver_options, tee=tee)

    # 成功分だけためる
    seeds_kept = []
    cost_list, r2_list, Te_list, fail_list, run_times = [], [], [], [], []
    train_true_list, train_vhat_list, test_vhat_list, train_r2_list = [], [], [], []

    # 追加リスト
    r2_sklearn_list, tr_r2_sklearn_list = [], []
    mse_list, tr_mse_list = [], []
    pve_list, tr_pve_list = [], []
    best_cost_opt_list, tr_best_cost_opt_list = [], []
    decision_error_test_list, decision_error_train_list = [], []

    print(f"=== Start: model={model_key}, solver={solver_name} ===")
    print(f"N={N}, d={d}, snr={snr}, rho={rho}, sigma={sigma}, res={res}, delta={delta}, runs={runs}, lambda_theta={lambda_theta}")
    if solver_options:
        print(f"solver_options={solver_options}")

    T0 = time.perf_counter()
    successes = 0
    trials = 0
    MAX_EXTRA = 1000  # 無限ループ防止。必要なら広げる

    while successes < runs and trials < runs + MAX_EXTRA:
        s = int(seed0 + trials)
        trials += 1
        try:
            (mc_true, r2_corr, Te, fr,
             tr_true, tr_vhat, te_vhat, rt,
             tr_r2_corr,
             r2_skl, tr_r2_skl,
             mse, tr_mse,
             pve, tr_pve,
             best_cost_opt, tr_best_cost_opt,
             dec_err_test, dec_err_train) = run_once(
                model_key=model_key,
                solver_spec=solver_spec,
                seed=s,
                N=N, d=d, snr=snr, rho=rho, sigma=sigma,
                res=res, delta=delta,
                tee=tee,
                use_true_cov=use_true_cov,
                reg_theta_l2=lambda_theta,
            )
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("solver_stopped:"):
                print(f"[SKIP] seed={s} {msg} -> skip")
                continue
            print(f"[WARN] seed={s} runtime error: {msg} -> skip")
            continue
        except Exception as e:
            print(f"[WARN] seed={s} failed with exception: {e} -> skip")
            continue

        # 成功判定（評価データあり & 主要指標が有限）
        ok = (Te is not None and Te > 0 and
              np.isfinite(mc_true))
        if not ok:
            print(f"[WARN] seed={s} produced invalid metrics (Te={Te},  cost={mc_true}) -> skip")
            continue

        # 採用
        seeds_kept.append(s)
        cost_list.append(mc_true)
        r2_list.append(r2_corr)           # corr^2
        Te_list.append(Te); fail_list.append(fr)
        train_true_list.append(tr_true); train_vhat_list.append(tr_vhat); test_vhat_list.append(te_vhat)
        train_r2_list.append(tr_r2_corr); run_times.append(rt)

        r2_sklearn_list.append(r2_skl); tr_r2_sklearn_list.append(tr_r2_skl)
        mse_list.append(mse); tr_mse_list.append(tr_mse)
        pve_list.append(pve); tr_pve_list.append(tr_pve)
        best_cost_opt_list.append(best_cost_opt); tr_best_cost_opt_list.append(tr_best_cost_opt)
        decision_error_test_list.append(dec_err_test); decision_error_train_list.append(dec_err_train)
        successes += 1

        if (successes % max(1, log_every)) == 0:
            print(f"[{successes}/{runs}] seed={s}  "
                  f"cost_test_vtrue={mc_true:.6f} cost_train_vtrue={tr_true:.6f}  "
                  f"corr2={r2_corr:.4f} R2={r2_skl:.4f} PVE={pve:.4f}  "
                  f"train_corr2={tr_r2_corr:.4f} train_R2={tr_r2_skl:.4f} train_PVE={tr_pve:.4f} "
                  f"mse={mse:.4e} train_mse={tr_mse:.4e} "
                  f"opt_cost={best_cost_opt:.6f} dec_err={dec_err_test:.4f} "
                  f"train_opt_cost={tr_best_cost_opt:.6f} train_dec_err={dec_err_train:.4f} "
                  f"rows={Te} fail={fr:.2%} rt={rt:.2f}s")

    total_elapsed = time.perf_counter() - T0
    print(f"\n[INFO] trials={trials}, successes={successes}, skipped={trials - successes}")

    if successes == 0:
        raise SystemExit("All runs failed. No results to summarize.")

    # 集計（成功分のみ）
    # 既存
    cost_mean, cost_se, cost_ci, n_cost = summarize(np.array(cost_list))
    r2_mean, r2_se, r2_ci, n_r2 = summarize(np.array(r2_list))  # corr^2

    # 追加
    r2sk_mean, r2sk_se, r2sk_ci, _ = summarize(np.array(r2_sklearn_list))
    mse_mean, mse_se, mse_ci, _     = summarize(np.array(mse_list))
    pve_mean, pve_se, pve_ci, _     = summarize(np.array(pve_list))

    tr_true_mean, _, _, _ = summarize(np.array(train_true_list))
    tr_vhat_mean, _, _, _ = summarize(np.array(train_vhat_list))
    te_vhat_mean, _, _, _ = summarize(np.array(test_vhat_list))
    tr_r2_mean, _, _, _   = summarize(np.array(train_r2_list))            # corr^2(train)

    tr_r2sk_mean, _, _, _ = summarize(np.array(tr_r2_sklearn_list))
    tr_mse_mean, _, _, _  = summarize(np.array(tr_mse_list))
    tr_pve_mean, _, _, _  = summarize(np.array(tr_pve_list))
    best_cost_mean, best_cost_se, best_cost_ci, _ = summarize(np.array(best_cost_opt_list))
    tr_best_cost_mean, tr_best_cost_se, tr_best_cost_ci, _ = summarize(np.array(tr_best_cost_opt_list))
    dec_err_mean, dec_err_se, dec_err_ci, _ = summarize(np.array(decision_error_test_list))
    tr_dec_err_mean, tr_dec_err_se, tr_dec_err_ci, _ = summarize(np.array(decision_error_train_list))

    print("\n==== Summary ====")
    print(f"  Mean MVO Cost (test, Vtrue): {cost_mean:.6f} (SE {cost_se:.6f}, 95% CI [{cost_ci[0]:.6f}, {cost_ci[1]:.6f}], n={n_cost})")
    print(f"  Mean corr^2 (test)         : {r2_mean:.6f} (SE {r2_se:.4f}, 95% CI [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}], n={n_r2})")
    print(f"  Mean R^2 (test, sklearn)   : {r2sk_mean:.6f} (SE {r2sk_se:.4f}, 95% CI [{r2sk_ci[0]:.4f}, {r2sk_ci[1]:.4f}])")
    print(f"  Mean MSE (test)            : {mse_mean:.6e} (SE {mse_se:.2e}, 95% CI [{mse_ci[0]:.2e}, {mse_ci[1]:.2e}])")
    print(f"  Mean PVE (test)            : {pve_mean:.6f} (SE {pve_se:.4f}, 95% CI [{pve_ci[0]:.4f}, {pve_ci[1]:.4f}])")
    print(f"  Optimal MVO Cost (test)    : {best_cost_mean:.6f} (SE {best_cost_se:.6f}, 95% CI [{best_cost_ci[0]:.6f}, {best_cost_ci[1]:.6f}])")
    print(f"  Decision Error (test)      : {dec_err_mean:.6f} (SE {dec_err_se:.6f}, 95% CI [{dec_err_ci[0]:.6f}, {dec_err_ci[1]:.6f}])")
    print(f"  Mean corr^2 (train)        : {tr_r2_mean:.6f}")
    print(f"  Mean R^2 (train, sklearn)  : {tr_r2sk_mean:.6f}")
    print(f"  Mean MSE (train)           : {tr_mse_mean:.6e}")
    print(f"  Mean PVE (train)           : {tr_pve_mean:.6f}")
    print(f"  Optimal MVO Cost (train)   : {tr_best_cost_mean:.6f}")
    print(f"  Decision Error (train)     : {tr_dec_err_mean:.6f}")
    print(f"  Train mean MVO (Vtrue)     : {tr_true_mean:.6f}")
    print(f"  Train mean MVO (Vhat)      : {tr_vhat_mean:.6f}")
    print(f"  Test  mean MVO (Vhat)      : {te_vhat_mean:.6f}")
    print(f"  Avg eval rows              : {np.nanmean(Te_list):.1f}")
    print(f"  Avg fail rate              : {np.nanmean(fail_list)*100:.2f}%")
    print(f"  Total time                 : {total_elapsed:.2f} sec")
    print(f"  Avg/run time               : {np.nanmean(run_times):.2f} sec")

    # ===== 保存 =====
    outdir = Path(args.outdir)
    day_dir = outdir / datetime.now().strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = (
        f"{args.model}_{args.solver}"
        f"_N{args.N}_d{args.d}_res{args.res}"
        f"_snr{args.snr}_rho{args.rho}_sigma{args.sigma}"
        f"_delta{args.delta}_{ts}"
        f"_lambda{lambda_theta}"
        f"_runs{successes}"

    )
    base = day_dir / tag

    # 1) per-run（成功シードのみ）
    run_df = pd.DataFrame({
        "seed": list(map(int, seeds_kept)),
        "mean_cost_test_vtrue": cost_list,
        "mean_cost_test_vhat":  test_vhat_list,
        "train_cost_vtrue":     train_true_list,
        "train_cost_vhat":      train_vhat_list,
        "mean_corr2_test":      r2_list,
        "mean_corr2_train":     train_r2_list,
        "mean_r2_test":         r2_sklearn_list,
        "mean_r2_train":        tr_r2_sklearn_list,
        "mse_test":             mse_list,
        "mse_train":            tr_mse_list,
        "pve_test":             pve_list,
        "pve_train":            tr_pve_list,
        "eval_rows":            Te_list,
        "fail_rate":            fail_list,
        "runtime_sec":          run_times,
        "optimal_cost_test_vtrue": best_cost_opt_list,
        "optimal_cost_train_vtrue": tr_best_cost_opt_list,
        "decision_error_test":   decision_error_test_list,
        "decision_error_train":  decision_error_train_list,
    })
    run_csv = str(base) + "_runs.csv"
    run_df.to_csv(run_csv, index=False)

    # 2) summary
    summary_row = {
        "timestamp": ts,
        "model": args.model,
        "solver": args.solver,
        "N": args.N, "d": args.d, "snr": args.snr, "rho": args.rho, "sigma": args.sigma,
        "res": args.res, "delta": args.delta, "runs": int(successes),
        "mean_cost_test_vtrue": float(cost_mean),
        "mean_cost_test_vtrue_se": float(cost_se),
        "mean_cost_test_vtrue_ci_lo": float(cost_ci[0]),
        "mean_cost_test_vtrue_ci_hi": float(cost_ci[1]),
        "mean_cost_test_vhat": float(te_vhat_mean),
        "train_cost_vtrue": float(tr_true_mean),
        "train_cost_vhat": float(tr_vhat_mean),

        # （注意）ここまでは “基本” 指標だけ
    }

    # ←←← ここで追加ブロックを差し込む
    summary_row.update({
        # corr^2（これまで r2_mean として集計していたもの）
        "mean_corr2_test": float(r2_mean),
        "mean_corr2_test_se": float(r2_se),
        "mean_corr2_test_ci_lo": float(r2_ci[0]),
        "mean_corr2_test_ci_hi": float(r2_ci[1]),
        "mean_corr2_train": float(tr_r2_mean),

        # sklearn の R^2（新規）
        "mean_r2_test": float(r2sk_mean),
        "mean_r2_test_se": float(r2sk_se),
        "mean_r2_test_ci_lo": float(r2sk_ci[0]),
        "mean_r2_test_ci_hi": float(r2sk_ci[1]),
        "mean_r2_train": float(tr_r2sk_mean),

        # MSE（新規）
        "mse_test": float(mse_mean),
        "mse_test_se": float(mse_se),
        "mse_test_ci_lo": float(mse_ci[0]),
        "mse_test_ci_hi": float(mse_ci[1]),
        "mse_train": float(tr_mse_mean),

        # PVE（新規）
        "pve_test": float(pve_mean),
        "pve_test_se": float(pve_se),
        "pve_test_ci_lo": float(pve_ci[0]),
        "pve_test_ci_hi": float(pve_ci[1]),
        "pve_train": float(tr_pve_mean),

        # Optimal costs & decision error（新規）
        "optimal_cost_test_vtrue": float(best_cost_mean),
        "optimal_cost_test_vtrue_se": float(best_cost_se),
        "optimal_cost_test_vtrue_ci_lo": float(best_cost_ci[0]),
        "optimal_cost_test_vtrue_ci_hi": float(best_cost_ci[1]),
        "optimal_cost_train_vtrue": float(tr_best_cost_mean),
        "optimal_cost_train_vtrue_se": float(tr_best_cost_se),
        "optimal_cost_train_vtrue_ci_lo": float(tr_best_cost_ci[0]),
        "optimal_cost_train_vtrue_ci_hi": float(tr_best_cost_ci[1]),
        "decision_error_test": float(dec_err_mean),
        "decision_error_test_se": float(dec_err_se),
        "decision_error_test_ci_lo": float(dec_err_ci[0]),
        "decision_error_test_ci_hi": float(dec_err_ci[1]),
        "decision_error_train": float(tr_dec_err_mean),
        "decision_error_train_se": float(tr_dec_err_se),
        "decision_error_train_ci_lo": float(tr_dec_err_ci[0]),
        "decision_error_train_ci_hi": float(tr_dec_err_ci[1]),

        # 既存の運用系メタ情報
        "avg_eval_T": float(np.nanmean(Te_list)),
        "avg_fail_rate": float(np.nanmean(fail_list)),
        "total_time_sec": float(total_elapsed),
        "avg_time_per_run_sec": float(np.nanmean(run_times)),
        "n_cost": int(n_cost),
        "n_r2": int(n_r2),
        "trials": int(trials),
        "skipped": int(trials - successes),
    })

    summary_df = pd.DataFrame([summary_row])
    summary_csv = str(base) + "_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 3) 累積ログ（モデル×ソルバーごと）
    log_path = day_dir / f"{args.model}_{args.solver}_summary_log.csv"
    header_needed = not log_path.exists()
    summary_df.to_csv(log_path, mode="a", header=header_needed, index=False)

    print("\n[Saved]")
    print(f"  per-run CSV : {run_csv}")
    print(f"  summary CSV : {summary_csv}")
    print(f"  summary log : {log_path}")
    if not args.no_plots:
        print(f"  plots       : {outdir}/{tag}_*.png")


if __name__ == "__main__":
    main()


# # 例1: KKT × Knitro（ログ表示）
# cd ~/VScode/GraduationResearch
# python DFL_Portfolio_Optimization2/experiments/run.py \
#     --model kkt \
#     --solver knitro \
#     --tee \
#     --N 2000 \
#     --d 10 \
#     --res 50 \
#     --snr 0.01 \
#     --rho 0.5 \
#     --sigma 0.0125 \
#     --delta 1.0 \
#     --lambda 0.0 \
#     --runs 10 \
#     --seed0 100 \
#     --outdir results/exp_unified \
#     --log-every 1


	# •	--model kkt → モデル選択（kkt or dual）
	# •	--solver knitro → ソルバー選択（gurobi, ipopt, knitro）
	# •	--tee → ソルバーのログをそのまま表示
	# •	--N 2000 → サンプル数
	# •	--d 10 → 資産数
	# •	--res 50 → ローリング共分散のウィンドウサイズ（burn-inにも反映）
	# •	--snr 0.01 → SNR
	# •	--rho 0.5 → 相関係数
	# •	--sigma 0.0125 → 各資産の標準偏差
	# •	--delta 1.0 → リスク回避パラメータ
	# •	--runs 10 → 繰り返し回数
	# •	--seed0 100 → 乱数シードの開始値
	# •	--outdir results/exp_unified → 保存先ディレクトリ
	# •	--log-every 1 → 1試行ごとに途中ログを出力
