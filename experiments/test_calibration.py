from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Callable

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# from data.synthetic import generate_simulation1_dataset
from data_stress.synthetic import generate_simulation1_dataset
from experiments.registry import SolverSpec, get_trainer
from models.covariance import estimate_epscov_rolling
from models.ols import train_ols, predict_yhat
from models.ols_gurobi import solve_series_mvo_gurobi


def mvo_cost(z: np.ndarray, y: np.ndarray, V: np.ndarray, delta: float = 1.0) -> float:
    return float(-z @ y + 0.5 * delta * (z @ V @ z))


@dataclass
class CalibrationResult:
    seed: int
    model: str
    s_star: float
    val_cost_base: float
    val_cost_cal: float
    test_cost_base: float
    test_cost_cal: float
    improvement_test: float
    std_ratio_test_base: float
    std_ratio_test_cal: float
    corr2_test_base: float
    corr2_test_cal: float

    def to_row(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "model": self.model,
            "s_star": self.s_star,
            "val_cost_base": self.val_cost_base,
            "val_cost_cal": self.val_cost_cal,
            "test_cost_base": self.test_cost_base,
            "test_cost_cal": self.test_cost_cal,
            "improvement_test": self.improvement_test,
            "std_ratio_test_base": self.std_ratio_test_base,
            "std_ratio_test_cal": self.std_ratio_test_cal,
            "corr2_test_base": self.corr2_test_base,
            "corr2_test_cal": self.corr2_test_cal,
        }


def compute_std_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for j in range(y_true.shape[1]):
        std_true = float(np.std(y_true[:, j], ddof=1))
        std_pred = float(np.std(y_pred[:, j], ddof=1))
        if std_true < 1e-12:
            continue
        vals.append(std_pred / std_true)
    return float(np.nanmean(vals)) if vals else np.nan


def compute_corr2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for j in range(y_true.shape[1]):
        std_true = float(np.std(y_true[:, j], ddof=1))
        std_pred = float(np.std(y_pred[:, j], ddof=1))
        if std_true < 1e-12 or std_pred < 1e-12:
            vals.append(np.nan)
            continue
        corr = np.corrcoef(y_true[:, j], y_pred[:, j])[0, 1]
        vals.append(corr**2)
    return float(np.nanmean(vals)) if vals else np.nan


def prepare_trainer(model_key: str, solver_name: str, tee: bool):
    solver_spec = SolverSpec(name=solver_name, options={}, tee=tee)
    trainer = get_trainer(model_key, solver_spec)
    return trainer


def split_indices(idx: Sequence[int], Vhats: Sequence[np.ndarray],
                  train_frac: float, val_frac: float) -> Tuple[List[int], List[np.ndarray],
                                                               List[int], List[np.ndarray],
                                                               List[int], List[np.ndarray]]:
    total = len(idx)
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    n_train = max(1, min(n_train, total - 2))
    n_val = max(1, min(n_val, total - n_train - 1))
    n_test = total - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough samples left for test split. Adjust fractions.")

    idx_arr = list(idx)
    V_arr = list(Vhats)

    idx_train = idx_arr[:n_train]
    V_train = V_arr[:n_train]

    idx_val = idx_arr[n_train:n_train + n_val]
    V_val = V_arr[n_train:n_train + n_val]

    idx_test = idx_arr[n_train + n_val:]
    V_test = V_arr[n_train + n_val:]

    return idx_train, V_train, idx_val, V_val, idx_test, V_test


def solve_mvo(Yhat_all: np.ndarray,
              Y: np.ndarray,
              idx_seq: Sequence[int],
              V_seq: Sequence[np.ndarray],
              delta: float,
              V_true: np.ndarray) -> Tuple[List[float], List[float]]:
    if not idx_seq:
        raise ValueError("Empty idx sequence passed to MVO solve.")
    Z = solve_series_mvo_gurobi(
        Yhat_all=Yhat_all,
        Vhats=V_seq,
        idx=idx_seq,
        delta=delta,
        psd_eps=1e-12,
        output=False,
        start_index=None,
    )
    Y_used = Y[idx_seq]
    costs_true = []
    costs_vhat = []
    for t, z in enumerate(Z):
        if np.isnan(z).any():
            costs_true.append(np.nan)
            costs_vhat.append(np.nan)
            continue
        costs_true.append(mvo_cost(z, Y_used[t], V_true, delta))
        costs_vhat.append(mvo_cost(z, Y_used[t], V_seq[t], delta))
    return costs_true, costs_vhat


def golden_search(func: Callable[[float], float],
                  lo: float,
                  hi: float,
                  iterations: int) -> Tuple[float, float]:
    phi = (1 + sqrt(5)) / 2
    a, b = lo, hi
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = func(c)
    fd = func(d)
    for _ in range(iterations):
        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - (b - a) / phi
            fc = func(c)
        else:
            a, fc = c, fd
            c = d
            d = a + (b - a) / phi
            fd = func(d)
    if fc < fd:
        return c, fc
    return d, fd


def run_one(model_key: str,
            solver: str,
            tee: bool,
            seed: int,
            N: int,
            d: int,
            snr: float,
            rho: float,
            sigma: float,
            res: int,
            delta: float,
            use_true_cov: bool,
            reg_theta_l2: float,
            train_frac: float,
            val_frac: float,
            s_lo: float,
            s_hi: float,
            s_iter: int) -> CalibrationResult:

    X, Y, V_true, theta0, tau = generate_simulation1_dataset(
        n_samples=N, n_assets=d, snr=snr, rho=rho, sigma=sigma, seed=seed
    )

    burn_in = int(res)
    if burn_in >= N:
        raise ValueError("burn_in(res) >= N")

    if use_true_cov:
        idx = list(range(burn_in, N))
        Vhats = [V_true.copy() for _ in idx]
    else:
        Vhats, idx = estimate_epscov_rolling(
            Y, X, theta0, tau, res=res, include_current=False
        )
        if not Vhats:
            raise RuntimeError("Rolling covariance estimation returned empty result")

    idx_train, V_train, idx_val, V_val, idx_test, V_test = split_indices(
        idx, Vhats, train_frac, val_frac
    )

    X_train = X[idx_train]
    Y_train = Y[idx_train]

    theta_init = train_ols(X_train, Y_train)

    if model_key == "ols":
        theta_hat = theta_init
    else:
        trainer = prepare_trainer(model_key, solver, tee)
        trainer_ret = trainer(
            X, Y, Vhats, idx,
            start_index=idx_train[0],
            end_index=idx_train[-1],
            delta=delta,
            theta_init=theta_init,
            tee=tee,
            reg_theta_l2=reg_theta_l2,
        )
        if len(trainer_ret) < 1:
            raise RuntimeError("Trainer returned empty result.")
        theta_hat = trainer_ret[0]

    Yhat_all = predict_yhat(X, theta_hat)

    # Base performance (no scaling)
    val_costs_base, _ = solve_mvo(Yhat_all, Y, idx_val, V_val, delta, V_true)
    test_costs_base, _ = solve_mvo(Yhat_all, Y, idx_test, V_test, delta, V_true)

    val_mean_base = float(np.nanmean(val_costs_base))
    test_mean_base = float(np.nanmean(test_costs_base))

    y_val = Y[idx_val]
    yhat_val = Yhat_all[idx_val]
    y_test = Y[idx_test]
    yhat_test = Yhat_all[idx_test]

    corr2_val = compute_corr2(y_val, yhat_val)
    corr2_test = compute_corr2(y_test, yhat_test)
    std_ratio_val = compute_std_ratio(y_val, yhat_val)
    std_ratio_test = compute_std_ratio(y_test, yhat_test)

    # Calibration objective
    def objective(s: float) -> float:
        Yhat_scaled = Yhat_all * s
        costs_val, _ = solve_mvo(Yhat_scaled, Y, idx_val, V_val, delta, V_true)
        return float(np.nanmean(costs_val))

    s_star, val_cost_best = golden_search(objective, s_lo, s_hi, s_iter)

    Yhat_scaled_star = Yhat_all * s_star
    val_costs_cal, _ = solve_mvo(Yhat_scaled_star, Y, idx_val, V_val, delta, V_true)
    test_costs_cal, _ = solve_mvo(Yhat_scaled_star, Y, idx_test, V_test, delta, V_true)

    val_mean_cal = float(np.nanmean(val_costs_cal))
    test_mean_cal = float(np.nanmean(test_costs_cal))

    yhat_test_cal = yhat_test * s_star
    corr2_test_cal = compute_corr2(y_test, yhat_test_cal)
    std_ratio_test_cal = compute_std_ratio(y_test, yhat_test_cal)

    improvement = test_mean_base - test_mean_cal

    return CalibrationResult(
        seed=seed,
        model=model_key,
        s_star=float(s_star),
        val_cost_base=val_mean_base,
        val_cost_cal=val_mean_cal,
        test_cost_base=test_mean_base,
        test_cost_cal=test_mean_cal,
        improvement_test=improvement,
        std_ratio_test_base=std_ratio_test,
        std_ratio_test_cal=std_ratio_test_cal,
        corr2_test_base=corr2_test,
        corr2_test_cal=corr2_test_cal,
    )


def summarize(results: List[CalibrationResult]) -> Dict[str, float]:
    arr = np.array([[r.val_cost_base, r.val_cost_cal,
                     r.test_cost_base, r.test_cost_cal,
                     r.improvement_test, r.std_ratio_test_base,
                     r.std_ratio_test_cal] for r in results], dtype=float)
    if arr.size == 0:
        return {}
    means = np.nanmean(arr, axis=0)
    return {
        "val_cost_base": float(means[0]),
        "val_cost_cal": float(means[1]),
        "test_cost_base": float(means[2]),
        "test_cost_cal": float(means[3]),
        "improvement_test": float(means[4]),
        "std_ratio_test_base": float(means[5]),
        "std_ratio_test_cal": float(means[6]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc scalar calibration diagnostics")
    ap.add_argument("--models", type=str, default="dual",
                    help="Comma separated list of models (dual,kkt,ols)")
    ap.add_argument("--solver", type=str, default="ipopt")
    ap.add_argument("--tee", action="store_true")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--snr", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.0125)
    ap.add_argument("--res", type=int, default=100)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--use-true-cov", action="store_true")
    ap.add_argument("--reg-theta-l2", type=float, default=0.0)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--s-lo", type=float, default=0.01)
    ap.add_argument("--s-hi", type=float, default=10.0)
    ap.add_argument("--s-iter", type=int, default=25)
    ap.add_argument("--out-csv", type=str, default=None)

    args = ap.parse_args()
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    results_by_model: Dict[str, List[CalibrationResult]] = {m: [] for m in models}

    for model in models:
        print(f"\n=== Model: {model} ===")
        for k in range(args.runs):
            seed = args.seed0 + k
            res_item = run_one(
                model_key=model,
                solver=args.solver,
                tee=args.tee,
                seed=seed,
                N=args.N,
                d=args.d,
                snr=args.snr,
                rho=args.rho,
                sigma=args.sigma,
                res=args.res,
                delta=args.delta,
                use_true_cov=args.use_true_cov,
                reg_theta_l2=args.reg_theta_l2,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                s_lo=args.s_lo,
                s_hi=args.s_hi,
                s_iter=args.s_iter,
            )
            results_by_model[model].append(res_item)
            print(f" seed={seed}  s*={res_item.s_star:.4f}  "
                  f"test_base={res_item.test_cost_base:.4f}  "
                  f"test_cal={res_item.test_cost_cal:.4f}  "
                  f"improve={res_item.improvement_test:.4f}  "
                  f"std_ratio_test: base={res_item.std_ratio_test_base:.6f}, "
                  f"cal={res_item.std_ratio_test_cal:.6f}")

    out_csv = args.out_csv
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = ["seed", "model", "s_star", "val_cost_base", "val_cost_cal",
                  "test_cost_base", "test_cost_cal", "improvement_test",
                  "std_ratio_test_base", "std_ratio_test_cal",
                  "corr2_test_base", "corr2_test_cal"]
        with out_path.open("w") as f:
            f.write(",".join(header) + "\n")
            for model, rows in results_by_model.items():
                for r in rows:
                    row = r.to_row()
                    f.write(",".join(str(row[h]) for h in header) + "\n")
        print(f"\nSaved calibration metrics to {out_path}")

    print("\n=== Summary ===")
    for model, rows in results_by_model.items():
        summary = summarize(rows)
        if not summary:
            continue
        print(f" {model}: "
              f"test_base={summary['test_cost_base']:.4f}, "
              f"test_cal={summary['test_cost_cal']:.4f}, "
              f"improve={summary['improvement_test']:.4f}, "
              f"std_ratio_base={summary['std_ratio_test_base']:.4f}, "
              f"std_ratio_cal={summary['std_ratio_test_cal']:.4f}")


if __name__ == "__main__":
    main()
