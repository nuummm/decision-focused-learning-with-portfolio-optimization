from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# プロジェクトルートを import パスに追加しておく
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# from data.synthetic import generate_simulation1_dataset
from data_stress.synthetic import generate_simulation1_dataset
from experiments.registry import SolverSpec, get_trainer
from models.covariance import estimate_epscov_rolling
from models.ols import train_ols, predict_yhat


@dataclass
class ScaleMetrics:
    seed: int
    model: str
    std_ratio_mean_test: float
    std_ratio_mean_train: float
    corr2_test: float
    corr2_train: float
    r2_test: float
    r2_train: float
    std_ratio_assets_test: List[float]
    std_ratio_assets_train: List[float]

    def to_row(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "model": self.model,
            "std_ratio_mean_test": self.std_ratio_mean_test,
            "std_ratio_mean_train": self.std_ratio_mean_train,
            "corr2_test": self.corr2_test,
            "corr2_train": self.corr2_train,
            "r2_test": self.r2_test,
            "r2_train": self.r2_train,
            "std_ratio_assets_test": json.dumps(self.std_ratio_assets_test),
            "std_ratio_assets_train": json.dumps(self.std_ratio_assets_train),
        }


def compute_std_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ratios: List[float] = []
    for j in range(y_true.shape[1]):
        std_true = float(np.std(y_true[:, j], ddof=1))
        std_pred = float(np.std(y_pred[:, j], ddof=1))
        if std_true < 1e-12:
            ratios.append(np.nan)
        else:
            ratios.append(std_pred / std_true)
    return np.array(ratios, dtype=float)


def compute_corr2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals: List[float] = []
    for j in range(y_true.shape[1]):
        if np.std(y_true[:, j], ddof=1) < 1e-12 or np.std(y_pred[:, j], ddof=1) < 1e-12:
            vals.append(np.nan)
            continue
        corr = np.corrcoef(y_true[:, j], y_pred[:, j])[0, 1]
        vals.append(corr**2)
    return float(np.nanmean(vals)) if vals else np.nan


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals: List[float] = []
    for j in range(y_true.shape[1]):
        var = float(np.var(y_true[:, j]))
        if var < 1e-12:
            vals.append(np.nan)
            continue
        mse = float(np.mean((y_true[:, j] - y_pred[:, j])**2))
        vals.append(1.0 - mse / var)
    return float(np.nanmean(vals)) if vals else np.nan


def prepare_trainer(model_key: str, solver_name: str, tee: bool) -> Tuple:
    solver_spec = SolverSpec(name=solver_name, options={}, tee=tee)
    trainer = get_trainer(model_key, solver_spec)
    return trainer, solver_spec


def collect_indices(idx: Sequence[int], Vhats: Sequence[np.ndarray],
                    burn_in: int, n_tr: int) -> Tuple[List[int], List[np.ndarray], List[int], List[np.ndarray]]:
    train_pairs = [(i, V) for i, V in zip(idx, Vhats) if burn_in <= i < burn_in + n_tr]
    test_pairs = [(i, V) for i, V in zip(idx, Vhats) if i >= burn_in + n_tr]
    train_idx = [i for i, _ in train_pairs]
    test_idx = [i for i, _ in test_pairs]
    V_train = [V for _, V in train_pairs]
    V_test = [V for _, V in test_pairs]
    return train_idx, V_train, test_idx, V_test


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
            reg_theta_l2: float) -> ScaleMetrics:

    X, Y, V_true, theta0, tau = generate_simulation1_dataset(
        n_samples=N, n_assets=d, snr=snr, rho=rho, sigma=sigma, seed=seed
    )

    burn_in = int(res)
    if burn_in >= N:
        raise ValueError("burn_in(res) >= N")

    n_eff = N - burn_in
    n_tr = n_eff // 2

    if use_true_cov:
        idx = list(range(burn_in, N))
        Vhats = [V_true.copy() for _ in idx]
    else:
        Vhats, idx = estimate_epscov_rolling(
            Y, X, theta0, tau, res=res, include_current=False
        )
        if len(Vhats) == 0:
            raise RuntimeError("Covariance estimation returned empty sequence.")

    train_idx, V_train, test_idx, V_test = collect_indices(idx, Vhats, burn_in, n_tr)
    if not train_idx or not test_idx:
        raise RuntimeError("Insufficient train/test indices after rolling window.")

    X_tr = X[burn_in: burn_in + n_tr]
    Y_tr = Y[burn_in: burn_in + n_tr]

    theta_init = train_ols(X_tr, Y_tr)

    if model_key == "ols":
        theta_hat = theta_init
    else:
        trainer, solver_spec = prepare_trainer(model_key, solver, tee)
        result = trainer(
            X, Y, Vhats, idx,
            start_index=train_idx[0],
            end_index=train_idx[-1],
            delta=delta,
            theta_init=theta_init,
            tee=tee,
            reg_theta_l2=reg_theta_l2,
        )
        if len(result) < 5:
            raise RuntimeError("Trainer returned unexpected result.")
        theta_hat = result[0]

    Yhat_all = predict_yhat(X, theta_hat)

    y_train = Y[train_idx]
    yhat_train = Yhat_all[train_idx]
    y_test = Y[test_idx]
    yhat_test = Yhat_all[test_idx]

    ratios_train = compute_std_ratio(y_train, yhat_train)
    ratios_test = compute_std_ratio(y_test, yhat_test)

    metrics = ScaleMetrics(
        seed=seed,
        model=model_key,
        std_ratio_mean_test=float(np.nanmean(ratios_test)),
        std_ratio_mean_train=float(np.nanmean(ratios_train)),
        corr2_test=compute_corr2(y_test, yhat_test),
        corr2_train=compute_corr2(y_train, yhat_train),
        r2_test=compute_r2(y_test, yhat_test),
        r2_train=compute_r2(y_train, yhat_train),
        std_ratio_assets_test=[float(x) for x in ratios_test.tolist()],
        std_ratio_assets_train=[float(x) for x in ratios_train.tolist()],
    )
    return metrics


def summarize(metrics: List[ScaleMetrics]) -> Dict[str, float]:
    def mean(values: List[float]) -> float:
        arr = np.array(values, dtype=float)
        return float(np.nanmean(arr)) if arr.size else np.nan

    return {
        "std_ratio_mean_test": mean([m.std_ratio_mean_test for m in metrics]),
        "std_ratio_mean_train": mean([m.std_ratio_mean_train for m in metrics]),
        "corr2_test": mean([m.corr2_test for m in metrics]),
        "corr2_train": mean([m.corr2_train for m in metrics]),
        "r2_test": mean([m.r2_test for m in metrics]),
        "r2_train": mean([m.r2_train for m in metrics]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Scale diagnostics without modifying run.py")
    ap.add_argument("--models", type=str, default="dual,ols",
                    help="Comma separated list of models (dual,kkt,ols)")
    ap.add_argument("--solver", type=str, default="ipopt", help="Solver for DFL models")
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
    ap.add_argument("--use-true-cov", action="store_true", help="Use V_true instead of rolling estimate")
    ap.add_argument("--reg-theta-l2", type=float, default=0.0)
    ap.add_argument("--out-csv", type=str, default=None, help="Optional path to write per-run metrics")

    args = ap.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    metrics_by_model: Dict[str, List[ScaleMetrics]] = {m: [] for m in models}

    for model in models:
        print(f"\n=== Model: {model} ===")
        for k in range(args.runs):
            seed = args.seed0 + k
            metrics = run_one(
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
            )
            metrics_by_model[model].append(metrics)
            ratios = ", ".join(f"{r:.6f}" if np.isfinite(r) else "nan" for r in metrics.std_ratio_assets_test)
            print(f" seed={seed}  std_ratio(test)={metrics.std_ratio_mean_test:.6f}  "
                  f"corr2(test)={metrics.corr2_test:.6f}  r2(test)={metrics.r2_test:.6f}  "
                  f"[per-asset {ratios}]")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            header = ["seed", "model", "std_ratio_mean_test", "std_ratio_mean_train",
                      "corr2_test", "corr2_train", "r2_test", "r2_train",
                      "std_ratio_assets_test", "std_ratio_assets_train"]
            f.write(",".join(header) + "\n")
            for model, rows in metrics_by_model.items():
                for m in rows:
                    row = m.to_row()
                    f.write(",".join(str(row[h]) for h in header) + "\n")
        print(f"\nSaved per-run metrics to {out_path}")

    print("\n=== Summary ===")
    for model, rows in metrics_by_model.items():
        summary = summarize(rows)
        print(f" {model}: "
              f"std_ratio_test={summary['std_ratio_mean_test']:.6f}, "
              f"std_ratio_train={summary['std_ratio_mean_train']:.6f}, "
              f"corr2_test={summary['corr2_test']:.6f}, "
              f"r2_test={summary['r2_test']:.6f}")


if __name__ == "__main__":
    main()
