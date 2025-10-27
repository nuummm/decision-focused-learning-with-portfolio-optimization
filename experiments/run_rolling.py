from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

# from data.synthetic import generate_simulation1_dataset
from data_stress.synthetic import generate_simulation1_dataset
from models.ols import train_ols, predict_yhat
from models.ols_gurobi import solve_mvo_gurobi
from models.dfl_p1_flex import fit_dfl_p1_flex


def mvo_cost(w: np.ndarray, y: np.ndarray, V: np.ndarray, delta: float = 1.0) -> float:
    return float(-w @ y + 0.5 * delta * (w @ V @ w))


def rolling_shrink_cov(
    Y: np.ndarray,
    window: int,
    alpha: float,
    eps: float,
) -> Tuple[List[np.ndarray], List[int]]:
    Y = np.asarray(Y, dtype=float)
    N, d = Y.shape
    if window <= 0 or window > N:
        raise ValueError("window must satisfy 0 < window <= N")
    Vhats: List[np.ndarray] = []
    idx: List[int] = []
    for t in range(window - 1, N):
        block = Y[t - window + 1 : t + 1]
        cov = np.cov(block, rowvar=False, bias=True)
        shrink = alpha * np.diag(np.diag(cov)) + (1.0 - alpha) * cov
        Vhat = 0.5 * (shrink + shrink.T) + eps * np.eye(d)
        Vhats.append(Vhat)
        idx.append(t)
    return Vhats, idx


@dataclass
class StepResult:
    seed: int
    t: int
    cost_pred: float
    cost_opt: float
    regret: float
    corr2: float
    r2: float
    train_time: float
    solve_time: float
    status: str


def compute_corr2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    vals: List[float] = []
    for j in range(y.shape[0]):
        if np.std(y[j]) < 1e-12 or np.std(yhat[j]) < 1e-12:
            vals.append(np.nan)
        else:
            vals.append(np.corrcoef(y[j], yhat[j])[0, 1] ** 2)
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def single_run(
    seed: int,
    model: str,
    N: int,
    d: int,
    snr: float,
    rho: float,
    sigma: float,
    res: int,
    min_train: int,
    delta: float,
    cov_alpha: float,
    cov_eps: float,
    flex_options: Optional[Dict[str, float]],
) -> Tuple[List[StepResult], Dict[str, float]]:
    X, Y, V_true, theta0, tau = generate_simulation1_dataset(
        n_samples=N, n_assets=d, snr=snr, rho=rho, sigma=sigma, seed=seed
    )

    Vhats_all, idx_all = rolling_shrink_cov(Y, window=res, alpha=cov_alpha, eps=cov_eps)
    idx_to_V = dict(zip(idx_all, Vhats_all))

    results: List[StepResult] = []
    for t in range(min_train, N):
        if t not in idx_to_V:
            continue
        train_start = 0
        train_end = t - 1
        if train_end < train_start:
            continue

        train_time = 0.0
        if model == "ols":
            X_tr = X[train_start : train_end + 1]
            Y_tr = Y[train_start : train_end + 1]
            t0 = time.perf_counter()
            theta_hat = train_ols(X_tr, Y_tr)
            train_time = time.perf_counter() - t0
        elif model == "flex":
            opts = dict(flex_options or {})
            opts.setdefault("formulation", "dual")
            lam_theta_anchor = float(opts.get("lambda_theta_anchor", 0.0) or 0.0)
            if lam_theta_anchor > 0.0:
                X_tr = X[train_start : train_end + 1]
                Y_tr = Y[train_start : train_end + 1]
                theta_anchor = train_ols(X_tr, Y_tr)
                opts["theta_anchor"] = theta_anchor
            t0 = time.perf_counter()
            try:
                theta_hat, *_ = fit_dfl_p1_flex(
                    X,
                    Y,
                    Vhats_all,
                    idx_all,
                    start_index=idx_all[0],
                    end_index=train_end,
                    delta=delta,
                    theta_init=None,
                    solver=opts.get("solver", "ipopt"),
                    solver_options=opts.get("solver_options", {}),
                    tee=bool(opts.get("tee", False)),
                    formulation=opts.get("formulation", "dual"),
                    lambda_theta_anchor=lam_theta_anchor,
                    theta_anchor=opts.get("theta_anchor"),
                    lambda_w_anchor=float(opts.get("lambda_w_anchor", 0.0) or 0.0),
                    w_anchor=opts.get("w_anchor"),
                    lambda_w_iso=float(opts.get("lambda_w_iso", 0.0) or 0.0),
                    dro_rho=float(opts.get("dro_rho", 0.0) or 0.0),
                )
            except Exception as e:
                train_time = time.perf_counter() - t0
                results.append(
                    StepResult(
                        seed=seed,
                        t=t,
                        cost_pred=float("nan"),
                        cost_opt=float("nan"),
                        regret=float("nan"),
                        corr2=float("nan"),
                        r2=float("nan"),
                        train_time=train_time,
                        solve_time=0.0,
                        status=f"solver_error:{e}",
                    )
                )
                continue
            train_time = time.perf_counter() - t0
        else:
            raise ValueError(f"unknown model: {model}")

        yhat_t = predict_yhat(X[[t]], theta_hat)[0]
        Vhat_t = idx_to_V[t]

        t1 = time.perf_counter()
        w_t = solve_mvo_gurobi(yhat_t, Vhat_t, delta=delta)
        solve_time = time.perf_counter() - t1
        status = "ok"
        if w_t is None:
            status = "mvo_failed"
            cost_pred = float("nan")
            cost_opt = float("nan")
            regret = float("nan")
        else:
            cost_pred = mvo_cost(w_t, Y[t], V_true, delta=delta)
            w_opt = solve_mvo_gurobi(Y[t], Vhat_t, delta=delta)
            if w_opt is None:
                cost_opt = float("nan")
                regret = float("nan")
            else:
                cost_opt = mvo_cost(w_opt, Y[t], V_true, delta=delta)
                regret = cost_pred - cost_opt

        corr = compute_corr2(Y[[t]].T, yhat_t.reshape(-1, 1))
        r2 = 1.0 - np.mean((Y[t] - yhat_t) ** 2) / (np.var(Y[t]) + 1e-12)

        results.append(
            StepResult(
                seed=seed,
                t=t,
                cost_pred=cost_pred,
                cost_opt=cost_opt,
                regret=regret,
                corr2=corr,
                r2=r2,
                train_time=train_time,
                solve_time=solve_time,
                status=status,
            )
        )

    summary = {}
    if results:
        costs = [r.cost_pred for r in results if np.isfinite(r.cost_pred)]
        regs = [r.regret for r in results if np.isfinite(r.regret)]
        summary = {
            "mean_cost": float(np.mean(costs)) if costs else float("nan"),
            "mean_regret": float(np.mean(regs)) if regs else float("nan"),
            "steps": len(results),
        }
    return results, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling window experiment (OLS vs Flex)")
    ap.add_argument("--model", type=str, choices=["ols", "flex"], required=True)
    ap.add_argument("--solver", type=str, default="ipopt")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--snr", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.0125)
    ap.add_argument("--res", type=int, default=20, help="covariance window size")
    ap.add_argument("--min-train", type=int, default=100)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--cov-alpha", type=float, default=0.1)
    ap.add_argument("--cov-eps", type=float, default=1e-4)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--outdir", type=str, default="results/rolling_exp")
    ap.add_argument("--flex-formulation", type=str, default="dual")
    ap.add_argument("--flex-lambda-theta-anchor", type=float, default=0.0)
    ap.add_argument("--flex-lambda-w-anchor", type=float, default=0.0)
    ap.add_argument("--flex-lambda-w-iso", type=float, default=0.0)
    ap.add_argument("--flex-dro-rho", type=float, default=0.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    flex_opts = None
    if args.model == "flex":
        flex_opts = {
            "formulation": args.flex_formulation,
            "lambda_theta_anchor": args.flex_lambda_theta_anchor,
            "lambda_w_anchor": args.flex_lambda_w_anchor,
            "lambda_w_iso": args.flex_lambda_w_iso,
            "dro_rho": args.flex_dro_rho,
            "solver": args.solver,
        }

    all_rows = []
    summary_rows = []
    for k in range(args.runs):
        seed = args.seed0 + k
        step_results, summary = single_run(
            seed=seed,
            model=args.model,
            N=args.N,
            d=args.d,
            snr=args.snr,
            rho=args.rho,
            sigma=args.sigma,
            res=args.res,
            min_train=args.min_train,
            delta=args.delta,
            cov_alpha=args.cov_alpha,
            cov_eps=args.cov_eps,
            flex_options=flex_opts,
        )
        for r in step_results:
            all_rows.append(r)
        summary_rows.append({"seed": seed, **summary})
        print(f"[RUN] seed={seed} steps={summary.get('steps',0)} mean_cost={summary.get('mean_cost', float('nan')):.4f} mean_regret={summary.get('mean_regret', float('nan')):.4f}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    per_step_csv = outdir / f"{args.model}_rolling_{timestamp}_steps.csv"
    summary_csv = outdir / f"{args.model}_rolling_{timestamp}_summary.csv"

    with per_step_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "t", "cost_pred", "cost_opt", "regret", "corr2", "r2", "train_time", "solve_time", "status"])
        for r in all_rows:
            writer.writerow([r.seed, r.t, r.cost_pred, r.cost_opt, r.regret, r.corr2, r.r2, r.train_time, r.solve_time, r.status])

    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean_cost", "mean_regret", "steps"])
        for row in summary_rows:
            writer.writerow([row["seed"], row.get("mean_cost", float("nan")), row.get("mean_regret", float("nan")), row.get("steps", 0)])

    print(f"[SAVED] per-step CSV: {per_step_csv}")
    print(f"[SAVED] summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
