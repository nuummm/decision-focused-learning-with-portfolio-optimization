from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Avoid GUI backends (safe for multiprocessing / threads on macOS)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getenv("TMPDIR", "/tmp"), "mplconfig"))

from dfl_portfolio.real_data.loader import MarketLoaderConfig
from dfl_portfolio.real_data.pipeline import PipelineConfig, build_data_bundle
from dfl_portfolio.real_data.cli import make_output_dir, parse_tickers, parse_trading_cost_map


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
RESULTS_BASE = PROJECT_ROOT / "results"
RESULTS_ROOT = RESULTS_BASE / "exp_localopt_A"


def _parse_int_list(value: str) -> List[int]:
    text = (value or "").strip()
    if not text:
        raise ValueError("seeds must be a comma-separated list of integers")
    out: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("seeds must contain at least one integer")
    return out


def _is_ok(status: object) -> bool:
    s = str(status or "").lower()
    return ("optimal" in s) or ("ok" in s) or ("completed" in s)


def _warn_flags_flex(reb_df: pd.DataFrame) -> Tuple[int, int, int]:
    # success / warning / no-solution counts per retrain event
    success = 0
    warning = 0
    no_solution = 0
    for _, row in reb_df.iterrows():
        status = row.get("solver_status", "")
        if not status:
            no_solution += 1
            continue
        if _is_ok(status):
            success += 1
        else:
            warning += 1
    return success, warning, no_solution


def _warn_flags_ipo_grad(reb_df: pd.DataFrame) -> Tuple[int, int, int]:
    success = 0
    warning = 0
    no_solution = 0
    eq_tol = 1e-4
    ineq_tol = 1e-8
    for _, row in reb_df.iterrows():
        status = row.get("solver_status", "")
        if not status:
            no_solution += 1
            continue
        eq = float(row.get("train_eq_viol_max", np.nan))
        ineq = float(row.get("train_ineq_viol_max", np.nan))
        if (np.isfinite(eq) and abs(eq) > eq_tol) or (np.isfinite(ineq) and abs(ineq) > ineq_tol):
            warning += 1
        elif _is_ok(status):
            success += 1
        else:
            warning += 1
    return success, warning, no_solution


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, "n": int(arr.size)}


def _plot_cumret_overlay(curves: Dict[int, pd.DataFrame], out_path: Path, title: str) -> None:
    if not curves:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    seeds = sorted(curves.keys())
    for seed in seeds:
        df = curves[seed]
        ax.plot(pd.to_datetime(df["date"]), df["wealth"], alpha=0.35, linewidth=1.0, label=None)
    # Mean curve (align by date)
    merged = None
    for seed in seeds:
        tmp = curves[seed][["date", "wealth"]].rename(columns={"wealth": f"w_{seed}"})
        merged = tmp if merged is None else merged.merge(tmp, on="date", how="inner")
    if merged is not None and merged.shape[0] > 0:
        wealth_cols = [c for c in merged.columns if c.startswith("w_")]
        mean_w = merged[wealth_cols].mean(axis=1)
        ax.plot(pd.to_datetime(merged["date"]), mean_w, color="black", linewidth=2.0, label="mean")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("wealth")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    formulation: Optional[str] = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment A: local-optimality robustness (multi-seed)")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--tickers", type=str, default="SPY,GLD,EEM,TLT")
    p.add_argument("--start", type=str, default="2006-01-01")
    p.add_argument("--end", type=str, default="2025-12-01")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--price-field", type=str, default="Close")
    p.add_argument("--return-kind", type=str, default="log", choices=["simple", "log"])
    p.add_argument("--frequency", type=str, default="weekly", choices=["daily", "weekly", "monthly"])
    p.add_argument("--resample-rule", type=str, default="W-FRI")
    p.add_argument("--momentum-window", type=int, default=26)
    p.add_argument("--return-horizon", type=int, default=1)
    p.add_argument("--cov-window", type=int, default=13)
    p.add_argument("--cov-method", type=str, default="oas", choices=["diag", "oas", "robust_lw", "mini_factor"])
    p.add_argument("--cov-shrinkage", type=float, default=0.94)
    p.add_argument("--cov-eps", type=float, default=1e-6)
    p.add_argument("--cov-robust-huber-k", type=float, default=1.5)
    p.add_argument("--cov-factor-rank", type=int, default=1)
    p.add_argument("--cov-factor-shrinkage", type=float, default=0.5)
    p.add_argument("--cov-ewma-alpha", type=float, default=0.97)
    p.add_argument("--no-auto-adjust", action="store_true")
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--train-window", type=int, default=26)
    p.add_argument("--rebal-interval", type=int, default=4)
    p.add_argument("--delta-up", type=float, default=0.5)
    p.add_argument("--delta-down", type=float, default=0.5)
    p.add_argument(
        "--trading-cost-per-asset",
        type=str,
        default="",
        help="Optional overrides like 'SPY:5,GLD:8' (basis points) applied per ticker.",
    )
    p.add_argument("--outdir", type=Path, default=None)

    # Flex solver
    p.add_argument("--flex-solver", type=str, default="knitro")
    p.add_argument("--flex-maxtime", type=float, default=180.0)
    p.add_argument("--flex-aux-init-sigma-w", type=float, default=0.05)
    p.add_argument("--flex-aux-init-sigma-lam", type=float, default=1e-2)
    p.add_argument("--flex-aux-init-sigma-mu", type=float, default=1e-2)

    # IPO-GRAD
    p.add_argument("--ipo-grad-epochs", type=int, default=500)
    p.add_argument("--ipo-grad-lr", type=float, default=1e-3)
    p.add_argument("--ipo-grad-batch-size", type=int, default=32)
    p.add_argument("--ipo-grad-qp-max-iter", type=int, default=5000)
    p.add_argument("--ipo-grad-qp-tol", type=float, default=1e-6)
    p.add_argument("--tee", action="store_true")
    p.add_argument("--debug-roll", action="store_true", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    # Heavy imports after parsing so `--help` stays lightweight.
    from dfl_portfolio.real_data.reporting import MODEL_COLOR_MAP, display_model_name
    from dfl_portfolio.experiments.real_data_run import run_rolling_experiment
    from dfl_portfolio.registry import SolverSpec, KNITRO_DEFAULTS
    seeds = _parse_int_list(args.seeds)

    raw_asset_costs = parse_trading_cost_map(args.trading_cost_per_asset) if args.trading_cost_per_asset else {}
    asset_cost_overrides_dec = {k.upper(): float(v) / 10000.0 for k, v in raw_asset_costs.items()}
    trading_costs_enabled = bool(asset_cost_overrides_dec)

    tickers = parse_tickers(args.tickers)
    loader_cfg = MarketLoaderConfig.for_cli(
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        price_field=args.price_field,
        return_kind=args.return_kind,
        frequency=args.frequency,
        resample_rule=args.resample_rule,
        momentum_window=args.momentum_window,
        return_horizon=args.return_horizon,
        cov_window=args.cov_window,
        cov_method=args.cov_method,
        cov_shrinkage=args.cov_shrinkage,
        cov_eps=args.cov_eps,
        cov_robust_huber_k=args.cov_robust_huber_k,
        cov_factor_rank=args.cov_factor_rank,
        cov_factor_shrinkage=args.cov_factor_shrinkage,
        cov_ewma_alpha=args.cov_ewma_alpha,
        auto_adjust=not args.no_auto_adjust,
        cache_dir=None,
        force_refresh=args.force_refresh,
        debug=False,
        train_window=args.train_window,
    )
    bundle = build_data_bundle(PipelineConfig(loader=loader_cfg, debug=False))

    outdir = make_output_dir(RESULTS_ROOT, args.outdir)
    analysis_dir = outdir / "analysis"
    analysis_csv = analysis_dir / "csv"
    analysis_fig = analysis_dir / "figures"
    analysis_csv.mkdir(parents=True, exist_ok=True)
    analysis_fig.mkdir(parents=True, exist_ok=True)

    # Model specs for A
    models = [
        ModelSpec(key="flex", label="flex_dual", formulation="dual"),
        ModelSpec(key="flex", label="flex_kkt", formulation="kkt"),
        ModelSpec(key="ipo_grad", label="ipo_grad"),
    ]

    # Solver specs (flex uses knitro; ipo_grad uses analytic wrapper)
    knitro_opts = dict(KNITRO_DEFAULTS)
    knitro_opts["maxtime_real"] = float(args.flex_maxtime)
    flex_solver_spec = SolverSpec(name=args.flex_solver, options=knitro_opts, tee=bool(args.tee))
    analytic_spec = SolverSpec(name="analytic", tee=bool(args.tee))

    # Options
    flex_base_options: Dict[str, Any] = {
        "theta_init_mode": "ipo",  # keep theta fixed across seeds
        "theta_anchor_mode": "ipo",
        "lambda_theta_anchor": 0.0,
        "lambda_theta_iso": 0.0,
        "aux_init_mode": "random",
        "aux_init_sigma_w": float(args.flex_aux_init_sigma_w),
        "aux_init_sigma_lam": float(args.flex_aux_init_sigma_lam),
        "aux_init_sigma_mu": float(args.flex_aux_init_sigma_mu),
    }
    ipo_grad_options: Dict[str, Any] = {
        "ipo_grad_epochs": int(args.ipo_grad_epochs),
        "ipo_grad_lr": float(args.ipo_grad_lr),
        "ipo_grad_batch_size": int(args.ipo_grad_batch_size),
        "ipo_grad_qp_max_iter": int(args.ipo_grad_qp_max_iter),
        "ipo_grad_qp_tol": float(args.ipo_grad_qp_tol),
    }

    # Aggregate outputs
    run_rows: List[Dict[str, Any]] = []
    retrain_frames: List[pd.DataFrame] = []
    wealth_curves: Dict[str, Dict[int, pd.DataFrame]] = {m.label: {} for m in models}

    total_jobs = len(seeds) * len(models)
    job_idx = 0
    job_times: List[float] = []
    print(f"[local-opt-A] start: seeds={len(seeds)} models={len(models)} jobs={total_jobs}")
    print(f"[local-opt-A] outdir={outdir}")

    for seed_i, seed in enumerate(seeds, start=1):
        print(f"[local-opt-A] seed {seed_i}/{len(seeds)} = {seed}")
        seed_dir = outdir / f"seed_{seed}"
        seed_debug = seed_dir / "debug"
        seed_models = seed_dir / "models"
        seed_debug.mkdir(parents=True, exist_ok=True)
        seed_models.mkdir(parents=True, exist_ok=True)

        for spec in models:
            job_idx += 1
            avg = (sum(job_times) / len(job_times)) if job_times else float("nan")
            eta_sec = (total_jobs - job_idx + 1) * avg if job_times else float("nan")
            eta_str = f"{eta_sec/60:.1f}m" if job_times else "n/a"
            print(f"[local-opt-A] job {job_idx}/{total_jobs} seed={seed} model={spec.label} (eta={eta_str})")
            solver_spec = flex_solver_spec if spec.key == "flex" else analytic_spec
            flex_options = None
            if spec.key == "flex":
                flex_options = dict(flex_base_options)
                flex_options["formulation"] = spec.formulation
            results_model_dir = seed_models / spec.label
            start_time = time.perf_counter()
            run_result = run_rolling_experiment(
                model_key=spec.key,
                model_label=spec.label,
                bundle=bundle,
                delta_up=float(args.delta_up),
                delta_down_candidates=[float(args.delta_down)],
                trading_cost_enabled=trading_costs_enabled,
                asset_cost_overrides=asset_cost_overrides_dec,
                solver_spec=solver_spec,
                flex_options=flex_options,
                spo_plus_options=None,
                ipo_grad_options=ipo_grad_options if spec.key == "ipo_grad" else None,
                train_window=int(args.train_window),
                rebal_interval=int(args.rebal_interval),
                debug_roll=bool(args.debug_roll),
                debug_dir=seed_debug,
                results_model_dir=results_model_dir,
                tee=bool(args.tee),
                asset_pred_dir=None,
                eval_start=pd.Timestamp(args.start),
                ipo_grad_debug_kkt=False,
                base_seed=int(seed),
            )
            elapsed_total = time.perf_counter() - start_time
            job_times.append(float(elapsed_total))

            stats = dict(run_result["stats"])
            stats["seed"] = int(seed)
            stats["model_key"] = spec.label
            stats["elapsed_total_run_sec"] = float(elapsed_total)

            reb_df = run_result.get("rebalance_df", pd.DataFrame())
            if isinstance(reb_df, pd.DataFrame) and not reb_df.empty:
                reb_df = reb_df.copy()
                reb_df["seed"] = int(seed)
                reb_df["model_key"] = spec.label
                retrain_frames.append(reb_df)
                fit_times = reb_df["elapsed_sec"].astype(float).to_numpy()
                stats["elapsed_total_fit_time"] = float(np.nansum(fit_times))
                stats["elapsed_fit_time_mean"] = float(np.nanmean(fit_times))
                stats["elapsed_fit_time_min"] = float(np.nanmin(fit_times))
                stats["elapsed_fit_time_max"] = float(np.nanmax(fit_times))
                if spec.key == "ipo_grad":
                    succ, warn, fail = _warn_flags_ipo_grad(reb_df)
                else:
                    succ, warn, fail = _warn_flags_flex(reb_df)
                stats["success_count"] = int(succ)
                stats["warning_count"] = int(warn)
                stats["no_solution_count"] = int(fail)
            else:
                stats["elapsed_total_fit_time"] = float("nan")
                stats["elapsed_fit_time_mean"] = float("nan")
                stats["elapsed_fit_time_min"] = float("nan")
                stats["elapsed_fit_time_max"] = float("nan")
                stats["success_count"] = 0
                stats["warning_count"] = 0
                stats["no_solution_count"] = 0

            run_rows.append(stats)
            wealth_df = run_result["wealth_df"][["date", "wealth"]].copy()
            wealth_curves[spec.label][int(seed)] = wealth_df
            print(
                f"[local-opt-A] done job {job_idx}/{total_jobs} seed={seed} model={spec.label} "
                f"elapsed={elapsed_total:.1f}s warn={stats.get('warning_count','?')} fail={stats.get('no_solution_count','?')}"
            )

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(analysis_csv / "runs.csv", index=False)
    if retrain_frames:
        retrain_df = pd.concat(retrain_frames, ignore_index=True)
        retrain_df.to_csv(analysis_csv / "retrain_log.csv", index=False)
    else:
        retrain_df = pd.DataFrame()

    # Summary table (mean/std over seeds)
    summary_rows: List[Dict[str, Any]] = []
    metrics = [
        "ann_return",
        "total_return",
        "terminal_wealth",
        "sharpe",
        "sortino",
        "ann_volatility",
        "max_drawdown",
        "cvar_95",
        "avg_turnover",
        "avg_trading_cost",
        "r2",
        "elapsed_total_fit_time",
        "elapsed_fit_time_mean",
        "warning_count",
        "no_solution_count",
    ]
    for model in sorted(runs_df["model_key"].unique()):
        part = runs_df[runs_df["model_key"] == model]
        row: Dict[str, Any] = {"model_key": model, "model": display_model_name(model)}
        for col in metrics:
            stats = _summary_stats(part[col].astype(float).to_numpy()) if col in part.columns else {"mean": np.nan, "std": np.nan, "n": 0}
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"] = stats["std"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(analysis_csv / "summary_table.csv", index=False)

    # Plot cumret overlays per model
    for model_label, curves in wealth_curves.items():
        if not curves:
            continue
        title = f"A: cumulative wealth overlay ({model_label})"
        color = MODEL_COLOR_MAP.get(model_label, None)
        _plot_cumret_overlay(curves, analysis_fig / f"cumret_{model_label}.png", title=title)

    # Save config
    cfg = vars(args).copy()
    cfg["seeds"] = seeds
    (analysis_csv / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[local-opt-A] finished. outputs -> {outdir}")


if __name__ == "__main__":  # pragma: no cover
    main()


"""
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.local_opt_study_a \
  --seeds "0,1,2,3,4,5,6,7,8,9" \
  --tickers "SPY,GLD,EEM,TLT" \
  --start 2006-01-01 --end 2025-12-01 \
  --flex-solver knitro \
"""
