from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dfl_portfolio.real_data.loader import MarketLoaderConfig
from dfl_portfolio.real_data.pipeline import (
    PipelineConfig,
    build_data_bundle,
)
from dfl_portfolio.real_data.cli import (
    build_parser,
    make_output_dir,
    parse_commalist,
    parse_model_train_window_spec,
    parse_tickers,
)
from dfl_portfolio.real_data.reporting import (
    WEIGHT_THRESHOLD,
    WEIGHT_PLOT_MAX_POINTS,
    PERIOD_WINDOWS,
    compute_correlation_stats,
    compute_period_metrics,
    compute_pairwise_mean_return_tests,
    compute_pairwise_performance_tests,
    compute_sortino_ratio,
    compute_cvar,
    display_model_name,
    export_average_weights,
    export_weight_threshold_frequency,
    export_weight_variance_correlation,
    export_max_return_winner_counts,
    max_drawdown,
    plot_asset_correlation,
    plot_flex_solver_debug,
    plot_multi_wealth,
    plot_time_series,
    plot_wealth_correlation_heatmap,
    plot_wealth_curve,
    plot_wealth_with_events,
    plot_wealth_window_normalized,
    plot_weight_comparison,
    plot_weight_histograms,
    plot_weight_paths,
    plot_delta_paths,
    plot_condition_numbers,
    update_experiment_ledger,
    run_extended_analysis,
    summarize_dfl_performance_significance,
    format_summary_for_output,
    build_cost_adjusted_summary,
)
from dfl_portfolio.registry import SolverSpec, get_trainer
from dfl_portfolio.models.ols import predict_yhat, train_ols
from dfl_portfolio.models.ipo_closed_form import fit_ipo_closed_form
from dfl_portfolio.models.ols_gurobi import solve_mvo_gurobi, solve_series_mvo_gurobi
from dfl_portfolio.models.dfl_p1_flex_delta import fit_dfl_p1_flex_delta
from dfl_portfolio.experiments.real_data_common import (
    mvo_cost,
    ScheduleItem,
    build_rebalance_schedule,
    prepare_flex_training_args,
    resolve_trading_cost_rates,
)
from dfl_portfolio.experiments.real_data_benchmarks import run_benchmark_suite

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
RESULTS_BASE = PROJECT_ROOT / "results"
RESULTS_ROOT = RESULTS_BASE / "exp_real_data_delta"
DEBUG_ROOT = RESULTS_BASE / "debug_outputs_delta"

# Suppress Pyomo-related warnings (only show errors)
logging.getLogger("pyomo").setLevel(logging.ERROR)
logging.getLogger("pyomo.solvers").setLevel(logging.ERROR)


def mirror_to_analysis_root(src: Path, analysis_dir: Path) -> None:
    """Copy ``src`` to the top-level ``analysis_dir`` for quick access."""

    if not src.exists():
        return
    dest = analysis_dir / src.name
    if dest == src:
        return
    shutil.copy2(src, dest)


def train_model_window(
    model_key: str,
    trainer,
    bundle,
    delta_up: float,
    delta_down: float,
    delta_min: float,
    delta_max: float,
    lambda_delta_up: float,
    solver_spec: SolverSpec,
    flex_options: Dict[str, Any] | None,
    train_start: int,
    train_end: int,
    tee: bool,
):
    delta_for_training = delta_down if model_key == "flex" else delta_up
    trainer_kwargs: Dict[str, object] = dict(
        X=bundle.dataset.X,
        Y=bundle.dataset.Y,
        Vhats=bundle.covariances,
        idx=bundle.cov_indices.tolist(),
        start_index=train_start,
        end_index=train_end,
        delta=delta_for_training,
        tee=tee,
    )
    theta_init_override: Optional[np.ndarray] = None
    if model_key == "flex" and flex_options:
        theta_init_override, resolved_flex, _ = prepare_flex_training_args(
            bundle, train_start, train_end, delta_for_training, tee, flex_options
        )
        trainer_kwargs.update(resolved_flex)
    if theta_init_override is not None:
        trainer_kwargs["theta_init"] = theta_init_override

    start_time = time.perf_counter()
    # flex だけは delta 学習版を使う（solver_spec に従ってソルバーを切り替える）
    if model_key == "flex":
        solver_name = solver_spec.name or "knitro"
        solver_opts = solver_spec.options or None
        trainer_ret = fit_dfl_p1_flex_delta(
            **trainer_kwargs,
            solver=solver_name,
            solver_options=solver_opts,
            delta_min=delta_min,
            delta_max=delta_max,
            lambda_delta_up=lambda_delta_up,
            delta_up=delta_up,
        )
    else:
        trainer_ret = trainer(**trainer_kwargs)
    elapsed = time.perf_counter() - start_time

    if not isinstance(trainer_ret, (list, tuple)) or len(trainer_ret) < 5:
        raise RuntimeError(f"Trainer {model_key} returned unexpected output")

    theta_hat = trainer_ret[0]
    info = trainer_ret[5] if len(trainer_ret) >= 6 else {}
    return theta_hat, info, elapsed


def run_rolling_experiment(
    model_key: str,
    model_label: str,
    bundle,
    delta_up: float,
    delta_down: float,
    delta_min: float,
    delta_max: float,
    lambda_delta_up: float,
    trading_cost_enabled: bool,
    trading_cost_default_bps: float,
    asset_cost_overrides: Dict[str, float] | None,
    solver_spec: SolverSpec,
    flex_options: Dict[str, Any] | None,
    train_window: int,
    rebal_interval: int,
    debug_roll: bool,
    debug_dir: Path,
    results_model_dir: Path,
    tee: bool,
    asset_pred_dir: Path | None = None,
    eval_start: Optional[pd.Timestamp] = None,
) -> Dict[str, object]:
    trainer = get_trainer(model_key, solver_spec)
    schedule = build_rebalance_schedule(bundle, train_window, rebal_interval, eval_start=eval_start)
    cov_lookup = {
        idx: (cov, stat)
        for idx, cov, stat in zip(
            bundle.cov_indices.tolist(), bundle.covariances, bundle.cov_stats
        )
    }

    tickers = bundle.dataset.config.tickers
    asset_cost_rates = resolve_trading_cost_rates(
        tickers,
        asset_cost_overrides or {},
        enable_default_costs=trading_cost_enabled,
        default_bps_if_missing=trading_cost_default_bps,
    )
    costs_active = bool(np.any(asset_cost_rates > 0.0))
    mean_defined_cost_bps = float(np.mean(asset_cost_rates) * 10000.0) if costs_active else 0.0

    wealth = 1.0
    wealth_net = 1.0
    wealth_dates: List[pd.Timestamp] = []
    wealth_values: List[float] = []
    wealth_net_values: List[float] = []
    wealth_labels: List[str] = []

    rebalance_rows: List[Dict[str, object]] = []
    asset_rows: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []

    theta_init: Optional[np.ndarray] = None
    X = bundle.dataset.X
    Y = bundle.dataset.Y

    prev_weights: Optional[np.ndarray] = None

    for cycle_id, item in enumerate(schedule, start=1):
        if debug_roll:
            progress = cycle_id / max(len(schedule), 1)
            bar = "#" * int(progress * 20)
            # ログは負荷軽減のため 20 サイクルごと＋最初／最後のみ出力する
            if cycle_id % 20 == 0 or cycle_id == 1 or cycle_id == len(schedule):
                print(
                    f"[roll-debug-delta] model={model_label} cycle={cycle_id}/{len(schedule)} "
                    f"idx={item.rebalance_idx} train=[{item.train_start},{item.train_end}] "
                    f"n_eval={len(item.eval_indices)} [{bar:<20}] {progress:.0%}"
                )

        start_time = time.perf_counter()
        theta_hat, info, elapsed = train_model_window(
            model_key=model_key,
            trainer=trainer,
            bundle=bundle,
            delta_up=delta_up,
            delta_down=delta_down,
            delta_min=delta_min,
            delta_max=delta_max,
            lambda_delta_up=lambda_delta_up,
            solver_spec=solver_spec,
            flex_options=flex_options,
            train_start=item.train_start,
            train_end=item.train_end,
            tee=tee,
        )
        theta_init = theta_hat
        # flex_delta では学習された delta_hat を meta から取得し、評価にも使う
        default_delta = delta_down if model_key == "flex" else delta_up
        cycle_delta = float(info.get("delta_hat", default_delta))
        rebalance_rows.append(
            {
                "cycle": cycle_id,
                "model": model_label,
                "rebalance_idx": item.rebalance_idx,
                "rebalance_date": bundle.dataset.timestamps[item.rebalance_idx].isoformat(),
                "train_start": item.train_start,
                "train_end": item.train_end,
            "n_eval": len(item.eval_indices),
            "elapsed_sec": elapsed,
            "delta_used": cycle_delta,
            "solver_status": (info or {}).get("status", ""),
            "solver_term": (info or {}).get("termination_condition", ""),
            }
        )

        if not wealth_dates and item.eval_indices:
            wealth_dates.append(bundle.dataset.timestamps[item.eval_indices[0]])
            wealth_values.append(wealth)
            wealth_net_values.append(wealth_net)
            wealth_labels.append("initial")

        for eval_idx in item.eval_indices:
            if eval_idx not in cov_lookup:
                continue
            cov, stat = cov_lookup[eval_idx]
            if model_key == "flex":
                x_row = X[eval_idx]
                yhat = predict_yhat(x_row[None, :], theta_hat)[0]
            else:
                yhat = predict_yhat(X[eval_idx : eval_idx + 1], theta_hat)[0]
            z = solve_mvo_gurobi(
                y_hat=yhat,
                V_hat=cov,
                delta=cycle_delta,
                psd_eps=1e-9,
                output=False,
            )
            if z is None or np.isnan(z).any():
                continue
            realized = float(z @ bundle.dataset.Y[eval_idx])
            if prev_weights is None or prev_weights.shape != z.shape:
                turnover = 0.0
                trading_cost = 0.0
            else:
                abs_changes = np.abs(z - prev_weights)
                turnover = float(0.5 * np.sum(abs_changes))
                if costs_active and asset_cost_rates.shape[0] == abs_changes.shape[0]:
                    trading_cost = float(0.5 * np.sum(abs_changes * asset_cost_rates))
                elif costs_active:
                    trading_cost = turnover * float(np.mean(asset_cost_rates))
                else:
                    trading_cost = 0.0
            prev_weights = z.copy()
            net_return = realized - trading_cost

            wealth *= (1.0 + realized)
            wealth_net *= (1.0 + net_return)
            wealth_dates.append(bundle.dataset.timestamps[eval_idx])
            wealth_values.append(wealth)
            wealth_net_values.append(wealth_net)
            wealth_labels.append("after_step")

            cost = mvo_cost(z, bundle.dataset.Y[eval_idx], cov, cycle_delta)
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals.size == 0:
                cond_number = float("nan")
            else:
                eig_min = float(np.max([np.min(eigvals), 1e-12]))
                eig_max = float(np.max(eigvals))
                cond_number = float(eig_max / eig_min) if eig_min > 0 else float("inf")
            step_rows.append(
                {
                    "cycle": cycle_id,
                    "model": model_label,
                    "date": bundle.dataset.timestamps[eval_idx].isoformat(),
                    "eval_idx": eval_idx,
                    "eig_min": stat.eigen_min,
                    "portfolio_return": realized,
                    "trading_cost": trading_cost,
                    "net_return": net_return,
                    "wealth": wealth,
                    "wealth_net": wealth_net,
                    "mvo_cost": cost,
                    "theta": json.dumps(theta_hat.tolist()),
                    "weights": json.dumps(z.tolist()),
                    "weight_sum": float(np.sum(z)),
                    "weight_min": float(np.min(z)),
                    "weight_max": float(np.max(z)),
                    "turnover": turnover,
                    "condition_number": cond_number,
                }
            )

            tickers = bundle.dataset.config.tickers
            yhat_vec = yhat
            yreal_vec = bundle.dataset.Y[eval_idx]
            ts = bundle.dataset.timestamps[eval_idx].isoformat()
            for j, ticker in enumerate(tickers):
                asset_rows.append(
                    {
                        "cycle": cycle_id,
                        "model": display_model_name(model_label),
                        "date": ts,
                        "eval_idx": eval_idx,
                        "ticker": ticker,
                        "weight": float(z[j]) if j < z.shape[0] else np.nan,
                        "pred_ret": float(yhat_vec[j]),
                        "real_ret": float(yreal_vec[j]),
                    }
                )

    if not step_rows:
        raise RuntimeError("No evaluation steps were executed. Check train_window/rebal_interval settings.")

    step_df = pd.DataFrame(step_rows)
    returns = step_df["portfolio_return"].to_numpy()
    returns_net = step_df["net_return"].to_numpy() if "net_return" in step_df.columns else returns.copy()
    avg_turnover = (
        float(step_df["turnover"].mean()) if "turnover" in step_df.columns and not step_df.empty else float("nan")
    )
    avg_trading_cost = (
        float(step_df["trading_cost"].mean())
        if "trading_cost" in step_df.columns and not step_df.empty
        else 0.0
    )
    avg_condition = (
        float(step_df["condition_number"].mean())
        if "condition_number" in step_df.columns and not step_df.empty
        else float("nan")
    )
    mean_step = float(np.mean(returns))
    std_step = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    mean_step_net = float(np.mean(returns_net))
    std_step_net = float(np.std(returns_net, ddof=1)) if returns_net.size > 1 else 0.0
    if wealth_dates and len(wealth_dates) >= 2:
        horizon_days = (wealth_dates[-1] - wealth_dates[0]).days
        horizon_years = max(horizon_days / 365.25, 1e-9)
        steps_per_year = len(step_rows) / horizon_years
    else:
        steps_per_year = 1.0
    mean_return = mean_step * steps_per_year
    std_return = std_step * math.sqrt(steps_per_year) if std_step > 0.0 else 0.0
    sharpe = mean_return / std_return if std_return > 1e-12 else np.nan
    mean_return_net = mean_step_net * steps_per_year
    std_return_net = std_step_net * math.sqrt(steps_per_year) if std_step_net > 0.0 else 0.0
    sharpe_net = mean_return_net / std_return_net if std_return_net > 1e-12 else np.nan
    sortino_step = compute_sortino_ratio(returns)
    sortino = (
        float(sortino_step) * math.sqrt(steps_per_year)
        if np.isfinite(sortino_step)
        else np.nan
    )
    sortino_step_net = compute_sortino_ratio(returns_net)
    sortino_net = (
        float(sortino_step_net) * math.sqrt(steps_per_year)
        if np.isfinite(sortino_step_net)
        else np.nan
    )

    model_debug_dir = debug_dir / f"model_{model_label}"
    model_debug_dir.mkdir(parents=True, exist_ok=True)

    step_path = model_debug_dir / "step_log.csv"
    step_df.to_csv(step_path, index=False)

    rebalance_df = pd.DataFrame(rebalance_rows)
    rebalance_df.to_csv(model_debug_dir / "rebalance_log.csv", index=False)

    wealth_df = pd.DataFrame(
        {
            "date": wealth_dates,
            "wealth": wealth_values,
            "wealth_net": wealth_net_values,
            "label": wealth_labels,
        }
    )
    wealth_df.to_csv(model_debug_dir / "wealth.csv", index=False)
    plot_wealth_curve(wealth_dates, wealth_values, model_debug_dir / "wealth.png")
    plot_condition_numbers(step_df, model_debug_dir / "condition_numbers.png")

    # Parse weights into tidy columns
    if step_df.empty:
        weights_df = pd.DataFrame()
    else:
        weight_records: List[Dict[str, float]] = []
        tickers = bundle.dataset.config.tickers
        for _, row in step_df.iterrows():
            weights = json.loads(row["weights"])
            record = {"date": row["date"]}
            for i, ticker in enumerate(tickers):
                record[ticker] = float(weights[i]) if i < len(weights) else np.nan
            record["portfolio_return_sq"] = float(row["portfolio_return"]) ** 2
            weight_records.append(record)
        weights_df = pd.DataFrame(weight_records)

    if not results_model_dir.exists():
        results_model_dir.mkdir(parents=True, exist_ok=True)
    wealth_df.to_csv(results_model_dir / "wealth.csv", index=False)
    if not weights_df.empty:
        weights_df.to_csv(results_model_dir / "weights.csv", index=False)
        plot_weight_paths(weights_df, model_label, results_model_dir / "weights.png")
    step_df.to_csv(results_model_dir / "step_metrics.csv", index=False)
    plot_condition_numbers(step_df, results_model_dir / "condition_numbers.png")

    lengths = (rebalance_df["train_end"] - rebalance_df["train_start"] + 1).tolist()
    status_counts = rebalance_df["solver_status"].value_counts().to_dict()
    reb_summary = {
        "n_cycles": len(schedule),
        "train_length_min": int(min(lengths)) if lengths else 0,
        "train_length_max": int(max(lengths)) if lengths else 0,
        "solver_status_counts": status_counts,
        "elapsed_mean": float(rebalance_df["elapsed_sec"].mean()) if len(rebalance_df) else 0.0,
        "elapsed_max": float(rebalance_df["elapsed_sec"].max()) if len(rebalance_df) else 0.0,
    }
    (model_debug_dir / "rebalance_summary.json").write_text(
        json.dumps(reb_summary, ensure_ascii=False, indent=2)
    )

    # CVaR (Expected Shortfall) at 95%: 下側 5% の平均リターン
    cvar_95 = compute_cvar(returns, alpha=0.05)

    # 決定係数 R^2（資産リターン予測の当てはまり；銘柄リターンベースの相関の2乗）
    r2: float = float("nan")
    try:
        if asset_rows:
            asset_df_r2 = pd.DataFrame(asset_rows)
            if {"pred_ret", "real_ret"}.issubset(asset_df_r2.columns):
                y = asset_df_r2["real_ret"].astype(float).to_numpy()
                yhat = asset_df_r2["pred_ret"].astype(float).to_numpy()
                if y.size > 1:
                    corr = float(np.corrcoef(y, yhat)[0, 1])
                    if np.isfinite(corr):
                        r2 = corr * corr
    except Exception:
        r2 = float("nan")

    terminal_wealth = float(wealth_values[-1]) if wealth_values else 1.0
    terminal_wealth_net = float(wealth_net_values[-1]) if wealth_net_values else 1.0
    total_return = terminal_wealth - 1.0
    total_return_net = terminal_wealth_net - 1.0

    stats_report = {
        "model": display_model_name(model_label),
        "n_retrain": len(schedule),
        "n_invest_steps": len(step_rows),
        "ann_return": mean_return,
        "ann_volatility": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_return_net": mean_return_net,
        "ann_volatility_net": std_return_net,
        "sharpe_net": sharpe_net,
        "sortino_net": sortino_net,
        "cvar_95": cvar_95,
        "r2": r2,
        "max_drawdown": max_drawdown(wealth_values),
        "terminal_wealth": terminal_wealth,
        "terminal_wealth_net": terminal_wealth_net,
        "total_return": total_return,
        "total_return_net": total_return_net,
        "train_window": train_window,
        "rebal_interval": rebal_interval,
        "avg_turnover": avg_turnover,
        "avg_trading_cost": avg_trading_cost,
        "trading_cost_bps": mean_defined_cost_bps,
        "avg_condition_number": avg_condition,
    }

    period_metrics = compute_period_metrics(step_df)

    if asset_pred_dir is not None and asset_rows:
        asset_pred_dir = Path(asset_pred_dir)
        asset_pred_dir.mkdir(parents=True, exist_ok=True)
        asset_df = pd.DataFrame(asset_rows)
        asset_df.to_csv(asset_pred_dir / f"{model_label}.csv", index=False)

    return {
        "stats": stats_report,
        "period_metrics": period_metrics,
        "wealth_df": wealth_df,
        "weights_df": weights_df,
        "rebalance_df": rebalance_df,
        "rebalance_summary": reb_summary,
    }


def main() -> None:
    parser = build_parser()
    # delta 範囲を CLI から指定できるようにする
    parser.add_argument(
        "--delta-min",
        type=float,
        default=0.01,
        help="Lower bound of learnable delta (inclusive), must be >=0 and < delta-max.",
    )
    parser.add_argument(
        "--delta-max",
        type=float,
        default=0.99,
        help="Upper bound of learnable delta (inclusive), must be <=1 and > delta-min.",
    )
    parser.add_argument(
        "--delta-reg-up",
        type=float,
        default=0.0,
        help="Regularization weight for (delta_down - delta_up)^2.",
    )
    args = parser.parse_args()
    model_train_windows = parse_model_train_window_spec(getattr(args, "model_train_window", ""))
    benchmark_specs = parse_commalist(getattr(args, "benchmarks", ""))
    if not benchmark_specs:
        fallback: List[str] = []
        if (args.benchmark_ticker or "").strip():
            fallback.append("spy")
        if getattr(args, "benchmark_equal_weight", False):
            fallback.append("equal_weight")
        benchmark_specs = fallback

    tickers = parse_tickers(args.tickers)
    outdir = make_output_dir(RESULTS_ROOT, args.outdir)
    analysis_dir = outdir / "analysis"
    analysis_csv_dir = analysis_dir / "csv"
    analysis_fig_dir = analysis_dir / "figures"
    asset_pred_dir = analysis_csv_dir / "asset_predictions"
    analysis_csv_dir.mkdir(parents=True, exist_ok=True)
    analysis_fig_dir.mkdir(parents=True, exist_ok=True)
    asset_pred_dir.mkdir(parents=True, exist_ok=True)
    model_outputs_dir = outdir / "model_outputs"
    model_outputs_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = DEBUG_ROOT / f"{outdir.name}_rolling"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # delta 学習版でも CLI の delta は「初期値」として使う
    if not (0.0 <= args.delta_min < args.delta_max <= 1.0):
        raise ValueError(
            f"delta-min/max must satisfy 0 <= delta-min < delta-max <= 1 "
            f"(got delta-min={args.delta_min}, delta-max={args.delta_max})"
        )
    base_delta = float(args.delta)
    cli_delta_up = getattr(args, "delta_up", None)
    cli_delta_down = getattr(args, "delta_down", None)
    delta_up = float(cli_delta_up) if cli_delta_up is not None else base_delta
    delta_down = float(cli_delta_down) if cli_delta_down is not None else delta_up
    if not (args.delta_min <= delta_down <= args.delta_max):
        raise ValueError(
            f"delta_down (initial) must be within [delta-min, delta-max]; "
            f"got delta_down={delta_down}, range=[{args.delta_min},{args.delta_max}]"
        )
    trading_costs_enabled = float(getattr(args, "trading_cost_bps", 0.0)) > 0.0
    trading_cost_default_bps = float(getattr(args, "trading_cost_bps", 0.0) or 0.0)
    raw_asset_costs: Dict[str, float] = getattr(args, "trading_cost_per_asset", {}) or {}
    asset_cost_overrides_dec = {
        ticker.upper(): max(float(rate), 0.0) / 10000.0 for ticker, rate in raw_asset_costs.items()
    }

    loader_cfg = MarketLoaderConfig.for_cli(
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        price_field=args.price_field,
        return_kind=args.return_kind,  # type: ignore[arg-type]
        frequency=args.frequency,  # type: ignore[arg-type]
        resample_rule=args.resample_rule,
        momentum_window=args.momentum_window,
        return_horizon=args.return_horizon,
        cov_window=args.cov_window,
        cov_method=args.cov_method,  # type: ignore[arg-type]
        cov_shrinkage=args.cov_shrinkage,
        cov_eps=args.cov_eps,
        cov_robust_huber_k=args.cov_robust_huber_k,
        cov_factor_rank=args.cov_factor_rank,
        cov_factor_shrinkage=args.cov_factor_shrinkage,
        cov_ewma_alpha=getattr(args, "cov_ewma_alpha", 0.94),
        auto_adjust=not args.no_auto_adjust,
        cache_dir=getattr(args, "cache_dir", None),
        force_refresh=args.force_refresh,
        debug=not args.no_debug,
        train_window=args.train_window,
    )

    pipeline_cfg = PipelineConfig(loader=loader_cfg, debug=not args.no_debug)
    bundle = build_data_bundle(pipeline_cfg)
    bundle_summary = bundle.summary()
    bundle_summary.update(
        {
            "train_window": args.train_window,
            "rebal_interval": args.rebal_interval,
            "covariance_samples": len(bundle.cov_indices),
            "train_window_overrides": model_train_windows,
            "cov_method": args.cov_method,
            "cov_shrinkage": args.cov_shrinkage,
            "cov_robust_huber_k": args.cov_robust_huber_k,
            "cov_factor_rank": args.cov_factor_rank,
            "cov_factor_shrinkage": args.cov_factor_shrinkage,
        }
    )

    summary_path = outdir / "experiment_summary.json"
    summary_path.write_text(json.dumps(bundle_summary, ensure_ascii=False, indent=2))

    config_records = []
    for key, value in sorted(vars(args).items()):
        config_records.append({"parameter": key, "value": value})
    # 他の実験スクリプトと同じファイル名に揃える
    pd.DataFrame(config_records).to_csv(analysis_csv_dir / "2-experiment_config.csv", index=False)

    start_ts = pd.Timestamp(loader_cfg.start)
    data_fig_dir = analysis_fig_dir / "data_overview"
    data_fig_dir.mkdir(parents=True, exist_ok=True)
    plot_time_series(
        bundle.dataset.prices,
        "Prices timeseries",
        start_ts,
        data_fig_dir / "data_prices.png",
    )
    plot_time_series(
        bundle.dataset.returns,
        "Returns timeseries",
        start_ts,
        data_fig_dir / "data_returns.png",
    )
    plot_time_series(
        bundle.dataset.momentum,
        "Momentum timeseries",
        start_ts,
        data_fig_dir / "data_momentum.png",
    )
    returns_matrix = bundle.dataset.returns.dropna(how="all")
    if not returns_matrix.empty:
        corr_df = returns_matrix.corr()
        corr_df.to_csv(analysis_csv_dir / "asset_return_correlation.csv")
        corr_stats = compute_correlation_stats(corr_df)
        pd.Series(corr_stats).to_csv(
            analysis_csv_dir / "asset_return_correlation_summary.csv", header=False
        )
        plot_asset_correlation(
            corr_df,
            analysis_fig_dir / "asset_return_correlation.png",
            corr_stats,
        )
        export_max_return_winner_counts(
            returns_matrix,
            analysis_csv_dir / "asset_max_return_wins.csv",
            analysis_fig_dir / "asset_max_return_wins.png",
        )

    stats_results: List[Dict[str, object]] = []
    period_rows: List[Dict[str, object]] = []
    wealth_dict: Dict[str, pd.DataFrame] = {}
    weight_dict: Dict[str, pd.DataFrame] = {}
    train_window_records: List[Dict[str, object]] = []
    rebalance_records: List[Dict[str, object]] = []
    # flex_delta 用のサイクルごとの rebalance ログ（solver 状態の時系列可視化用）
    flex_rebalance_logs: List[pd.DataFrame] = []
    delta_records: List[Dict[str, object]] = []

    flex_base_options = dict(
        lambda_theta_anchor=args.flex_lambda_theta_anchor,
        lambda_theta_iso=args.flex_lambda_theta_iso,
        theta_anchor_mode=args.flex_theta_anchor_mode,
        theta_init_mode=args.flex_theta_init_mode,
    )

    # flex formulation と ensemble 指定の解釈（real_data_run.py と揃える）
    flex_forms_raw = parse_commalist(args.flex_formulation)
    if not flex_forms_raw:
        flex_forms_raw = ["dual"]

    ensemble_aliases = {"dual&kkt", "dual_kkt", "dual_kkt_ens", "ensemble", "ens"}
    flex_ensemble_enabled = any(f in ensemble_aliases for f in flex_forms_raw)

    flex_formulations = [f for f in flex_forms_raw if f not in ensemble_aliases]
    if not flex_formulations:
        flex_formulations = ["dual"]

    model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    for model_key in model_list:
        if model_key not in {"ols", "ipo", "flex"}:
            print(f"[real-data-delta] skipping unsupported model '{model_key}'")
            continue
        if model_key == "flex":
            formulations = flex_formulations
            valid_forms = {"dual", "kkt"}
            for form in formulations:
                if form not in valid_forms:
                    raise ValueError(f"Unknown flex formulation '{form}'. Use 'dual' or 'kkt'.")
        else:
            formulations = [None]

        for form in formulations:
            label = model_key
            if model_key == "flex" and (len(formulations) > 1 or (form and form != "dual")):
                label = f"{model_key}_{form}"
            print(f"[real-data-delta] rolling model={label}")
            solver_name = args.flex_solver if model_key == "flex" else "analytic"
            solver_spec = SolverSpec(name=solver_name, tee=args.tee)
            flex_options = None
            if model_key == "flex":
                flex_options = dict(flex_base_options)
                flex_options["formulation"] = form or "dual"
            results_dir = model_outputs_dir / label
            effective_train_window = model_train_windows.get(model_key, args.train_window)
            train_window_records.append(
                {
                    "model": label,
                    "base_model": model_key,
                    "train_window": effective_train_window,
                    "override": "yes" if model_key in model_train_windows else "no",
                }
            )
            if model_key in model_train_windows:
                print(
                    f"[real-data-delta] overriding train_window for {label}: "
                    f"{effective_train_window} (default {args.train_window})"
                )
            eval_start_ts = pd.Timestamp(args.start)
            run_result = run_rolling_experiment(
                model_key=model_key,
                model_label=label,
                bundle=bundle,
                delta_up=delta_up,
                delta_down=delta_down,
                delta_min=args.delta_min,
                delta_max=args.delta_max,
                lambda_delta_up=args.delta_reg_up,
                trading_cost_enabled=trading_costs_enabled,
                trading_cost_default_bps=trading_cost_default_bps,
                asset_cost_overrides=asset_cost_overrides_dec,
                solver_spec=solver_spec,
                flex_options=flex_options,
                train_window=effective_train_window,
                rebal_interval=args.rebal_interval,
                debug_roll=args.debug_roll,
                debug_dir=debug_dir,
                results_model_dir=results_dir,
                tee=args.tee,
                asset_pred_dir=asset_pred_dir,
                eval_start=eval_start_ts,
            )
            stats_results.append(run_result["stats"])
            reb_df = run_result.get("rebalance_df", pd.DataFrame())
            if model_key == "flex" and isinstance(reb_df, pd.DataFrame) and not reb_df.empty:
                flex_rebalance_logs.append(reb_df.copy())
            reb_summary = run_result.get("rebalance_summary", {})
            if reb_summary:
                record = {
                    "model": label,
                    "base_model": model_key,
                    "n_cycles": reb_summary.get("n_cycles"),
                    "train_length_min": reb_summary.get("train_length_min"),
                    "train_length_max": reb_summary.get("train_length_max"),
                    "elapsed_mean": reb_summary.get("elapsed_mean"),
                    "elapsed_max": reb_summary.get("elapsed_max"),
                    "solver_status_counts": json.dumps(
                        reb_summary.get("solver_status_counts", {}) or {}, ensure_ascii=False
                    ),
                }
                status_counts = reb_summary.get("solver_status_counts", {}) or {}
                for status, count in status_counts.items():
                    record[f"solver_status_{status}"] = count
                record["solver_status_optimal_total"] = status_counts.get("optimal", 0)
                rebalance_records.append(record)
                # collect per-cycle delta_used from model debug rebalance_log
                model_debug_dir = debug_dir / f"model_{label}"
                rebalance_log_path = model_debug_dir / "rebalance_log.csv"
                if rebalance_log_path.exists():
                    reb_log_df = pd.read_csv(rebalance_log_path)
                    if "delta_used" in reb_log_df.columns and "cycle" in reb_log_df.columns:
                        for _, row in reb_log_df.iterrows():
                            delta_records.append(
                                {
                                    "model": label,
                                    "cycle": int(row.get("cycle", 0)),
                                    "rebalance_idx": int(row.get("rebalance_idx", -1)),
                                    "rebalance_date": row.get("rebalance_date", ""),
                                    "train_start": int(row.get("train_start", -1)),
                                    "train_end": int(row.get("train_end", -1)),
                                    "delta_used": float(row["delta_used"]),
                                    "elapsed_sec": float(row.get("elapsed_sec", 0.0)),
                                    "solver_status": row.get("solver_status", ""),
                                    "solver_term": row.get("solver_term", ""),
                                }
                            )
            for row in run_result["period_metrics"]:
                period_entry = dict(row)
                period_entry["model"] = display_model_name(label)
                period_entry["train_window"] = run_result["stats"].get(
                    "train_window", effective_train_window
                )
                period_rows.append(period_entry)
            wealth_dict[label] = run_result["wealth_df"][["date", "wealth"]]
            if not run_result["weights_df"].empty:
                weight_dict[label] = run_result["weights_df"]

    # flex dual/kkt ensemble (delta 版) を step_log ベースで構成
    if flex_ensemble_enabled:
        from dfl_portfolio.experiments.real_data_common import build_flex_dual_kkt_ensemble

        build_flex_dual_kkt_ensemble(
            bundle=bundle,
            debug_dir=debug_dir,
            args=args,
            wealth_dict=wealth_dict,
            weight_dict=weight_dict,
            stats_results=stats_results,
            period_rows=period_rows,
            log_prefix="[ensemble]",
        )

    bench_stats, bench_wealth = run_benchmark_suite(
        bundle,
        benchmarks=benchmark_specs,
        args=args,
        trading_costs_enabled=trading_costs_enabled,
        asset_cost_overrides=asset_cost_overrides_dec,
        eval_start=eval_start_ts,
    )
    if bench_stats:
        stats_results.extend(bench_stats)
    for label, df in bench_wealth.items():
        wealth_dict[label] = df

    summary_csv_path = analysis_csv_dir / "1-summary.csv"
    summary_cost_csv_path = analysis_csv_dir / "1-summary_cost.csv"
    summary_raw_df = pd.DataFrame(stats_results)
    if not summary_raw_df.empty:
        if "model" in summary_raw_df.columns:
            summary_raw_df["model"] = summary_raw_df["model"].map(display_model_name)
        summary_raw_df["max_drawdown"] = summary_raw_df["max_drawdown"].astype(float)
        summary_df = format_summary_for_output(summary_raw_df)
        summary_df.to_csv(summary_csv_path, index=False)
        summary_cost_df = format_summary_for_output(
            build_cost_adjusted_summary(summary_raw_df)
        )
        summary_cost_df.to_csv(summary_cost_csv_path, index=False)
    else:
        summary_csv_path.write_text("")
        summary_cost_csv_path.write_text("")
    mirror_to_analysis_root(summary_csv_path, analysis_dir)
    mirror_to_analysis_root(summary_cost_csv_path, analysis_dir)

    if period_rows:
        period_df = pd.DataFrame(period_rows)
        if "model" in period_df.columns:
            period_df["model"] = period_df["model"].map(display_model_name)
        period_df.to_csv(analysis_csv_dir / "period_metrics.csv", index=False)

    solver_events_by_model: Dict[str, pd.DataFrame] = {}
    if flex_rebalance_logs:
        tmp_events: Dict[str, List[pd.DataFrame]] = {}
        for log in flex_rebalance_logs:
            df = log.copy()
            if df.empty or "rebalance_date" not in df.columns:
                continue
            df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
            statuses = df.get("solver_status", "").astype(str)
            mask = ~statuses.str.contains("optimal", case=False, na=False)
            mask &= ~statuses.str.contains("ok", case=False, na=False)
            df = df.loc[mask]
            if df.empty:
                continue
            need_cols = {"model", "rebalance_date", "solver_status"}
            missing = need_cols - set(df.columns)
            if missing:
                continue
            for model, sub in df[list(need_cols)].groupby("model"):
                tmp_events.setdefault(model, []).append(sub)
        solver_events_by_model = {
            model: pd.concat(chunks, ignore_index=True).sort_values("rebalance_date")
            for model, chunks in tmp_events.items()
        }

    if wealth_dict:
        wealth_merge = None
        for model, wdf in wealth_dict.items():
            display_label = display_model_name(model)
            df_model = wdf.rename(columns={"wealth": display_label})
            if wealth_merge is None:
                wealth_merge = df_model
            else:
                wealth_merge = wealth_merge.merge(df_model, on="date", how="outer")
        if wealth_merge is not None:
            wealth_merge = wealth_merge.sort_values("date")
            wealth_merge = wealth_merge.groupby("date", as_index=False).last()
            wealth_merge.to_csv(analysis_csv_dir / "wealth_comparison.csv", index=False)
            plot_multi_wealth(
                {m: df for m, df in wealth_dict.items()},
                analysis_fig_dir / "wealth_comparison.png",
            )
            wealth_events_base_path = analysis_dir / "wealth_events.png"
            wealth_events_primary_path = analysis_dir / "2-wealth_events.png"
            plot_wealth_with_events(
                {m: df for m, df in wealth_dict.items()},
                wealth_events_base_path,
            )
            if wealth_events_primary_path.exists():
                shutil.copy2(
                    wealth_events_primary_path,
                    analysis_fig_dir / wealth_events_primary_path.name,
                )
            flex_only = {m: df for m, df in wealth_dict.items() if "flex" in str(m)}
            if flex_only:
                plot_wealth_with_events(
                    flex_only,
                    analysis_fig_dir / "wealth_events_dfl_only.png",
                )
            flex_dual_key = None
            for key in wealth_dict.keys():
                key_str = str(key)
                if "flex" in key_str and ("dual" in key_str or key_str == "flex"):
                    flex_dual_key = key
                    break
            if flex_dual_key is not None:
                dual_vs_others: Dict[str, pd.DataFrame] = {}
                dual_vs_others[flex_dual_key] = wealth_dict[flex_dual_key]
                for key, df in wealth_dict.items():
                    if key == flex_dual_key:
                        continue
                    if "flex" in str(key):
                        continue
                    dual_vs_others[key] = df
                if len(dual_vs_others) > 1:
                    plot_wealth_with_events(
                        dual_vs_others,
                        analysis_fig_dir / "wealth_events_flex_dual_vs_baselines.png",
                    )
            event_fig_dir = analysis_fig_dir / "event_windows"
            event_fig_dir.mkdir(parents=True, exist_ok=True)
            for name, start, end in PERIOD_WINDOWS:
                sub_dict: Dict[str, pd.DataFrame] = {}
                for m, df in wealth_dict.items():
                    if df.empty:
                        continue
                    tmp = df.copy()
                    tmp["date"] = pd.to_datetime(tmp["date"])
                    mask = (tmp["date"] >= pd.Timestamp(start)) & (tmp["date"] <= pd.Timestamp(end))
                    tmp = tmp.loc[mask]
                    if not tmp.empty:
                        sub_dict[m] = tmp[["date", "wealth"]].copy()
                if sub_dict:
                    plot_wealth_window_normalized(
                        sub_dict,
                        name,
                        start,
                        end,
                        event_fig_dir / f"wealth_window_{name}.png",
                        events_by_model=solver_events_by_model,
                    )
            five_year_dir = analysis_fig_dir / "wealth_windows_5y"
            five_year_dir_all = five_year_dir / "all_models"
            five_year_dir_dfl = five_year_dir / "dfl_only"
            five_year_dir_all.mkdir(parents=True, exist_ok=True)
            five_year_dir_dfl.mkdir(parents=True, exist_ok=True)
            dates_all = pd.to_datetime(wealth_merge["date"])
            if not dates_all.empty:
                start_all = dates_all.min()
                end_all = dates_all.max()
                start_win = start_all
                while start_win < end_all:
                    end_win = start_win + pd.DateOffset(years=5)
                    if end_win > end_all:
                        end_win = end_all
                    window_label = f"{start_win.year}_{end_win.year}"
                    sub_dict: Dict[str, pd.DataFrame] = {}
                    for m, df in wealth_dict.items():
                        if df.empty:
                            continue
                        tmp = df.copy()
                        tmp["date"] = pd.to_datetime(tmp["date"])
                        mask = (tmp["date"] >= start_win) & (tmp["date"] < end_win)
                        tmp = tmp.loc[mask]
                        if not tmp.empty:
                            sub_dict[m] = tmp[["date", "wealth"]].copy()
                    if sub_dict:
                        plot_wealth_window_normalized(
                            sub_dict,
                            window_label,
                            start_win,
                            end_win,
                            five_year_dir_all / f"wealth_window_5y_{window_label}.png",
                            events_by_model=solver_events_by_model,
                        )
                        flex_only_win = {m: df for m, df in sub_dict.items() if "flex" in str(m)}
                        if flex_only_win:
                            plot_wealth_window_normalized(
                                flex_only_win,
                                window_label,
                                start_win,
                                end_win,
                                five_year_dir_dfl / f"wealth_window_5y_{window_label}.png",
                                events_by_model=solver_events_by_model,
                            )
                    start_win = start_win + pd.DateOffset(years=5)
            wealth_returns = wealth_merge.copy()
            wealth_returns["date"] = pd.to_datetime(wealth_returns["date"])
            wealth_returns = (
                wealth_returns.set_index("date").pct_change(fill_method=None).dropna(how="all")
            )
            if not wealth_returns.empty:
                corr = wealth_returns.corr()
                corr.to_csv(analysis_csv_dir / "wealth_correlation.csv")
                plot_wealth_correlation_heatmap(corr, analysis_fig_dir / "wealth_correlation.png")
                sig_df = compute_pairwise_mean_return_tests(wealth_returns)
                if not sig_df.empty:
                    for col in ("model_a", "model_b"):
                        if col in sig_df.columns:
                            sig_df[col] = sig_df[col].map(display_model_name)
                    sig_df.to_csv(analysis_csv_dir / "return_significance.csv", index=False)
                perf_df = compute_pairwise_performance_tests(wealth_returns)
                if not perf_df.empty:
                    for col in ("model_a", "model_b"):
                        if col in perf_df.columns:
                            perf_df[col] = perf_df[col].map(display_model_name)
                    perf_df.to_csv(analysis_csv_dir / "performance_significance.csv", index=False)
                    summarize_dfl_performance_significance(analysis_csv_dir)

    if weight_dict:
        plot_weight_comparison(weight_dict, analysis_fig_dir / "weights_comparison.png")
        for name, start, end in [
            ("lehman_2008", "2007-07-01", "2009-06-30"),
            ("covid_2020", "2020-02-01", "2020-12-31"),
            ("inflation_2022", "2022-01-01", "2023-12-31"),
        ]:
            sub_weights: Dict[str, pd.DataFrame] = {}
            for m, df in weight_dict.items():
                if df.empty:
                    continue
                tmp = df.copy()
                tmp["date"] = pd.to_datetime(tmp["date"])
                mask = (tmp["date"] >= pd.Timestamp(start)) & (tmp["date"] <= pd.Timestamp(end))
                tmp = tmp.loc[mask]
                if not tmp.empty:
                    sub_weights[m] = tmp
            if sub_weights:
                plot_weight_comparison(
                    sub_weights, analysis_fig_dir / f"weights_comparison_{name}.png"
                )
        weights_5y_dir = analysis_fig_dir / "weights_windows_5y"
        weights_5y_dir.mkdir(parents=True, exist_ok=True)
        all_dates: List[pd.Timestamp] = []
        for df in weight_dict.values():
            if df.empty or "date" not in df.columns:
                continue
            d = pd.to_datetime(df["date"])
            if not d.empty:
                all_dates.append(d.min())
                all_dates.append(d.max())
        if all_dates:
            start_all = min(all_dates)
            end_all = max(all_dates)
            start_win = start_all
            while start_win < end_all:
                end_win = start_win + pd.DateOffset(years=5)
                if end_win > end_all:
                    end_win = end_all
                window_label = f"{start_win.year}_{end_win.year}"
                sub_weights: Dict[str, pd.DataFrame] = {}
                for m, df in weight_dict.items():
                    if df.empty or "date" not in df.columns:
                        continue
                    tmp = df.copy()
                    tmp["date"] = pd.to_datetime(tmp["date"])
                    mask = (tmp["date"] >= start_win) & (tmp["date"] < end_win)
                    tmp = tmp.loc[mask]
                    if not tmp.empty:
                        sub_weights[m] = tmp
                if sub_weights:
                    plot_weight_comparison(
                        sub_weights,
                        weights_5y_dir / f"weights_comparison_5y_{window_label}.png",
                        max_points=None,
                    )
                start_win = start_win + pd.DateOffset(years=5)
        export_weight_variance_correlation(
            weight_dict,
            analysis_csv_dir / "weight_variance_correlation.csv",
            analysis_fig_dir / "weight_variance_correlation.png",
        )
        export_average_weights(
            weight_dict,
            analysis_csv_dir / "average_weights.csv",
            analysis_fig_dir / "average_weights.png",
        )
        plot_weight_histograms(weight_dict, analysis_fig_dir / "weight_histograms.png")
        export_weight_threshold_frequency(
            weight_dict,
            WEIGHT_THRESHOLD,
            analysis_csv_dir / "weight_threshold_freq.csv",
            analysis_fig_dir / "weight_threshold_freq.png",
        )
    if train_window_records:
        pd.DataFrame(train_window_records).to_csv(
            analysis_csv_dir / "model_train_windows.csv", index=False
        )
    if rebalance_records:
        rebalance_df = pd.DataFrame(rebalance_records)
        rebalance_df.to_csv(analysis_csv_dir / "rebalance_summary.csv", index=False)
    if flex_rebalance_logs:
        flex_debug_df = pd.concat(flex_rebalance_logs, ignore_index=True)
        plot_flex_solver_debug(flex_debug_df, analysis_fig_dir / "flex_solver_debug.png")
    if delta_records:
        delta_df = pd.DataFrame(delta_records)
        delta_df.to_csv(analysis_csv_dir / "delta_trajectory.csv", index=False)
        plot_delta_paths(delta_df, analysis_fig_dir / "delta_paths.png")
    if not summary_df.empty:
        update_experiment_ledger(RESULTS_ROOT, outdir, args, summary_df, analysis_csv_dir, bundle_summary)

    run_extended_analysis(analysis_csv_dir, analysis_fig_dir, model_outputs_dir, asset_pred_dir)

    print(f"[real-data-delta] finished. outputs -> {outdir}")
    print(f"[real-data-delta] debug artifacts -> {debug_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.real_data_run_delta \
  --tickers "SPY,GLD,EEM,TLT" \
  --frequency weekly \
  --resample-rule W-FRI \
  --momentum-window 26 \
  --cov-window 13 \
  --cov-method oas \
  --train-window 26 \
  --rebal-interval 4 \
  --delta-min 0.3 \
  --delta-max 0.7 \
  --delta-reg-up 0.1 \
  --delta-up 0.5 \
  --delta-down 0.5 \
  --models ols,ipo,flex \
  --flex-solver knitro \
  --flex-formulation 'dual' \
  --flex-lambda-theta-anchor 10.0 \
  --flex-theta-anchor-mode ipo \
  --benchmark-ticker SPY \
  --benchmark-equal-weight \
  --debug-roll


"""
