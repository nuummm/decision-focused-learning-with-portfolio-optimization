from __future__ import annotations

import argparse
import json
import logging
import math
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
    compute_benchmark_series,
    compute_equal_weight_benchmark,
    compute_correlation_stats,
    compute_period_metrics,
    compute_pairwise_mean_return_tests,
    compute_pairwise_performance_tests,
    compute_sortino_ratio,
    display_model_name,
    export_average_weights,
    export_weight_threshold_frequency,
    export_weight_variance_correlation,
    max_drawdown,
    plot_asset_correlation,
    plot_flex_solver_debug,
    plot_multi_wealth,
    plot_time_series,
    plot_wealth_correlation_heatmap,
    plot_wealth_curve,
    plot_wealth_with_events,
    plot_weight_comparison,
    plot_weight_histograms,
    plot_weight_paths,
    plot_phi_paths,
    plot_beta_paths,
    update_experiment_ledger,
    run_extended_analysis,
    summarize_dfl_performance_significance,
    format_summary_for_output,
)
from dfl_portfolio.real_data.covariance import estimate_shrinkage_covariances
from dfl_portfolio.registry import SolverSpec, get_trainer
from dfl_portfolio.models.ols import predict_yhat, train_ols
from dfl_portfolio.models.ipo_closed_form import fit_ipo_closed_form
from dfl_portfolio.models.ols_gurobi import solve_mvo_gurobi, solve_series_mvo_gurobi
from dfl_portfolio.models.dfl_p1_flex_V import fit_dfl_p1_flex_V
from dfl_portfolio.experiments.real_data_common import (
    mvo_cost,
    ScheduleItem,
    build_rebalance_schedule,
    prepare_flex_training_args,
)

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
RESULTS_BASE = PROJECT_ROOT / "results"
RESULTS_ROOT = RESULTS_BASE / "exp_real_data_V"
DEBUG_ROOT = RESULTS_BASE / "debug_outputs_V"

# Suppress Pyomo-related warnings (only show errors)
logging.getLogger("pyomo").setLevel(logging.ERROR)
logging.getLogger("pyomo.solvers").setLevel(logging.ERROR)


def train_model_window(
    model_key: str,
    trainer,
    bundle,
    delta: float,
    solver_spec: SolverSpec,
    flex_options: Dict[str, Any] | None,
    train_start: int,
    train_end: int,
    tee: bool,
    V_sample_list: Sequence[np.ndarray],
    V_diag_list: Sequence[np.ndarray],
):
    idx_list = bundle.cov_indices.tolist()
    base_kwargs: Dict[str, object] = dict(
        X=bundle.dataset.X,
        Y=bundle.dataset.Y,
        idx=idx_list,
        start_index=train_start,
        end_index=train_end,
        delta=delta,
        tee=tee,
    )
    theta_init_override: Optional[np.ndarray] = None
    if model_key == "flex" and flex_options:
        theta_init_override, resolved_flex = prepare_flex_training_args(
            bundle, train_start, train_end, delta, tee, flex_options
        )
        base_kwargs.update(resolved_flex)
    if theta_init_override is not None:
        base_kwargs["theta_init"] = theta_init_override

    start_time = time.perf_counter()
    if model_key == "flex":
        trainer_ret = fit_dfl_p1_flex_V(
            V_sample_list=V_sample_list,
            V_diag_list=V_diag_list,
            solver_options=solver_spec.options,
            **base_kwargs,
        )
    else:
        trainer_kwargs = dict(base_kwargs)
        trainer_kwargs["Vhats"] = bundle.covariances
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
    delta: float,
    solver_spec: SolverSpec,
    flex_options: Dict[str, Any] | None,
    train_window: int,
    rebal_interval: int,
    debug_roll: bool,
    debug_dir: Path,
    results_model_dir: Path,
    V_sample_list: Sequence[np.ndarray],
    V_diag_list: Sequence[np.ndarray],
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

    wealth = 1.0
    wealth_dates: List[pd.Timestamp] = []
    wealth_values: List[float] = []
    wealth_labels: List[str] = []

    step_rows: List[Dict[str, object]] = []
    rebalance_rows: List[Dict[str, object]] = []
    asset_rows: List[Dict[str, object]] = []

    total_cycles = len(schedule)
    for cycle_id, item in enumerate(schedule):
        theta_hat, info, elapsed = train_model_window(
            model_key,
            trainer,
            bundle,
            delta,
            solver_spec,
            flex_options,
            item.train_start,
            item.train_end,
            tee,
            V_sample_list,
            V_diag_list,
        )
        Yhat_all = predict_yhat(bundle.dataset.X, theta_hat)

        if debug_roll:
            progress = (cycle_id + 1) / max(total_cycles, 1)
            bar = "#" * int(progress * 20)
            print(
                f"[roll-debug-V] model={model_label} cycle={cycle_id+1}/{total_cycles} "
                f"idx={item.rebalance_idx} train=[{item.train_start},{item.train_end}] "
                f"n_eval={len(item.eval_indices)} [{bar:<20}] {progress:.0%}"
            )

        phi_hat = info.get("phi_hat", None)
        rebalance_rows.append(
            {
                "cycle": cycle_id,
                "model": model_label,
                "rebalance_idx": item.rebalance_idx,
                "rebalance_date": bundle.dataset.timestamps[item.rebalance_idx].isoformat(),
                "train_start": item.train_start,
                "train_end": item.train_end,
                "solver_status": (info or {}).get("status", ""),
                "solver_term": (info or {}).get("termination_condition", ""),
                "elapsed_sec": elapsed,
                "phi_used": float(phi_hat) if phi_hat is not None else np.nan,
            }
        )

        if not wealth_dates and item.eval_indices:
            wealth_dates.append(bundle.dataset.timestamps[item.eval_indices[0]])
            wealth_values.append(wealth)
            wealth_labels.append("initial")

        for eval_idx in item.eval_indices:
            if eval_idx not in cov_lookup:
                continue
            cov, stat = cov_lookup[eval_idx]
            yhat = Yhat_all[eval_idx]
            z = solve_mvo_gurobi(
                y_hat=yhat,
                V_hat=cov,
                delta=delta,
                psd_eps=1e-9,
                output=False,
            )
            if z is None or np.isnan(z).any():
                continue
            realized = float(z @ bundle.dataset.Y[eval_idx])
            wealth *= (1.0 + realized)
            wealth_dates.append(bundle.dataset.timestamps[eval_idx])
            wealth_values.append(wealth)
            wealth_labels.append("after_step")

            cost = mvo_cost(z, bundle.dataset.Y[eval_idx], cov, delta)
            step_rows.append(
                {
                    "cycle": cycle_id,
                    "model": model_label,
                    "date": bundle.dataset.timestamps[eval_idx].isoformat(),
                    "eval_idx": eval_idx,
                    "eig_min": stat.eigen_min,
                    "portfolio_return": realized,
                    "wealth": wealth,
                    "mvo_cost": cost,
                    "theta": json.dumps(theta_hat.tolist()),
                    "weights": json.dumps(z.tolist()),
                    "weight_sum": float(np.sum(z)),
                    "weight_min": float(np.min(z)),
                    "weight_max": float(np.max(z)),
                }
            )

            tickers = bundle.dataset.config.tickers
            yhat_vec = Yhat_all[eval_idx]
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
    mean_step = float(np.mean(returns))
    std_step = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if wealth_dates and len(wealth_dates) >= 2:
        horizon_days = (wealth_dates[-1] - wealth_dates[0]).days
        horizon_years = max(horizon_days / 365.25, 1e-9)
        steps_per_year = len(step_rows) / horizon_years
    else:
        steps_per_year = 1.0
    mean_return = mean_step * steps_per_year
    std_return = std_step * math.sqrt(steps_per_year) if std_step > 0.0 else 0.0
    sharpe = mean_return / std_return if std_return > 1e-12 else np.nan
    sortino_step = compute_sortino_ratio(returns)
    sortino = (
        float(sortino_step) * math.sqrt(steps_per_year)
        if np.isfinite(sortino_step)
        else np.nan
    )

    model_debug_dir = debug_dir / f"model_{model_label}"
    model_debug_dir.mkdir(parents=True, exist_ok=True)

    step_path = model_debug_dir / "step_log.csv"
    step_df.to_csv(step_path, index=False)

    rebalance_df = pd.DataFrame(rebalance_rows)
    rebalance_df.to_csv(model_debug_dir / "rebalance_log.csv", index=False)

    wealth_df = pd.DataFrame({"date": wealth_dates, "wealth": wealth_values, "label": wealth_labels})
    wealth_df.to_csv(model_debug_dir / "wealth.csv", index=False)
    plot_wealth_curve(wealth_dates, wealth_values, model_debug_dir / "wealth.png")

    if step_df.empty:
        weights_df = pd.DataFrame()
    else:
        weight_records: List[Dict[str, float]] = []
        tickers = list(bundle.dataset.config.tickers)
        for _, row in step_df.iterrows():
            weights = np.asarray(json.loads(row["weights"]), dtype=float)
            record: Dict[str, float] = {
                "date": row["date"],
                "portfolio_return": float(row["portfolio_return"]),
                "portfolio_return_sq": float(row["portfolio_return"] ** 2),
            }
            for idx, t in enumerate(tickers):
                if idx < weights.shape[0]:
                    record[t] = float(weights[idx])
            weight_records.append(record)
        weights_df = pd.DataFrame(weight_records)

    stats_report = {
        "model": display_model_name(model_label),
        "n_cycles": len(rebalance_rows),
        "n_steps": int(len(step_df)),
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown(wealth_values),
        "final_wealth": float(wealth_values[-1]) if wealth_values else 1.0,
        "train_window": train_window,
        "rebal_interval": rebal_interval,
    }

    period_metrics = compute_period_metrics(step_df)

    if asset_pred_dir is not None and asset_rows:
        asset_pred_dir = Path(asset_pred_dir)
        asset_pred_dir.mkdir(parents=True, exist_ok=True)
        asset_df = pd.DataFrame(asset_rows)
        asset_df.to_csv(asset_pred_dir / f"{model_label}.csv", index=False)

    return {
        "stats": stats_report,
        "step_df": step_df,
        "weights_df": weights_df,
        "wealth_df": wealth_df,
        "rebalance_df": rebalance_df,
        "cov_stats": {},
        "period_metrics": period_metrics,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    # real_data_run_V 専用: 共分散の時間減衰率 β（EWMA）
    parser.add_argument(
        "--cov-ewma-beta",
        type=float,
        default=0.94,
        help=(
            "EWMA 共分散の時間減衰率 β (0<β<1)。"
            "β が小さいほど短期寄り、β が 1 に近いほど長期寄りの共分散となる。"
            "real_data_run_V のみ有効。"
        ),
    )
    args = parser.parse_args(argv)

    if args.cov_method != "diag":
        raise ValueError("real_data_run_V は cov-method=diag のみ対応です。")
    if str(args.flex_solver).lower() != "knitro":
        raise ValueError("real_data_run_V は --flex-solver knitro のみ対応です。")
    if not (0.0 < float(args.cov_ewma_beta) < 1.0):
        raise ValueError("--cov-ewma-beta は 0 と 1 の間の値を指定してください。")

    tickers = parse_tickers(args.tickers)
    loader_cfg = MarketLoaderConfig(
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
        auto_adjust=not args.no_auto_adjust,
        force_refresh=args.force_refresh,
        train_window=args.train_window,
    )
    pipeline_cfg = PipelineConfig(loader=loader_cfg, debug=not args.no_debug)

    outdir = make_output_dir(RESULTS_ROOT, args.outdir)
    debug_dir = DEBUG_ROOT / outdir.name
    debug_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_data_bundle(pipeline_cfg)
    bundle_summary = bundle.summary()

    # V 学習用に、等重みではなく時間減衰率 β を用いた EWMA 共分散 S_t(β) を構成。
    # Shrinkage φ 自体は DFL モデル内部 (fit_dfl_p1_flex_V) で学習させる。
    returns_df = bundle.dataset.returns.dropna(how="any")
    cfg = bundle.dataset.config
    beta = float(args.cov_ewma_beta)

    values = returns_df.to_numpy(dtype=float)
    times = list(returns_df.index)
    d = values.shape[1]
    window = int(cfg.cov_window)
    if window < 2:
        raise ValueError("cov_window は 2 以上を指定してください。")

    sample_by_ts: Dict[pd.Timestamp, np.ndarray] = {}
    for i in range(len(values)):
        if i + 1 < window:
            continue
        window_vals = values[i - window + 1 : i + 1]
        # S_t(β) = (1-β) * Σ_{k=0}^{W-1} β^k r_{t-k} r_{t-k}^T
        S = np.zeros((d, d), dtype=float)
        # 直近から過去へさかのぼって加重
        for k in range(window_vals.shape[0]):
            r = window_vals[-1 - k].reshape(-1, 1)
            w = (1.0 - beta) * (beta ** k)
            S += w * (r @ r.T)
        S = 0.5 * (S + S.T) + cfg.cov_eps * np.eye(d)
        sample_by_ts[times[i]] = S

    V_sample_all: List[np.ndarray] = []
    V_diag_all: List[np.ndarray] = []
    for idx in bundle.cov_indices.tolist():
        ts = bundle.dataset.timestamps[idx]
        cov_sample = sample_by_ts.get(ts)
        if cov_sample is None:
            raise RuntimeError(f"no EWMA covariance found for timestamp {ts}")
        V_sample_all.append(cov_sample)
        V_diag_all.append(np.diag(np.diag(cov_sample)))

    model_specs = parse_commalist(args.models)
    flex_formulations = parse_commalist(args.flex_formulation) if args.flex_formulation else ["dual"]
    flex_ensemble_enabled = "dual&kkt" in flex_formulations
    flex_formulations = [f for f in flex_formulations if f != "dual&kkt"]

    model_train_windows = parse_model_train_window_spec(args.model_train_window)

    outdir.mkdir(parents=True, exist_ok=True)
    analysis_fig_dir = outdir / "analysis" / "figures"
    analysis_csv_dir = outdir / "analysis" / "csv"
    asset_pred_dir = analysis_csv_dir / "asset_predictions"
    model_outputs_dir = outdir / "models"
    analysis_fig_dir.mkdir(parents=True, exist_ok=True)
    analysis_csv_dir.mkdir(parents=True, exist_ok=True)
    asset_pred_dir.mkdir(parents=True, exist_ok=True)
    model_outputs_dir.mkdir(parents=True, exist_ok=True)

    # 実験設定の記録
    start_ts = pd.Timestamp(loader_cfg.start)
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
    pd.DataFrame(config_records).to_csv(analysis_csv_dir / "experiment_config.csv", index=False)

    # データの可視化（real_data_run.py と同様の構造）
    plot_time_series(
        bundle.dataset.prices,
        "Prices timeseries",
        start_ts,
        analysis_fig_dir / "data_prices.png",
    )
    plot_time_series(
        bundle.dataset.returns,
        "Returns timeseries",
        start_ts,
        analysis_fig_dir / "data_returns.png",
    )
    plot_time_series(
        bundle.dataset.momentum,
        "Momentum timeseries",
        start_ts,
        analysis_fig_dir / "data_momentum.png",
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

    stats_results: List[Dict[str, object]] = []
    period_rows: List[Dict[str, object]] = []
    rebalance_records: List[Dict[str, object]] = []
    train_window_records: List[Dict[str, object]] = []
    phi_records: List[Dict[str, object]] = []
    beta_records: List[Dict[str, object]] = []

    wealth_dict: Dict[str, pd.DataFrame] = {}
    weight_dict: Dict[str, pd.DataFrame] = {}

    for model_key in model_specs:
        model_key = model_key.strip()
        if model_key not in {"ols", "ipo", "flex"}:
            print(f"[real-data-V] skipping unsupported model '{model_key}'")
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
            print(f"[real-data-V] rolling model={label}")
            solver_spec = SolverSpec(name="knitro", tee=args.tee)
            flex_options = None
            if model_key == "flex":
                flex_options = {
                    "formulation": form or "dual",
                    "lambda_theta_anchor": args.flex_lambda_theta_anchor,
                    "lambda_theta_anchor_l1": 0.0,
                    "lambda_theta_iso": args.flex_lambda_theta_iso,
                    "theta_anchor_mode": args.flex_theta_anchor_mode,
                    "theta_init_mode": args.flex_theta_init_mode,
                }
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
                    f"[real-data-V] overriding train_window for {label}: "
                    f"{effective_train_window} (default {args.train_window})"
                )
            eval_start_ts = pd.Timestamp(args.start)
            run_result = run_rolling_experiment(
                model_key=model_key,
                model_label=label,
                bundle=bundle,
                delta=args.delta,
                solver_spec=solver_spec,
                flex_options=flex_options,
                train_window=effective_train_window,
                rebal_interval=args.rebal_interval,
                debug_roll=args.debug_roll,
                debug_dir=debug_dir,
                results_model_dir=results_dir,
                tee=args.tee,
                V_sample_list=V_sample_all,
                V_diag_list=V_diag_all,
                asset_pred_dir=asset_pred_dir,
                eval_start=eval_start_ts,
            )
            stats_results.append(run_result["stats"])
            reb_df = run_result["rebalance_df"]
            if not reb_df.empty:
                status_counts = reb_df["solver_status"].value_counts().to_dict()
                record = {
                    "model": label,
                    "base_model": model_key,
                    "n_cycles": len(reb_df),
                    "train_length_min": int(reb_df["train_end"].sub(reb_df["train_start"]).min() + 1),
                    "train_length_max": int(reb_df["train_end"].sub(reb_df["train_start"]).max() + 1),
                    "elapsed_mean": float(reb_df["elapsed_sec"].mean()),
                    "elapsed_max": float(reb_df["elapsed_sec"].max()),
                    "solver_status_counts": json.dumps(status_counts, ensure_ascii=False),
                }
                for status, count in status_counts.items():
                    record[f"solver_status_{status}"] = count
                record["solver_status_optimal_total"] = status_counts.get("optimal", 0)
                rebalance_records.append(record)
                # phi の推移を記録（flex モデルのみ想定）
                reb_log_df = reb_df
                if "phi_used" in reb_log_df.columns and "cycle" in reb_log_df.columns:
                    for _, row in reb_log_df.iterrows():
                        phi_records.append(
                            {
                                "model": label,
                                "cycle": int(row.get("cycle", 0)),
                                "rebalance_idx": int(row.get("rebalance_idx", -1)),
                                "rebalance_date": row.get("rebalance_date", ""),
                                "train_start": int(row.get("train_start", -1)),
                                "train_end": int(row.get("train_end", -1)),
                                "phi_used": float(row.get("phi_used", np.nan)),
                                "elapsed_sec": float(row.get("elapsed_sec", 0.0)),
                                "solver_status": row.get("solver_status", ""),
                                "solver_term": row.get("solver_term", ""),
                            }
                        )
                        # beta は CLI で指定された定数だが、phi と同じ軸で推移を可視化できるよう記録する
                        beta_records.append(
                            {
                                "model": label,
                                "cycle": int(row.get("cycle", 0)),
                                "rebalance_idx": int(row.get("rebalance_idx", -1)),
                                "rebalance_date": row.get("rebalance_date", ""),
                                "train_start": int(row.get("train_start", -1)),
                                "train_end": int(row.get("train_end", -1)),
                                "beta_used": float(args.cov_ewma_beta),
                                "elapsed_sec": float(row.get("elapsed_sec", 0.0)),
                                "solver_status": row.get("solver_status", ""),
                                "solver_term": row.get("solver_term", ""),
                            }
                        )
            for row in run_result["period_metrics"]:
                period_entry = dict(row)
                period_entry["model"] = display_model_name(label)
                period_entry["train_window"] = run_result["stats"].get("train_window", effective_train_window)
                period_rows.append(period_entry)
            wealth_dict[label] = run_result["wealth_df"][["date", "wealth"]]
            if not run_result["weights_df"].empty:
                weight_dict[label] = run_result["weights_df"]

    # flex dual/kkt ensemble（V 版）。step_log のリターンを凸結合で再構成。
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

    benchmark_spec = (args.benchmark_ticker or "").strip()
    if benchmark_spec and wealth_dict:
        min_date: Optional[pd.Timestamp] = None
        for df in wealth_dict.values():
            if df.empty:
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            dates = pd.to_datetime(df["date"])
            if dates.empty:
                continue
            current_min = dates.min()
            if pd.isna(current_min):
                continue
            if min_date is None or current_min < min_date:
                min_date = current_min
        benchmark_info = compute_benchmark_series(bundle, benchmark_spec, start_date=min_date)
        if benchmark_info:
            stats_results.append(benchmark_info["stats"])
            wealth_dict[benchmark_info["label"]] = benchmark_info["wealth_df"]
    if getattr(args, "benchmark_equal_weight", False) and wealth_dict:
        min_date_eq: Optional[pd.Timestamp] = None
        for df in wealth_dict.values():
            if df.empty:
                continue
            dates = pd.to_datetime(df["date"])
            if dates.empty:
                continue
            current_min = dates.min()
            if pd.isna(current_min):
                continue
            if min_date_eq is None or current_min < min_date_eq:
                min_date_eq = current_min
        eq_info = compute_equal_weight_benchmark(bundle, start_date=min_date_eq)
        if eq_info:
            stats_results.append(eq_info["stats"])
            wealth_dict[eq_info["label"]] = eq_info["wealth_df"]

    summary_df = pd.DataFrame(stats_results)
    if not summary_df.empty:
        if "model" in summary_df.columns:
            summary_df["model"] = summary_df["model"].map(display_model_name)
        summary_df["max_drawdown"] = summary_df["max_drawdown"].astype(float)
        summary_df = format_summary_for_output(summary_df)
        summary_df.to_csv(analysis_csv_dir / "summary.csv", index=False)
    else:
        (analysis_csv_dir / "summary.csv").write_text("")

    if period_rows:
        period_df = pd.DataFrame(period_rows)
        if "model" in period_df.columns:
            period_df["model"] = period_df["model"].map(display_model_name)
        period_df.to_csv(analysis_csv_dir / "period_metrics.csv", index=False)

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
            plot_multi_wealth({m: df for m, df in wealth_dict.items()}, analysis_fig_dir / "wealth_comparison.png")
            plot_wealth_with_events({m: df for m, df in wealth_dict.items()}, analysis_fig_dir / "wealth_events.png")
            wealth_returns = wealth_merge.copy()
            wealth_returns["date"] = pd.to_datetime(wealth_returns["date"])
            wealth_returns = wealth_returns.set_index("date").pct_change(fill_method=None).dropna(how="all")
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
        flex_debug_df = rebalance_df[rebalance_df["base_model"] == "flex"] if "base_model" in rebalance_df.columns else pd.DataFrame()
        if not flex_debug_df.empty:
            plot_flex_solver_debug(flex_debug_df, analysis_fig_dir / "flex_solver_debug.png")
    if phi_records:
        phi_df = pd.DataFrame(phi_records)
        phi_df.to_csv(analysis_csv_dir / "phi_trajectory.csv", index=False)
        plot_phi_paths(phi_df, analysis_fig_dir / "phi_paths.png")
    if beta_records:
        beta_df = pd.DataFrame(beta_records)
        beta_df.to_csv(analysis_csv_dir / "beta_trajectory.csv", index=False)
        plot_beta_paths(beta_df, analysis_fig_dir / "beta_paths.png")
    if not summary_df.empty:
        update_experiment_ledger(RESULTS_ROOT, outdir, args, summary_df, analysis_csv_dir, bundle_summary)

    run_extended_analysis(analysis_csv_dir, analysis_fig_dir, model_outputs_dir, asset_pred_dir)

    print(f"[real-data-V] finished. outputs -> {outdir}")
    print(f"[real-data-V] debug artifacts -> {debug_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.real_data_run_V \
  --tickers "SPY,GLD,EEM,TLT" \
  --start 2006-10-31 --end 2025-10-31 \
  --interval 1d \
  --price-field Close \
  --return-kind log \
  --frequency weekly \
  --resample-rule W-FRI \
  --momentum-window 26 \
  --return-horizon 1 \
  --cov-window 13 \
  --cov-method diag \
  --cov-shrinkage 0.94 \
  --cov-ewma-beta 0.94 \
  --cov-eps 1e-6 \
  --train-window 26 \
  --rebal-interval 4 \
  --delta 0.5 \
  --models ols,ipo,flex \
  --flex-solver knitro \
  --flex-formulation 'dual' \
  --flex-lambda-theta-anchor 0.0 \
  --flex-lambda-theta-iso 0.0 \
  --flex-theta-anchor-mode ols \
  --flex-theta-init-mode none \
  --benchmark-ticker SPY \
  --benchmark-equal-weight \
  --debug-roll
"""
