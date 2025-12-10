from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dfl_portfolio.real_data.loader import MarketLoaderConfig
from dfl_portfolio.real_data.pipeline import PipelineConfig, build_data_bundle
from dfl_portfolio.real_data.cli import (
    build_parser,
    make_output_dir,
    parse_tickers,
    parse_model_train_window_spec,
    parse_commalist,
)
from dfl_portfolio.real_data.reporting import (
    WEIGHT_THRESHOLD,
    compute_benchmark_series,
    compute_equal_weight_benchmark,
    compute_period_metrics,
    compute_pairwise_mean_return_tests,
    compute_sortino_ratio,
    display_model_name,
    export_average_weights,
    export_weight_threshold_frequency,
    export_weight_variance_correlation,
    max_drawdown,
    plot_multi_wealth,
    plot_wealth_correlation_heatmap,
    plot_wealth_curve,
    plot_wealth_with_events,
    plot_weight_comparison,
    plot_weight_histograms,
    plot_weight_paths,
    run_extended_analysis,
    compute_pairwise_performance_tests,
    summarize_dfl_performance_significance,
    format_summary_for_output,
)
from dfl_portfolio.models.ols_gurobi import solve_mvo_gurobi
from dfl_portfolio.models.ols_multi import train_ols_multi, predict_yhat_multi
from dfl_portfolio.models.dfl_p1_flex_multi import fit_dfl_p1_flex_multi
from dfl_portfolio.experiments.real_data_common import (
    mvo_cost,
    build_rebalance_schedule,
    ScheduleItem,
    build_flex_dual_kkt_ensemble,
)


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
RESULTS_BASE = PROJECT_ROOT / "results"
RESULTS_ROOT = RESULTS_BASE / "exp_real_data_multi"
DEBUG_ROOT = RESULTS_BASE / "debug_outputs_multi"


logging.getLogger("pyomo").setLevel(logging.ERROR)
logging.getLogger("pyomo.solvers").setLevel(logging.ERROR)


def build_multi_features(bundle, short_window: int, vol_window: int) -> np.ndarray:
    """returns, momentum から (N,d,K) 形式の特徴量テンソルを構成する。

    現在は 2 変量:
      1. 中期モメンタム: momentum_window
      2. ボラティリティ: vol_window ローリング標準偏差

    短期リターン特徴量は計算負荷を抑えるため削除。
    """
    tickers = bundle.dataset.config.tickers
    idx = pd.Index(bundle.dataset.timestamps)

    returns = bundle.dataset.returns.copy()
    momentum = bundle.dataset.momentum.copy()

    momentum_feat = momentum.loc[idx]
    vol_feat = returns.rolling(vol_window).std().loc[idx]

    feats = []
    for df in (momentum_feat, vol_feat):
        df = df[tickers]
        feats.append(df.to_numpy(dtype=float))

    X_feat = np.stack(feats, axis=-1)  # (N, d, K=2)
    return X_feat


def run_rolling_experiment_multi(
    model_key: str,
    model_label: str,
    bundle,
    X_feat: np.ndarray,
    delta: float,
    train_window: int,
    rebal_interval: int,
    debug_roll: bool,
    debug_dir: Path,
    results_model_dir: Path,
    asset_pred_dir: Path | None = None,
    formulation: Optional[str] = None,
    eval_start: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """OLS / flex (multi-feature) 用ローリング実験ループ。"""
    schedule = build_rebalance_schedule(bundle, train_window, rebal_interval, eval_start=eval_start)
    if debug_roll:
        print(f"[real-data-multi] schedule length = {len(schedule)}")
    cov_lookup = {
        idx: (cov, stat)
        for idx, cov, stat in zip(
            bundle.cov_indices.tolist(), bundle.covariances, bundle.cov_stats
        )
    }

    Y = bundle.dataset.Y
    wealth = 1.0
    wealth_dates: List[pd.Timestamp] = []
    wealth_values: List[float] = []
    wealth_labels: List[str] = []

    step_rows: List[Dict[str, Any]] = []
    rebalance_rows: List[Dict[str, Any]] = []
    asset_rows: List[Dict[str, Any]] = []

    total_cycles = len(schedule)
    short_model_label = model_label

    for cycle_id, item in enumerate(schedule):
        if debug_roll:
            progress = (cycle_id + 1) / max(total_cycles, 1)
            bar = "#" * int(progress * 20)
            print(
                f"[roll-debug-multi] model={short_model_label} cycle={cycle_id+1}/{total_cycles} "
                f"idx={item.rebalance_idx} train=[{item.train_start},{item.train_end}] "
                f"n_eval={len(item.eval_indices)} [{bar:<20}] {progress:.0%}"
            )

        train_slice = slice(item.train_start, item.train_end + 1)
        start_time = time.perf_counter()
        if model_key == "ols":
            theta_hat = train_ols_multi(X_feat[train_slice], Y[train_slice])
            info: Dict[str, Any] = {}
        elif model_key == "flex":
            form = (formulation or "dual").lower()
            theta_hat, _, _, _, _, info = fit_dfl_p1_flex_multi(
                X_feat,
                Y,
                bundle.covariances,
                bundle.cov_indices.tolist(),
                start_index=item.train_start,
                end_index=item.train_end,
                formulation=form,
                delta=delta,
                solver="knitro",
                solver_options=None,
                tee=False,
                lambda_theta_anchor=0.0,
                lambda_theta_anchor_l1=0.0,
                lambda_theta_iso=0.0,
            )
        else:
            raise ValueError(f"Unsupported model_key for multi experiment: {model_key}")
        elapsed = time.perf_counter() - start_time

        Yhat_all = predict_yhat_multi(X_feat, theta_hat)

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
            realized = float(z @ Y[eval_idx])
            wealth *= (1.0 + realized)
            wealth_dates.append(bundle.dataset.timestamps[eval_idx])
            wealth_values.append(wealth)
            wealth_labels.append("after_step")

            cost = mvo_cost(z, Y[eval_idx], cov, delta)
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
            yreal_vec = Y[eval_idx]
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
        raise RuntimeError("No evaluation steps were executed in multi experiment.")

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

    # weights for analysis
    weights_df: pd.DataFrame
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

    results_model_dir.mkdir(parents=True, exist_ok=True)
    wealth_df.to_csv(results_model_dir / "wealth.csv", index=False)
    if not weights_df.empty:
        weights_df.to_csv(results_model_dir / "weights.csv", index=False)
        plot_weight_paths(weights_df, model_label, results_model_dir / "weights.png")
    step_df.to_csv(results_model_dir / "step_metrics.csv", index=False)

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

    terminal_wealth = float(wealth_values[-1]) if wealth_values else 1.0
    total_return = terminal_wealth - 1.0

    stats_report = {
        "model": display_model_name(model_label),
        "n_retrain": len(schedule),
        "n_invest_steps": len(step_rows),
        "ann_return": mean_return,
        "ann_volatility": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown(wealth_values),
        "terminal_wealth": terminal_wealth,
        "total_return": total_return,
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
        "period_metrics": period_metrics,
        "wealth_df": wealth_df,
        "weights_df": weights_df,
        "rebalance_summary": reb_summary,
    }


def main() -> None:
    parser = build_parser()
    parser.add_argument(
        "--multi-short-window",
        type=int,
        default=0,
        help="Rolling window (in periods) for short-horizon features (0 -> momentum_window//4, min 5).",
    )
    parser.add_argument(
        "--multi-vol-window",
        type=int,
        default=0,
        help="Rolling window (in periods) for volatility feature (0 -> same as multi-short-window).",
    )
    args = parser.parse_args()
    model_train_windows = parse_model_train_window_spec(getattr(args, "model_train_window", ""))

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
        cov_ewma_alpha=getattr(args, "cov_ewma_alpha", 0.94),
        auto_adjust=not args.no_auto_adjust,
        cache_dir=None,
        force_refresh=args.force_refresh,
        debug=not args.no_debug,
        train_window=args.train_window,
    )
    pipe_cfg = PipelineConfig(loader=loader_cfg, debug=not args.no_debug)
    bundle = build_data_bundle(pipe_cfg)

    # 実験設定の記録
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
    config_records: List[Dict[str, object]] = []
    for key, value in sorted(vars(args).items()):
        config_records.append({"parameter": key, "value": value})
    pd.DataFrame(config_records).to_csv(analysis_csv_dir / "2-experiment_config.csv", index=False)

    # 多変量特徴量の構築
    if args.multi_short_window and args.multi_short_window > 0:
        short_window = int(args.multi_short_window)
    else:
        short_window = max(5, args.momentum_window // 4)
    if args.multi_vol_window and args.multi_vol_window > 0:
        vol_window = int(args.multi_vol_window)
    else:
        vol_window = short_window
    X_feat = build_multi_features(bundle, short_window=short_window, vol_window=vol_window)

    # モデルリスト: ols と flex のみ対応
    model_keys = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    supported = {"ols", "flex"}
    filtered = [m for m in model_keys if m in supported]
    if not filtered:
        raise ValueError(f"Multi-feature experiment supports only models {supported}, got {model_keys}")
    skipped = [m for m in model_keys if m not in supported]
    if skipped:
        print(f"[real-data-multi] skipping unsupported models in multi experiment: {skipped}")

    stats_results: List[Dict[str, Any]] = []
    period_rows: List[Dict[str, Any]] = []
    wealth_dict: Dict[str, pd.DataFrame] = {}
    weight_dict: Dict[str, pd.DataFrame] = {}
    train_window_records: List[Dict[str, Any]] = []
    rebalance_records: List[Dict[str, Any]] = []

    # flex の formulation 展開
    flex_forms = parse_commalist(getattr(args, "flex_formulation", "dual"))
    expanded_models: List[tuple[str, str, Optional[str]]] = []
    for base in filtered:
        if base == "flex":
            for form in flex_forms:
                f = form.lower()
                if f in {"dual", "kkt"}:
                    label = f"flex_{f}"
                    expanded_models.append(("flex", label, f))
        else:
            expanded_models.append((base, base, None))

    for model_key, label, formulation in expanded_models:
        if args.debug_roll:
            print(f"[real-data-multi] rolling model={label}")
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
                f"[real-data-multi] overriding train_window for {label}: "
                f"{effective_train_window} (default {args.train_window})"
            )

        eval_start_ts = pd.Timestamp(args.start)
        run_result = run_rolling_experiment_multi(
            model_key=model_key,
            model_label=label,
            bundle=bundle,
            X_feat=X_feat,
            delta=args.delta,
            train_window=effective_train_window,
            rebal_interval=args.rebal_interval,
            debug_roll=args.debug_roll,
            debug_dir=debug_dir,
            results_model_dir=results_dir,
            asset_pred_dir=asset_pred_dir,
            formulation=formulation,
            eval_start=eval_start_ts,
        )
        stats_results.append(run_result["stats"])
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
        for row in run_result["period_metrics"]:
            period_entry = dict(row)
            period_entry["model"] = display_model_name(label)
            period_entry["train_window"] = run_result["stats"].get("train_window", effective_train_window)
            period_rows.append(period_entry)
        wealth_dict[label] = run_result["wealth_df"][["date", "wealth"]]
        if not run_result["weights_df"].empty:
            weight_dict[label] = run_result["weights_df"]

    # flex dual/kkt ensemble (multi) を構成
    if "dual&kkt" in flex_forms:
        build_flex_dual_kkt_ensemble(
            bundle=bundle,
            debug_dir=debug_dir,
            args=args,
            wealth_dict=wealth_dict,
            weight_dict=weight_dict,
            stats_results=stats_results,
            period_rows=period_rows,
            log_prefix="[ensemble-multi]",
        )

    # ベンチマーク（シングルティッカー）
    benchmark_spec = (args.benchmark_ticker or "").strip()
    if benchmark_spec and wealth_dict:
        min_date: Optional[pd.Timestamp] = None
        for df in wealth_dict.values():
            if df.empty:
                continue
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

    # 等配分ベンチマーク
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

    # summary.csv など出力
    summary_df = pd.DataFrame(stats_results)
    if not summary_df.empty:
        if "model" in summary_df.columns:
            summary_df["model"] = summary_df["model"].map(display_model_name)
        summary_df["max_drawdown"] = summary_df["max_drawdown"].astype(float)
        summary_df = format_summary_for_output(summary_df)
        summary_df.to_csv(analysis_csv_dir / "1-summary.csv", index=False)
    else:
        (analysis_csv_dir / "1-summary.csv").write_text("")

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

    run_extended_analysis(analysis_csv_dir, analysis_fig_dir, model_outputs_dir, asset_pred_dir)

    print(f"[real-data-multi] finished. outputs -> {outdir}")
    print(f"[real-data-multi] debug artifacts -> {debug_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""
source /Users/kensei/Documents/VScode/GraduationResearch/gurobi-env/bin/activate
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.real_data_run_multi \
  --tickers "SPY,GLD,EEM,TLT" \
  --start 2006-10-31 --end 2025-10-31 \
  --interval 1d \
  --price-field Close \
  --return-kind log \
  --frequency weekly \
  --resample-rule W-FRI \
  --return-horizon 1 \
  --cov-window 10 \
  --cov-method diag \
  --cov-shrinkage 0.94 \
  --cov-eps 1e-6 \
  --train-window 25 \
  --rebal-interval 4 \
  --delta 0.5 \
  --models ols,flex \
  --flex-solver knitro \
  --flex-formulation 'dual' \
  --benchmark-ticker SPY \
  --benchmark-equal-weight \
  --momentum-window 30 \
  --multi-vol-window 30 \
  --debug-roll




"""
