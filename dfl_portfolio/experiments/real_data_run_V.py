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
    plot_phi_paths,
    plot_beta_paths,
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
from dfl_portfolio.models.dfl_p1_flex_V import fit_dfl_p1_flex_V
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
RESULTS_ROOT = RESULTS_BASE / "exp_real_data_V"
DEBUG_ROOT = RESULTS_BASE / "debug_outputs_V"

# Suppress Pyomo-related warnings (only show errors)
logging.getLogger("pyomo").setLevel(logging.ERROR)
logging.getLogger("pyomo.solvers").setLevel(logging.ERROR)


def mirror_to_analysis_root(src: Path, analysis_dir: Path) -> None:
    """Copy ``src`` to the parent ``analysis_dir`` for quick access."""

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
    solver_spec: SolverSpec,
    flex_options: Dict[str, Any] | None,
    train_start: int,
    train_end: int,
    tee: bool,
    V_sample_list: Sequence[np.ndarray],
    V_diag_list: Sequence[np.ndarray],
):
    delta_for_training = delta_down if model_key == "flex" else delta_up
    idx_list = bundle.cov_indices.tolist()
    base_kwargs: Dict[str, object] = dict(
        X=bundle.dataset.X,
        Y=bundle.dataset.Y,
        idx=idx_list,
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
        base_kwargs.update(resolved_flex)
        # OAS×EWMA の shrinkage 係数 δ_t から、train window 上の平均を φ のアンカーにする。
        phi_vals: List[float] = []
        cov_idx_seq = bundle.cov_indices.tolist()
        for pos, cov_idx in enumerate(cov_idx_seq):
            if cov_idx < train_start or cov_idx > train_end:
                continue
            stat = bundle.cov_stats[pos]
            delta_oas = getattr(stat, "oas_delta", None)
            if delta_oas is None:
                continue
            try:
                v = float(delta_oas)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                phi_vals.append(v)
        phi_anchor_val = float(np.mean(phi_vals)) if phi_vals else 0.0
        base_kwargs["phi_anchor"] = phi_anchor_val

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


def _parse_delta_list(value: object, fallback: float, *, allow_multiple: bool) -> List[float]:
    if value is None:
        return [fallback]
    if isinstance(value, (int, float)):
        values = [float(value)]
    else:
        text = str(value).strip()
        if not text:
            return [fallback]
        tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
        if not tokens:
            raise ValueError("delta value list must contain at least one numeric entry.")
        try:
            values = [float(tok) for tok in tokens]
        except ValueError as exc:
            raise ValueError(f"Invalid delta specification '{value}'. Expected comma-separated floats.") from exc
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"delta values must be within [0,1]; got {v}")
    if not allow_multiple and len(values) > 1:
        raise ValueError("Multiple values are not allowed for this delta option.")
    return values


def run_rolling_experiment(
    model_key: str,
    model_label: str,
    bundle,
    delta_up: float,
    delta_down_candidates: Sequence[float],
    trading_cost_enabled: bool,
    asset_cost_overrides: Dict[str, float] | None,
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

    tickers = bundle.dataset.config.tickers
    asset_cost_rates = resolve_trading_cost_rates(
        tickers,
        asset_cost_overrides or {},
        enable_default_costs=trading_cost_enabled,
    )
    costs_active = bool(np.any(asset_cost_rates > 0.0))
    mean_defined_cost_bps = float(np.mean(asset_cost_rates) * 10000.0) if costs_active else 0.0

    wealth = 1.0
    wealth_net = 1.0
    wealth_dates: List[pd.Timestamp] = []
    wealth_values: List[float] = []
    wealth_net_values: List[float] = []
    wealth_labels: List[str] = []

    step_rows: List[Dict[str, object]] = []
    rebalance_rows: List[Dict[str, object]] = []
    asset_rows: List[Dict[str, object]] = []
    delta_history: List[Dict[str, object]] = []
    prev_weights: Optional[np.ndarray] = None

    total_cycles = len(schedule)
    for cycle_id, item in enumerate(schedule):
        candidate_list = (
            list(delta_down_candidates) if model_key == "flex" else [delta_down_candidates[0]]
        )
        selected_delta_down = candidate_list[0]
        train_objective = float("nan")
        grid_stats: List[Dict[str, float]] = []
        if model_key == "flex" and len(candidate_list) > 1:
            best: Optional[Tuple[np.ndarray, Dict[str, object], float, float]] = None
            elapsed_total = 0.0
            for cand in candidate_list:
                theta_hat_c, info_c, elapsed_c = train_model_window(
                    model_key,
                    trainer,
                    bundle,
                    delta_up,
                    cand,
                    solver_spec,
                    flex_options,
                    item.train_start,
                    item.train_end,
                    tee,
                    V_sample_list,
                    V_diag_list,
                )
                elapsed_total += elapsed_c
                obj_raw = info_c.get("objective_value") if isinstance(info_c, dict) else None
                obj_val = float(obj_raw) if obj_raw is not None else float("inf")
                grid_stats.append({"delta_down": float(cand), "objective": float(obj_val)})
                if best is None or obj_val < best[2]:
                    best = (theta_hat_c, info_c, obj_val, cand)
            if best is None:
                raise RuntimeError("Delta grid search failed to produce a valid solution.")
            theta_hat, info, train_objective, selected_delta_down = best
            elapsed = elapsed_total
        else:
            selected_delta = candidate_list[0]
            theta_hat, info, elapsed = train_model_window(
                model_key,
                trainer,
                bundle,
                delta_up,
                selected_delta,
                solver_spec,
                flex_options,
                item.train_start,
                item.train_end,
                tee,
                V_sample_list,
                V_diag_list,
            )
            obj_raw = info.get("objective_value") if isinstance(info, dict) else None
            train_objective = float(obj_raw) if obj_raw is not None else float("nan")
            grid_stats.append({"delta_down": float(selected_delta), "objective": train_objective})
            selected_delta_down = selected_delta
        Yhat_all = predict_yhat(bundle.dataset.X, theta_hat)
        Yhat_all = predict_yhat(bundle.dataset.X, theta_hat)

        if debug_roll:
            progress = (cycle_id + 1) / max(total_cycles, 1)
            bar = "#" * int(progress * 20)
            # ログは負荷軽減のため 20 サイクルごと＋最初／最後のみ出力する
            if (cycle_id + 1) % 20 == 0 or cycle_id == 0 or (cycle_id + 1) == total_cycles:
                print(
                    f"[roll-debug-V] model={model_label} cycle={cycle_id+1}/{total_cycles} "
                    f"idx={item.rebalance_idx} train=[{item.train_start},{item.train_end}] "
                    f"n_eval={len(item.eval_indices)} [{bar:<20}] {progress:.0%}"
                )

        phi_hat = info.get("phi_hat", None)
        phi_anchor = info.get("phi_anchor", None)
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
                "phi_anchor": float(phi_anchor) if phi_anchor is not None else np.nan,
                "delta_down_selected": float(selected_delta_down),
                "train_objective": float(train_objective),
            }
        )

        if model_key == "flex":
            delta_history.append(
                {
                    "model": display_model_name(model_label),
                    "cycle": cycle_id,
                    "rebalance_idx": item.rebalance_idx,
                    "rebalance_date": bundle.dataset.timestamps[item.rebalance_idx].isoformat(),
                    "delta_used": float(selected_delta_down),
                    "objective_value": float(train_objective),
                    "delta_candidates": json.dumps(grid_stats, ensure_ascii=False),
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
            # 学習済みの φ を用いて、テスト期間でも
            # V_eff = (1-φ) * V + φ * diag(V) を使って配分を決定する。
            V_used = cov
            if model_key == "flex" and phi_hat is not None and np.isfinite(phi_hat):
                phi_val = float(phi_hat)
                diag_cov = np.diag(np.diag(cov))
                V_used = (1.0 - phi_val) * cov + phi_val * diag_cov
            yhat = Yhat_all[eval_idx]
            z = solve_mvo_gurobi(
                y_hat=yhat,
                V_hat=V_used,
                delta=delta_up,
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

            cost = mvo_cost(z, bundle.dataset.Y[eval_idx], V_used, delta_up)
            eigvals = np.linalg.eigvalsh(V_used)
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

    # CVaR (Expected Shortfall) at 95%: 下側 5% の平均リターン
    from dfl_portfolio.real_data.reporting import compute_cvar

    cvar_95 = compute_cvar(returns, alpha=0.05)

    # 決定係数 R^2（資産リターン予測の当てはまり）
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
        # R^2 は診断用なので、例外時は NaN のままにして続行する
        r2 = float("nan")

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

    terminal_wealth = float(wealth_values[-1]) if wealth_values else 1.0
    terminal_wealth_net = float(wealth_net_values[-1]) if wealth_net_values else 1.0
    total_return = terminal_wealth - 1.0
    total_return_net = terminal_wealth_net - 1.0

    stats_report = {
        "model": display_model_name(model_label),
        "n_retrain": len(rebalance_rows),
        "n_invest_steps": int(len(step_df)),
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
        "step_df": step_df,
        "weights_df": weights_df,
        "wealth_df": wealth_df,
        "rebalance_df": rebalance_df,
        "cov_stats": {},
        "period_metrics": period_metrics,
        "delta_history": delta_history,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    benchmark_specs = parse_commalist(getattr(args, "benchmarks", ""))
    if not benchmark_specs:
        legacy: List[str] = []
        if (args.benchmark_ticker or "").strip():
            legacy.append("spy")
        if getattr(args, "benchmark_equal_weight", False):
            legacy.append("equal_weight")
        benchmark_specs = legacy

    # V 学習版でも通常版と同じ「EWMA+OAS」共分散を使う。
    if args.cov_method != "oas":
        raise ValueError("real_data_run_V は cov-method=oas のみ対応です。")
    if str(args.flex_solver).lower() != "knitro":
        raise ValueError("real_data_run_V は --flex-solver knitro のみ対応です。")

    tickers = parse_tickers(args.tickers)
    base_delta = float(args.delta)
    if not (0.0 <= base_delta <= 1.0):
        raise ValueError(f"delta must be within [0,1]; got {base_delta}")
    cli_delta_up = getattr(args, "delta_up", None)
    cli_delta_down = getattr(args, "delta_down", None)
    delta_up_list = _parse_delta_list(cli_delta_up, base_delta, allow_multiple=False)
    delta_up = float(delta_up_list[0])
    delta_down_list = _parse_delta_list(cli_delta_down, delta_up, allow_multiple=True)
    trading_costs_enabled = float(getattr(args, "trading_cost_bps", 0.0)) > 0.0
    raw_asset_costs: Dict[str, float] = getattr(args, "trading_cost_per_asset", {}) or {}
    asset_cost_overrides_dec = {
        ticker.upper(): max(float(rate), 0.0) / 10000.0 for ticker, rate in raw_asset_costs.items()
    }
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
        cov_ewma_alpha=getattr(args, "cov_ewma_alpha", 0.97),
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
    analysis_dir = outdir / "analysis"
    analysis_csv_dir = analysis_dir / "csv"
    analysis_fig_dir = analysis_dir / "figures"
    asset_pred_dir = analysis_csv_dir / "asset_predictions"
    model_outputs_dir = outdir / "models"
    debug_dir = DEBUG_ROOT / outdir.name
    debug_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_data_bundle(pipeline_cfg)
    bundle_summary = bundle.summary()

    # 通常版と同じ EWMA+OAS 共分散 Σ_t(α) をベースとして使用する。
    # V 学習版では、この Σ_t(α) を V_sample_list として渡し、
    # その対角成分を V_diag_list として渡す。
    V_sample_all: List[np.ndarray] = [np.asarray(C, float) for C in bundle.covariances]
    V_diag_all: List[np.ndarray] = [np.diag(np.diag(C)) for C in bundle.covariances]

    model_specs = parse_commalist(args.models)
    flex_formulations = parse_commalist(args.flex_formulation) if args.flex_formulation else ["dual"]
    flex_ensemble_enabled = "dual&kkt" in flex_formulations
    flex_formulations = [f for f in flex_formulations if f != "dual&kkt"]

    model_train_windows = parse_model_train_window_spec(args.model_train_window)

    outdir.mkdir(parents=True, exist_ok=True)
    analysis_csv_dir.mkdir(parents=True, exist_ok=True)
    analysis_fig_dir.mkdir(parents=True, exist_ok=True)
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
    pd.DataFrame(config_records).to_csv(analysis_csv_dir / "2-experiment_config.csv", index=False)

    # データの可視化（real_data_run.py と同様の構造）
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
    rebalance_records: List[Dict[str, object]] = []
    train_window_records: List[Dict[str, object]] = []
    phi_records: List[Dict[str, object]] = []
    beta_records: List[Dict[str, object]] = []
    # flex 用のサイクルごとの rebalance ログ（solver 状態の時系列可視化用）
    flex_rebalance_logs: List[pd.DataFrame] = []
    delta_records: List[Dict[str, object]] = []

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
                    "lambda_phi_anchor": args.flex_lambda_phi_anchor,
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
                delta_up=delta_up,
                delta_down_candidates=delta_down_list,
                trading_cost_enabled=trading_costs_enabled,
                asset_cost_overrides=asset_cost_overrides_dec,
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
            if model_key == "flex":
                hist = run_result.get("delta_history", [])
                if hist:
                    delta_records.extend(hist)
            reb_df = run_result["rebalance_df"]
            if model_key == "flex" and not reb_df.empty:
                flex_rebalance_logs.append(reb_df.copy())
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
                                "phi_anchor": float(row.get("phi_anchor", np.nan)),
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
                                "beta_used": float(getattr(args, "cov_ewma_alpha", 0.97)),
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

    bench_stats, bench_wealth = run_benchmark_suite(
        bundle,
        benchmarks=benchmark_specs,
        args=args,
        trading_costs_enabled=trading_costs_enabled,
        asset_cost_overrides=asset_cost_overrides_dec,
        eval_start=pd.Timestamp(args.start),
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
            if not need_cols.issubset(df.columns):
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
    if phi_records:
        phi_df = pd.DataFrame(phi_records)
        phi_df.to_csv(analysis_csv_dir / "phi_trajectory.csv", index=False)
        plot_phi_paths(phi_df, analysis_fig_dir / "phi_paths.png")
    if beta_records:
        beta_df = pd.DataFrame(beta_records)
        beta_df.to_csv(analysis_csv_dir / "beta_trajectory.csv", index=False)
        plot_beta_paths(beta_df, analysis_fig_dir / "beta_paths.png")
    if delta_records:
        delta_df = pd.DataFrame(delta_records)
        delta_df.to_csv(analysis_csv_dir / "delta_trajectory.csv", index=False)
        plot_delta_paths(delta_df, analysis_fig_dir / "delta_paths.png")
    if not summary_df.empty:
        update_experiment_ledger(RESULTS_ROOT, outdir, args, summary_df, analysis_csv_dir, bundle_summary)

    run_extended_analysis(analysis_csv_dir, analysis_fig_dir, model_outputs_dir, asset_pred_dir)

    print(f"[real-data-V] finished. outputs -> {outdir}")
    print(f"[real-data-V] debug artifacts -> {debug_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""
source /Users/kensei/Documents/VScode/GraduationResearch/gurobi-env/bin/activate
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.real_data_run_V \
  --tickers "SPY,GLD,EEM,TLT" \
  --start 2006-10-31 --end 2025-10-31 \
  --frequency weekly \
  --resample-rule W-FRI \
  --momentum-window 26 \
  --return-horizon 1 \
  --cov-window 13 \
  --cov-method oas \
  --train-window 26 \
  --rebal-interval 4 \
  --delta 0.5 \
  --delta-up 0.5 \
  --delta-down 0.5 \
  --models ols,ipo,flex \
  --flex-solver knitro \
  --flex-formulation 'dual' \
  --flex-lambda-theta-anchor 10.0 \
  --flex-theta-anchor-mode ipo \
  --flex-theta-init-mode none \
  --flex-lambda-phi-anchor 0.0 \
  --benchmark-ticker SPY \
  --benchmark-equal-weight \
  --debug-roll

"""
