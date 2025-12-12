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

try:  # matplotlib is only needed for custom window plots
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - best effort
    plt = None

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
    plot_solver_summary_bars,
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
RESULTS_ROOT = RESULTS_BASE / "exp_real_data"
DEBUG_ROOT = RESULTS_BASE / "debug_outputs"

# Suppress Pyomo-related warnings (only show errors)
logging.getLogger("pyomo").setLevel(logging.ERROR)
logging.getLogger("pyomo.solvers").setLevel(logging.ERROR)


def plot_weight_window_with_connections(weight_dict: Dict[str, pd.DataFrame], path: Path) -> None:
    """Render stacked weights and effective connection counts per model."""

    if plt is None or not weight_dict:
        return
    entries: List[Tuple[str, pd.DataFrame]] = []
    for model_key, df in weight_dict.items():
        if df.empty or "date" not in df.columns:
            continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.sort_values("date")
        entries.append((str(model_key), tmp))
    if not entries:
        return

    n_models = len(entries)
    fig_height = max(2.5 * n_models + 2, 4)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(12, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    axes_list = list(axes)
    conn_ax = axes_list[-1]
    model_axes = axes_list[:-1]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])

    for idx, ((model_key, df_model), ax) in enumerate(zip(entries, model_axes)):
        asset_cols = [c for c in df_model.columns if c not in {"date", "portfolio_return_sq"}]
        if not asset_cols:
            continue
        values = df_model[asset_cols].astype(float)
        dates = pd.to_datetime(df_model["date"])
        bottom = np.zeros(len(values))
        for col_idx, col in enumerate(asset_cols):
            label = col if idx == 0 else None
            ax.bar(dates, values[col], bottom=bottom, width=5, label=label)
            bottom += values[col].to_numpy()
        ax.set_ylabel(display_model_name(model_key))
        ax.set_ylim(0.0, 1.0)
        if idx == 0 and asset_cols:
            ax.legend(loc="upper right", fontsize=8)

        weights = values
        h_val = (weights.pow(2).sum(axis=1)).replace(0.0, np.nan)
        neff = 1.0 / h_val
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        conn_ax.plot(
            dates,
            neff,
            label=display_model_name(model_key),
            color=color,
        )

    conn_ax.set_ylabel("N_eff")
    conn_ax.set_xlabel("date")
    conn_ax.set_title("Effective connection count (N_eff)")
    conn_ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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
    ipo_grad_debug_kkt: bool = False,
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
        theta_init_override, resolved_flex = prepare_flex_training_args(
            bundle, train_start, train_end, delta_for_training, tee, flex_options
        )
        trainer_kwargs.update(resolved_flex)
    if model_key == "ipo_grad":
        # Warm-start IPO-GRAD from the analytical IPO closed-form solution
        try:
            theta_ipo, _, _, _, _, _ = fit_ipo_closed_form(
                bundle.dataset.X,
                bundle.dataset.Y,
                bundle.covariances,
                bundle.cov_indices.tolist(),
                start_index=train_start,
                end_index=train_end,
                delta=delta_for_training,
                mode="budget",
                psd_eps=1e-12,
                ridge_theta=1e-10,
                tee=tee,
            )
            theta_init_override = np.asarray(theta_ipo, dtype=float)
        except Exception as exc:  # pragma: no cover - best-effort warm start
            if tee:
                print(f"[IPO-GRAD] IPO closed-form warm-start failed: {exc}")
    if theta_init_override is not None:
        trainer_kwargs["theta_init"] = theta_init_override
    if model_key == "ipo_grad":
        trainer_kwargs["ipo_grad_debug_kkt"] = bool(ipo_grad_debug_kkt)

    start_time = time.perf_counter()
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
    tee: bool,
    asset_pred_dir: Path | None = None,
    eval_start: Optional[pd.Timestamp] = None,
    ipo_grad_debug_kkt: bool = False,
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
    cost_label_entries: List[str] = []
    if costs_active:
        for ticker, rate in zip(tickers, asset_cost_rates):
            bps_val = float(rate * 10000.0)
            if not np.isfinite(bps_val):
                continue
            if abs(bps_val - round(bps_val)) < 1e-9:
                entry = f"{ticker}:{int(round(bps_val))}"
            else:
                entry = f"{ticker}:{bps_val:.2f}"
            cost_label_entries.append(entry)
    trading_cost_label = ",".join(cost_label_entries)

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
                    ipo_grad_debug_kkt=ipo_grad_debug_kkt,
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
                ipo_grad_debug_kkt=ipo_grad_debug_kkt,
            )
            obj_raw = info.get("objective_value") if isinstance(info, dict) else None
            train_objective = float(obj_raw) if obj_raw is not None else float("nan")
            grid_stats.append({"delta_down": float(selected_delta), "objective": train_objective})
            selected_delta_down = selected_delta
        Yhat_all = predict_yhat(bundle.dataset.X, theta_hat)

        if debug_roll:
            progress = (cycle_id + 1) / max(total_cycles, 1)
            bar = "#" * int(progress * 20)
            # ログは負荷軽減のため 20 サイクルごとにのみ出力する
            if (cycle_id + 1) % 20 == 0:
                print(
                    f"[roll-debug] model={model_label} cycle={cycle_id+1}/{total_cycles} "
                    f"idx={item.rebalance_idx} train=[{item.train_start},{item.train_end}] "
                    f"n_eval={len(item.eval_indices)} [{bar:<20}] {progress:.0%}"
                )

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
                "train_eq_viol_max": (info or {}).get("eq_viol_max", float("nan")),
                "train_ineq_viol_max": (info or {}).get("ineq_viol_max", float("nan")),
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
            yhat = Yhat_all[eval_idx]
            z = solve_mvo_gurobi(
                y_hat=yhat,
                V_hat=cov,
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

            cost = mvo_cost(z, bundle.dataset.Y[eval_idx], cov, delta_up)
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

            # asset-level predictions / realized returns
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
    # Sortino も Sharpe と整合するように年率換算する
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
        "trading_cost_bps": trading_cost_label,
        "avg_condition_number": avg_condition,
    }

    report_path = model_debug_dir / "debug_notes.txt"
    report_path.write_text(
        "\n".join(
            [
                "# Rolling debug notes",
                f"model: {model_key}",
                f"cycles: {len(schedule)}",
                f"steps: {len(step_rows)}",
                "Check points:",
                "- eig_min > 0 (see step_log.csv) ⇒ 共分散が正定値",
                "- wealth.csv の initial 行 = 1.0 で始まり、以降 0 未満になっていないか",
                "- step_log の weight_sum≈1, weight_min>=0 か（制約違反チェック）",
                "- rebalance_summary.json で solver_status が異常値を持っていないか",
            ]
        ),
        encoding="utf-8",
    )

    period_metrics = compute_period_metrics(step_df)

    # asset-level CSV
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
        "delta_history": delta_history,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model_train_windows = parse_model_train_window_spec(getattr(args, "model_train_window", ""))
    benchmark_specs = parse_commalist(getattr(args, "benchmarks", ""))
    if not benchmark_specs:
        legacy_list: List[str] = []
        if (args.benchmark_ticker or "").strip():
            legacy_list.append("spy")
        if getattr(args, "benchmark_equal_weight", False):
            legacy_list.append("equal_weight")
        benchmark_specs = legacy_list

    tickers = parse_tickers(args.tickers)
    outdir = make_output_dir(RESULTS_ROOT, args.outdir)
    analysis_dir = outdir / "analysis"
    analysis_csv_dir = analysis_dir / "csv"
    analysis_fig_dir = analysis_dir / "figures"
    asset_pred_dir = analysis_csv_dir / "asset_predictions"
    analysis_csv_dir.mkdir(parents=True, exist_ok=True)
    analysis_fig_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = DEBUG_ROOT / f"{outdir.name}_rolling"
    debug_dir.mkdir(parents=True, exist_ok=True)
    model_outputs_dir = debug_dir
    model_outputs_dir.mkdir(parents=True, exist_ok=True)

    # Safety check: delta in [0,1] (enforced by CLI type) and IPO+delta=0 is invalid
    base_delta = float(args.delta)
    if base_delta < 0.0 or base_delta > 1.0:
        raise ValueError(f"delta must be in [0, 1], got {base_delta}")
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

    flex_forms_raw = parse_commalist(args.flex_formulation)
    if not flex_forms_raw:
        flex_forms_raw = ["dual"]

    ensemble_aliases = {"dual&kkt", "dual_kkt", "dual_kkt_ens", "ensemble", "ens"}
    flex_ensemble_enabled = any(f in ensemble_aliases for f in flex_forms_raw)

    flex_formulations = [f for f in flex_forms_raw if f not in ensemble_aliases]
    if not flex_formulations:
        flex_formulations = ["dual"]

    valid_forms = {"dual", "kkt"}
    for form in flex_formulations:
        if form not in valid_forms:
            raise ValueError(f"Unknown flex formulation '{form}'. Use 'dual' or 'kkt'.")

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
        cache_dir=None,
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
            data_fig_dir / "asset_return_correlation.png",
            corr_stats,
        )
        # 各日ごとに「最大リターンを出した資産」の回数を集計し、CSV と棒グラフを出力
        export_max_return_winner_counts(
            returns_matrix,
            analysis_csv_dir / "asset_max_return_wins.csv",
            data_fig_dir / "asset_max_return_wins.png",
        )

    stats_results: List[Dict[str, object]] = []
    period_rows: List[Dict[str, object]] = []
    wealth_dict: Dict[str, pd.DataFrame] = {}
    weight_dict: Dict[str, pd.DataFrame] = {}
    weight_dict: Dict[str, pd.DataFrame] = {}
    train_window_records: List[Dict[str, object]] = []
    rebalance_records: List[Dict[str, object]] = []
    # flex モデルのサイクルごとの rebalance ログ（solver 状態の時系列可視化用）
    flex_rebalance_logs: List[pd.DataFrame] = []
    delta_records: List[Dict[str, object]] = []

    flex_base_options = dict(
        lambda_theta_anchor=args.flex_lambda_theta_anchor,
        lambda_theta_iso=args.flex_lambda_theta_iso,
        theta_anchor_mode=args.flex_theta_anchor_mode,
        theta_init_mode=args.flex_theta_init_mode,
    )

    model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    # 以前は IPO 解析解が δ=0 に対応していなかったためエラーにしていたが、
    # 現在は δ=0 のとき solve_mvo_gurobi 側で「期待リターン最大資産へのフルベット」
    # を解析的に返すようにしているため、ここでは弾かずにそのまま実験を許可する。

    eval_start_ts = pd.Timestamp(args.start)

    for model_key in model_list:
        if model_key not in {"ols", "ipo", "ipo_grad", "flex"}:
            print(f"[real-data] skipping unsupported model '{model_key}'")
            continue
        formulations = flex_formulations if model_key == "flex" else [None]
        for form in formulations:
            label = model_key
            if model_key == "flex" and (len(formulations) > 1 or (form and form != "dual")):
                label = f"{model_key}_{form}"
            print(f"[real-data] rolling model={label}")
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
                    f"[real-data] overriding train_window for {label}: "
                    f"{effective_train_window} (default {args.train_window})"
                )
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
                asset_pred_dir=asset_pred_dir,
                eval_start=eval_start_ts,
                ipo_grad_debug_kkt=getattr(args, "ipo_grad_debug_kkt", False),
            )
            stats_results.append(run_result["stats"])
            if model_key == "flex":
                hist = run_result.get("delta_history", [])
                if hist:
                    delta_records.extend(hist)
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

    # Build flex dual/kkt ensemble if requested, using step_log.csv (convex combination of returns).
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

    summary_output_path = analysis_dir / "1-summary.csv"
    summary_csv_path = analysis_csv_dir / "1-summary.csv"
    summary_cost_path = analysis_dir / "1-summary_cost.csv"
    summary_cost_csv_path = analysis_csv_dir / "1-summary_cost.csv"
    summary_raw_df = pd.DataFrame(stats_results)
    if not summary_raw_df.empty:
        summary_raw_df["max_drawdown"] = summary_raw_df["max_drawdown"].astype(float)
        gross_df = summary_raw_df.drop(
            columns=[c for c in summary_raw_df.columns if c.endswith("_net")],
            errors="ignore",
        ).drop(columns=["trading_cost_bps"], errors="ignore")
        summary_df = format_summary_for_output(gross_df)
        summary_df.to_csv(summary_output_path, index=False)

        net_keep_cols = ["model"] + [c for c in summary_raw_df.columns if c.endswith("_net")]
        net_keep_cols += [
            "avg_turnover",
            "avg_trading_cost",
            "trading_cost_bps",
            "r2",
            "n_retrain",
            "n_invest_steps",
            "max_drawdown",
            "cvar_95",
        ]
        summary_cost_base = summary_raw_df[[c for c in net_keep_cols if c in summary_raw_df.columns]].copy()
        summary_cost_df = format_summary_for_output(summary_cost_base)
        summary_cost_df.to_csv(summary_cost_path, index=False)
    else:
        summary_output_path.write_text("")
        summary_cost_path.write_text("")
    shutil.copy2(summary_output_path, summary_csv_path)
    shutil.copy2(summary_cost_path, summary_cost_csv_path)

    if period_rows:
        period_df = pd.DataFrame(period_rows)
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
            # 同一日付に複数行（初期値とステップ後など）があるので、日付ごとに最後の行を採用して正規化する
            wealth_merge = wealth_merge.groupby("date", as_index=False).last()
            wealth_merge.to_csv(analysis_csv_dir / "wealth_comparison.csv", index=False)
            plot_multi_wealth(
                {m: df for m, df in wealth_dict.items()},
                analysis_fig_dir / "wealth_comparison.png",
            )
            wealth_events_base_path = analysis_dir / "wealth_events.png"
            wealth_events_primary_path = analysis_dir / "2-wealth_events.png"
            # 1) 全モデル版（従来どおり）
            plot_wealth_with_events(
                {m: df for m, df in wealth_dict.items()},
                wealth_events_base_path,
            )
            if wealth_events_primary_path.exists():
                shutil.copy2(
                    wealth_events_primary_path,
                    analysis_fig_dir / wealth_events_primary_path.name,
                )
            # 2) DFL 系モデルのみ（flex 系だけ抽出）
            flex_only = {m: df for m, df in wealth_dict.items() if "flex" in str(m)}
            if flex_only:
                plot_wealth_with_events(
                    flex_only,
                    analysis_fig_dir / "wealth_events_dfl_only.png",
                )
            # 3) DFL-dual vs その他（flex_dual を基準に比較）
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
                    # 比較対象として flex 系以外のモデルのみ追加
                    if "flex" in str(key):
                        continue
                    dual_vs_others[key] = df
                if len(dual_vs_others) > 1:
                    plot_wealth_with_events(
                        dual_vs_others,
                        analysis_fig_dir / "wealth_events_flex_dual_vs_baselines.png",
                    )
            # 4) 各危機ウィンドウごとに、期間開始時点を 1 とした累積推移を描画
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
            # 5) 実験全期間を 5 年ごとに区切り、各ウィンドウの開始時点を 1 に正規化した推移を出力
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
                    # 最終ウィンドウは末尾まで含める
                    if end_win > end_all:
                        end_win = end_all
                    window_label = f"{start_win.year}_{end_win.year}"
                    sub_dict: Dict[str, pd.DataFrame] = {}
                    for m, df in wealth_dict.items():
                        if df.empty:
                            continue
                        tmp = df.copy()
                        tmp["date"] = pd.to_datetime(tmp["date"])
                        # 5年ウィンドウは [start_win, end_win) とする
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
                        flex_only = {m: df for m, df in sub_dict.items() if "flex" in str(m)}
                        if flex_only:
                            plot_wealth_window_normalized(
                                flex_only,
                                window_label,
                                start_win,
                                end_win,
                                five_year_dir_dfl / f"wealth_window_5y_{window_label}.png",
                                events_by_model=solver_events_by_model,
                            )
                    # 次の 5 年ウィンドウへ
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
        # 主要イベント期間ごとの weights_comparison も追加
        for name, start, end in [
            # リーマンショック〜世界金融危機
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
                    sub_weights, event_fig_dir / f"weights_comparison_{name}.png"
                )
        # 実験全期間を 2 年ごとに区切り、各ウィンドウの weights_comparison を出力
        weights_2y_dir = analysis_fig_dir / "weights_windows_2y"
        weights_all_dir = weights_2y_dir / "all_models"
        weights_dfl_dir = weights_2y_dir / "dfl_only"
        weights_all_dir.mkdir(parents=True, exist_ok=True)
        weights_dfl_dir.mkdir(parents=True, exist_ok=True)
        # 全モデルの weight 日付の min/max を取得
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
                end_win = start_win + pd.DateOffset(years=2)
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
                    plot_weight_window_with_connections(
                        sub_weights,
                        weights_all_dir / f"weights_window_2y_{window_label}.png",
                    )
                    flex_only = {k: v for k, v in sub_weights.items() if "flex" in str(k)}
                    if flex_only:
                        plot_weight_window_with_connections(
                            flex_only,
                            weights_dfl_dir / f"weights_window_2y_{window_label}.png",
                        )
                start_win = start_win + pd.DateOffset(years=2)
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
        solver_debug_dir = analysis_fig_dir / "solver_debug"
        solver_debug_dir.mkdir(parents=True, exist_ok=True)
        plot_solver_summary_bars(rebalance_df, solver_debug_dir)
    # flex solver debug: サイクルごとの elapsed / status を時系列で表示
    if flex_rebalance_logs:
        flex_debug_df = pd.concat(flex_rebalance_logs, ignore_index=True)
        solver_debug_dir = analysis_fig_dir / "solver_debug"
        solver_debug_dir.mkdir(parents=True, exist_ok=True)
        plot_flex_solver_debug(flex_debug_df, solver_debug_dir / "flex_solver_debug.png")
    if delta_records:
        delta_df = pd.DataFrame(delta_records)
        delta_df.to_csv(analysis_csv_dir / "delta_trajectory.csv", index=False)
        delta_plot_df = delta_df.copy()
        if "model" in delta_plot_df.columns and "delta_used" in delta_plot_df.columns:
            means = (
                delta_plot_df.groupby("model")["delta_used"]
                .mean()
                .to_dict()
            )
            delta_plot_df["model"] = delta_plot_df["model"].map(
                lambda m: (
                    f"{m} (avg={means.get(m, float('nan')):.3f})"
                    if m in means and np.isfinite(means.get(m, np.nan))
                    else str(m)
                )
            )
        plot_delta_paths(delta_plot_df, analysis_fig_dir / "delta_paths.png")
    if not summary_df.empty:
        update_experiment_ledger(RESULTS_ROOT, outdir, args, summary_df, analysis_csv_dir, bundle_summary)

    # 追加分析: 集中度・MSE・バイアス
    run_extended_analysis(analysis_csv_dir, analysis_fig_dir, model_outputs_dir, asset_pred_dir)

    print(f"[real-data] finished. outputs -> {outdir}")
    print(f"[real-data] debug artifacts -> {debug_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""
Baseline execution (defaults match the CLI parser):
----------------------------------------------------------
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.real_data_run \
  --tickers "SPY,GLD,EEM,TLT" \
  --start 2006-01-01 --end 2025-12-01 \
  --momentum-window 26 \
  --cov-window 13 \
  --cov-method oas \
  --train-window 26 \
  --rebal-interval 4 \
  --delta-up 0.5 \
  --delta-down 0.5 \
  --models "ols,ipo,ipo_grad,flex" \
  --flex-solver knitro \
  --flex-formulation 'dual,kkt,dual&kkt' \
  --flex-lambda-theta-anchor 10.0 \
  --flex-theta-anchor-mode ipo \
  --benchmarks "spy,1/n,tsmom_spy" \
  --trading-cost-bps 1 \
  --trading-cost-per-asset "SPY:5,GLD:10,EEM:10,TLT:5" \
  --debug-roll


CLI options (defaults follow the parser definitions)
---------------------------------------------------
データ取得・前処理
- `--tickers` (str, default `SPY,GLD,EEM,TLT`): ティッカーをカンマ区切りで指定。
- `--start` / `--end` (str, default `2006-01-01` / `2025-12-01`): 取得期間。
- `--interval` (str, default `1d`): Yahoo Finance から取得する原系列の足。
- `--price-field` (str, default `Close`): 使用する価格列。
- `--return-kind` (str, `simple|log`, default `log`): 目的変数のリターン種別。
- `--frequency` (str, `daily|weekly|monthly`, default `weekly`): リバランス頻度。
- `--resample-rule` (str, default `W-FRI`): pandas resample ルール。
- `--momentum-window` (int, default `26`): モメンタム特徴の計算窓。
- `--return-horizon` (int, default `1`): 何期間先のリターンを予測するか。
- `--no-auto-adjust`: 調整後リターン補正を無効化。
- `--force-refresh`: 価格キャッシュを無視して再取得。

共分散推定
- `--cov-window` (int, default `13`): 共分散計算のローリング窓。
- `--cov-method` (str, `diag|oas|robust_lw|mini_factor`, default `oas`).
- `--cov-shrinkage` / `--cov-eps` / `--cov-robust-huber-k` / `--cov-factor-rank` / `--cov-factor-shrinkage` / `--cov-ewma-alpha`: method 固有の調整値。

ローリング設定
- `--train-window` (int, default `26`)
- `--rebal-interval` (int, default `4`)
- `--model-train-window` (str, default `""`): 例 `ols:60,flex:40` でモデル別に上書き。

モデル・Flex 設定
- `--delta` (float ∈ [0,1], default `0.5`): 既定の δ。
- `--delta-up` (float ∈ [0,1], default `None` → `--delta`): 目的関数で使う δ。
- `--delta-down` (float ∈ [0,1], default `None` → `--delta-up`): DFL 制約で使う δ。
- `--models` (str, default `ols,ipo,flex`): 走らせるモデル一覧（例 `ols,ipo,ipo_grad,flex`）。
- `--flex-solver` (str, default `knitro`).
- `--flex-formulation` (str, default `dual,kkt,dual&kkt`): Flex の定式化。
- `--flex-ensemble-weight-dual` (float ∈ [0,1], default `0.5`): dual/kkt ensemble の重み。
- `--flex-lambda-theta-anchor` / `--flex-lambda-theta-iso` (float, default `10.0` / `0.0`): θ 罰則。
- `--flex-theta-anchor-mode` (str, default `ipo`), `--flex-theta-init-mode` (str, default `none`): θ アンカー / 初期化モード。
- `--flex-lambda-phi-anchor` (float, default `0.0`): V-learning の φ アンカー罰則。

IPO-GRAD（IPO-NN）設定
- `--ipo-grad-epochs` (int, default `500`): IPO-GRAD の学習エポック数。
- `--ipo-grad-lr` (float, default `1e-3`): IPO-GRAD の学習率（Adam）。
- `--ipo-grad-batch-size` (int, default `0`): IPO-GRAD のバッチサイズ（0 の場合はフルバッチ）。
- `--ipo-grad-qp-max-iter` (int, default `5000`): IPO-GRAD 内部の QP ソルバの最大反復回数。
- `--ipo-grad-qp-tol` (float, default `1e-6`): IPO-GRAD 内部の QP ソルバの収束許容誤差。
- `--ipo-grad-debug-kkt`: IPO-GRAD の各リバランスで KKT 条件のデバッグチェックを有効化。

取引コスト
- `--trading-cost-bps` (float, default `0.0`): 正の値を指定すると、内蔵のティッカー別コスト表（bps）を有効化。
- `--trading-cost-per-asset` (str, default `""`): `SPY:5,GLD:8` のようにティッカー別コストを上書き。

実行制御・出力
- `--tee`: ソルバログを表示。
- `--debug-roll` / `--no-debug-roll`: ローリング進捗ログの表示切替（デフォルト有効）。
- `--benchmarks` (str, default `""`): 新しいベンチマーク指定（例 `spy,1/n,tsmom_spy`）。空文字のときは従来フラグにフォールバック。
- `--outdir` (Path, default `None` → `results/exp_real_data/<timestamp>` に自動作成)。
- `--no-debug`: データローダのデバッグログを抑止。
"""
