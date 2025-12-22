from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dfl_portfolio.experiments.real_data_common import (
    ScheduleItem,
    build_rebalance_schedule,
    resolve_trading_cost_rates,
)
from dfl_portfolio.real_data.reporting import (
    compute_benchmark_series,
    compute_cvar,
    compute_equal_weight_benchmark,
    compute_sortino_ratio,
    max_drawdown,
)


@dataclass
class BenchmarkResult:
    label: str
    stats: Dict[str, object]
    wealth_df: pd.DataFrame


STRATEGY_LABELS = {
    "tsmom_spy": "TSMOM (SPY)",
}


def _ensure_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    total = np.sum(w)
    if total <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w /= total
    return w


def _tsmom_spy_weights(
    returns_df: pd.DataFrame,
    train_end_ts: pd.Timestamp,
    window: int,
    benchmark_ticker: str,
) -> np.ndarray:
    """
    Time-series momentum (TSMOM) on the benchmark index (e.g. SPY).

    A simple Moskowitz-style signal: if the cumulative past `window`-step return
    of the benchmark is positive, take a long position in the benchmark;
    otherwise move to cash (no risky asset exposure).
    """
    tickers = list(returns_df.columns)
    n_assets = len(tickers)
    weights = np.zeros(n_assets, dtype=float)
    if benchmark_ticker not in returns_df.columns:
        # Fallback: equal weight across assets (keeps the run alive)
        return np.ones(n_assets, dtype=float) / max(n_assets, 1)
    window_data = returns_df[benchmark_ticker].loc[:train_end_ts].tail(window)
    if window_data.empty or len(window_data) < max(window, 1):
        # Not enough history: fall back to equal weight
        return np.ones(n_assets, dtype=float) / max(n_assets, 1)
    # Time-series momentum signal: sign of cumulative past return
    cum_ret = float((1.0 + window_data).prod() - 1.0)
    if cum_ret > 0.0:
        idx = tickers.index(benchmark_ticker)
        weights[idx] = 1.0
    # If cum_ret <= 0: stay in cash (all zeros)
    return weights


def _simulate_strategy(
    bundle,
    schedule: Sequence[ScheduleItem],
    trading_cost_rates: np.ndarray,
    strategy: str,
    *,
    benchmark_ticker: str,
    momentum_window: int,
    trading_costs_enabled: bool,
    train_window: int,
    rebal_interval: int,
) -> Optional[BenchmarkResult]:
    if not schedule:
        return None
    returns_df = bundle.dataset.returns.copy()
    tickers = list(bundle.dataset.config.tickers)
    returns_df = returns_df.reindex(columns=tickers)
    n_assets = len(tickers)
    mean_cost_bps = float(np.mean(trading_cost_rates) * 10000.0) if trading_costs_enabled else 0.0

    wealth = 1.0
    wealth_net = 1.0
    wealth_dates: List[pd.Timestamp] = []
    wealth_values: List[float] = []
    wealth_net_values: List[float] = []
    step_rows: List[Dict[str, object]] = []
    prev_weights: Optional[np.ndarray] = None
    condition_values: List[float] = []

    for cycle in schedule:
        weights: Optional[np.ndarray] = None
        if strategy == "tsmom_spy":
            train_end_ts = bundle.dataset.timestamps[cycle.train_end]
            weights = _tsmom_spy_weights(
                returns_df,
                train_end_ts,
                momentum_window,
                benchmark_ticker=benchmark_ticker,
            )
        else:
            raise ValueError(f"Unknown benchmark strategy '{strategy}'")

        if weights is None:
            continue

        if not wealth_dates and cycle.eval_indices:
            wealth_dates.append(bundle.dataset.timestamps[cycle.eval_indices[0]])
            wealth_values.append(wealth)
            wealth_net_values.append(wealth_net)

        for eval_idx in cycle.eval_indices:
            returns_vec = bundle.dataset.Y[eval_idx]
            realized = float(np.dot(weights, returns_vec))
            if prev_weights is None or prev_weights.shape != weights.shape:
                turnover = 0.0
                trading_cost = 0.0
            else:
                abs_changes = np.abs(weights - prev_weights)
                turnover = float(0.5 * np.sum(abs_changes))
                trading_cost = (
                    float(0.5 * np.sum(abs_changes * trading_cost_rates))
                    if trading_costs_enabled
                    else 0.0
                )
            prev_weights = weights.copy()
            net_return = realized - trading_cost
            wealth *= (1.0 + realized)
            wealth_net *= (1.0 + net_return)
            ts = bundle.dataset.timestamps[eval_idx]
            wealth_dates.append(ts)
            wealth_values.append(wealth)
            wealth_net_values.append(wealth_net)
            step_rows.append(
                {
                    "date": ts.isoformat(),
                    "portfolio_return": realized,
                    "net_return": net_return,
                    "turnover": turnover,
                    "trading_cost": trading_cost,
                    "weights": json.dumps(weights.tolist()),
                    "weight_min": float(np.min(weights)),
                    "weight_max": float(np.max(weights)),
                    "weight_sum": float(np.sum(weights)),
                }
            )

    if not step_rows:
        return None

    step_df = pd.DataFrame(step_rows)
    returns = step_df["portfolio_return"].to_numpy(dtype=float)
    net_returns = step_df["net_return"].to_numpy(dtype=float)
    avg_turnover = float(step_df["turnover"].mean()) if not step_df.empty else float("nan")
    avg_trading_cost = float(step_df["trading_cost"].mean()) if not step_df.empty else 0.0
    mean_step = float(np.mean(returns))
    std_step = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    mean_step_net = float(np.mean(net_returns))
    std_step_net = float(np.std(net_returns, ddof=1)) if net_returns.size > 1 else 0.0
    if wealth_dates and len(wealth_dates) >= 2:
        horizon_days = (wealth_dates[-1] - wealth_dates[0]).days
        horizon_years = max(horizon_days / 365.25, 1e-9)
        steps_per_year = len(step_rows) / horizon_years
    else:
        steps_per_year = 1.0
    mean_return = mean_step * steps_per_year
    std_return = std_step * math.sqrt(steps_per_year) if std_step > 0.0 else 0.0
    mean_return_net = mean_step_net * steps_per_year
    std_return_net = std_step_net * math.sqrt(steps_per_year) if std_step_net > 0.0 else 0.0
    sharpe = mean_return / std_return if std_return > 1e-12 else np.nan
    sharpe_net = mean_return_net / std_return_net if std_return_net > 1e-12 else np.nan
    sortino = compute_sortino_ratio(returns)
    sortino_net = compute_sortino_ratio(net_returns)
    if np.isfinite(sortino):
        sortino = float(sortino) * math.sqrt(steps_per_year)
    if np.isfinite(sortino_net):
        sortino_net = float(sortino_net) * math.sqrt(steps_per_year)
    cvar_95 = compute_cvar(returns, alpha=0.05)
    terminal_wealth = float(wealth_values[-1]) if wealth_values else 1.0
    terminal_wealth_net = float(wealth_net_values[-1]) if wealth_net_values else 1.0
    total_return = terminal_wealth - 1.0
    total_return_net = terminal_wealth_net - 1.0
    condition_mean = float(np.mean(condition_values)) if condition_values else float("nan")

    stats = {
        "model": STRATEGY_LABELS.get(strategy, strategy),
        "n_retrain": int(len(schedule)),
        "n_invest_steps": int(len(step_rows)),
        "ann_return": mean_return,
        "ann_volatility": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_return_net": mean_return_net,
        "ann_volatility_net": std_return_net,
        "sharpe_net": sharpe_net,
        "sortino_net": sortino_net,
        "cvar_95": cvar_95,
        "r2": float("nan"),
        "max_drawdown": max_drawdown(wealth_values),
        "terminal_wealth": terminal_wealth,
        "terminal_wealth_net": terminal_wealth_net,
        "total_return": total_return,
        "total_return_net": total_return_net,
        "train_window": train_window,
        "rebal_interval": rebal_interval,
        "avg_turnover": avg_turnover,
        "avg_trading_cost": avg_trading_cost,
        "trading_cost_bps": mean_cost_bps,
        "avg_condition_number": condition_mean,
    }

    wealth_df = pd.DataFrame({"date": wealth_dates, "wealth": wealth_values})
    label = f"benchmark_{strategy}"
    return BenchmarkResult(label=label, stats=stats, wealth_df=wealth_df)


def _enrich_basic_benchmark(stats: Dict[str, object], mean_cost_bps: float) -> Dict[str, object]:
    enriched = dict(stats)
    enriched.setdefault("ann_return_net", stats.get("ann_return", float("nan")))
    enriched.setdefault("ann_volatility_net", stats.get("ann_volatility", float("nan")))
    enriched.setdefault("sharpe_net", stats.get("sharpe", float("nan")))
    enriched.setdefault("sortino_net", stats.get("sortino", float("nan")))
    enriched.setdefault("total_return_net", stats.get("total_return", float("nan")))
    enriched.setdefault("terminal_wealth_net", stats.get("terminal_wealth", float("nan")))
    enriched.setdefault("avg_turnover", 0.0)
    enriched.setdefault("avg_trading_cost", 0.0)
    enriched.setdefault("trading_cost_bps", mean_cost_bps)
    enriched.setdefault("avg_condition_number", float("nan"))
    enriched.setdefault("train_window", 0)
    enriched.setdefault("rebal_interval", 0)
    return enriched


def run_benchmark_suite(
    bundle,
    *,
    benchmarks: Sequence[str],
    args,
    trading_costs_enabled: bool,
    asset_cost_overrides: Dict[str, float],
    eval_start: Optional[pd.Timestamp],
) -> Tuple[List[Dict[str, object]], Dict[str, pd.DataFrame]]:
    if not benchmarks:
        return [], {}

    tickers = bundle.dataset.config.tickers
    trading_cost_rates = resolve_trading_cost_rates(
        tickers,
        asset_cost_overrides,
        enable_default_costs=trading_costs_enabled,
        default_bps_if_missing=float(getattr(args, "trading_cost_bps", 0.0) or 0.0),
    )
    mean_cost_bps = float(np.mean(trading_cost_rates) * 10000.0) if trading_costs_enabled else 0.0

    schedule = build_rebalance_schedule(
        bundle,
        args.train_window,
        args.rebal_interval,
        eval_start=eval_start,
    )

    stats_results: List[Dict[str, object]] = []
    wealth_entries: Dict[str, pd.DataFrame] = {}
    handled = set()
    start_ts = eval_start

    for bench in benchmarks:
        name = bench.strip().lower()
        if not name or name in handled:
            continue
        handled.add(name)
        if name == "spy":
            info = compute_benchmark_series(bundle, args.benchmark_ticker or "SPY", start_date=start_ts)
            if not info:
                continue
            stats = _enrich_basic_benchmark(info["stats"], mean_cost_bps)
            stats["model"] = f"Buy&Hold({args.benchmark_ticker or 'SPY'})"
            stats_results.append(stats)
            wealth_entries[info["label"]] = info["wealth_df"]
        elif name in {"equal_weight", "1/n", "bh_equal"}:
            info = compute_equal_weight_benchmark(bundle, start_date=start_ts)
            if not info:
                continue
            stats = _enrich_basic_benchmark(info["stats"], mean_cost_bps)
            stats_results.append(stats)
            wealth_entries[info["label"]] = info["wealth_df"]
        elif name in {"tsmom_spy", "momentum_tsmom_spy"}:
            result = _simulate_strategy(
                bundle,
                schedule,
                trading_cost_rates,
                strategy="tsmom_spy",
                benchmark_ticker=args.benchmark_ticker or "SPY",
                momentum_window=args.momentum_window,
                trading_costs_enabled=trading_costs_enabled,
                train_window=args.train_window,
                rebal_interval=args.rebal_interval,
            )
            if result:
                stats_results.append(result.stats)
                wealth_entries[result.label] = result.wealth_df
        else:
            print(f"[benchmark] Unknown benchmark '{name}', skipping.")

    return stats_results, wealth_entries
