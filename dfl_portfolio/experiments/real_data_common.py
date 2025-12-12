from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dfl_portfolio.real_data.reporting import (
    compute_correlation_stats,
    plot_asset_correlation,
    plot_time_series,
    compute_sortino_ratio,
    max_drawdown,
    compute_period_metrics,
    display_model_name,
)
from dfl_portfolio.models.ols import predict_yhat, train_ols


def mvo_cost(z: np.ndarray, y: np.ndarray, V: np.ndarray, delta: float = 1.0) -> float:
    """共通の MVO 目的関数 (1-δ)リターン + δリスク。"""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    V = np.asarray(V, dtype=float)
    return float(-(1.0 - delta) * (z @ y) + 0.5 * delta * (z @ V @ z))


DEFAULT_TRADING_COST_BPS: Dict[str, float] = {
    "SPY": 5.0,
    "TLT": 5.0,
    "GLD": 10.0,
    "EEM": 10.0,
    "EURUSD=X": 2.0,
}
DEFAULT_TRADING_COST_RATES: Dict[str, float] = {
    ticker: bps / 10000.0 for ticker, bps in DEFAULT_TRADING_COST_BPS.items()
}


def resolve_trading_cost_rates(
    tickers: Sequence[str],
    overrides: Dict[str, float] | None,
    *,
    enable_default_costs: bool,
) -> np.ndarray:
    """Return per-asset trading cost rates (decimal) with safety checks."""
    overrides_upper = {str(k).upper(): float(v) for k, v in (overrides or {}).items()}
    rates: List[float] = []
    for ticker in tickers:
        key = ticker.upper()
        if key in overrides_upper:
            rates.append(max(overrides_upper[key], 0.0))
            continue
        if enable_default_costs:
            if key not in DEFAULT_TRADING_COST_RATES:
                raise ValueError(
                    f"Trading cost is enabled but ticker '{ticker}' has no default cost entry. "
                    "Specify the cost via --trading-cost-per-asset."
                )
            rates.append(DEFAULT_TRADING_COST_RATES[key])
        else:
            rates.append(0.0)
    return np.asarray(rates, dtype=float)


@dataclass
class ScheduleItem:
    rebalance_idx: int
    train_start: int
    train_end: int
    eval_indices: List[int]


def build_rebalance_schedule(
    bundle,
    train_window: int,
    rebal_interval: int,
    eval_start: Optional[pd.Timestamp] = None,
) -> List[ScheduleItem]:
    """共通のローリング・リバランススケジュール構築ロジック。

    eval_start を指定した場合、その日付以上のタイムスタンプを持つ
    最初のサンプルを評価開始日とし、それ以前はスキップする。
    """
    cov_indices = bundle.cov_indices.tolist()
    cov_set = set(cov_indices)
    test_cov = cov_indices[:]

    # 評価開始位置（cov_indices 上のインデックス）を決める
    start_pos = 0
    if eval_start is not None:
        ts_list = bundle.dataset.timestamps
        for p, idx in enumerate(test_cov):
            if ts_list[idx] >= eval_start:
                start_pos = p
                break
        else:
            # eval_start 以降に有効な共分散が無ければスケジュールは空
            return []

    schedule: List[ScheduleItem] = []
    pos = start_pos
    while pos < len(test_cov):
        rebalance_idx = test_cov[pos]
        train_end = rebalance_idx - 1
        train_start = train_end - train_window + 1
        if train_start < 0:
            pos += 1
            continue
        if train_start not in cov_set or train_end not in cov_set:
            pos += 1
            continue
        eval_indices = test_cov[pos : pos + rebal_interval]
        if not eval_indices:
            break
        schedule.append(
            ScheduleItem(
                rebalance_idx=rebalance_idx,
                train_start=train_start,
                train_end=train_end,
                eval_indices=eval_indices,
            )
        )
        pos += rebal_interval
    return schedule


def prepare_flex_training_args(
    bundle,
    train_start: int,
    train_end: int,
    delta: float,
    tee: bool,
    flex_options: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """flex モデル共通の θ 初期値・アンカー準備ロジック。"""
    from dfl_portfolio.models.ipo_closed_form import fit_ipo_closed_form  # 局所 import

    flex_kwargs: Dict[str, Any] = dict(flex_options or {})
    theta_init_mode = str(flex_kwargs.pop("theta_init_mode", "none") or "none").lower()
    theta_anchor_mode = str(flex_kwargs.pop("theta_anchor_mode", "none") or "none").lower()
    flex_kwargs["formulation"] = str(flex_kwargs.get("formulation", "dual") or "dual").lower()

    X = np.asarray(bundle.dataset.X, float)
    Y = np.asarray(bundle.dataset.Y, float)
    idx_list = bundle.cov_indices.tolist()
    n_samples = X.shape[0]
    start_idx = max(0, int(train_start))
    end_idx = min(n_samples - 1, int(train_end))
    if end_idx < start_idx:
        raise ValueError(f"Invalid train window [{train_start}, {train_end}]")
    train_slice = slice(start_idx, end_idx + 1)
    X_train = X[train_slice]
    Y_train = Y[train_slice]

    theta_sources: Dict[str, np.ndarray] = {}

    def ensure_theta_source(mode: str, *, fallback_to_ols: bool, context: str) -> Optional[np.ndarray]:
        m = (mode or "none").lower()
        if m in {"", "none"}:
            return None
        if m == "ols":
            if "ols" not in theta_sources:
                theta_sources["ols"] = np.asarray(train_ols(X_train, Y_train), dtype=float)
            return theta_sources["ols"].copy()
        if m == "ipo":
            # delta=0 のときは IPO 解析解が定義されないので、
            # 「リターン項だけの解析解」として OLS 回帰係数を用いる。
            if delta == 0.0:
                if "ipo" not in theta_sources:
                    theta_sources["ipo"] = np.asarray(train_ols(X_train, Y_train), dtype=float)
                return theta_sources["ipo"].copy()
            if "ipo" in theta_sources:
                return theta_sources["ipo"].copy()
            try:
                theta_ipo, *_ = fit_ipo_closed_form(
                    X,
                    Y,
                    bundle.covariances,
                    idx_list,
                    start_index=train_start,
                    end_index=train_end,
                    delta=delta,
                    mode="budget",
                    tee=tee,
                )
                theta_sources["ipo"] = np.asarray(theta_ipo, dtype=float)
                return theta_sources["ipo"].copy()
            except Exception as exc:
                if fallback_to_ols:
                    print(f"[WARN] IPO anchor construction failed ({exc}); using OLS for {context}")
                    if "ols" not in theta_sources:
                        theta_sources["ols"] = np.asarray(train_ols(X_train, Y_train), dtype=float)
                    return theta_sources["ols"].copy()
                raise
        raise ValueError(f"Unsupported theta source '{mode}' for {context}")

    theta_init = None
    if theta_init_mode not in {"", "none"}:
        theta_init = ensure_theta_source(theta_init_mode, fallback_to_ols=True, context="theta_init")

    lam_theta_anchor = float(flex_kwargs.get("lambda_theta_anchor", 0.0) or 0.0)
    lam_theta_anchor_l1 = float(flex_kwargs.get("lambda_theta_anchor_l1", 0.0) or 0.0)
    if (
        "theta_anchor" not in flex_kwargs
        and (lam_theta_anchor > 0.0 or lam_theta_anchor_l1 > 0.0 or theta_anchor_mode not in {"", "none"})
    ):
        theta_anchor_vec = ensure_theta_source(theta_anchor_mode, fallback_to_ols=False, context="theta_anchor")
        if theta_anchor_vec is None:
            raise ValueError("theta_anchor_mode requires a valid reference but none was available.")
        flex_kwargs["theta_anchor"] = theta_anchor_vec

    flex_kwargs["lambda_theta_anchor"] = lam_theta_anchor
    flex_kwargs["lambda_theta_anchor_l1"] = float(flex_kwargs.get("lambda_theta_anchor_l1", 0.0) or 0.0)
    flex_kwargs["lambda_theta_iso"] = float(flex_kwargs.get("lambda_theta_iso", 0.0) or 0.0)
    return theta_init, flex_kwargs


__all__ = [
    "mvo_cost",
    "ScheduleItem",
    "build_rebalance_schedule",
    "prepare_flex_training_args",
    "build_flex_dual_kkt_ensemble",
    "resolve_trading_cost_rates",
]


def build_flex_dual_kkt_ensemble(
    *,
    bundle,
    debug_dir: Path,
    args: argparse.Namespace,
    wealth_dict: Dict[str, pd.DataFrame],
    weight_dict: Dict[str, pd.DataFrame],
    stats_results: List[Dict[str, object]],
    period_rows: List[Dict[str, object]],
    log_prefix: str = "[ensemble]",
) -> None:
    """flex dual/kkt の step_log からアンサンブルモデルを構成する共通ヘルパー。"""
    dual_label = None
    for cand in ("flex_dual", "flex"):
        if cand in wealth_dict:
            dual_label = cand
            break
    kkt_label = None
    for cand in ("flex_kkt",):
        if cand in wealth_dict:
            kkt_label = cand
            break
    ens_label = "flex_dual_kkt_ens"
    alpha = float(getattr(args, "flex_ensemble_weight_dual", 0.5))
    if dual_label is None or kkt_label is None:
        print(f"{log_prefix} flex dual/kkt outputs not both available; ensemble disabled.")
        return

    dual_step_path = debug_dir / f"model_{dual_label}" / "step_log.csv"
    kkt_step_path = debug_dir / f"model_{kkt_label}" / "step_log.csv"
    if not dual_step_path.exists() or not kkt_step_path.exists():
        print(f"{log_prefix} step_log.csv for flex dual/kkt not found; skipping ensemble.")
        return

    dual_df = pd.read_csv(dual_step_path)
    kkt_df = pd.read_csv(kkt_step_path)
    required_cols = {"date", "portfolio_return"}
    if not required_cols.issubset(dual_df.columns) or not required_cols.issubset(kkt_df.columns):
        print(f"{log_prefix} step_log.csv missing required columns; skipping ensemble.")
        return

    dual_df = dual_df[["date", "portfolio_return", "weights"]]
    kkt_df = kkt_df[["date", "portfolio_return", "weights"]]
    merged = dual_df.rename(
        columns={"portfolio_return": "ret_dual", "weights": "weights_dual"}
    ).merge(
        kkt_df.rename(columns={"portfolio_return": "ret_kkt", "weights": "weights_kkt"}),
        on="date",
        how="inner",
    )
    if merged.empty:
        print(f"{log_prefix} no overlapping dates for flex dual/kkt; skipping ensemble.")
        return

    merged = merged.sort_values("date")
    ens_rows: List[Dict[str, object]] = []
    wealth = 1.0
    wealth_series_dates: List[pd.Timestamp] = [pd.to_datetime(merged["date"].iloc[0])]
    wealth_series_values: List[float] = [1.0]
    skipped_both_nan = 0
    tickers = list(bundle.dataset.config.tickers)

    for _, row in merged.iterrows():
        r_dual = float(row["ret_dual"])
        r_kkt = float(row["ret_kkt"])
        port_ret = alpha * r_dual + (1.0 - alpha) * r_kkt
        wealth *= 1.0 + port_ret
        wealth_series_dates.append(pd.to_datetime(row["date"]))
        wealth_series_values.append(wealth)
        try:
            w_dual = np.asarray(json.loads(row["weights_dual"]), dtype=float)
            w_kkt = np.asarray(json.loads(row["weights_kkt"]), dtype=float)
            dual_valid = np.all(np.isfinite(w_dual))
            kkt_valid = np.all(np.isfinite(w_kkt))
            if not dual_valid and not kkt_valid:
                skipped_both_nan += 1
                continue
            if dual_valid and not kkt_valid:
                w_ens = w_dual
            elif kkt_valid and not dual_valid:
                w_ens = w_kkt
            else:
                w_ens = alpha * w_dual + (1.0 - alpha) * w_kkt
            rec: Dict[str, object] = {"date": row["date"], "portfolio_return": port_ret}
            for idx, t in enumerate(tickers):
                if idx < w_ens.shape[0]:
                    rec[t] = float(w_ens[idx])
            rec["portfolio_return_sq"] = port_ret ** 2
            ens_rows.append(rec)
        except Exception:
            rec = {
                "date": row["date"],
                "portfolio_return": port_ret,
                "portfolio_return_sq": port_ret ** 2,
            }
            ens_rows.append(rec)

    if skipped_both_nan > 0:
        print(
            f"{log_prefix} skipped {skipped_both_nan} steps where both dual and kkt weights were NaN."
        )
    if not ens_rows:
        print(f"{log_prefix} no valid ensemble rows constructed; skipping ensemble.")
        return

    ens_df = pd.DataFrame(ens_rows).sort_values("date")
    if "wealth" not in ens_df.columns:
        ens_df = ens_df.copy()
        ens_df["wealth"] = wealth_series_values[1:]
    wealth_df = pd.DataFrame({"date": wealth_series_dates, "wealth": wealth_series_values})

    weight_cols = ["date"] + tickers
    if set(weight_cols).issubset(ens_df.columns):
        weights_df = ens_df[weight_cols + ["portfolio_return_sq"]].copy()
    else:
        weights_df = pd.DataFrame()

    avg_turnover = float("nan")
    if not weights_df.empty:
        weights_sorted = weights_df.sort_values("date")
        prev_vec: Optional[np.ndarray] = None
        turnover_vals: List[float] = []
        for _, row in weights_sorted.iterrows():
            vec = np.array([row.get(t, np.nan) for t in tickers], dtype=float)
            if not np.all(np.isfinite(vec)):
                prev_vec = None
                continue
            if prev_vec is not None:
                turnover_vals.append(0.5 * float(np.sum(np.abs(vec - prev_vec))))
            prev_vec = vec
        if turnover_vals:
            avg_turnover = float(np.mean(turnover_vals))

    returns = ens_df["portfolio_return"].to_numpy()
    mean_step = float(np.mean(returns)) if returns.size else 0.0
    std_step = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    # 年率換算: dual / kkt と同じく「全期間のステップ数 / 年数」でスケール
    if wealth_series_dates and len(wealth_series_dates) >= 2:
        horizon_days = (wealth_series_dates[-1] - wealth_series_dates[0]).days
        horizon_years = max(horizon_days / 365.25, 1e-9)
        steps_per_year = len(ens_df) / horizon_years
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
    # CVaR (Expected Shortfall) at 95% for ensemble
    from dfl_portfolio.real_data.reporting import compute_cvar

    cvar_95 = compute_cvar(returns, alpha=0.05)
    terminal_wealth = float(wealth_series_values[-1]) if wealth_series_values else 1.0
    total_return = terminal_wealth - 1.0

    stats_report = {
        "model": display_model_name(ens_label),
        "n_retrain": int(len(ens_df)),
        "n_invest_steps": int(len(ens_df)),
        "ann_return": mean_return,
        "ann_volatility": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_return_net": mean_return,
        "ann_volatility_net": std_return,
        "sharpe_net": sharpe,
        "sortino_net": sortino,
        "cvar_95": cvar_95,
        "max_drawdown": max_drawdown(wealth_series_values),
        "terminal_wealth": terminal_wealth,
        "terminal_wealth_net": terminal_wealth,
        "total_return": total_return,
        "total_return_net": total_return,
        "train_window": args.train_window,
        "rebal_interval": args.rebal_interval,
        "avg_turnover": avg_turnover,
        "avg_trading_cost": 0.0,
        "trading_cost_bps": 0.0,
        "avg_condition_number": float("nan"),
    }
    stats_results.append(stats_report)

    period_metrics = compute_period_metrics(
        ens_df[["date", "portfolio_return", "wealth"]].assign(model=display_model_name(ens_label))
    )
    for row in period_metrics:
        entry = dict(row)
        entry["model"] = display_model_name(ens_label)
        entry["train_window"] = args.train_window
        period_rows.append(entry)

    wealth_dict[ens_label] = wealth_df
    if not weights_df.empty:
        weight_dict[ens_label] = weights_df
