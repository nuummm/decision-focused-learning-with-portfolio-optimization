from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.real_data.loader import MarketLoaderConfig
from experiments.real_data.data_pipeline import (
    PipelineConfig,
    build_data_bundle,
)
from experiments.registry import (
    SolverSpec,
    get_trainer,
    KNITRO_DEFAULTS,
    GUROBI_DEFAULTS,
    IPOPT_DEFAULTS,
)
from models.ols import predict_yhat, train_ols
from models.ipo_closed_form import fit_ipo_closed_form
from models.ols_gurobi import solve_mvo_gurobi, solve_series_mvo_gurobi

try:  # Optional plotting for debug artifacts
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

HERE = Path(__file__).resolve()
RESULTS_BASE = HERE.parents[3] / "results"
RESULTS_ROOT = RESULTS_BASE / "exp_real_data"
DEBUG_ROOT = RESULTS_BASE / "debug_outputs"

PERIOD_WINDOWS = [
    ("gfc_2008", "2007-01-01", "2009-12-31"),
    ("covid_2020", "2020-02-01", "2020-12-31"),
    ("inflation_2022", "2022-01-01", "2023-12-31"),
]

WEIGHT_THRESHOLD = 0.95


def mvo_cost(z: np.ndarray, y: np.ndarray, V: np.ndarray, delta: float = 1.0) -> float:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    V = np.asarray(V, dtype=float)
    return float(-z @ y + 0.5 * delta * (z @ V @ z))


def parse_tickers(value: str) -> List[str]:
    return [t.strip().upper() for t in value.split(",") if t.strip()]


def parse_commalist(value: str) -> List[str]:
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def parse_model_train_window_spec(value: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    text = (value or "").strip()
    if not text:
        return mapping
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid model-train-window spec '{token}'. Use model:window format.")
        name, window_str = token.split(":", 1)
        model_name = name.strip().lower()
        if not model_name:
            raise ValueError(f"Missing model name in spec '{token}'.")
        try:
            window_val = int(window_str.strip())
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid train window '{window_str}' for model '{model_name}'.") from exc
        if window_val <= 0:
            raise ValueError(f"Train window for model '{model_name}' must be positive.")
        mapping[model_name] = window_val
    return mapping


def make_output_dir(base: Path | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = base or (RESULTS_ROOT / timestamp)
    root.mkdir(parents=True, exist_ok=True)
    return root


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
) -> List[ScheduleItem]:
    cov_indices = bundle.cov_indices.tolist()
    cov_set = set(cov_indices)
    test_cov = cov_indices[:]

    schedule: List[ScheduleItem] = []
    pos = 0
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


def plot_wealth_curve(dates: Sequence[pd.Timestamp], wealth: Sequence[float], path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(dates, wealth, label="wealth")
    plt.xlabel("date")
    plt.ylabel("wealth")
    plt.title("Portfolio wealth trajectory")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_weight_paths(weights_df: pd.DataFrame, model: str, path: Path) -> None:
    if plt is None or weights_df.empty:
        return
    dates = pd.to_datetime(weights_df["date"])
    value_cols = [c for c in weights_df.columns if c not in {"date", "portfolio_return_sq"}]
    values = weights_df[value_cols].astype(float)
    value_cols_sorted = sorted(value_cols)
    values = values[value_cols_sorted]
    plt.figure(figsize=(10, 4))
    bottom = np.zeros(len(values))
    if len(dates) > 1:
        date_series = pd.Series(dates)
        diffs = date_series.diff().dt.days.dropna()
        median_diff = diffs.median() if not diffs.empty else 1
        interval_days = max(int(median_diff), 1)
    else:
        interval_days = 7
    width = interval_days
    for col in value_cols_sorted:
        plt.bar(dates, values[col], bottom=bottom, label=col, width=width, align="center")
        bottom += values[col].to_numpy()
    plt.ylim(0, 1)
    plt.title(f"Weight allocation (stacked) - {model}")
    plt.xlabel("date")
    plt.ylabel("weight share")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    # also plot individual lines for reference
    plt.figure(figsize=(10, 4))
    for col in value_cols_sorted:
        plt.plot(dates, values[col], label=col)
    plt.ylim(0, 1)
    plt.title(f"Weight trajectories ({model})")
    plt.xlabel("date")
    plt.ylabel("weight")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path.with_name(path.stem + "_lines" + path.suffix))
    plt.close()


def plot_weight_comparison(weight_dict: Dict[str, pd.DataFrame], path: Path) -> None:
    if plt is None or not weight_dict:
        return
    models = list(weight_dict.keys())
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        df = weight_dict[model]
        dates = pd.to_datetime(df["date"])
        value_cols = [c for c in df.columns if c not in {"date", "portfolio_return_sq"}]
        values = df[value_cols].astype(float)
        bottom = np.zeros(len(values))
        for col in value_cols:
            ax.bar(dates, values[col], bottom=bottom, label=col, width=5)
            bottom += values[col].to_numpy()
        ax.set_ylim(0, 1)
        ax.set_ylabel(model)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("date")
    fig.suptitle("Weight allocation (stacked) per model")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_weight_variance_correlation(
    weight_dict: Dict[str, pd.DataFrame],
    csv_path: Path,
    fig_path: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for model, df in weight_dict.items():
        if "portfolio_return_sq" not in df.columns:
            continue
        ticker_cols = [c for c in df.columns if c not in {"date", "portfolio_return_sq"}]
        if not ticker_cols:
            continue
        returns_sq = df["portfolio_return_sq"].astype(float)
        if returns_sq.std(ddof=0) == 0:
            continue
        for ticker in ticker_cols:
            corr = np.corrcoef(df[ticker].astype(float), returns_sq)[0, 1]
            rows.append(
                {
                    "model": model,
                    "ticker": ticker,
                    "corr_weight_vs_return_var": float(corr),
                }
            )
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        if plt is not None:
            pivot = df.pivot(index="model", columns="ticker", values="corr_weight_vs_return_var")
            fig, ax = plt.subplots(figsize=(6, 4))
            data = pivot.to_numpy()
            cax = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title("Weight vs variance correlation")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")
            fig.colorbar(cax)
            fig.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)


def plot_wealth_correlation_heatmap(corr_df: pd.DataFrame, path: Path) -> None:
    if plt is None or corr_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    data = corr_df.to_numpy()
    cax = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title("Wealth return correlation")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(cax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def max_drawdown(wealth_series: Sequence[float]) -> float:
    arr = np.asarray(wealth_series, dtype=float)
    if arr.size == 0:
        return np.nan
    running_max = np.maximum.accumulate(arr)
    dd = 1.0 - arr / running_max
    return float(np.nanmax(dd))


def compute_period_metrics(step_df: pd.DataFrame) -> List[Dict[str, object]]:
    if step_df.empty:
        return []
    df = step_df.copy()
    df["date_ts"] = pd.to_datetime(df["date"])
    results: List[Dict[str, object]] = []
    for name, start, end in PERIOD_WINDOWS:
        mask = (df["date_ts"] >= pd.Timestamp(start)) & (df["date_ts"] <= pd.Timestamp(end))
        subset = df.loc[mask]
        if subset.empty:
            continue
        returns = subset["portfolio_return"].to_numpy()
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
        sharpe = mean_ret / std_ret if std_ret > 1e-12 else np.nan
        wealth = subset["wealth"].to_numpy()
        dd = max_drawdown(wealth)
        results.append(
            {
                "period": name,
                "start": start,
                "end": end,
                "mean_return": mean_ret,
                "std_return": std_ret,
                "sharpe": sharpe,
                "max_drawdown": dd,
                "n_steps": int(len(subset)),
            }
        )
    return results


def plot_multi_wealth(wealth_dict: Dict[str, pd.DataFrame], path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(10, 4))
    for model, df in wealth_dict.items():
        plt.plot(pd.to_datetime(df["date"]), df["wealth"], label=model)
    plt.xlabel("date")
    plt.ylabel("wealth")
    plt.title("Wealth comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_wealth_with_events(wealth_dict: Dict[str, pd.DataFrame], path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(10, 4))
    for model, df in wealth_dict.items():
        plt.plot(pd.to_datetime(df["date"]), df["wealth"], label=model)
    for name, start, end in PERIOD_WINDOWS:
        plt.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15, label=name)
    plt.xlabel("date")
    plt.ylabel("wealth")
    plt.title("Wealth comparison with crisis windows")
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)
    plt.legend(unique_handles, unique_labels, loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_time_series(df: pd.DataFrame, title: str, start_date: pd.Timestamp, path: Path) -> None:
    if plt is None or df.empty:
        return
    plt.figure(figsize=(10, 4))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    if not pd.isna(start_date):
        plt.axvline(start_date, color="red", linestyle="--", label="start_date")
    plt.title(title)
    plt.xlabel("date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def reduce_weight_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    value_cols = [c for c in df.columns if c not in {"date", "portfolio_return_sq"}]
    value_cols_sorted = sorted(value_cols)
    values = df[value_cols_sorted].astype(float)
    return values, value_cols_sorted


def export_average_weights(weight_dict: Dict[str, pd.DataFrame], csv_path: Path, fig_path: Path) -> None:
    rows: List[Dict[str, object]] = []
    for model, df in weight_dict.items():
        values, cols = reduce_weight_columns(df)
        mean_vals = values.mean(skipna=True)
        for ticker in cols:
            rows.append({"model": model, "ticker": ticker, "avg_weight": float(mean_vals[ticker])})
    if not rows:
        return
    avg_df = pd.DataFrame(rows)
    avg_df.to_csv(csv_path, index=False)
    if plt is None:
        return
    pivot = avg_df.pivot(index="model", columns="ticker", values="avg_weight")
    pivot = pivot.fillna(0.0)
    ind = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    plt.figure(figsize=(8, 4))
    for i, ticker in enumerate(pivot.columns):
        plt.bar(ind + i * width, pivot[ticker], width=width, label=ticker)
    plt.xticks(ind + width * (len(pivot.columns) - 1) / 2, pivot.index)
    plt.ylabel("average weight")
    plt.title("Average allocation per model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def plot_weight_histograms(weight_dict: Dict[str, pd.DataFrame], path: Path) -> None:
    if plt is None or not weight_dict:
        return
    models = list(weight_dict.keys())
    tickers = reduce_weight_columns(next(iter(weight_dict.values())))[1]
    n_rows = len(models)
    n_cols = len(tickers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), sharex=False)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    for r, model in enumerate(models):
        values, _ = reduce_weight_columns(weight_dict[model])
        for c, ticker in enumerate(tickers):
            ax = axes[r, c]
            ax.hist(values[ticker].dropna(), bins=20, range=(0, 1), color="tab:blue", alpha=0.7)
            if r == n_rows - 1:
                ax.set_xlabel(ticker)
            if c == 0:
                ax.set_ylabel(model)
    fig.suptitle("Weight distributions")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_flex_solver_debug(df: pd.DataFrame, path: Path) -> None:
    if plt is None or df.empty:
        return
    flex_df = df.copy()
    for col in flex_df.columns:
        if col.startswith("solver_status_"):
            flex_df[col] = pd.to_numeric(flex_df[col], errors="coerce")
    models = flex_df["model"].tolist()
    x = np.arange(len(models))
    status_cols = [c for c in flex_df.columns if c.startswith("solver_status_")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if status_cols:
        width = 0.8 / len(status_cols)
        for i, col in enumerate(status_cols):
            axes[0].bar(
                x + i * width,
                flex_df[col].astype(float),
                width=width,
                label=col.replace("solver_status_", ""),
            )
        axes[0].set_xticks(x + width * (len(status_cols) - 1) / 2)
    else:
        axes[0].bar(x, [0] * len(models), width=0.6, label="none")
        axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha="right")
    axes[0].set_ylabel("count")
    axes[0].set_title("Solver status counts")
    axes[0].legend()

    width2 = 0.35
    axes[1].bar(x - width2 / 2, flex_df["elapsed_mean"].astype(float), width=width2, label="elapsed_mean")
    axes[1].bar(x + width2 / 2, flex_df["elapsed_max"].astype(float), width=width2, label="elapsed_max")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha="right")
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Solver elapsed time")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compute_benchmark_series(
    bundle,
    ticker: str,
    start_date: Optional[pd.Timestamp] = None,
) -> Optional[Dict[str, object]]:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None
    returns_df = bundle.dataset.returns
    if ticker not in returns_df.columns:
        print(f"[benchmark] ticker '{ticker}' not found in returns; skipping benchmark.")
        return None
    series = returns_df[ticker].dropna()
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    if start_ts is not None:
        series = series.loc[series.index >= start_ts]
    if series.empty:
        print(f"[benchmark] ticker '{ticker}' has no valid returns; skipping benchmark.")
        return None
    dates = pd.to_datetime(series.index)
    returns = series.to_numpy(dtype=float)
    wealth = np.cumprod(1.0 + returns)
    dates_list = list(dates)
    wealth_list = list(wealth)
    initial_date = start_ts if start_ts is not None else dates_list[0]
    dates_list.insert(0, initial_date)
    wealth_list.insert(0, 1.0)
    wealth_df = pd.DataFrame({"date": dates_list, "wealth": wealth_list})
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = mean_return / std_return if std_return > 1e-12 else np.nan
    label = f"benchmark_{ticker}"
    stats = {
        "model": label,
        "n_cycles": int(len(returns)),
        "n_steps": int(len(returns)),
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(wealth_list),
        "final_wealth": float(wealth_list[-1]),
        "train_window": 0,
        "rebal_interval": 0,
    }
    return {"label": label, "wealth_df": wealth_df, "stats": stats}
def compute_correlation_stats(corr_df: pd.DataFrame) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if corr_df.empty:
        return stats
    n = corr_df.shape[0]
    corr_values = corr_df.to_numpy(dtype=float)
    if n > 1:
        upper = corr_values[np.triu_indices(n, 1)]
        if upper.size > 0:
            stats["mean_corr"] = float(np.mean(upper))
            stats["mean_abs_corr"] = float(np.mean(np.abs(upper)))
    try:
        det = float(np.linalg.det(corr_values))
        stats["determinant"] = det
    except np.linalg.LinAlgError:
        stats["determinant"] = float("nan")
    try:
        eigvals = np.linalg.eigvalsh(corr_values)
        eigvals = eigvals[eigvals > 0]
        if eigvals.size > 0:
            entropy = -np.sum((eigvals / eigvals.sum()) * np.log(eigvals / eigvals.sum()))
            stats["entropy"] = float(entropy)
    except np.linalg.LinAlgError:
        stats["entropy"] = float("nan")
    return stats


def plot_asset_correlation(corr_df: pd.DataFrame, path: Path, stats: Optional[Dict[str, float]] = None) -> None:
    if plt is None or corr_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    data = corr_df.to_numpy()
    cax = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    title = "Asset return correlation"
    if stats:
        mean_abs = stats.get("mean_abs_corr")
        if mean_abs is not None and not np.isnan(mean_abs):
            title += f" (avg |ρ|={mean_abs:.2f})"
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(cax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def update_experiment_ledger(
    outdir: Path,
    args: argparse.Namespace,
    summary_df: pd.DataFrame,
    analysis_csv_dir: Path,
    bundle_summary: Dict[str, object],
) -> None:
    if summary_df.empty:
        return
    ledger_dir = RESULTS_ROOT / "experiment_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "ledger.csv"
    asset_corr_csv = analysis_csv_dir / "asset_return_correlation.csv"
    asset_corr_summary_csv = analysis_csv_dir / "asset_return_correlation_summary.csv"
    corr_json = ""
    if asset_corr_csv.exists():
        try:
            corr_df = pd.read_csv(asset_corr_csv, index_col=0)
            corr_json = corr_df.to_json()
        except Exception:
            corr_json = ""
    corr_summary_json = ""
    if asset_corr_summary_csv.exists():
        try:
            _corr_summary_df = pd.read_csv(
                asset_corr_summary_csv, index_col=0, header=None
            )
            corr_summary_series = _corr_summary_df.iloc[:, 0]
            corr_summary_json = corr_summary_series.to_json()
        except Exception:
            corr_summary_json = ""
    summary_path = analysis_csv_dir / "summary.csv"
    period_path = analysis_csv_dir / "period_metrics.csv"
    wealth_csv_path = analysis_csv_dir / "wealth_comparison.csv"
    rebalance_csv_path = analysis_csv_dir / "rebalance_summary.csv"
    base_record = {
        "experiment_id": outdir.name,
        "experiment_path": str(outdir),
        "timestamp": datetime.now().isoformat(),
        "tickers": args.tickers,
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "frequency": args.frequency,
        "resample_rule": args.resample_rule,
        "momentum_window": args.momentum_window,
        "return_horizon": args.return_horizon,
        "cov_window": args.cov_window,
        "cov_method": args.cov_method,
        "cov_shrinkage": args.cov_shrinkage,
        "delta": args.delta,
        "train_window_default": args.train_window,
        "rebal_interval": args.rebal_interval,
        "models": args.models,
        "benchmark_ticker": getattr(args, "benchmark_ticker", ""),
        "asset_corr_csv": str(asset_corr_csv) if asset_corr_csv.exists() else "",
        "asset_corr_json": corr_json,
        "asset_corr_summary_csv": (
            str(asset_corr_summary_csv) if asset_corr_summary_csv.exists() else ""
        ),
        "asset_corr_summary_json": corr_summary_json,
        "summary_csv": str(summary_path),
        "period_metrics_csv": str(period_path),
        "wealth_comparison_csv": str(wealth_csv_path),
        "rebalance_summary_csv": str(rebalance_csv_path),
        "covariance_samples": bundle_summary.get("covariance_samples", ""),
        "n_assets": bundle_summary.get("n_assets", ""),
    }
    rows: List[Dict[str, object]] = []
    for _, row in summary_df.iterrows():
        entry = dict(base_record)
        for key, value in row.items():
            entry[key] = value
        rows.append(entry)
    if not rows:
        return
    ledger_df = pd.DataFrame(rows)
    ledger_df["experiment_id"] = ledger_df["experiment_id"].astype(str)
    if ledger_path.exists():
        try:
            existing = pd.read_csv(ledger_path)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, ledger_df], ignore_index=True)
        combined.drop_duplicates(subset=["experiment_id", "model"], keep="last", inplace=True)
        combined.to_csv(ledger_path, index=False)
    else:
        ledger_df.to_csv(ledger_path, index=False)


def export_weight_threshold_frequency(
    weight_dict: Dict[str, pd.DataFrame],
    threshold: float,
    csv_path: Path,
    fig_path: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for model, df in weight_dict.items():
        values, cols = reduce_weight_columns(df)
        counts = (values >= threshold).sum(axis=0)
        total = len(values)
        for ticker in cols:
            freq = float(counts[ticker]) / total if total > 0 else 0.0
            rows.append({"model": model, "ticker": ticker, "freq_ge_thresh": freq})
    if not rows:
        return
    freq_df = pd.DataFrame(rows)
    freq_df.to_csv(csv_path, index=False)
    if plt is None:
        return
    pivot = freq_df.pivot(index="model", columns="ticker", values="freq_ge_thresh").fillna(0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    data = pivot.to_numpy()
    cax = ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Freq(weight >= {threshold:.2f})")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(cax)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def resolved_solver_options(name: str, options: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if options:
        return dict(options)
    lower = (name or "").lower()
    if lower == "knitro":
        return dict(KNITRO_DEFAULTS)
    if lower == "gurobi":
        return dict(GUROBI_DEFAULTS)
    if lower == "ipopt":
        return dict(IPOPT_DEFAULTS)
    return {}


def prepare_flex_training_args(
    bundle,
    train_start: int,
    train_end: int,
    delta: float,
    tee: bool,
    flex_options: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    flex_kwargs: Dict[str, Any] = dict(flex_options or {})
    theta_init_mode = str(flex_kwargs.pop("theta_init_mode", "none") or "none").lower()
    theta_anchor_mode = str(flex_kwargs.pop("theta_anchor_mode", "none") or "none").lower()
    w_anchor_mode = str(flex_kwargs.pop("w_anchor_mode", "none") or "none").lower()
    theta_clamp_tol = float(flex_kwargs.pop("theta_clamp_tol", 0.0) or 0.0)
    theta_clamp_floor = float(flex_kwargs.pop("theta_clamp_floor", 0.0) or 0.0)
    w_clamp_tol = float(flex_kwargs.pop("w_clamp_tol", 0.0) or 0.0)
    w_clamp_floor = float(flex_kwargs.pop("w_clamp_floor", 0.0) or 0.0)
    flex_kwargs.pop("theta_clamp_penalty", None)
    flex_kwargs.pop("w_clamp_penalty", None)

    anchor_clamp_tol = float(flex_kwargs.get("anchor_clamp_tol", 0.0) or 0.0)
    anchor_clamp_floor = float(flex_kwargs.get("anchor_clamp_floor", 0.0) or 0.0)
    anchor_clamp_tol = max(anchor_clamp_tol, theta_clamp_tol, w_clamp_tol)
    anchor_clamp_floor = max(anchor_clamp_floor, theta_clamp_floor, w_clamp_floor)
    flex_kwargs["anchor_clamp_tol"] = anchor_clamp_tol
    flex_kwargs["anchor_clamp_floor"] = anchor_clamp_floor
    flex_kwargs["theta_clamp_enable"] = bool(flex_kwargs.get("theta_clamp_enable", False))
    flex_kwargs["w_clamp_enable"] = bool(flex_kwargs.get("w_clamp_enable", False))
    flex_kwargs["theta_clamp_source"] = str(flex_kwargs.get("theta_clamp_source", "none") or "none").lower()
    flex_kwargs["w_clamp_source"] = str(flex_kwargs.get("w_clamp_source", "none") or "none").lower()
    flex_kwargs["formulation"] = str(flex_kwargs.get("formulation", "dual") or "dual").lower()

    X = np.asarray(bundle.dataset.X, dtype=float)
    Y = np.asarray(bundle.dataset.Y, dtype=float)
    n_samples = X.shape[0]
    start_idx = max(0, int(train_start))
    end_idx = min(n_samples - 1, int(train_end))
    if end_idx < start_idx:
        raise ValueError(f"Invalid train window [{train_start}, {train_end}]")
    train_slice = slice(start_idx, end_idx + 1)
    X_train = X[train_slice]
    Y_train = Y[train_slice]

    idx_list = bundle.cov_indices.tolist()
    cov_pairs = [
        (idx_val, cov)
        for idx_val, cov in zip(idx_list, bundle.covariances)
        if train_start <= idx_val <= train_end
    ]
    used_train_idx = [idx_val for idx_val, _ in cov_pairs]
    cov_train = [cov for _, cov in cov_pairs]

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

    lam_w_anchor = float(flex_kwargs.get("lambda_w_anchor", 0.0) or 0.0)
    lam_w_anchor_l1 = float(flex_kwargs.get("lambda_w_anchor_l1", 0.0) or 0.0)
    need_w_anchor = (
        "w_anchor" not in flex_kwargs
        and (lam_w_anchor > 0.0 or lam_w_anchor_l1 > 0.0 or w_anchor_mode not in {"", "none"})
    )
    if need_w_anchor:
        theta_for_w_anchor = ensure_theta_source(w_anchor_mode, fallback_to_ols=False, context="w_anchor")
        if theta_for_w_anchor is None:
            raise ValueError("w_anchor_mode requires a reference theta but none was produced.")
        if not used_train_idx or not cov_train:
            raise ValueError("Cannot construct w_anchor: no covariance pairs in the training window.")
        Yhat_anchor_all = predict_yhat(X, theta_for_w_anchor)
        w_anchor_mat = solve_series_mvo_gurobi(
            Yhat_all=Yhat_anchor_all,
            Vhats=cov_train,
            idx=used_train_idx,
            delta=delta,
            psd_eps=1e-12,
            output=False,
            start_index=None,
        )
        if w_anchor_mat.size == 0:
            raise ValueError("solve_series_mvo_gurobi returned empty matrix for w_anchor.")
        flex_kwargs["w_anchor"] = w_anchor_mat

    flex_kwargs["lambda_theta_anchor"] = lam_theta_anchor
    flex_kwargs["lambda_theta_anchor_l1"] = float(flex_kwargs.get("lambda_theta_anchor_l1", 0.0) or 0.0)
    flex_kwargs["lambda_theta_iso"] = float(flex_kwargs.get("lambda_theta_iso", 0.0) or 0.0)
    flex_kwargs["lambda_w_anchor"] = lam_w_anchor
    flex_kwargs["lambda_w_anchor_l1"] = float(flex_kwargs.get("lambda_w_anchor_l1", 0.0) or 0.0)
    flex_kwargs["lambda_w_iso"] = float(flex_kwargs.get("lambda_w_iso", 0.0) or 0.0)

    return theta_init, flex_kwargs


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
):
    trainer_kwargs: Dict[str, object] = dict(
        X=bundle.dataset.X,
        Y=bundle.dataset.Y,
        Vhats=bundle.covariances,
        idx=bundle.cov_indices.tolist(),
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
        trainer_kwargs.update(resolved_flex)
    if theta_init_override is not None:
        trainer_kwargs["theta_init"] = theta_init_override

    start_time = time.perf_counter()
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
    tee: bool,
) -> Dict[str, object]:
    trainer = get_trainer(model_key, solver_spec)
    schedule = build_rebalance_schedule(bundle, train_window, rebal_interval)
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
        )
        Yhat_all = predict_yhat(bundle.dataset.X, theta_hat)

        if debug_roll:
            progress = (cycle_id + 1) / max(total_cycles, 1)
            bar = "#" * int(progress * 20)
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

    if not step_rows:
        raise RuntimeError("No evaluation steps were executed. Check train_window/rebal_interval settings.")

    step_df = pd.DataFrame(step_rows)
    returns = step_df["portfolio_return"].to_numpy()
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = mean_return / std_return if std_return > 1e-12 else np.nan

    model_debug_dir = debug_dir / f"model_{model_label}"
    model_debug_dir.mkdir(parents=True, exist_ok=True)

    step_path = model_debug_dir / "step_log.csv"
    step_df.to_csv(step_path, index=False)

    rebalance_df = pd.DataFrame(rebalance_rows)
    rebalance_df.to_csv(model_debug_dir / "rebalance_log.csv", index=False)

    wealth_df = pd.DataFrame({"date": wealth_dates, "wealth": wealth_values, "label": wealth_labels})
    wealth_df.to_csv(model_debug_dir / "wealth.csv", index=False)
    plot_wealth_curve(wealth_dates, wealth_values, model_debug_dir / "wealth.png")

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

    stats_report = {
        "model": model_label,
        "n_cycles": len(schedule),
        "n_steps": len(step_rows),
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(wealth_values),
        "final_wealth": float(wealth_values[-1]) if wealth_values else 1.0,
        "train_window": train_window,
        "rebal_interval": rebal_interval,
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

    return {
        "stats": stats_report,
        "period_metrics": period_metrics,
        "wealth_df": wealth_df,
        "weights_df": weights_df,
        "rebalance_summary": reb_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-data rolling experiment runner")
    parser.add_argument("--tickers", type=str, default="SPY,TLT,DBC,BIL")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--price-field", type=str, default="Close")
    parser.add_argument("--return-kind", type=str, default="log", choices=["simple", "log"])
    parser.add_argument("--frequency", type=str, default="weekly", choices=["daily", "weekly"])
    parser.add_argument("--resample-rule", type=str, default="W-FRI")
    parser.add_argument("--momentum-window", type=int, default=52)
    parser.add_argument("--return-horizon", type=int, default=1)
    parser.add_argument("--cov-window", type=int, default=60)
    parser.add_argument("--cov-method", type=str, default="diag", choices=["diag", "ledoit_wolf"])
    parser.add_argument("--cov-shrinkage", type=float, default=0.94)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    parser.add_argument("--no-auto-adjust", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")

    parser.add_argument("--train-window", type=int, default=25)
    parser.add_argument("--rebal-interval", type=int, default=1)
    parser.add_argument(
        "--model-train-window",
        type=str,
        default="",
        help="Optional overrides e.g. 'ols:60,flex:25' to use per-model train windows.",
    )

    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--models", type=str, default="ols,ipo,dual,flex")
    parser.add_argument("--dual-solver", type=str, default="gurobi")
    parser.add_argument("--flex-solver", type=str, default="gurobi")
    parser.add_argument(
        "--flex-formulation",
        type=str,
        default="dual",
        help="Comma-separated flex base models to run (e.g., 'dual' or 'dual,kkt').",
    )
    parser.add_argument("--flex-lambda-theta-anchor", type=float, default=0.0)
    parser.add_argument("--flex-lambda-w-anchor", type=float, default=0.0)
    parser.add_argument("--flex-lambda-theta-iso", type=float, default=0.0)
    parser.add_argument("--flex-lambda-w-iso", type=float, default=0.0)
    parser.add_argument("--flex-theta-anchor-mode", type=str, default="ols")
    parser.add_argument("--flex-w-anchor-mode", type=str, default="ols")
    parser.add_argument("--flex-theta-init-mode", type=str, default="ols")
    parser.add_argument("--flex-theta-clamp-enable", action="store_true")
    parser.add_argument("--flex-theta-clamp-source", type=str, default="ols")
    parser.add_argument("--flex-theta-clamp-tol", type=float, default=0.0)
    parser.add_argument("--flex-theta-clamp-floor", type=float, default=0.0)
    parser.add_argument("--flex-theta-clamp-penalty", type=float, default=0.0)
    parser.add_argument("--flex-w-clamp-enable", action="store_true")
    parser.add_argument("--flex-w-clamp-source", type=str, default="ols")
    parser.add_argument("--flex-w-clamp-tol", type=float, default=0.0)
    parser.add_argument("--flex-w-clamp-floor", type=float, default=0.0)
    parser.add_argument("--flex-w-clamp-penalty", type=float, default=0.0)
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--debug-roll", action="store_true")
    parser.add_argument(
        "--benchmark-ticker",
        type=str,
        default="SPY",
        help="Ticker used for buy-and-hold benchmark (leave empty to disable).",
    )

    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--no-debug", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model_train_windows = parse_model_train_window_spec(getattr(args, "model_train_window", ""))

    tickers = parse_tickers(args.tickers)
    outdir = make_output_dir(args.outdir)
    analysis_dir = outdir / "analysis"
    analysis_csv_dir = analysis_dir / "csv"
    analysis_fig_dir = analysis_dir / "figures"
    analysis_csv_dir.mkdir(parents=True, exist_ok=True)
    analysis_fig_dir.mkdir(parents=True, exist_ok=True)
    model_outputs_dir = outdir / "model_outputs"
    model_outputs_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = DEBUG_ROOT / f"{outdir.name}_rolling"
    debug_dir.mkdir(parents=True, exist_ok=True)

    flex_formulations = parse_commalist(args.flex_formulation)
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
        auto_adjust=not args.no_auto_adjust,
        cache_dir=None,
        force_refresh=args.force_refresh,
        debug=not args.no_debug,
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
        }
    )

    summary_path = outdir / "experiment_summary.json"
    summary_path.write_text(json.dumps(bundle_summary, ensure_ascii=False, indent=2))

    config_records = []
    for key, value in sorted(vars(args).items()):
        config_records.append({"parameter": key, "value": value})
    pd.DataFrame(config_records).to_csv(analysis_csv_dir / "experiment_config.csv", index=False)

    start_ts = pd.Timestamp(loader_cfg.start)
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
    wealth_dict: Dict[str, pd.DataFrame] = {}
    weight_dict: Dict[str, pd.DataFrame] = {}
    weight_dict: Dict[str, pd.DataFrame] = {}
    train_window_records: List[Dict[str, object]] = []
    rebalance_records: List[Dict[str, object]] = []

    flex_base_options = dict(
        lambda_theta_anchor=args.flex_lambda_theta_anchor,
        lambda_w_anchor=args.flex_lambda_w_anchor,
        lambda_theta_iso=args.flex_lambda_theta_iso,
        lambda_w_iso=args.flex_lambda_w_iso,
        theta_anchor_mode=args.flex_theta_anchor_mode,
        w_anchor_mode=args.flex_w_anchor_mode,
        theta_init_mode=args.flex_theta_init_mode,
        theta_clamp_enable=args.flex_theta_clamp_enable,
        theta_clamp_source=args.flex_theta_clamp_source,
        theta_clamp_tol=args.flex_theta_clamp_tol,
        theta_clamp_floor=args.flex_theta_clamp_floor,
        theta_clamp_penalty=args.flex_theta_clamp_penalty,
        w_clamp_enable=args.flex_w_clamp_enable,
        w_clamp_source=args.flex_w_clamp_source,
        w_clamp_tol=args.flex_w_clamp_tol,
        w_clamp_floor=args.flex_w_clamp_floor,
        w_clamp_penalty=args.flex_w_clamp_penalty,
    )

    model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    for model_key in model_list:
        if model_key not in {"ols", "ipo", "dual", "flex"}:
            print(f"[real-data] skipping unsupported model '{model_key}'")
            continue
        formulations = flex_formulations if model_key == "flex" else [None]
        for form in formulations:
            label = model_key
            if model_key == "flex" and (len(formulations) > 1 or (form and form != "dual")):
                label = f"{model_key}_{form}"
            print(f"[real-data] rolling model={label}")
            solver_name = args.dual_solver if model_key == "dual" else args.flex_solver if model_key == "flex" else "analytic"
            if model_key in {"ols", "ipo"}:
                solver_spec = SolverSpec(name="analytic", tee=args.tee)
            else:
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
                delta=args.delta,
                solver_spec=solver_spec,
                flex_options=flex_options,
                train_window=effective_train_window,
                rebal_interval=args.rebal_interval,
                debug_roll=args.debug_roll,
                debug_dir=debug_dir,
                results_model_dir=results_dir,
                tee=args.tee,
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
                    period_entry["model"] = label
                    period_entry["train_window"] = run_result["stats"].get("train_window", effective_train_window)
                    period_rows.append(period_entry)
                wealth_dict[label] = run_result["wealth_df"][["date", "wealth"]]
                if not run_result["weights_df"].empty:
                    weight_dict[label] = run_result["weights_df"]
                if not run_result["weights_df"].empty:
                    weight_dict[label] = run_result["weights_df"]

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

    summary_df = pd.DataFrame(stats_results)
    if not summary_df.empty:
        summary_df["max_drawdown"] = summary_df["max_drawdown"].astype(float)
        summary_df.to_csv(analysis_csv_dir / "summary.csv", index=False)
    else:
        (analysis_csv_dir / "summary.csv").write_text("")

    if period_rows:
        period_df = pd.DataFrame(period_rows)
        period_df.to_csv(analysis_csv_dir / "period_metrics.csv", index=False)

    if wealth_dict:
        wealth_merge = None
        for model, wdf in wealth_dict.items():
            df_model = wdf.rename(columns={"wealth": model})
            if wealth_merge is None:
                wealth_merge = df_model
            else:
                wealth_merge = wealth_merge.merge(df_model, on="date", how="outer")
        if wealth_merge is not None:
            wealth_merge = wealth_merge.sort_values("date")
            wealth_merge.to_csv(analysis_csv_dir / "wealth_comparison.csv", index=False)
            plot_multi_wealth({m: df for m, df in wealth_dict.items()}, analysis_fig_dir / "wealth_comparison.png")
            plot_wealth_with_events({m: df for m, df in wealth_dict.items()}, analysis_fig_dir / "wealth_events.png")
            wealth_returns = wealth_merge.copy()
            wealth_returns["date"] = pd.to_datetime(wealth_returns["date"])
            wealth_returns = wealth_returns.set_index("date").pct_change().dropna(how="all")
            if not wealth_returns.empty:
                corr = wealth_returns.corr()
                corr.to_csv(analysis_csv_dir / "wealth_correlation.csv")
                plot_wealth_correlation_heatmap(corr, analysis_fig_dir / "wealth_correlation.png")

    if weight_dict:
        plot_weight_comparison(weight_dict, analysis_fig_dir / "weights_comparison.png")
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
        flex_debug_df = rebalance_df[rebalance_df["base_model"] == "flex"]
        if not flex_debug_df.empty:
            plot_flex_solver_debug(flex_debug_df, analysis_fig_dir / "flex_solver_debug.png")
    if not summary_df.empty:
        update_experiment_ledger(outdir, args, summary_df, analysis_csv_dir, bundle_summary)

    print(f"[real-data] finished. outputs -> {outdir}")
    print(f"[real-data] debug artifacts -> {debug_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""
cd /Users/kensei/VScode/GraduationResearch/DFL_Portfolio_Optimization2

python -m experiments.real_data.run_real_eval \
  --tickers "SPY,GLD,EEM,TLT" \
  --start 2006-10-31 --end 2025-10-31 \
  --interval 1d \
  --price-field Close \
  --return-kind log \
  --frequency weekly \
  --resample-rule W-FRI \
  --momentum-window 30 \
  --return-horizon 1 \
  --cov-window 10 \
  --cov-method diag \
  --cov-shrinkage 0.94 \
  --cov-eps 1e-6 \
  --train-window 25 \
  --rebal-interval 4 \
  --delta 1.0 \
  --models ols,ipo,flex \
  --flex-solver knitro \
  --flex-formulation dual,kkt \
  --flex-lambda-theta-anchor 0 \
  --flex-lambda-w-anchor 0.0 \
  --flex-lambda-theta-iso 0.0 \
  --flex-lambda-w-iso 0.0 \
  --flex-theta-anchor-mode ols \
  --flex-w-anchor-mode ols \
  --flex-theta-init-mode none \
  --benchmark-ticker SPY \
  --debug-roll \
  
#   --model-train-window ols:100
  
#   --flex-w-clamp-enable --flex-w-clamp-source ols --flex-anchor-clamp-tol 0 --flex-anchor-clamp-floor 0.0 \
#   --flex-theta-clamp-enable --flex-theta-clamp-source ols --flex-anchor-clamp-tol 0 --flex-anchor-clamp-floor 0.0 \



--tickers : 例 "SPY,GLD,EEM,TLT"
--start, --end : 日付文字列 (YYYY-MM-DD)
--interval : Yahoo Finance から取得する間隔 (1d, 1wk, 1mo など)
--price-field : Close, Adj Close など
--return-kind : log / simple
--frequency : daily / weekly
--resample-rule : 週次時のリサンプル規則 (W-FRI など)
--momentum-window : モメンタム期間
--return-horizon : 予測先 (例: 1 = 1 期間先)
--cov-window : 共分散のローリング窓
--cov-method : diag / ledoit_wolf
--cov-shrinkage : diag shrinkage の λ
--cov-eps : 数値安定化用 epsilon
--no-auto-adjust : Yahoo の調整終値を使わない
--force-refresh : キャッシュ無視で再取得
ローリング設定:

--train-window : 各サイクルの学習窓サイズ
--rebal-interval : リバランス間隔 H
モデル設定:

--delta : リスク回避係数
--models : ols,ipo,dual,flex などカンマ区切り
--flex-formulation : dual,kkt のように指定可
--flex-lambda-theta-anchor, --flex-lambda-w-anchor, --flex-lambda-theta-iso, --flex-lambda-w-iso : Flex の正則化パラメータ
実行制御:

--tee : ソルバーのログ表示
--debug-roll : ローリング進捗ログを表示
--outdir : 出力先ディレクトリ（省略時は results/exp_real_data/<timestamp>）
--no-debug : ローデータ読み込み時のデバッグ出力を抑制

"""
