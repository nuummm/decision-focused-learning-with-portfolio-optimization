from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import math
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

from dfl_portfolio.registry import KNITRO_DEFAULTS, GUROBI_DEFAULTS

PERIOD_WINDOWS = [
    ("gfc_2008", "2007-01-01", "2009-12-31"),
    ("covid_2020", "2020-02-01", "2020-12-31"),
    ("inflation_2022", "2022-01-01", "2023-12-31"),
]

WEIGHT_THRESHOLD = 0.95
WEIGHT_PLOT_MAX_POINTS = 60


# 表示用のモデル名マッピング
_MODEL_DISPLAY_MAP = {
    "flex_dual": "DFL-QCQP-dual",
    "flex_kkt": "DFL-QCQP-kkt",
    "flex_dual_kkt_ens": "DFL-QCQP-ens",
    "benchmark_equal_weight": "1/N",
}


def display_model_name(model: str) -> str:
    """集計・可視化用にモデル名を整形する."""
    name = str(model)
    # ベンチマークティッカー系: benchmark_SPY → [SPY]
    if name.startswith("benchmark_") and name != "benchmark_equal_weight":
        ticker = name[len("benchmark_") :]
        return f"[{ticker}]"
    return _MODEL_DISPLAY_MAP.get(name, name)


def _compute_steps_per_year(dates: Sequence[pd.Timestamp], n_steps: int) -> float:
    """観測期間から 1 年あたりのステップ数を推定するヘルパー."""
    if not dates or n_steps <= 0:
        return 1.0
    start = pd.to_datetime(dates[0])
    end = pd.to_datetime(dates[-1])
    horizon_days = (end - start).days
    horizon_years = max(horizon_days / 365.25, 1e-9)
    return float(n_steps) / horizon_years


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
        if len(df) > WEIGHT_PLOT_MAX_POINTS:
            df = df.tail(WEIGHT_PLOT_MAX_POINTS)
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


def plot_delta_paths(delta_df: pd.DataFrame, path: Path) -> None:
    """Plot delta trajectories per model over time (approximate calendar on x-axis)."""
    if plt is None or delta_df.empty:
        return
    if "delta_used" not in delta_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    # rebalance_date があればそれを横軸に、なければ従来どおり cycle を使う
    use_date_axis = "rebalance_date" in delta_df.columns
    for model, sub in delta_df.groupby("model"):
        sub_sorted = sub.sort_values("cycle") if "cycle" in sub.columns else sub.copy()
        if use_date_axis:
            x = pd.to_datetime(sub_sorted["rebalance_date"])
        else:
            if "cycle" not in sub_sorted.columns:
                continue
            x = sub_sorted["cycle"]
        ax.plot(x, sub_sorted["delta_used"], label=model)

    # 金融ショック期間を背景に表示（wealth_events と同じ PERIOD_WINDOWS を使用）
    if use_date_axis:
        for name, start, end in PERIOD_WINDOWS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15, label=name)

    ax.set_xlabel("date" if use_date_axis else "cycle")
    ax.set_ylabel("delta")
    ax.set_title("Learned delta trajectories per model")
    ax.set_ylim(0.0, 1.0)
    # ラベル重複を除去
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)
    ax.legend(uniq_h, uniq_l, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_phi_paths(phi_df: pd.DataFrame, path: Path) -> None:
    """Plot shrinkage parameter phi trajectories per model over time."""
    if plt is None or phi_df.empty:
        return
    if "phi_used" not in phi_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    use_date_axis = "rebalance_date" in phi_df.columns
    for model, sub in phi_df.groupby("model"):
        sub_sorted = sub.sort_values("cycle") if "cycle" in sub.columns else sub.copy()
        if use_date_axis:
            x = pd.to_datetime(sub_sorted["rebalance_date"])
        else:
            if "cycle" not in sub_sorted.columns:
                continue
            x = sub_sorted["cycle"]
        ax.plot(x, sub_sorted["phi_used"], label=model)

    if use_date_axis:
        for name, start, end in PERIOD_WINDOWS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15, label=name)

    ax.set_xlabel("date" if use_date_axis else "cycle")
    ax.set_ylabel("phi")
    ax.set_title("Learned covariance shrinkage trajectories per model")
    ax.set_ylim(0.0, 1.0)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)
    ax.legend(uniq_h, uniq_l, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_multi_wealth(wealth_dict: Dict[str, pd.DataFrame], path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(10, 4))
    color_map = {
        "ols": "blue",
        "ipo": "orange",
        "flex_dual": "green",
        "flex_kkt": "red",
        "flex_dual_kkt_ens": "black",
        "benchmark_SPY": "purple",
        "benchmark_equal_weight": "grey",
    }
    for model, df in wealth_dict.items():
        color = color_map.get(model, None)
        display_label = display_model_name(model)
        kwargs = {"label": display_label}
        if color is not None:
            kwargs["color"] = color
        # flex_dual / flex_kkt は半透明で重なりを見やすく
        if model in {"flex_dual", "flex_kkt"}:
            kwargs["linestyle"] = "-"
            kwargs["alpha"] = 0.4
        plt.plot(pd.to_datetime(df["date"]), df["wealth"], **kwargs)
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
    color_map = {
        "ols": "blue",
        "ipo": "orange",
        "flex_dual": "green",
        "flex_kkt": "red",
        "flex_dual_kkt_ens": "black",
        "benchmark_SPY": "purple",
        "benchmark_equal_weight": "grey",
    }
    for model, df in wealth_dict.items():
        color = color_map.get(model, None)
        display_label = display_model_name(model)
        kwargs = {"label": display_label}
        if color is not None:
            kwargs["color"] = color
        if model in {"flex_dual", "flex_kkt"}:
            kwargs["linestyle"] = "-"
            kwargs["alpha"] = 0.4
        plt.plot(pd.to_datetime(df["date"]), df["wealth"], **kwargs)
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
    # 重み列として扱うカラムを抽出する。
    # 日付とポートフォリオリターン関連（portfolio_return, portfolio_return_sq）は除外する。
    value_cols = [
        c
        for c in df.columns
        if c not in {"date", "portfolio_return", "portfolio_return_sq"}
    ]
    value_cols_sorted = sorted(value_cols)
    values = df[value_cols_sorted].astype(float)
    return values, value_cols_sorted


def export_weight_threshold_frequency(
    weight_dict: Dict[str, pd.DataFrame],
    threshold: float,
    csv_path: Path,
    fig_path: Path,
) -> None:
    """Export frequency with which weights exceed a given threshold, plus heatmap."""
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

    # 共通のビン（0〜1 の範囲を等間隔に分割）
    bins = np.linspace(0.0, 1.0, 21)

    for r, model in enumerate(models):
        values, _ = reduce_weight_columns(weight_dict[model])
        for c, ticker in enumerate(tickers):
            ax = axes[r, c]
            data = values[ticker].dropna().to_numpy()
            if data.size == 0:
                ax.set_visible(False)
                continue
            ax.hist(data, bins=bins, range=(0.0, 1.0), color="tab:blue", alpha=0.7)
            ax.set_xlim(0.0, 1.0)
            if r == n_rows - 1:
                ax.set_xlabel(ticker)
            if c == 0:
                ax.set_ylabel(model)
    fig.suptitle("Weight distributions (histograms)")
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

    # 年率換算: 実際の期間から 1 年あたりのステップ数を推定
    mean_step = float(np.mean(returns))
    std_step = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    steps_per_year = _compute_steps_per_year(dates_list, returns.size)
    mean_return = mean_step * steps_per_year
    std_return = std_step * math.sqrt(steps_per_year) if std_step > 0.0 else 0.0
    sharpe = mean_return / std_return if std_return > 1e-12 else np.nan
    sortino_step = compute_sortino_ratio(returns)
    sortino = (
        float(sortino_step) * math.sqrt(steps_per_year)
        if np.isfinite(sortino_step)
        else np.nan
    )
    label = f"benchmark_{ticker}"
    stats = {
        "model": label,
        "n_cycles": int(len(returns)),
        "n_steps": int(len(returns)),
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown(wealth_list),
        "final_wealth": float(wealth_list[-1]),
        "train_window": 0,
        "rebal_interval": 0,
    }
    return {"label": label, "wealth_df": wealth_df, "stats": stats}


def compute_equal_weight_benchmark(
    bundle,
    start_date: Optional[pd.Timestamp] = None,
) -> Optional[Dict[str, object]]:
    """等配分（全ティッカー同ウエイト）のベンチマーク系列を構成する。"""
    returns_df = bundle.dataset.returns
    if returns_df.empty:
        return None
    series = returns_df.mean(axis=1).dropna()
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    if start_ts is not None:
        series = series.loc[series.index >= start_ts]
    if series.empty:
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
    mean_step = float(np.mean(returns))
    std_step = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    steps_per_year = _compute_steps_per_year(dates_list, returns.size)
    mean_return = mean_step * steps_per_year
    std_return = std_step * math.sqrt(steps_per_year) if std_step > 0.0 else 0.0
    sharpe = mean_return / std_return if std_return > 1e-12 else np.nan
    sortino_step = compute_sortino_ratio(returns)
    sortino = (
        float(sortino_step) * math.sqrt(steps_per_year)
        if np.isfinite(sortino_step)
        else np.nan
    )
    label = "benchmark_equal_weight"
    stats = {
        # 表示名としては 1/N を使う
        "model": "1/N",
        "n_cycles": int(len(returns)),
        "n_steps": int(len(returns)),
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
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


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_pairwise_mean_return_tests(
    wealth_returns: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """各モデルのリターン系列どうしの「平均リターン差」の有意性検定（対応あり t 検定近似）。

    前提:
        - wealth_returns: インデックス=date, 列=model のリターン DataFrame。
        - 行ごとに同じ日付のリターンが並んでいる（対応のある観測）。

    出力:
        model_a, model_b, n_obs, mean_diff, t_stat, p_value, significant_5pct
    """
    if wealth_returns.empty or wealth_returns.shape[1] < 2:
        return pd.DataFrame()

    models = list(wealth_returns.columns)
    rows: List[Dict[str, object]] = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            diff = wealth_returns[a] - wealth_returns[b]
            diff = diff.dropna()
            n = int(diff.shape[0])
            if n < 2:
                continue
            mean_diff = float(diff.mean())
            std_diff = float(diff.std(ddof=1))
            if std_diff <= 0.0:
                continue
            t_stat = mean_diff / (std_diff / math.sqrt(n))
            p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "n_obs": n,
                    "mean_diff": mean_diff,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "significant_5pct": bool(p_value < alpha),
                }
            )

    return pd.DataFrame(rows)


def compute_pairwise_performance_tests(
    wealth_returns: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """平均リターン・分散・Sharpe・Sortino・累積リターンの差の有意性を
    ブートストラップで評価する。

    - 各ペア (model_a, model_b) について、同じ日付のリターン差を前提に
      時点方向の再標本化（ブートストラップ）を行う。
    - 各ブートストラップサンプルごとにメトリクス差を計算し、
      その符号が 0 からどれだけ安定して離れているかで両側 p 値を近似する。
    """
    if wealth_returns.empty or wealth_returns.shape[1] < 2:
        return pd.DataFrame()

    models = list(wealth_returns.columns)
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(42)

    def _metrics(r: np.ndarray) -> Dict[str, float]:
        r = np.asarray(r, dtype=float)
        mean = float(np.mean(r))
        std = float(np.std(r, ddof=1)) if r.size > 1 else float("nan")
        sharpe = mean / std if std > 1e-12 else float("nan")
        sortino = compute_sortino_ratio(r)
        final_wealth = float(np.prod(1.0 + r))
        return {
            "mean": mean,
            "std": std,
            "sharpe": sharpe,
            "sortino": sortino,
            "final_wealth": final_wealth,
        }

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            ra = wealth_returns[a]
            rb = wealth_returns[b]
            mask = ra.notna() & rb.notna()
            ra = ra[mask].to_numpy(dtype=float)
            rb = rb[mask].to_numpy(dtype=float)
            n = int(ra.shape[0])
            if n < 2:
                continue

            m_a = _metrics(ra)
            m_b = _metrics(rb)
            diff_obs = {k: m_a[k] - m_b[k] for k in m_a.keys()}

            diff_boot: Dict[str, List[float]] = {k: [] for k in m_a.keys()}
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                m_a_b = _metrics(ra[idx])
                m_b_b = _metrics(rb[idx])
                for k in diff_boot.keys():
                    diff_boot[k].append(m_a_b[k] - m_b_b[k])

            row: Dict[str, object] = {
                "model_a": a,
                "model_b": b,
                "n_obs": n,
            }
            for k in ["mean", "std", "sharpe", "sortino", "final_wealth"]:
                obs = diff_obs.get(k, float("nan"))
                boot_arr = np.asarray(diff_boot.get(k, []), dtype=float)
                boot_arr = boot_arr[np.isfinite(boot_arr)]
                if boot_arr.size == 0 or not np.isfinite(obs):
                    p_val = float("nan")
                else:
                    greater = float(np.mean(boot_arr >= 0.0))
                    less = float(np.mean(boot_arr <= 0.0))
                    p_val = 2.0 * min(greater, less)
                    p_val = min(p_val, 1.0)
                row[f"{k}_diff"] = obs
                row[f"{k}_p_5pct"] = p_val
                row[f"{k}_p_1pct"] = p_val
            rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
def summarize_dfl_performance_significance(analysis_csv_dir: Path) -> None:
    """提案手法（DFL-QCQP 系）が他モデルと比べて有意かどうかの概要 CSV を作成する。

    入力: performance_significance.csv
    出力: performance_significance_dfl_summary.csv
    """
    perf_path = Path(analysis_csv_dir) / "performance_significance.csv"
    if not perf_path.exists():
        return
    df = pd.read_csv(perf_path)
    if df.empty:
        return

    all_models: set[str] = set(df["model_a"]).union(set(df["model_b"]))
    # 提案手法モデル（display 名ベース）。display 前の "flex" もフォールバックとして拾う。
    dfl_models = [m for m in all_models if "DFL-QCQP" in str(m)]
    if not dfl_models:
        dfl_models = [m for m in all_models if "flex" in str(m)]
    if not dfl_models:
        return

    rows_out: list[dict[str, object]] = []
    metrics = ["mean", "std", "sharpe", "sortino", "final_wealth"]

    for dfl in dfl_models:
        for other in sorted(m for m in all_models if m != dfl):
            mask_ab = (df["model_a"] == dfl) & (df["model_b"] == other)
            mask_ba = (df["model_a"] == other) & (df["model_b"] == dfl)
            if mask_ab.any():
                rec = df.loc[mask_ab].iloc[0]
                sign = 1.0
            elif mask_ba.any():
                rec = df.loc[mask_ba].iloc[0]
                sign = -1.0
            else:
                continue

            row: dict[str, object] = {
                "model_dfl": dfl,
                "model_other": other,
                "n_obs": int(rec.get("n_obs", 0)),
            }
            for k in metrics:
                diff_col = f"{k}_diff"
                p_col = f"{k}_p_5pct"
                diff_val = rec.get(diff_col, np.nan)
                try:
                    diff = float(diff_val) * sign
                except Exception:
                    diff = float("nan")
                p_val_raw = rec.get(p_col, np.nan)
                try:
                    p_val = float(p_val_raw)
                except Exception:
                    p_val = float("nan")
                row[f"{k}_diff"] = diff
                row[f"{k}_p"] = p_val
                row[f"{k}_sig_5pct"] = bool(np.isfinite(p_val) and p_val < 0.05)
                row[f"{k}_sig_1pct"] = bool(np.isfinite(p_val) and p_val < 0.01)
            rows_out.append(row)

    if rows_out:
        out_df = pd.DataFrame(rows_out)
        out_df.to_csv(
            Path(analysis_csv_dir) / "performance_significance_dfl_summary.csv",
            index=False,
        )


# ------------------------------------------------------------
# 追加分析: 集中度・MSE・バイアス
# ------------------------------------------------------------


def run_concentration_analysis(
    analysis_csv_dir: Path,
    analysis_fig_dir: Path,
    model_outputs_dir: Path,
) -> None:
    """weights.csv から集中度指標 (Neff, max_w, cap_hit) を計算・可視化する。"""
    model_outputs_dir = Path(model_outputs_dir)
    rows: List[Dict[str, object]] = []
    EPS_W = 1e-3
    CAP_THRESH = 0.95

    for model_dir in sorted(model_outputs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        weights_path = model_dir / "weights.csv"
        if not weights_path.exists():
            continue
        try:
            df = pd.read_csv(weights_path)
        except Exception:
            continue
        if "date" not in df.columns:
            continue
        asset_cols = [c for c in df.columns if c not in {"date", "portfolio_return_sq"}]
        if not asset_cols:
            continue
        w = df[asset_cols].astype(float)
        w = w.mask(w.abs() < EPS_W, 0.0)
        H = (w**2).sum(axis=1)
        N_eff = 1.0 / H.replace(0.0, np.nan)
        max_w = w.max(axis=1)
        cap_hit = (max_w >= CAP_THRESH).astype(float)

        rows.append(
            {
                "model": display_model_name(model_dir.name),
                "H_mean": float(H.mean()),
                "N_eff_mean": float(N_eff.mean()),
                "max_w_mean": float(max_w.mean()),
                "cap_hit_freq": float(cap_hit.mean()),
            }
        )

    if not rows:
        return

    conc_df = pd.DataFrame(rows).set_index("model")
    conc_df.to_csv(analysis_csv_dir / "concentration_summary.csv")

    models = conc_df.index.tolist()
    x = np.arange(len(models))

    if plt is None:
        return

    # 一枚の図に 3 つの指標を並べて表示
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    if len(axes.shape) == 0:  # pragma: no cover - defensive
        axes = [axes]

    # モデルごとの色を固定
    color_map = {
        "ols": "blue",
        "ipo": "orange",
        "DFL-QCQP-dual": "green",
        "DFL-QCQP-kkt": "red",
        "DFL-QCQP-ens": "black",
        "1/N": "grey",
    }
    bar_colors = []
    for i, m in enumerate(models):
        c = color_map.get(m)
        if c is None:
            c = f"C{i % 10}"
        bar_colors.append(c)

    # Neff
    ax = axes[0]
    ax.bar(x, conc_df["N_eff_mean"], color=bar_colors)
    ax.set_ylabel("Effective #assets (avg)")
    ax.set_title("Neff")

    # max weight
    ax = axes[1]
    ax.bar(x, conc_df["max_w_mean"], color=bar_colors)
    ax.set_ylabel("Average max weight")
    ax.set_title("Max weight")

    # cap frequency
    ax = axes[2]
    ax.bar(x, conc_df["cap_hit_freq"], color=bar_colors)
    ax.set_ylabel("Freq. of w_max ≥ 0.95")
    ax.set_title("Concentration freq.")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(analysis_fig_dir / "concentration_summary_bar.png")
    plt.close(fig)


def run_mse_and_bias_analysis(
    analysis_csv_dir: Path,
    analysis_fig_dir: Path,
    asset_pred_dir: Path,
) -> None:
    """asset_predictions から MSE・バイアス関連の統計と図を作成する。"""
    asset_pred_dir = Path(asset_pred_dir)
    if not asset_pred_dir.exists():
        return

    dfs: List[pd.DataFrame] = []
    for path in sorted(asset_pred_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "pred_ret" not in df.columns or "real_ret" not in df.columns:
            continue
        if "model" not in df.columns:
            df["model"] = path.stem
        dfs.append(df)
    if not dfs:
        return

    df = pd.concat(dfs, ignore_index=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df["model"] = df["model"].map(display_model_name)

    # MSE by model
    df["err2"] = (df["pred_ret"] - df["real_ret"]) ** 2
    mse_df = df.groupby("model")["err2"].mean().to_frame("mse")

    # Sharpe from summary.csv
    summary_path = Path(analysis_csv_dir) / "summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if "model" not in summary.columns:
        return
    summary = summary.set_index("model")
    # left join にして、1/N など「MSE を持たない」ベンチマークも残す
    plot_df = summary.join(mse_df, how="left")
    if plot_df.empty or plt is None:
        return

    # MSE vs Sharpe scatter
    plt.figure()
    max_mse = float(plot_df["mse"].max(skipna=True)) if "mse" in plot_df.columns else float("nan")
    default_x = 0.0 if not np.isfinite(max_mse) else max_mse * 1.05
    for model, row in plot_df.iterrows():
        # ベンチマーク系は散布図から除外（SPY, 1/N）
        mstr = str(model)
        if ("SPY" in mstr and "benchmark" in mstr) or mstr in {"1/N", "[SPY]", "benchmark_SPY"}:
            continue
        x = row.get("mse", np.nan)
        if not np.isfinite(x):
            # MSE が定義されないベンチマーク（1/N など）は右端にプロット
            x = default_x
        y = row.get("sharpe", np.nan)
        if not np.isfinite(y):
            continue
        plt.scatter(x, y)
        plt.text(x, y, model, fontsize=8)
    plt.xlabel("Prediction MSE (per step)")
    plt.ylabel("Sharpe ratio (annualized)")
    plt.title("Prediction accuracy vs decision quality")
    plt.tight_layout()
    plt.savefig(analysis_fig_dir / "mse_vs_sharpe_scatter.png")
    plt.close()

    # Bias-related analyses
    df["bias"] = df["pred_ret"] - df["real_ret"]

    # Up / Down
    df["updown"] = np.where(df["pred_ret"] > 0, "Up", "Down")
    updown_stats = df.groupby(["model", "updown"])["bias"].agg(
        mean_bias="mean",
        median_bias="median",
        std_bias="std",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    updown_stats.to_csv(analysis_csv_dir / "updown_bias_stats.csv")

    if sns is not None and plt is not None:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="updown", y="bias", hue="model")
        plt.axhline(0.0, linestyle="--", linewidth=1, color="black")
        plt.ylabel("Bias = pred_ret - real_ret")
        plt.title("Prediction bias by Up/Down group")
        plt.yscale("symlog", linthresh=1e-2)
        plt.tight_layout()
        plt.savefig(analysis_fig_dir / "updown_bias_boxplot.png")
        plt.close()

    # IN / OUT (weight threshold)
    TAU = 0.05
    df["inout"] = np.where(df["weight"] > TAU, "IN", "OUT")
    inout_stats = df.groupby(["model", "inout"])["bias"].agg(
        mean_bias="mean",
        median_bias="median",
        std_bias="std",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    inout_stats.to_csv(analysis_csv_dir / "inout_bias_stats.csv")

    if sns is not None and plt is not None:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="inout", y="bias", hue="model")
        plt.axhline(0.0, linestyle="--", linewidth=1, color="black")
        plt.ylabel("Bias = pred_ret - real_ret")
        plt.title("Prediction bias for IN/OUT assets")
        plt.yscale("symlog", linthresh=1e-2)
        plt.tight_layout()
        plt.savefig(analysis_fig_dir / "inout_bias_boxplot.png")
        plt.close()

    # ---- flex 系モデルに対するヒストグラム（Up/Down, IN/OUT） ----
    flex_models = [m for m in df["model"].unique() if "DFL-QCQP" in str(m)]
    if not flex_models:
        # まだ display_model_name を通していない実験では "flex" ラベルを拾う
        flex_models = [m for m in df["model"].unique() if "flex" in str(m)]
    if plt is not None and flex_models:
        # 共通のビンを決める（全体の 1〜99% 区間）
        bias_q01 = float(df["bias"].quantile(0.01))
        bias_q99 = float(df["bias"].quantile(0.99))
        if not np.isfinite(bias_q01) or not np.isfinite(bias_q99) or bias_q01 == bias_q99:
            bias_q01, bias_q99 = -0.1, 0.1
        bins = np.linspace(bias_q01, bias_q99, 40)

        # Up/Down ヒストグラム
        n = len(flex_models)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]
        for ax, model in zip(axes, flex_models):
            sub = df[df["model"] == model]
            up = sub[sub["updown"] == "Up"]["bias"].to_numpy()
            down = sub[sub["updown"] == "Down"]["bias"].to_numpy()
            if up.size == 0 and down.size == 0:
                continue
            if down.size > 0:
                ax.hist(
                    down,
                    bins=bins,
                    alpha=0.5,
                    density=True,
                    color="red",
                    label="Down",
                )
            if up.size > 0:
                ax.hist(
                    up,
                    bins=bins,
                    alpha=0.5,
                    density=True,
                    color="green",
                    label="Up",
                )
            ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
            ax.set_title(model)
            ax.set_xlim(bias_q01, bias_q99)
        axes[0].set_ylabel("Density")
        fig.suptitle("Prediction bias (Up vs Down) for DFL-QCQP models")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.tight_layout(rect=[0, 0, 0.95, 0.9])
        fig.savefig(analysis_fig_dir / "updown_bias_hist_flex.png")
        plt.close(fig)

        # IN/OUT ヒストグラム
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]
        for ax, model in zip(axes, flex_models):
            sub = df[df["model"] == model]
            inside = sub[sub["inout"] == "IN"]["bias"].to_numpy()
            outside = sub[sub["inout"] == "OUT"]["bias"].to_numpy()
            if inside.size == 0 and outside.size == 0:
                continue
            if outside.size > 0:
                ax.hist(
                    outside,
                    bins=bins,
                    alpha=0.5,
                    density=True,
                    color="red",
                    label="OUT",
                )
            if inside.size > 0:
                ax.hist(
                    inside,
                    bins=bins,
                    alpha=0.5,
                    density=True,
                    color="green",
                    label="IN",
                )
            ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
            ax.set_title(model)
            ax.set_xlim(bias_q01, bias_q99)
        axes[0].set_ylabel("Density")
        fig.suptitle("Prediction bias (IN vs OUT) for DFL-QCQP models")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.tight_layout(rect=[0, 0, 0.95, 0.9])
        fig.savefig(analysis_fig_dir / "inout_bias_hist_flex.png")
        plt.close(fig)

    # Top/Bottom bias timeseries
    def _top_bottom_bias(group: pd.DataFrame) -> pd.Series:
        g_sorted = group.sort_values("weight", ascending=False)
        top_bias = float(g_sorted.iloc[0]["bias"])
        bottom_bias = float(g_sorted.iloc[-1]["bias"])
        return pd.Series({"top_bias": top_bias, "bottom_bias": bottom_bias})

    tb = (
        df.groupby(["model", "date"])[["weight", "bias"]]
        .apply(_top_bottom_bias)
        .reset_index()
        .sort_values("date")
    )
    tb.to_csv(analysis_csv_dir / "topbottom_bias_timeseries.csv", index=False)

    if plt is not None and not tb.empty:
        plt.figure(figsize=(10, 5))
        # 提案手法（DFL-QCQP 系）のうち 1 モデルだけ可視化（優先的に dual）
        candidates = [m for m in tb["model"].unique() if "DFL-QCQP" in str(m)]
        if "DFL-QCQP-dual" in candidates:
            target_model = "DFL-QCQP-dual"
        elif candidates:
            target_model = sorted(candidates)[0]
        else:
            # DFL-QCQP が無い場合は、先頭のモデルだけを対象にする
            target_model = str(tb["model"].iloc[0])

        sub = tb[tb["model"] == target_model]
        plt.plot(sub["date"], sub["top_bias"], label=f"{target_model} Top", linestyle="-")
        plt.plot(sub["date"], sub["bottom_bias"], label=f"{target_model} Bottom", linestyle="--")
        plt.axhline(0.0, linestyle=":", linewidth=1, color="black")
        plt.ylabel("Bias = pred_ret - real_ret")
        plt.xlabel("date")
        plt.title("Bias of top/bottom-weighted assets over time")
        plt.yscale("symlog", linthresh=1e-2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(analysis_fig_dir / "topbottom_bias_timeseries.png")
        plt.close()


def run_extended_analysis(
    analysis_csv_dir: Path,
    analysis_fig_dir: Path,
    model_outputs_dir: Path,
    asset_pred_dir: Path,
) -> None:
    """集中度・MSE・バイアスをまとめて実行するラッパ."""
    try:
        run_concentration_analysis(analysis_csv_dir, analysis_fig_dir, model_outputs_dir)
    except Exception as exc:  # pragma: no cover - 解析失敗は本体を止めない
        print(f"[analysis] concentration analysis failed: {exc}")
    try:
        run_mse_and_bias_analysis(analysis_csv_dir, analysis_fig_dir, asset_pred_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[analysis] mse/bias analysis failed: {exc}")


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


def compute_sortino_ratio(returns: Sequence[float], target: float = 0.0) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return np.nan
    downside = arr - target
    downside = downside[downside < 0.0]
    if downside.size == 0:
        return np.nan
    downside_dev = np.sqrt(np.mean(np.square(downside)))
    if downside_dev <= 1e-12:
        return np.nan
    return float((np.mean(arr) - target) / downside_dev)


def update_experiment_ledger(
    results_root: Path,
    outdir: Path,
    args: argparse.Namespace,
    summary_df: pd.DataFrame,
    analysis_csv_dir: Path,
    bundle_summary: Dict[str, object],
) -> None:
    if summary_df.empty:
        return
    ledger_dir = results_root / "experiment_ledger"
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
        "cov_eps": args.cov_eps,
        "cov_robust_huber_k": args.cov_robust_huber_k,
        "cov_factor_rank": args.cov_factor_rank,
        "cov_factor_shrinkage": args.cov_factor_shrinkage,
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


def resolved_solver_options(name: str, options: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Return a concrete solver options dict, filling in defaults if needed."""
    if options:
        return dict(options)
    lower = (name or "").lower()
    if lower == "knitro":
        return dict(KNITRO_DEFAULTS)
    if lower == "gurobi":
        return dict(GUROBI_DEFAULTS)
    return {}
