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
    import matplotlib.dates as mdates
except ImportError:  # pragma: no cover
    plt = None
    mdates = None
try:  # Optional: 日本語フォント環境があれば自動設定
    import japanize_matplotlib  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

from dfl_portfolio.registry import KNITRO_DEFAULTS, GUROBI_DEFAULTS

PERIOD_WINDOWS = [
    ("lehman_2008", "2007-01-01", "2009-12-31"),
    ("covid_2020", "2020-02-01", "2020-12-31"),
    ("inflation_2022", "2022-01-01", "2023-12-31"),
]

WEIGHT_THRESHOLD = 0.95
WEIGHT_PLOT_MAX_POINTS = 60


# 表示用のモデル名マッピング
_MODEL_DISPLAY_MAP = {
    "ols": "OLS",
    "ipo": "IPO-analytic",
    "ipo_grad": "IPO-GRAD",
    "spo_plus": "SPO+",
    "flex": "DFL-QCQP",
    "flex_dual": "DFL-QCQP-dual",
    "flex_kkt": "DFL-QCQP-kkt",
    "flex_dual_kkt_ens": "DFL-QCQP-ens",
    "benchmark_equal_weight": "1/N",
    "benchmark_tsmom_spy": "TSMOM (SPY)",
}

# 一貫した色指定（内部名ベース）
MODEL_COLOR_MAP: Dict[str, str] = {
    "ols": "tab:blue",
    "ipo": "tab:orange",
    "ipo_grad": "tab:brown",
    "spo_plus": "tab:cyan",
    "flex": "tab:green",
    "flex_dual": "tab:green",
    "flex_kkt": "tab:red",
    "flex_dual_kkt_ens": "black",
    "benchmark_SPY": "tab:purple",
    "benchmark_tsmom_spy": "tab:olive",
    "benchmark_equal_weight": "grey",
}

# 線種やアルファをモデルごとに統一的に指定（DFL系は同系色、ensは目立ちすぎない点線）
MODEL_LINESTYLE_MAP: Dict[str, str] = {
    # すべて実線に揃え、ens のみ点線で補助扱い
    "ols": "-",
    "ipo": "-",
    "ipo_grad": "-",
    "spo_plus": "-",
    "flex": "-",
    "flex_dual": "-",
    "flex_kkt": "-",
    "flex_dual_kkt_ens": ":",
    "benchmark_SPY": "-",
    "benchmark_tsmom_spy": "-",
    "benchmark_equal_weight": "-",
}
MODEL_LINEWIDTH_MAP: Dict[str, float] = {
    "flex_dual_kkt_ens": 1.8,
}
MODEL_ALPHA_MAP: Dict[str, float] = {
    # DFL 主系統をやや淡く重ねやすく、ens はさらに抑える
    "flex_dual": 0.75,
    "flex_kkt": 0.75,
    "flex_dual_kkt_ens": 0.65,
}

# 資産系列の基本カラー（データ概観プロットで使用）
ASSET_COLOR_SEQUENCE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

# 表示用の並び順（summary と統一）
PREFERRED_MODEL_ORDER = [
    "DFL-QCQP-dual",
    "DFL-QCQP-kkt",
    "DFL-QCQP-ens",
    "SPO+",
    "IPO-GRAD",
    "IPO-analytic",
    "OLS",
    "Buy&Hold SPY",
    "1/N",
    "TSMOM (SPY)",
]


def _model_plot_kwargs(model: str, base_alpha: float = 1.0) -> Dict[str, object]:
    """統一した色・線種を返すヘルパー。"""
    kwargs: Dict[str, object] = {}
    color = MODEL_COLOR_MAP.get(model)
    if color:
        kwargs["color"] = color
    linestyle = MODEL_LINESTYLE_MAP.get(model)
    if linestyle:
        kwargs["linestyle"] = linestyle
    linewidth = MODEL_LINEWIDTH_MAP.get(model)
    if linewidth:
        kwargs["linewidth"] = linewidth
    alpha = MODEL_ALPHA_MAP.get(model, 1.0) * base_alpha
    if alpha != 1.0:
        kwargs["alpha"] = alpha
    return kwargs


def _sort_models_for_display(models: Sequence[str]) -> List[str]:
    """display 名ベースで PREFERRED_MODEL_ORDER に従って並べ替える。"""
    order_map = {name: idx for idx, name in enumerate(PREFERRED_MODEL_ORDER)}
    return sorted(
        models,
        key=lambda m: order_map.get(display_model_name(m), len(order_map)),
    )


def _add_range_markers(ax, start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> None:
    """開始・終了を示す縦線を控えめに追加する（ラベルは付けない）。"""
    if start_ts is None or end_ts is None:
        return
    for ts in [start_ts, end_ts]:
        if pd.isna(ts):
            continue
        ts_dt = pd.to_datetime(ts)
        ax.axvline(ts_dt, color="0.35", linestyle=":", linewidth=0.9, alpha=0.8)


def display_model_name(model: str) -> str:
    """集計・可視化用にモデル名を整形する."""
    name = str(model)
    if name in _MODEL_DISPLAY_MAP:
        return _MODEL_DISPLAY_MAP[name]
    # ベンチマークティッカー系: benchmark_SPY → Buy&Hold SPY
    if name.startswith("benchmark_") and name != "benchmark_equal_weight":
        ticker = name[len("benchmark_") :]
        return f"Buy&Hold {ticker}"
    return name


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
    plt.xlabel("日付")
    plt.ylabel("累積リターン")
    plt.title("ポートフォリオ累積リターン")
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
    plt.title(f"資産ウェイト推移 (積み上げ) - {model}")
    plt.xlabel("日付")
    plt.ylabel("ウェイト")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plt.figure(figsize=(10, 4))
    for col in value_cols_sorted:
        plt.plot(dates, values[col], label=col)
    plt.ylim(0, 1)
    plt.title(f"資産ウェイト推移 ({model})")
    plt.xlabel("日付")
    plt.ylabel("ウェイト")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path.with_name(path.stem + "_lines" + path.suffix))
    plt.close()


def plot_weight_comparison(
    weight_dict: Dict[str, pd.DataFrame],
    path: Path,
    *,
    max_points: int | None = WEIGHT_PLOT_MAX_POINTS,
) -> None:
    if plt is None or not weight_dict:
        return
    models = _sort_models_for_display(list(weight_dict.keys()))
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        df = weight_dict[model]
        if max_points is not None and len(df) > max_points:
            df = df.tail(max_points)
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
    axes[-1].set_xlabel("日付")
    fig.suptitle("資産ウェイト推移（直近ポイント）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_weight_variance_correlation(
    weight_dict: Dict[str, pd.DataFrame],
    csv_path: Path,
    fig_path: Path,
) -> None:
    """残してはおくが、現在は CSV のみ出力し図は作らない（fig_path は未使用）。"""
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


def plot_wealth_correlation_heatmap(corr_df: pd.DataFrame, path: Path) -> None:
    if plt is None or corr_df.empty:
        return
    # 並び順を summary に合わせる
    desired = _sort_models_for_display(list(corr_df.index))
    corr_df = corr_df.reindex(index=desired, columns=desired)

    fig, ax = plt.subplots(figsize=(7, 6))
    data = corr_df.to_numpy()
    cax = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title("累積リターン相関")
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
    # 内部では正の割合（例: 0.25）を返す。CSV 出力時に -25.00 のように符号・%変換する。
    return float(np.nanmax(dd))


def compute_cvar(
    returns: Sequence[float],
    alpha: float = 0.05,
) -> float:
    """Compute CVaR (Expected Shortfall) of returns at level alpha.

    Returns are assumed to be per-period arithmetic returns. The function returns
    the expected value of returns conditional on being in the worst ``alpha``
    fraction (typically a negative number). When there are too few observations,
    NaN is returned.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return np.nan
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    # 並べ替えて下側 alpha 部分の平均をとる
    sorted_ret = np.sort(arr)
    k = int(np.floor(alpha * sorted_ret.size))
    if k <= 0:
        return np.nan
    tail = sorted_ret[:k]
    return float(np.mean(tail))


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


def format_summary_for_output(summary_df: pd.DataFrame) -> pd.DataFrame:
    """summary.csv 用にメトリクスを人間が読みやすい形に整形する。

    - ann_return, ann_volatility, total_return を [%] として扱い、
      100 倍して小数第 2 位までに丸める（total_return は累積リターン）。
    - sharpe, sortino はスケーリングせず生の値を小数第 4 位までに丸める。
    - max_drawdown は「下落率」を表現するために負の値に変換（例: 0.25 → -25.00）。
    - final_wealth も小数第 2 位までに丸める。
    """
    if summary_df.empty:
        return summary_df
    df = summary_df.copy()
    # summary では train_window / rebal_interval / avg_condition_number は列から除外する
    for col in ["train_window", "rebal_interval", "avg_condition_number"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    # % として表示する列（年率リターン・年率ボラティリティ・累積リターン）
    percent_cols = [
        "ann_return",
        "ann_volatility",
        "total_return",
        "ann_return_net",
        "ann_volatility_net",
        "total_return_net",
        "avg_turnover",
    ]
    for col in percent_cols:
        if col in df.columns:
            df[col] = (df[col].astype(float) * 100.0).round(2)
    # Sharpe / Sortino は基本 4 桁だが net 系は 2 桁に抑える
    if "sharpe" in df.columns:
        df["sharpe"] = df["sharpe"].astype(float).round(4)
    if "sortino" in df.columns:
        df["sortino"] = df["sortino"].astype(float).round(4)
    if "sharpe_net" in df.columns:
        df["sharpe_net"] = df["sharpe_net"].astype(float).round(2)
    if "sortino_net" in df.columns:
        df["sortino_net"] = df["sortino_net"].astype(float).round(2)
    # R^2 は高精度確認用に 6 桁
    if "r2" in df.columns:
        df["r2"] = df["r2"].astype(float).round(6)
    # RMSE も小さい値になりがちなので 6 桁
    if "rmse" in df.columns:
        df["rmse"] = df["rmse"].astype(float).round(6)
    # CVaR は「損失期待値」なので負の値のまま % 表示
    if "cvar_95" in df.columns:
        df["cvar_95"] = (df["cvar_95"].astype(float) * 100.0).round(2)
    if "max_drawdown" in df.columns:
        df["max_drawdown"] = (-df["max_drawdown"].astype(float) * 100.0).round(2)
    if "avg_trading_cost" in df.columns:
        series = pd.to_numeric(df["avg_trading_cost"], errors="coerce")
        df["avg_trading_cost"] = series.apply(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    if "terminal_wealth" in df.columns:
        df["terminal_wealth"] = df["terminal_wealth"].astype(float).round(2)
    if "terminal_wealth_net" in df.columns:
        df["terminal_wealth_net"] = df["terminal_wealth_net"].astype(float).round(2)

    # 列の並び替え: 投資成績→リスク→補助指標→カウント系の順に並べる
    preferred_order = [
        "model",
        # 成果・リターン系
        "ann_return",
        "total_return",
        "terminal_wealth",
        "ann_return_net",
        "total_return_net",
        "terminal_wealth_net",
        "sharpe",
        "sharpe_net",
        "sortino",
        "sortino_net",
        # リスク系
        "ann_volatility",
        "ann_volatility_net",
        "max_drawdown",
        "cvar_95",
        "avg_turnover",
        "avg_trading_cost",
        # サブ指標
        "r2",
        "rmse",
        # カウント系
        "n_retrain",
        "n_invest_steps",
    ]
    existing_cols = list(df.columns)
    ordered_cols = [c for c in preferred_order if c in existing_cols]
    # 上記以外の列があれば末尾に付ける
    remaining = [c for c in existing_cols if c not in ordered_cols]
    df = df[ordered_cols + remaining]

    # モデル表示順（提案手法 → 近縁手法 → ベンチマーク）
    preferred_model_order = [
        "DFL-QCQP-dual",
        "DFL-QCQP-kkt",
        "DFL-QCQP-ens",
        "SPO+",
        "IPO-GRAD",
        "IPO-analytic",
        "OLS",
        "Buy&Hold SPY",
        "1/N",
        "TSMOM (SPY)",
    ]
    if "model" in df.columns:
        order_map = {name: idx for idx, name in enumerate(preferred_model_order)}
        df = df.assign(_sort_key=df["model"].map(order_map).fillna(len(order_map)))
        df = df.sort_values(by="_sort_key", kind="mergesort").drop(columns=["_sort_key"])
    return df


def build_cost_adjusted_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Construct a cost-adjusted view by swapping gross metrics with their net counterparts."""
    if summary_df.empty:
        return summary_df.copy()
    df = summary_df.copy()
    replacements = {
        "ann_return": "ann_return_net",
        "total_return": "total_return_net",
        "terminal_wealth": "terminal_wealth_net",
        "ann_volatility": "ann_volatility_net",
        "sharpe": "sharpe_net",
        "sortino": "sortino_net",
    }
    for gross_col, net_col in replacements.items():
        if net_col in df.columns:
            df[gross_col] = df[net_col]
    return df


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
        delta_vals = pd.to_numeric(sub_sorted["delta_used"], errors="coerce")
        delta_mean = float(delta_vals.mean()) if delta_vals.notna().any() else float("nan")
        label = (
            f"{model} (avg={delta_mean:.3f})"
            if np.isfinite(delta_mean)
            else str(model)
        )
        if use_date_axis:
            x = pd.to_datetime(sub_sorted["rebalance_date"])
        else:
            if "cycle" not in sub_sorted.columns:
                continue
            x = sub_sorted["cycle"]
        ax.plot(x, sub_sorted["delta_used"], label=label)

    # 金融ショック期間を背景に表示（wealth_events と同じ PERIOD_WINDOWS を使用）
    if use_date_axis:
        for _, start, end in PERIOD_WINDOWS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15)

    ax.set_xlabel("日付" if use_date_axis else "サイクル")
    ax.set_ylabel("デルタ")
    ax.set_title("学習されたデルタ推移（モデル別）")
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

    # DFL 系（flex 系）のみを対象に可視化する
    if "model" in phi_df.columns:
        mask_flex = phi_df["model"].astype(str).str.contains("flex", case=False)
        phi_df = phi_df.loc[mask_flex].copy()
        if phi_df.empty:
            return

    fig, ax = plt.subplots(figsize=(8, 4))
    use_date_axis = "rebalance_date" in phi_df.columns
    anchor_levels: dict[str, float] = {}
    # OAS 由来のアンカー φ_t の推移も描画するため、モデルごとの系列を保持
    anchor_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model, sub in phi_df.groupby("model"):
        sub_sorted = sub.sort_values("cycle") if "cycle" in sub.columns else sub.copy()
        if use_date_axis:
            x = pd.to_datetime(sub_sorted["rebalance_date"])
        else:
            if "cycle" not in sub_sorted.columns:
                continue
            x = sub_sorted["cycle"]
        ax.plot(x, sub_sorted["phi_used"], label=model)
        # モデルごとのアンカー値（phi_anchor 列があればその平均、なければ 0.0）
        if "phi_anchor" in sub_sorted.columns:
            vals = pd.to_numeric(sub_sorted["phi_anchor"], errors="coerce")
            vals_arr_full = vals.to_numpy(dtype=float)
            mask = np.isfinite(vals_arr_full)
            vals_arr = vals_arr_full[mask]
            if vals_arr.size > 0:
                anchor_levels[str(model)] = float(vals_arr.mean())
                # アンカーの推移そのものも描画できるように保存
                if use_date_axis:
                    x_raw = pd.to_datetime(sub_sorted["rebalance_date"]).to_numpy()
                else:
                    x_raw = np.asarray(x)
                x_anchor = x_raw[mask]
                anchor_series[str(model)] = (x_anchor, vals_arr)

    # OAS+EWMA ベースラインの φ アンカーを点線で表示
    # ・phi_anchor 列がある場合は、その平均値を使用
    # ・無い場合は φ=0.0（追加 shrinkage 無し＝OAS のみ）を基準とする
    if anchor_levels:
        # flex 系モデルのアンカーの代表値として平均を用いる
        phi_anchor_val = float(np.mean(list(anchor_levels.values())))
        label = f"OAS baseline (phi≈{phi_anchor_val:.2f})"
    else:
        phi_anchor_val = 0.0
        label = "OAS baseline (phi=0.0)"

    ax.axhline(
        y=phi_anchor_val,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=label,
    )

    # 各モデルについて、OAS 由来のアンカー φ_t の推移も点線で描画する
    for model, (x_anchor, vals_arr) in anchor_series.items():
        ax.plot(
            x_anchor,
            vals_arr,
            color="grey",
            linestyle=":",
            linewidth=1.0,
            label=f"OAS phi_anchor ({model})",
        )

    if use_date_axis:
        for _, start, end in PERIOD_WINDOWS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15)

    ax.set_xlabel("日付" if use_date_axis else "サイクル")
    ax.set_ylabel("φ (縮小パラメータ)")
    ax.set_title("共分散縮小パラメータ φ の推移（モデル別）")
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


def plot_beta_paths(beta_df: pd.DataFrame, path: Path) -> None:
    """Plot EWMA decay parameter beta trajectories per model over time."""
    if plt is None or beta_df.empty:
        return
    if "beta_used" not in beta_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    use_date_axis = "rebalance_date" in beta_df.columns
    for model, sub in beta_df.groupby("model"):
        sub_sorted = sub.sort_values("cycle") if "cycle" in sub.columns else sub.copy()
        if use_date_axis:
            x = pd.to_datetime(sub_sorted["rebalance_date"])
        else:
            if "cycle" not in sub_sorted.columns:
                continue
            x = sub_sorted["cycle"]
        ax.plot(x, sub_sorted["beta_used"], label=model)

    if use_date_axis:
        for _, start, end in PERIOD_WINDOWS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15)

    ax.set_xlabel("日付" if use_date_axis else "サイクル")
    ax.set_ylabel("β (EWMA)")
    ax.set_title("EWMA β の推移（モデル別）")
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
    for model, df in wealth_dict.items():
        display_label = display_model_name(model)
        kwargs = {"label": display_label}
        kwargs.update(_model_plot_kwargs(model))
        plt.plot(pd.to_datetime(df["date"]), df["wealth"], **kwargs)
    plt.xlabel("日付")
    plt.ylabel("累積リターン")
    plt.title("累積リターン比較")
    # 開始・終了を控えめにマーキング
    all_dates = pd.concat([pd.to_datetime(df["date"]) for df in wealth_dict.values() if not df.empty])
    if not all_dates.empty:
        ax = plt.gca()
        _add_range_markers(ax, all_dates.min(), all_dates.max())
    plt.legend()
    plt.tight_layout()
    fig_path = path
    if path.name == "wealth_comparison.png":
        fig_path = path.with_name("1-wealth_comparison.png")
    plt.savefig(fig_path)
    plt.close()


def plot_wealth_with_events(wealth_dict: Dict[str, pd.DataFrame], path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(10, 4))
    for model, df in wealth_dict.items():
        display_label = display_model_name(model)
        kwargs = {"label": display_label}
        kwargs.update(_model_plot_kwargs(model))
        plt.plot(pd.to_datetime(df["date"]), df["wealth"], **kwargs)
    for _, start, end in PERIOD_WINDOWS:
        plt.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15)
    all_dates = pd.concat([pd.to_datetime(df["date"]) for df in wealth_dict.values() if not df.empty])
    start_ts = all_dates.min() if not all_dates.empty else None
    end_ts = all_dates.max() if not all_dates.empty else None
    if start_ts is not None and end_ts is not None:
        ax = plt.gca()
        _add_range_markers(ax, start_ts, end_ts)
        title_range = f"{start_ts.date()} – {end_ts.date()}"
    else:
        title_range = ""
    plt.xlabel("日付")
    plt.ylabel("累積リターン")
    plt.title(f"累積リターン {title_range}".strip())
    # 凡例をモデル順で並べ替え
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    order_map = {name: idx for idx, name in enumerate(PREFERRED_MODEL_ORDER)}
    sorted_items = sorted(seen.items(), key=lambda kv: order_map.get(kv[0], len(order_map)))
    uniq_labels = [k for k, _ in sorted_items]
    uniq_handles = [v for _, v in sorted_items]
    plt.legend(uniq_handles, uniq_labels, loc="upper left")
    plt.tight_layout()
    fig_path = path
    if path.name == "wealth_events.png":
        fig_path = path.with_name("2-wealth_events.png")
    plt.savefig(fig_path)
    plt.close()


def plot_wealth_window_normalized(
    wealth_dict: Dict[str, pd.DataFrame],
    period_name: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    path: Path,
    events_by_model: Dict[str, pd.DataFrame] | None = None,
) -> None:
    """Plot wealth trajectories per model, normalized to 1 at the period start.

    For each model, we take wealth over [start, end], rebase the first value to 1.0,
    and plot the normalized path. This is useful to compare performance within a
    specific crisis window.
    """
    if plt is None or not wealth_dict:
        return
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    plt.figure(figsize=(8, 4))
    any_plotted = False
    norm_series_by_model: Dict[str, pd.DataFrame] = {}
    line_colors: Dict[str, str] = {}
    for model, df in wealth_dict.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.sort_values("date")
        mask = (tmp["date"] >= start_ts) & (tmp["date"] <= end_ts)
        tmp = tmp.loc[mask]
        if tmp.empty:
            continue
        base = float(tmp["wealth"].iloc[0])
        if not np.isfinite(base) or base == 0.0:
            continue
        series = tmp.copy()
        series["wealth_norm"] = series["wealth"].astype(float) / base
        display_label = display_model_name(model)
        kwargs = {"label": display_label}
        kwargs.update(_model_plot_kwargs(model))
        (line_handle,) = plt.plot(series["date"], series["wealth_norm"], **kwargs)
        norm_series_by_model[model] = series[["date", "wealth_norm"]].copy()
        line_colors[model] = line_handle.get_color()
        any_plotted = True
    if not any_plotted:
        plt.close()
        return
    plt.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.xlabel("日付")
    plt.ylabel("累積リターン（開始を1に正規化）")
    plt.title(f"期間別累積リターン（開始=1）: {period_name}")
    ax = plt.gca()
    _add_range_markers(ax, start_ts, end_ts)
    plt.legend(loc="upper left")

    if events_by_model:
        for model, events in events_by_model.items():
            if model not in norm_series_by_model:
                continue
            if events.empty:
                continue
            events = events.copy()
            events["rebalance_date"] = pd.to_datetime(events["rebalance_date"])
            mask = (events["rebalance_date"] >= start_ts) & (events["rebalance_date"] <= end_ts)
            events = events.loc[mask]
            if events.empty:
                continue
            series = norm_series_by_model[model].sort_values("date")
            if series.empty:
                continue
            x_series = series["date"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
            y_series = series["wealth_norm"].to_numpy(dtype=float)
            color = line_colors.get(model, "red")
            x_evts = []
            y_evts = []
            for evt_date in events["rebalance_date"]:
                evt_ord = float(pd.Timestamp(evt_date).toordinal())
                if evt_ord < x_series[0] or evt_ord > x_series[-1]:
                    continue
                y_evt = np.interp(evt_ord, x_series, y_series)
                x_evts.append(evt_date)
                y_evts.append(y_evt)
            if x_evts:
                plt.scatter(
                    x_evts,
                    y_evts,
                    marker="x",
                    s=25,
                    color=color,
                    alpha=0.9,
                    linewidths=1.2,
                )
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_time_series(df: pd.DataFrame, title: str, start_date: pd.Timestamp, path: Path) -> None:
    if plt is None or df.empty:
        return
    plt.figure(figsize=(10, 4))
    cols = list(df.columns)
    for idx, col in enumerate(cols):
        color = ASSET_COLOR_SEQUENCE[idx % len(ASSET_COLOR_SEQUENCE)]
        plt.plot(df.index, df[col], label=col, color=color, linewidth=1.2)
    if not pd.isna(start_date):
        plt.axvline(start_date, color="red", linestyle="--", label="start_date")
    plt.title(title)
    plt.xlabel("日付")
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
    """Export frequency with which weights exceed a given threshold.

    現在は CSV のみを出力し、図は生成しない（fig_path は未使用）。
    """
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


def export_winner_capture_frequency(
    *,
    returns_df: pd.DataFrame,
    weight_dict: Dict[str, pd.DataFrame],
    threshold: float,
    csv_path: Path,
    fig_path: Path,
    by_asset_csv_path: Path | None = None,
    by_asset_fig_path: Path | None = None,
    title: str = "週次最大リターン資産の捕捉率（w>=閾値）",
) -> None:
    """Compute how often each model allocates >=threshold weight to the period winner asset.

    Winner asset is defined by realized returns (argmax across assets) at each date.
    A "capture" occurs if the model's portfolio weight for that winner asset is >= threshold.
    """
    if returns_df is None or returns_df.empty or not weight_dict:
        return
    df_ret = returns_df.copy()
    df_ret = df_ret.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if df_ret.empty:
        return
    df_ret.index = pd.to_datetime(df_ret.index)

    # Winner ticker per date
    winner = df_ret.idxmax(axis=1)
    winner_df = pd.DataFrame({"date": df_ret.index, "winner": winner.astype(str).to_numpy()}).dropna()
    if winner_df.empty:
        return
    tickers = [c for c in df_ret.columns]
    ticker_set = set(map(str, tickers))

    def _model_label(model_key: str) -> str:
        try:
            return display_model_name(model_key)
        except Exception:
            return str(model_key)

    rows: List[Dict[str, object]] = []
    by_asset_rows: List[Dict[str, object]] = []
    # For plotting: store (captured_by_ticker, missed_count, n_total) per model
    plot_cache: Dict[str, Tuple[Dict[str, int], int, int]] = {}

    for model_key, wdf in weight_dict.items():
        if wdf is None or wdf.empty or "date" not in wdf.columns:
            continue
        df_w = wdf.copy()
        df_w["date"] = pd.to_datetime(df_w["date"])
        df_w = df_w.dropna(subset=["date"])
        df_w = df_w.set_index("date").sort_index()
        # keep only tickers present in returns (robust to extra columns)
        keep_cols = [c for c in df_w.columns if str(c) in ticker_set]
        if not keep_cols:
            continue
        df_w = df_w[keep_cols].astype(float)

        merged = winner_df.merge(
            df_w.reset_index(),
            on="date",
            how="inner",
        )
        if merged.empty:
            continue

        captured_total = 0
        captured_by_ticker: Dict[str, int] = {t: 0 for t in tickers}
        missed_total = 0

        # Per-date capture decision
        for _, r in merged.iterrows():
            w_ticker = str(r["winner"])
            if w_ticker not in df_w.columns:
                missed_total += 1
                continue
            w_val = float(r.get(w_ticker, np.nan))
            if np.isfinite(w_val) and (w_val >= float(threshold)):
                captured_total += 1
                captured_by_ticker[w_ticker] = captured_by_ticker.get(w_ticker, 0) + 1
            else:
                missed_total += 1

        n_total = int(merged.shape[0])
        capture_rate = float(captured_total) / float(n_total) if n_total > 0 else float("nan")
        rows.append(
            {
                "model": str(_model_label(str(model_key))),
                "model_key": str(model_key),
                "threshold": float(threshold),
                "n_dates": n_total,
                "n_captured": int(captured_total),
                "n_missed": int(missed_total),
                "capture_rate": float(capture_rate),
            }
        )
        plot_cache[str(model_key)] = (captured_by_ticker, missed_total, n_total)

        # By-asset conditional capture: P(captured | ticker is winner)
        for t in tickers:
            win_mask = merged["winner"].astype(str) == str(t)
            n_win = int(win_mask.sum())
            n_cap = int(captured_by_ticker.get(str(t), 0))
            rate_t = float(n_cap) / float(n_win) if n_win > 0 else float("nan")
            by_asset_rows.append(
                {
                    "model": str(_model_label(str(model_key))),
                    "model_key": str(model_key),
                    "ticker": str(t),
                    "threshold": float(threshold),
                    "n_wins": n_win,
                    "n_captured": n_cap,
                    "capture_rate_when_wins": rate_t,
                }
            )

    if not rows:
        return
    out_df = pd.DataFrame(rows).sort_values("capture_rate", ascending=False)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_path, index=False)

    if by_asset_csv_path is not None and by_asset_rows:
        by_asset_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(by_asset_rows).to_csv(by_asset_csv_path, index=False)

    if plt is None:
        return

    # Plot 1: stacked bar (captured-by-asset + missed)
    labels = [str(_model_label(k)) for k in plot_cache.keys()]
    model_keys = list(plot_cache.keys())
    x = np.arange(len(model_keys), dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(model_keys)), 4.2))
    bottom = np.zeros(len(model_keys), dtype=float)
    # Asset colors
    asset_colors = {t: ASSET_COLOR_SEQUENCE[i % len(ASSET_COLOR_SEQUENCE)] for i, t in enumerate(tickers)}
    for t in tickers:
        vals = []
        for k in model_keys:
            captured_by_t, _, n_total = plot_cache[k]
            vals.append(float(captured_by_t.get(str(t), 0)) / float(n_total) if n_total > 0 else 0.0)
        ax.bar(x, vals, bottom=bottom, label=str(t), color=asset_colors.get(str(t), None), alpha=0.85)
        bottom += np.asarray(vals, dtype=float)
    missed_vals = []
    for k in model_keys:
        _, missed, n_total = plot_cache[k]
        missed_vals.append(float(missed) / float(n_total) if n_total > 0 else 0.0)
    ax.bar(x, missed_vals, bottom=bottom, label="miss", color="#b0b0b0", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("頻度")
    ax.set_title(f"{title}（threshold={float(threshold):.2f}）")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)

    # Plot 2 (optional): conditional capture by asset (heatmap-like table)
    if by_asset_fig_path is not None and by_asset_rows:
        try:
            df_ba = pd.DataFrame(by_asset_rows)
            pivot = df_ba.pivot_table(
                index="model",
                columns="ticker",
                values="capture_rate_when_wins",
                aggfunc="mean",
            )
            fig2, ax2 = plt.subplots(figsize=(max(7.5, 1.4 * len(pivot.index)), 3.5))
            im = ax2.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGn")
            ax2.set_yticks(np.arange(pivot.shape[0]))
            ax2.set_yticklabels(list(pivot.index))
            ax2.set_xticks(np.arange(pivot.shape[1]))
            ax2.set_xticklabels(list(pivot.columns))
            ax2.set_title(f"勝者資産ごとの捕捉率（w>=閾値, threshold={float(threshold):.2f}）")
            fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.iat[i, j]
                    if not np.isfinite(v):
                        continue
                    ax2.text(j, i, f"{float(v):.2f}", ha="center", va="center", fontsize=8, color="black")
            fig2.tight_layout()
            by_asset_fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig2.savefig(by_asset_fig_path)
            plt.close(fig2)
        except Exception:
            pass

def export_average_weights(weight_dict: Dict[str, pd.DataFrame], csv_path: Path, fig_path: Path) -> None:
    rows: List[Dict[str, object]] = []
    for model, df in weight_dict.items():
        values, cols = reduce_weight_columns(df)
        if values.empty:
            continue
        mean_vals = values.mean(skipna=True)
        # 資産ごとの「w >= WEIGHT_THRESHOLD」の頻度
        freq_vals = (values >= WEIGHT_THRESHOLD).mean(axis=0)
        for ticker in cols:
            rows.append(
                {
                    "model": model,
                    "ticker": ticker,
                    "avg_weight": float(mean_vals[ticker]),
                    "freq_ge_thresh": float(freq_vals[ticker]),
                }
            )
    if not rows:
        return
    avg_df = pd.DataFrame(rows)
    avg_df.to_csv(csv_path, index=False)
    if plt is None:
        return
    # 1枚の図に avg と「w>=threshold の頻度」を並べる
    pivot_avg = avg_df.pivot(index="model", columns="ticker", values="avg_weight").fillna(0.0)
    pivot_freq = avg_df.pivot(index="model", columns="ticker", values="freq_ge_thresh").fillna(0.0)
    models = _sort_models_for_display(list(pivot_avg.index))
    pivot_avg = pivot_avg.reindex(index=models)
    pivot_freq = pivot_freq.reindex(index=models)
    tickers = list(pivot_avg.columns)
    x = np.arange(len(models))
    width = 0.8 / max(len(tickers), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    hatch_patterns = ["", "//", "\\\\", "++", "xx", "..", "**"]
    # ティッカーごとに明確な色＋ハッチを割り当て、モデルで並べる
    asset_colors = [ASSET_COLOR_SEQUENCE[i % len(ASSET_COLOR_SEQUENCE)] for i in range(len(tickers))]

    # 平均ウェイト
    ax = axes[0]
    for i, ticker in enumerate(tickers):
        hatch = hatch_patterns[i % len(hatch_patterns)]
        ax.bar(
            x + i * width,
            pivot_avg[ticker],
            width=width,
            label=ticker,
            color=asset_colors[i],
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
            alpha=0.75,
        )
    ax.set_title("平均ウェイト（資産別）")
    ax.set_ylabel("平均ウェイト")

    # 95%以上採用頻度
    ax = axes[1]
    for i, ticker in enumerate(tickers):
        hatch = hatch_patterns[i % len(hatch_patterns)]
        ax.bar(
            x + i * width,
            pivot_freq[ticker],
            width=width,
            label=ticker,
            color=asset_colors[i],
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
            alpha=0.75,
        )
    ax.set_title(f"ウェイトが {WEIGHT_THRESHOLD:.2f} 以上となる頻度（資産別）")
    ax.set_ylabel("頻度")

    # x 軸ラベルは display_model_name で人間向けに整形
    display_labels = [display_model_name(m) for m in models]
    for ax in axes:
        ax.set_xticks(x + width * (len(tickers) - 1) / 2)
        ax.set_xticklabels(display_labels, rotation=45, ha="right")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.95, 1.0])
    fig.savefig(fig_path)
    plt.close(fig)


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

    # 共通のビン（0〜1 をより細かく等間隔に分割）
    bins = np.linspace(0.0, 1.0, 41)

    for r, model in enumerate(models):
        values, _ = reduce_weight_columns(weight_dict[model])
        for c, ticker in enumerate(tickers):
            ax = axes[r, c]
            data = values[ticker].dropna().to_numpy()
            if data.size == 0:
                ax.set_visible(False)
                continue
            ax.hist(data, bins=bins, range=(0.0, 1.0), color="tab:blue", alpha=0.7)
            ax.set_yscale("log")
            ax.set_xlim(0.0, 1.0)
            if r == n_rows - 1:
                ax.set_xlabel(ticker)
            if c == 0:
                ax.set_ylabel(model)
    fig.suptitle("ウェイト分布（ヒストグラム）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_condition_numbers(step_df: pd.DataFrame, path: Path, *, max_points: int | None = None) -> None:
    """Plot covariance condition numbers over time as a bar chart."""
    if plt is None or step_df.empty:
        return
    required_cols = {"date", "condition_number"}
    if not required_cols.issubset(step_df.columns):
        return
    df = step_df[list(required_cols)].copy()
    df = df.dropna()
    if df.empty:
        return
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if max_points is not None and int(max_points) > 0 and len(df) > int(max_points):
        df = df.iloc[-int(max_points):]
    dates = df["date"]
    values = df["condition_number"].astype(float)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(dates, values, width=3, color="tab:blue", alpha=0.75)
    ax.set_xlabel("日付")
    ax.set_ylabel("条件数")
    ax.set_title("共分散行列の条件数推移")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_flex_solver_debug(df: pd.DataFrame, path: Path) -> None:
    """Visualize flex solver behavior over time (per rebalance).

    Expects per-cycle rebalance logs with at least:
        - model
        - rebalance_date (ISO string) or cycle
        - elapsed_sec
        - solver_status
    """
    if plt is None or df.empty:
        return
    flex_df = df.copy()
    if "model" not in flex_df.columns:
        return
    def _display_model_name(model: object) -> str:
        name = str(model)
        if "flex_dual" in name:
            return "DFL-QCQP-dual"
        if "flex_kkt" in name:
            return "DFL-QCQP-kkt"
        return name

    flex_df["model_display"] = flex_df["model"].map(_display_model_name)
    # 時系列ソート
    if "rebalance_date" in flex_df.columns:
        flex_df["rebalance_date"] = pd.to_datetime(flex_df["rebalance_date"])
        flex_df = flex_df.sort_values(["model", "rebalance_date"])
        x_col = "rebalance_date"
        x_label = "date"
    else:
        if "cycle" not in flex_df.columns:
            return
        flex_df = flex_df.sort_values(["model", "cycle"])
        x_col = "cycle"
        x_label = "cycle"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)

    # --- 左: solver status の時系列（optimal 以外のみマーカー表示, モデルごとに Y を分ける） ---
    ax_status = axes[0]
    # ok とみなすステータス（optimal / ok を含むもの）
    def _is_ok(status: object) -> bool:
        s = str(status).lower()
        return ("optimal" in s) or ("ok" in s)

    # 異常系ステータスだけ抽出
    flex_df["solver_status_str"] = flex_df["solver_status"].astype(str)
    non_ok = flex_df[~flex_df["solver_status_str"].map(_is_ok)]
    if not non_ok.empty:
        # モデルごとに Y 軸を離して配置（flex_dual / flex_kkt の違いを見る）
        uniq_models = sorted(non_ok["model_display"].astype(str).unique())
        y_map = {m: i for i, m in enumerate(uniq_models)}
        marker_cycle = ["x", "^", "v", "s", "D", "P", "*", "o"]
        marker_map = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
        color_map = {}

        for i, status in enumerate(sorted(non_ok["solver_status_str"].unique())):
            marker_map[status] = marker_cycle[i % len(marker_cycle)]
            color_map[status] = color_cycle[i % len(color_cycle)]

        for status, sub in non_ok.groupby("solver_status_str"):
            x_vals = sub[x_col]
            y_vals = [y_map[str(m)] for m in sub["model_display"]]
            ax_status.scatter(
                x_vals,
                y_vals,
                marker=marker_map.get(status, "x"),
                color=color_map.get(status, "C0"),
                label=status,
            )
        ax_status.set_yticks(list(y_map.values()))
        ax_status.set_yticklabels(list(y_map.keys()))
        legend_labels = []
        for status in sorted(non_ok["solver_status_str"].unique()):
            parts = []
            sub = non_ok[non_ok["solver_status_str"] == status]
            for model, cnt in sub["model_display"].value_counts().items():
                parts.append(f"{model}: {cnt}")
            label = f"{status} ({', '.join(parts)})"
            legend_labels.append(label)
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker=marker_map.get(status, "x"),
                color=color_map.get(status, "C0"),
                linestyle="",
            )
            for status in sorted(non_ok["solver_status_str"].unique())
        ]
        ax_status.legend(handles, legend_labels, loc="best", fontsize=8)
    ax_status.set_xlabel(x_label)
    ax_status.set_ylabel("model (non-optimal statuses)")
    ax_status.set_title("Flex solver status over time")

    # --- 右: elapsed time のヒストグラム比較 ---
    ax_time = axes[1]
    use_log_scale = False
    if "elapsed_sec" in flex_df.columns:
        grouped = []
        for model, sub in flex_df.groupby("model_display"):
            values = sub["elapsed_sec"].astype(float).dropna()
            if not values.empty:
                grouped.append((str(model), values))
        if grouped:
            eps = 1e-3
            min_val = min(float(vals.min()) for _, vals in grouped)
            max_val = max(float(vals.max()) for _, vals in grouped)
            if not np.isfinite(min_val) or not np.isfinite(max_val):
                min_val, max_val = eps, 1.0
            if max_val <= min_val:
                max_val = min_val + eps
            min_for_bins = max(min_val, eps)
            bins = np.logspace(np.log10(min_for_bins), np.log10(max_val + eps), 80)
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
            for idx, (model, values) in enumerate(grouped):
                clipped = np.clip(values.to_numpy(dtype=float), min_for_bins, None)
                color = color_cycle[idx % len(color_cycle)]
                ax_time.hist(
                    clipped,
                    bins=bins,
                    alpha=0.45,
                    color=color,
                    edgecolor=color,
                    linewidth=0.8,
                    label=model,
                )
                mean_elapsed = float(np.mean(clipped))
                ax_time.axvline(
                    max(mean_elapsed, min_for_bins),
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.9,
                )
            use_log_scale = True
    if use_log_scale:
        ax_time.set_xscale("log")
    ax_time.set_xlabel("解計算時間 (秒)")
    ax_time.set_ylabel("頻度")
    ax_time.set_title("Flex ソルバー計算時間の分布")
    ax_time.legend(loc="best", fontsize=8)

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
    cvar_95 = compute_cvar(returns, alpha=0.05)
    sortino_step = compute_sortino_ratio(returns)
    sortino = (
        float(sortino_step) * math.sqrt(steps_per_year)
        if np.isfinite(sortino_step)
        else np.nan
    )
    label = f"benchmark_{ticker}"
    terminal_wealth = float(wealth_list[-1])
    total_return = terminal_wealth - 1.0
    stats = {
        "model": label,
        "n_retrain": int(len(returns)),
        "n_invest_steps": int(len(returns)),
        "ann_return": mean_return,
        "ann_volatility": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_return_net": mean_return,
        "ann_volatility_net": std_return,
        "sharpe_net": sharpe,
        "sortino_net": sortino,
        "cvar_95": cvar_95,
        "max_drawdown": max_drawdown(wealth_list),
        "terminal_wealth": terminal_wealth,
        "terminal_wealth_net": terminal_wealth,
        "total_return": total_return,
        "total_return_net": total_return,
        "train_window": 0,
        "rebal_interval": 0,
        "avg_turnover": 0.0,
        "avg_trading_cost": 0.0,
        "trading_cost_bps": 0.0,
        "avg_condition_number": float("nan"),
    }
    return {"label": label, "wealth_df": wealth_df, "stats": stats}


def plot_solver_summary_bars(rebalance_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot solver warning counts and mean elapsed time for key models.

    Focuses on flex_dual, flex_kkt, ipo_grad, and spo_plus. For flex models, a warning
    is counted when solver_status is not 'optimal'/'ok'. For ipo_grad, a warning
    is counted when the recorded maximum constraint violation exceeds a small
    tolerance. For spo_plus, a warning is counted when the oracle reports any
    failures/fallbacks.
    """
    if plt is None:
        return
    if rebalance_df.empty or "model" not in rebalance_df.columns:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    df = rebalance_df.copy()

    def _base_model(name: object) -> Optional[str]:
        s = str(name)
        if "flex_dual" in s:
            return "flex_dual"
        if "flex_kkt" in s:
            return "flex_kkt"
        if s == "ipo_grad":
            return "ipo_grad"
        if s == "spo_plus":
            return "spo_plus"
        return None

    df["model_base"] = df["model"].map(_base_model)
    df = df[df["model_base"].notna()].copy()
    if df.empty:
        return

    # Define warnings
    def _is_ok(status: object) -> bool:
        s = str(status).lower()
        return ("optimal" in s) or ("ok" in s)

    df["is_warning"] = False
    mask_flex = df["model_base"].isin(["flex_dual", "flex_kkt"])
    df.loc[mask_flex, "is_warning"] = ~df.loc[mask_flex, "solver_status"].map(_is_ok)

    mask_ipo = df["model_base"] == "ipo_grad"
    if mask_ipo.any():
        eq = df.loc[mask_ipo, "train_eq_viol_max"].astype(float).abs()
        ineq = df.loc[mask_ipo, "train_ineq_viol_max"].astype(float).abs()
        # Treat violations above small tolerances as warnings.
        eq_tol = 1e-4
        ineq_tol = 1e-8
        df.loc[mask_ipo, "is_warning"] = (eq > eq_tol) | (ineq > ineq_tol)

    mask_spo = df["model_base"] == "spo_plus"
    if mask_spo.any():
        fail_true = df.loc[mask_spo, "spo_oracle_fail_true"].astype(float).fillna(0.0)
        fail_tilde = df.loc[mask_spo, "spo_oracle_fail_tilde"].astype(float).fillna(0.0)
        fallback_tilde = df.loc[mask_spo, "spo_oracle_fallback_tilde"].astype(float).fillna(0.0)
        df.loc[mask_spo, "is_warning"] = (fail_true > 0) | (fail_tilde > 0) | (fallback_tilde > 0)

    # Aggregate statistics
    warn_counts = df.groupby("model_base")["is_warning"].sum()
    total_counts = df.groupby("model_base")["is_warning"].count()
    elapsed_mean = df.groupby("model_base")["elapsed_sec"].mean()

    models = ["flex_dual", "flex_kkt", "ipo_grad", "spo_plus"]
    display_names = {
        "flex_dual": "DFL-QCQP-dual",
        "flex_kkt": "DFL-QCQP-kkt",
        "ipo_grad": "IPO-GRAD",
        "spo_plus": "SPO+",
    }

    # Warning counts bar chart
    values = [float(warn_counts.get(m, 0.0)) for m in models]
    totals = [int(total_counts.get(m, 0)) for m in models]
    if any(totals):
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(models))
        colors = [MODEL_COLOR_MAP.get(m, "tab:gray") for m in models]
        bars = ax.bar(x, values, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([display_names[m] for m in models], rotation=30, ha="right")
        ax.set_ylabel("警告件数")
        ax.set_title("ソルバー警告件数（flex / IPO-GRAD / SPO+）")
        for xi, bar, val, tot in zip(x, bars, values, totals):
            ax.text(
                xi,
                bar.get_height() + 0.05,
                f"{int(val)}/{tot}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(out_dir / "solver_warning_counts.png")
        plt.close(fig)

    # Elapsed time distribution (boxplot)
    elapsed_data: List[np.ndarray] = []
    for m in models:
        sub = pd.to_numeric(df.loc[df["model_base"] == m, "elapsed_sec"], errors="coerce").dropna()
        # For log-scale, remove non-positive values.
        sub = sub[sub > 0.0]
        if sub.empty:
            elapsed_data.append(np.asarray([np.nan], dtype=float))
        else:
            elapsed_data.append(sub.to_numpy(dtype=float))
    if any(arr.size for arr in elapsed_data):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        colors = [MODEL_COLOR_MAP.get(m, "tab:gray") for m in models]
        box = ax.boxplot(
            elapsed_data,
            tick_labels=[display_names[m] for m in models],
            patch_artist=True,
            showfliers=True,
            widths=0.55,
        )
        for patch, c in zip(box.get("boxes", []), colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for whisker in box.get("whiskers", []):
            whisker.set_color("#444444")
        for cap in box.get("caps", []):
            cap.set_color("#444444")
        for median in box.get("medians", []):
            median.set_color("#111111")
            median.set_linewidth(1.6)
        for flier in box.get("fliers", []):
            flier.set_markeredgecolor("#666666")
            flier.set_alpha(0.6)

        ax.set_yscale("log")
        ax.set_ylabel("1サイクルあたり解計算時間 (秒) [log]")
        ax.set_title("ソルバー計算時間の分布（flex / IPO-GRAD / SPO+）")
        ax.grid(axis="y", alpha=0.2)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        # Keep filename for backwards compatibility.
        fig.savefig(out_dir / "solver_elapsed_mean.png")
        plt.close(fig)


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
    cvar_95 = compute_cvar(returns, alpha=0.05)
    sortino_step = compute_sortino_ratio(returns)
    sortino = (
        float(sortino_step) * math.sqrt(steps_per_year)
        if np.isfinite(sortino_step)
        else np.nan
    )
    label = "benchmark_equal_weight"
    terminal_wealth = float(wealth_list[-1])
    total_return = terminal_wealth - 1.0
    stats = {
        # 表示名としては 1/N を使う
        "model": "1/N",
        "n_retrain": int(len(returns)),
        "n_invest_steps": int(len(returns)),
        "ann_return": mean_return,
        "ann_volatility": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_return_net": mean_return,
        "ann_volatility_net": std_return,
        "sharpe_net": sharpe,
        "sortino_net": sortino,
        "cvar_95": cvar_95,
        "max_drawdown": max_drawdown(wealth_list),
        "terminal_wealth": terminal_wealth,
        "terminal_wealth_net": terminal_wealth,
        "total_return": total_return,
        "total_return_net": total_return,
        "train_window": 0,
        "rebal_interval": 0,
        "avg_turnover": 0.0,
        "avg_trading_cost": 0.0,
        "trading_cost_bps": 0.0,
        "avg_condition_number": float("nan"),
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


def export_max_return_winner_counts(
    returns_df: pd.DataFrame,
    csv_path: Path,
    fig_path: Path,
    *,
    mse_fig_path: Path | None = None,
) -> None:
    """Count how often each asset achieves the maximum return per date.

    For each row in ``returns_df`` (one date), we find the column with the
    largest return, and count how often each asset is the winner. Rows where all
    returns are NaN are ignored. The counts are exported as CSV and visualized
    as a bar chart.
    """
    if returns_df.empty:
        return
    df = returns_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    # ドローダウンと同様、全 NaN 行は無視
    df = df.dropna(how="all")
    if df.empty:
        return
    # 日付ごとに最大リターンとなった列名を取得
    winners = df.idxmax(axis=1)
    counts = winners.value_counts()
    # 元の並び（列順）を維持した順序で並べる
    ordered_index = [c for c in df.columns if c in counts.index]
    counts = counts.reindex(ordered_index, fill_value=0)
    counts.to_csv(csv_path, header=["max_ret_wins"])

    if plt is None:
        return
    plt.figure(figsize=(8, 4))
    x = np.arange(len(counts.index))
    labels = list(counts.index)
    values = counts.to_numpy(dtype=float)
    colors = [ASSET_COLOR_SEQUENCE[i % len(ASSET_COLOR_SEQUENCE)] for i in range(len(labels))]
    hatches = ["", "//", "\\\\", "++", "xx", "..", "**"]
    hatch_list = [hatches[i % len(hatches)] for i in range(len(labels))]
    for xi, val, color, hatch in zip(x, values, colors, hatch_list):
        plt.bar(
            xi,
            val,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
        )
        plt.text(xi, val + max(values) * 0.01, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("最大リターン獲得日数")
    plt.title("各資産がその日の最高リターンとなった回数")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    # 予測誤差（MSE）を横軸にした散布図を追加（asset_predictions を集計）
    if mse_fig_path and plt is not None:
        try:
            asset_pred_dir = mse_fig_path.parent.parent / "csv" / "asset_predictions"
            if asset_pred_dir.exists():
                dfs: list[pd.DataFrame] = []
                for csv_file in asset_pred_dir.glob("*.csv"):
                    df_pred = pd.read_csv(csv_file)
                    if {"pred_ret", "real_ret", "model", "ticker"}.issubset(df_pred.columns):
                        dfs.append(df_pred[["model", "ticker", "pred_ret", "real_ret"]])
                if dfs:
                    big = pd.concat(dfs, ignore_index=True)
                    big["pred_ret"] = pd.to_numeric(big["pred_ret"], errors="coerce")
                    big["real_ret"] = pd.to_numeric(big["real_ret"], errors="coerce")
                    big = big.dropna(subset=["pred_ret", "real_ret", "ticker"])
                    mse_df = (
                        big.groupby("ticker")
                        .apply(lambda g: float(np.mean((g["pred_ret"] - g["real_ret"]) ** 2)))
                        .reset_index(name="mse")
                    )
                    plt.figure(figsize=(8, 4))
                    plt.bar(
                        mse_df["ticker"],
                        mse_df["mse"],
                        color=[ASSET_COLOR_SEQUENCE[i % len(ASSET_COLOR_SEQUENCE)] for i in range(len(mse_df))],
                        edgecolor="black",
                        linewidth=0.6,
                        alpha=0.85,
                    )
                    for xi, (_, row) in enumerate(mse_df.iterrows()):
                        plt.text(xi, row["mse"] * 1.01, f"{row['mse']:.4f}", ha="center", va="bottom", fontsize=9)
                    plt.xlabel("ティッカー")
                    plt.ylabel("MSE")
                    plt.title("資産別予測誤差 (MSE)")
                    plt.tight_layout()
                    plt.savefig(mse_fig_path)
                    plt.close()
        except Exception:
            pass


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
        # CVaR (Expected Shortfall) at 95% (lower 5% mean return; higher is better).
        cvar_95 = compute_cvar(r, alpha=0.05)
        final_wealth = float(np.prod(1.0 + r))
        return {
            "mean": mean,
            "std": std,
            "sharpe": sharpe,
            "sortino": sortino,
            "cvar_95": cvar_95,
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
            for k in ["mean", "std", "sharpe", "sortino", "cvar_95", "final_wealth"]:
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
    出力: 3-performance_significance_dfl_summary.csv
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

    # 比較対象とする指標
    metrics = ["mean", "std", "sharpe", "sortino", "cvar_95", "final_wealth"]

    rows_out: list[dict[str, object]] = []
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
        # 出力は True/False の有意フラグのみ（値や p 値は不要）
        keep_cols: list[str] = ["model_dfl", "model_other"]
        for k in metrics:
            keep_cols.append(f"{k}_sig_5pct")
            keep_cols.append(f"{k}_sig_1pct")
        keep_cols = [c for c in keep_cols if c in out_df.columns]
        out_df = out_df[keep_cols]

        # 列名を日本語にして、CSV 単体でも解釈しやすくする
        metric_name_ja = {
            "mean": "年率リターン",
            "std": "年率ボラティリティ",
            "sharpe": "シャープ",
            "sortino": "ソルティノ",
            "cvar_95": "CVaR(95%)",
            "final_wealth": "最終資産",
        }
        rename_map: Dict[str, str] = {
            "model_dfl": "DFLモデル",
            "model_other": "比較モデル",
        }
        for k in metrics:
            ja = metric_name_ja.get(k, k)
            col_5 = f"{k}_sig_5pct"
            col_1 = f"{k}_sig_1pct"
            if col_5 in out_df.columns:
                rename_map[col_5] = f"{ja} 有意差(5%)"
            if col_1 in out_df.columns:
                rename_map[col_1] = f"{ja} 有意差(1%)"
        out_df = out_df.rename(columns=rename_map)

        out_df.to_csv(
            Path(analysis_csv_dir) / "3-performance_significance_dfl_summary.csv",
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

    # モデルごとの色は共通マップに合わせる
    bar_colors = [MODEL_COLOR_MAP.get(str(m).lower(), MODEL_COLOR_MAP.get(str(m), f"C{i%10}")) for i, m in enumerate(models)]

    # Neff
    ax = axes[0]
    ax.bar(x, conc_df["N_eff_mean"], color=bar_colors)
    ax.set_ylabel("有効資産数 (平均)")
    ax.set_title("Neff")

    # max weight
    ax = axes[1]
    ax.bar(x, conc_df["max_w_mean"], color=bar_colors)
    ax.set_ylabel("最大ウェイト (平均)")
    ax.set_title("最大ウェイト")

    # cap frequency
    ax = axes[2]
    ax.bar(x, conc_df["cap_hit_freq"], color=bar_colors)
    ax.set_ylabel("最大ウェイトが0.95以上となる頻度")
    ax.set_title("ウェイト集中度の頻度")

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
    # Sharpe from 1-summary.csv (gross) and 1-summary_cost.csv (net, if available)
    summary_path = Path(analysis_csv_dir) / "1-summary.csv"
    summary_cost_path = Path(analysis_csv_dir) / "1-summary_cost.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if "model" not in summary.columns:
        return
    summary = summary.set_index("model")
    summary_cost: Optional[pd.DataFrame] = None
    if summary_cost_path.exists():
        try:
            tmp = pd.read_csv(summary_cost_path)
            if "model" in tmp.columns:
                summary_cost = tmp.set_index("model")
        except Exception:
            summary_cost = None
    # left join にして、1/N など「MSE を持たない」ベンチマークも残す
    plot_df = summary.join(mse_df, how="left")
    plot_df_cost = (
        summary_cost.join(mse_df, how="left")
        if isinstance(summary_cost, pd.DataFrame)
        else None
    )
    # RMSE: if not already present in 1-summary.csv, derive from MSE.
    if "rmse" not in plot_df.columns and "mse" in plot_df.columns:
        plot_df["rmse"] = pd.to_numeric(plot_df["mse"], errors="coerce").map(
            lambda v: float(np.sqrt(v)) if np.isfinite(v) and v >= 0.0 else np.nan
        )
    if plot_df.empty or plt is None:
        return

    # バイアス関連図の保存先ディレクトリ
    bias_fig_dir = analysis_fig_dir / "bias_analysis"
    bias_fig_dir.mkdir(parents=True, exist_ok=True)

    def _display_color_map() -> Dict[str, str]:
        # Prefer a display-name keyed palette so downstream plots stay consistent even
        # when DataFrames already contain display_model_name(model).
        out: Dict[str, str] = {}
        for internal_key, color in MODEL_COLOR_MAP.items():
            out.setdefault(display_model_name(internal_key), color)
            out.setdefault(str(internal_key), color)
        # Common display-name aliases
        out.setdefault("IPO-analytic", MODEL_COLOR_MAP.get("ipo", "tab:orange"))
        out.setdefault("IPO-GRAD", MODEL_COLOR_MAP.get("ipo_grad", "tab:brown"))
        out.setdefault("SPO+", MODEL_COLOR_MAP.get("spo_plus", "tab:cyan"))
        out.setdefault("OLS", MODEL_COLOR_MAP.get("ols", "tab:blue"))
        out.setdefault("DFL-QCQP-dual", MODEL_COLOR_MAP.get("flex_dual", "tab:green"))
        out.setdefault("DFL-QCQP-kkt", MODEL_COLOR_MAP.get("flex_kkt", "tab:red"))
        out.setdefault("DFL-QCQP-ens", MODEL_COLOR_MAP.get("flex_dual_kkt_ens", "black"))
        return out

    def _resolve_label_overlaps(fig, ax, labels, *, max_iter: int = 120) -> None:
        """Heuristic label repulsion in screen space (no external deps)."""
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax_bbox = ax.get_window_extent(renderer)
        except Exception:  # pragma: no cover
            return
        px_to_pt = 72.0 / float(fig.dpi or 100.0)

        for _ in range(int(max_iter)):
            moved = False
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
            except Exception:  # pragma: no cover
                break
            bboxes = [t.get_window_extent(renderer).expanded(1.05, 1.15) for t in labels]
            centers = [np.array([(bb.x0 + bb.x1) / 2.0, (bb.y0 + bb.y1) / 2.0]) for bb in bboxes]
            for i in range(len(labels)):
                for j in range(i):
                    if not bboxes[i].overlaps(bboxes[j]):
                        continue
                    v = centers[i] - centers[j]
                    if not np.isfinite(v).all() or float(np.linalg.norm(v)) < 1e-9:
                        v = np.array([1.0, 0.0])
                    v = v / float(np.linalg.norm(v))
                    # Move in opposite directions (points units via px->pt)
                    step_pt = 6.0
                    delta_pt = v * (step_pt)
                    pi = np.array(labels[i].get_position(), dtype=float)
                    pj = np.array(labels[j].get_position(), dtype=float)
                    labels[i].set_position(tuple(pi + delta_pt))
                    labels[j].set_position(tuple(pj - delta_pt))
                    moved = True

            # Keep labels inside axes bounds (soft clamp)
            if moved:
                try:
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    for t in labels:
                        bb = t.get_window_extent(renderer)
                        dx = 0.0
                        dy = 0.0
                        if bb.x0 < ax_bbox.x0:
                            dx += (ax_bbox.x0 - bb.x0) * px_to_pt
                        if bb.x1 > ax_bbox.x1:
                            dx -= (bb.x1 - ax_bbox.x1) * px_to_pt
                        if bb.y0 < ax_bbox.y0:
                            dy += (ax_bbox.y0 - bb.y0) * px_to_pt
                        if bb.y1 > ax_bbox.y1:
                            dy -= (bb.y1 - ax_bbox.y1) * px_to_pt
                        if dx != 0.0 or dy != 0.0:
                            p = np.array(t.get_position(), dtype=float)
                            t.set_position(tuple(p + np.array([dx, dy])))
                except Exception:  # pragma: no cover
                    pass

            if not moved:
                break

    def _should_skip_scatter(mstr: str) -> bool:
        # ベンチマーク系は散布図から除外（SPY, 1/N）
        return (
            ("SPY" in mstr and "benchmark" in mstr)
            or mstr in {"1/N", "[SPY]", "benchmark_SPY", "Buy&Hold SPY", "Buy&Hold(SPY)"}
        )

    # R^2 vs Sharpe scatter (summary の R^2 を使用)
    if plt is not None:
        palette = _display_color_map()
        label_offsets = [
            (6, 6),
            (6, -10),
            (-10, 6),
            (-10, -10),
            (10, 0),
            (0, 10),
            (-12, 0),
            (0, -12),
        ]
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        labels: List[Any] = []
        for model, row in plot_df.iterrows():
            mstr = str(model)
            if _should_skip_scatter(mstr):
                continue
            x = row.get("r2", np.nan)
            if not np.isfinite(x):
                continue
            y = row.get("sharpe", np.nan)
            if not np.isfinite(y):
                continue
            color = palette.get(mstr, "tab:blue")
            ax.scatter(x, y, color=color, s=46, edgecolors="white", linewidths=0.8, zorder=3)
            dx, dy = label_offsets[len(labels) % len(label_offsets)]
            labels.append(
                ax.annotate(
                    mstr,
                    xy=(x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
            )
        _resolve_label_overlaps(fig, ax, labels)
        ax.set_xlabel("アウトオブサンプル $R^2$（全観測: 資産×時点）")
        ax.set_ylabel("シャープレシオ（年率換算）")
        ax.set_title("予測精度と意思決定品質の関係")
        fig.tight_layout()
        fig.savefig(bias_fig_dir / "r2_vs_sharpe_scatter.png")
        plt.close(fig)

        # RMSE vs Sharpe scatter (asset_predictions 由来の MSE を sqrt して使用)
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        labels = []
        for model, row in plot_df.iterrows():
            mstr = str(model)
            if _should_skip_scatter(mstr):
                continue
            x = row.get("rmse", np.nan)
            if not np.isfinite(x):
                continue
            y = row.get("sharpe", np.nan)
            if not np.isfinite(y):
                continue
            color = palette.get(mstr, "tab:blue")
            ax.scatter(x, y, color=color, s=46, edgecolors="white", linewidths=0.8, zorder=3)
            dx, dy = label_offsets[len(labels) % len(label_offsets)]
            labels.append(
                ax.annotate(
                    mstr,
                    xy=(x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
            )
        _resolve_label_overlaps(fig, ax, labels)
        ax.set_xlabel("アウトオブサンプル RMSE（全観測: 資産×時点）")
        ax.set_ylabel("シャープレシオ（年率換算）")
        ax.set_title("予測誤差と意思決定品質の関係")
        fig.tight_layout()
        fig.savefig(bias_fig_dir / "rmse_vs_sharpe_scatter.png")
        plt.close(fig)

        # Turnover vs Sharpe/CVaR scatter (decision quality vs trading intensity)
        def _maybe_float(x: object) -> float:
            try:
                v = float(x)
                return v
            except Exception:
                return float("nan")

        if "avg_turnover" in plot_df.columns:
            # 1-summary.csv の avg_turnover は format_summary_for_output で % 表示（100倍）済み。
            use_cost_view = isinstance(plot_df_cost, pd.DataFrame)
            use_df = plot_df_cost if use_cost_view else plot_df
            sharpe_col = "sharpe_net" if "sharpe_net" in use_df.columns else "sharpe"
            cvar_col = "cvar_95_net" if "cvar_95_net" in use_df.columns else "cvar_95"

            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            labels = []
            for model, row in use_df.iterrows():
                mstr = str(model)
                if _should_skip_scatter(mstr):
                    continue
                x = _maybe_float(row.get("avg_turnover", np.nan))
                y = _maybe_float(row.get(sharpe_col, np.nan))
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                color = palette.get(mstr, "tab:blue")
                ax.scatter(x, y, color=color, s=46, edgecolors="white", linewidths=0.8, zorder=3)
                dx, dy = label_offsets[len(labels) % len(label_offsets)]
                labels.append(
                    ax.annotate(
                        mstr,
                        xy=(x, y),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                    )
                )
            _resolve_label_overlaps(fig, ax, labels)
            ax.set_xlabel("平均ターンオーバー（%）")
            ax.set_ylabel(
                "シャープレシオ（取引コスト控除後, 年率換算）"
                if use_cost_view
                else "シャープレシオ（年率換算）"
            )
            ax.set_title(
                "売買量とリスク調整後リターンの関係（取引コスト控除後）"
                if use_cost_view
                else "売買量とリスク調整後リターンの関係"
            )
            fig.tight_layout()
            fig.savefig(bias_fig_dir / "turnover_vs_sharpe_scatter.png")
            plt.close(fig)

            if cvar_col in use_df.columns:
                fig, ax = plt.subplots(figsize=(7.2, 5.2))
                labels = []
                for model, row in use_df.iterrows():
                    mstr = str(model)
                    if _should_skip_scatter(mstr):
                        continue
                    x = _maybe_float(row.get("avg_turnover", np.nan))
                    y = _maybe_float(row.get(cvar_col, np.nan))
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    color = palette.get(mstr, "tab:blue")
                    ax.scatter(x, y, color=color, s=46, edgecolors="white", linewidths=0.8, zorder=3)
                    dx, dy = label_offsets[len(labels) % len(label_offsets)]
                    labels.append(
                        ax.annotate(
                            mstr,
                            xy=(x, y),
                            xytext=(dx, dy),
                            textcoords="offset points",
                            fontsize=8,
                            color=color,
                        )
                    )
                _resolve_label_overlaps(fig, ax, labels)
                ax.set_xlabel("平均ターンオーバー（%）")
                ax.set_ylabel(
                    "CVaR(95%)（取引コスト控除後, %）"
                    if use_cost_view
                    else "CVaR(95%)（%）"
                )
                ax.set_title(
                    "売買量と下方リスク（CVaR）の関係（取引コスト控除後）"
                    if use_cost_view
                    else "売買量と下方リスク（CVaR）の関係"
                )
                fig.tight_layout()
                fig.savefig(bias_fig_dir / "turnover_vs_cvar_scatter.png")
                plt.close(fig)

        # RMSE vs CVaR scatter (predictive fit ≠ decision quality)
        if "rmse" in plot_df.columns and "cvar_95" in plot_df.columns:
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            labels = []
            for model, row in plot_df.iterrows():
                mstr = str(model)
                if _should_skip_scatter(mstr):
                    continue
                x = _maybe_float(row.get("rmse", np.nan))
                y = _maybe_float(row.get("cvar_95", np.nan))
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                color = palette.get(mstr, "tab:blue")
                ax.scatter(x, y, color=color, s=46, edgecolors="white", linewidths=0.8, zorder=3)
                dx, dy = label_offsets[len(labels) % len(label_offsets)]
                labels.append(
                    ax.annotate(
                        mstr,
                        xy=(x, y),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                    )
                )
            _resolve_label_overlaps(fig, ax, labels)
            ax.set_xlabel("アウトオブサンプル RMSE（全観測: 資産×時点）")
            ax.set_ylabel("CVaR(95%)（%）")
            ax.set_title("予測誤差と下方リスク（CVaR）の関係")
            fig.tight_layout()
            fig.savefig(bias_fig_dir / "rmse_vs_cvar_scatter.png")
            plt.close(fig)

        # Decision Sensitivity Plot:
        # x = |predicted return|, y = |portfolio contribution| = |w * realized return|
        # This is per-(model,ticker,date) point cloud.
        try:
            needed = {"pred_ret", "real_ret", "weight", "model"}
            if needed.issubset(df.columns):
                dd = df.copy()
                dd["pred_ret"] = pd.to_numeric(dd["pred_ret"], errors="coerce")
                dd["real_ret"] = pd.to_numeric(dd["real_ret"], errors="coerce")
                dd["weight"] = pd.to_numeric(dd["weight"], errors="coerce")
                dd = dd.dropna(subset=["pred_ret", "real_ret", "weight", "model"])
                if not dd.empty:
                    dd["abs_pred"] = dd["pred_ret"].abs()
                    dd["abs_contrib"] = (dd["weight"] * dd["real_ret"]).abs()

                    fig, ax = plt.subplots(figsize=(7.6, 5.4))
                    # Cap points per model for readability.
                    max_points_per_model = 8000
                    for model_name, sub in dd.groupby("model"):
                        mstr = str(model_name)
                        if _should_skip_scatter(mstr):
                            continue
                        if len(sub) > max_points_per_model:
                            sub = sub.sample(n=max_points_per_model, random_state=0)
                        color = palette.get(mstr, "tab:blue")
                        ax.scatter(
                            sub["abs_pred"].to_numpy(),
                            sub["abs_contrib"].to_numpy(),
                            s=8,
                            alpha=0.18,
                            color=color,
                            edgecolors="none",
                            label=mstr,
                        )
                    ax.set_xlabel(r"$|\\hat r|$（予測リターンの絶対値）")
                    ax.set_ylabel(r"$|w\\times r|$（寄与の絶対値）")
                    ax.set_title("意思決定感度プロット（予測の強さに対する反応）")
                    ax.grid(alpha=0.2)
                    # Keep legend compact: sort and limit columns
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(
                            handles,
                            labels,
                            loc="upper right",
                            frameon=True,
                            fontsize=8,
                            ncol=1,
                        )
                    fig.tight_layout()
                    fig.savefig(bias_fig_dir / "decision_sensitivity_scatter.png")
                    plt.close(fig)
        except Exception:
            pass

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

    if plt is not None:
        try:
            plt.figure(figsize=(8, 5))
            # 各モデルごとに Down（赤）/Up（緑）の箱ひげを並べる
            models_order = sorted(df["model"].unique(), key=lambda m: PREFERRED_MODEL_ORDER.index(m) if m in PREFERRED_MODEL_ORDER else len(PREFERRED_MODEL_ORDER))
            positions: List[float] = []
            data: List[np.ndarray] = []
            colors: List[str] = []
            width = 0.15
            x_base = np.arange(len(models_order))
            for i, m in enumerate(models_order):
                sub_m = df[df["model"] == m]
                down_bias = sub_m[sub_m["updown"] == "Down"]["bias"].to_numpy()
                up_bias = sub_m[sub_m["updown"] == "Up"]["bias"].to_numpy()
                if down_bias.size:
                    positions.append(x_base[i] - width)
                    data.append(down_bias)
                    colors.append("lightcoral")
                if up_bias.size:
                    positions.append(x_base[i] + width)
                    data.append(up_bias)
                    colors.append("lightseagreen")
            if data:
                box = plt.boxplot(
                    data,
                    positions=positions,
                    widths=width * 1.6,
                    patch_artist=True,
                )
                for patch, c in zip(box["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.8)
                # median の色を見やすい濃いグレーに
                for med in box["medians"]:
                    med.set_color("dimgray")
                    med.set_linewidth(1.5)
            # x 軸はモデル名のみ表示
            display_labels = [display_model_name(m) for m in models_order]
            plt.xticks(x_base, display_labels, rotation=45, ha="right")
            # 色の意味は凡例で示す
            from matplotlib.patches import Patch  # type: ignore
            legend_handles = [
                Patch(facecolor="lightcoral", alpha=0.8, label="Down"),
                Patch(facecolor="lightseagreen", alpha=0.8, label="Up"),
            ]
            plt.legend(handles=legend_handles, loc="upper right")
            plt.axhline(0.0, linestyle="--", linewidth=1, color="black")
            plt.ylabel("バイアス = 予測リターン - 実現リターン")
            plt.title("Up/Down別の予測バイアス")
            # 差分は正負を取り得るため symlog を使って対数スケール化（0 近傍は線形）
            # 併せて表示範囲は固定して比較しやすくする。
            y_min, y_max = -0.1, 0.1
            plt.yscale("symlog", linthresh=1e-2)
            plt.ylim(y_min, y_max)
            if data:
                # 各箱の中央値（線）の値を表示（混み合う場合があるので小さめに）
                offset = (y_max - y_min) * 0.02
                for med in box.get("medians", []):
                    x = float(np.mean(med.get_xdata()))
                    y = float(np.mean(med.get_ydata()))
                    if not np.isfinite(y):
                        continue
                    plt.text(
                        x,
                        min(y + offset, y_max - offset),
                        f"{y:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="dimgray",
                        rotation=90,
                    )
            plt.tight_layout()
            plt.savefig(bias_fig_dir / "updown_bias_boxplot.png")
        except Exception as exc:  # pragma: no cover - 図生成失敗時も後続解析は続行
            print(f"[analysis] updown bias boxplot failed: {exc}")
        finally:
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

    if plt is not None:
        try:
            plt.figure(figsize=(8, 5))
            # 各モデルごとに OUT/IN の箱ひげを並べる（summary と同じモデル順）
            models_order = sorted(
                df["model"].unique(),
                key=lambda m: PREFERRED_MODEL_ORDER.index(m)
                if m in PREFERRED_MODEL_ORDER
                else len(PREFERRED_MODEL_ORDER),
            )
            positions: List[float] = []
            data: List[np.ndarray] = []
            colors: List[str] = []
            width = 0.15
            x_base = np.arange(len(models_order))
            for i, m in enumerate(models_order):
                sub_m = df[df["model"] == m]
                out_bias = sub_m[sub_m["inout"] == "OUT"]["bias"].to_numpy()
                in_bias = sub_m[sub_m["inout"] == "IN"]["bias"].to_numpy()
                if out_bias.size:
                    positions.append(x_base[i] - width)
                    data.append(out_bias)
                    colors.append("lightcoral")
                if in_bias.size:
                    positions.append(x_base[i] + width)
                    data.append(in_bias)
                    colors.append("lightseagreen")
            if data:
                box = plt.boxplot(
                    data,
                    positions=positions,
                    widths=width * 1.6,
                    patch_artist=True,
                )
                for patch, c in zip(box["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.8)
                # 中央値線を見やすいダークグレーに
                for med in box["medians"]:
                    med.set_color("dimgray")
                    med.set_linewidth(1.5)
            display_labels = [display_model_name(m) for m in models_order]
            plt.xticks(x_base, display_labels, rotation=45, ha="right")
            from matplotlib.patches import Patch  # type: ignore
            legend_handles = [
                Patch(facecolor="lightcoral", alpha=0.8, label="OUT（低ウェイト）"),
                Patch(facecolor="lightseagreen", alpha=0.8, label="IN（高ウェイト）"),
            ]
            plt.legend(handles=legend_handles, loc="upper right")
            plt.axhline(0.0, linestyle="--", linewidth=1, color="black")
            plt.ylabel("バイアス = 予測リターン - 実現リターン")
            plt.title("IN/OUT（ウェイト大小別）の予測バイアス")
            # 差分は正負を取り得るため symlog を使って対数スケール化（0 近傍は線形）
            # 併せて表示範囲は固定して比較しやすくする。
            y_min, y_max = -0.1, 0.1
            plt.yscale("symlog", linthresh=1e-2)
            plt.ylim(y_min, y_max)
            if data:
                offset = (y_max - y_min) * 0.02
                for med in box.get("medians", []):
                    x = float(np.mean(med.get_xdata()))
                    y = float(np.mean(med.get_ydata()))
                    if not np.isfinite(y):
                        continue
                    plt.text(
                        x,
                        min(y + offset, y_max - offset),
                        f"{y:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="dimgray",
                        rotation=90,
                    )
            plt.tight_layout()
            plt.savefig(bias_fig_dir / "inout_bias_boxplot.png")
        except Exception as exc:  # pragma: no cover
            print(f"[analysis] inout bias boxplot failed: {exc}")
        finally:
            plt.close()

    # ---- flex 系モデルに対するヒストグラム（Up/Down, IN/OUT） ----
    flex_models = [m for m in df["model"].unique() if "DFL-QCQP" in str(m)]
    if not flex_models:
        # まだ display_model_name を通していない実験では "flex" ラベルを拾う
        flex_models = [m for m in df["model"].unique() if "flex" in str(m)]
    if plt is not None and flex_models:
        # 共通のビンを決める（全体の最小値〜最大値）
        bias_min = float(df["bias"].min())
        bias_max = float(df["bias"].max())
        if not np.isfinite(bias_min) or not np.isfinite(bias_max) or bias_min == bias_max:
            bias_min, bias_max = -0.1, 0.1
        # 分布形状をなめらかに見るため、ビン数はやや多めに設定
        bins = np.linspace(bias_min, bias_max, 80)

        # Up/Down ヒストグラム
        n = len(flex_models)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharex=True, sharey=True)
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
                    label="Down（予測<0）",
                )
            if up.size > 0:
                ax.hist(
                    up,
                    bins=bins,
                    alpha=0.5,
                    density=True,
                    color="green",
                    label="Up（予測>=0）",
                )
            ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
            ax.set_title(model)
            ax.set_xlim(bias_min, bias_max)
            ax.set_yscale("log")
        axes[0].set_ylabel("密度（対数スケール）")
        # 凡例は最初の軸にまとめて表示
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(bias_fig_dir / "updown_bias_hist_flex.png")
        plt.close(fig)

        # IN/OUT ヒストグラム
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharex=True, sharey=True)
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
                    label="OUT（低ウェイト）",
                )
            if inside.size > 0:
                ax.hist(
                    inside,
                    bins=bins,
                    alpha=0.5,
                    density=True,
                    color="green",
                    label="IN（高ウェイト）",
                )
            ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
            ax.set_title(model)
            ax.set_xlim(bias_min, bias_max)
            ax.set_yscale("log")
        axes[0].set_ylabel("密度（対数スケール）")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(bias_fig_dir / "inout_bias_hist_flex.png")
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
    # 将来の解析用に CSV は残すが、図は生成しない
    tb.to_csv(analysis_csv_dir / "topbottom_bias_timeseries.csv", index=False)


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
    try:
        analysis_dir = Path(analysis_fig_dir).parent
        export_analysis_csv_tables_as_png(analysis_csv_dir, analysis_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[analysis] csv table png export failed: {exc}")


def export_dataframe_table_png(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "",
    float_digits: int = 3,
    float_digits_by_col: Optional[Dict[str, int]] = None,
    formatters_by_col: Optional[Dict[str, Any]] = None,
    max_rows: int | None = None,
    max_cols: int | None = None,
    highlight_top_k: int = 0,
    lower_is_better_cols: Sequence[str] = (),
    exclude_highlight_cols: Sequence[str] = (),
    exclude_highlight_rows_by_col: Optional[Dict[str, Sequence[str]]] = None,
    row_id_col: str = "model",
    highlight_truthy_cells: bool = False,
    truthy_cell_color: str = "#b7f7c1",
) -> None:
    """Render a DataFrame as a PNG table for quick visual inspection."""
    if plt is None or df is None:
        return
    if df.empty:
        return
    # Important: reset index so that table row positions line up with numeric ranking indices
    # (callers often filter rows without resetting the index).
    df_disp = df.copy().reset_index(drop=True)
    if max_cols is not None and df_disp.shape[1] > max_cols:
        df_disp = df_disp.iloc[:, :max_cols]
    if max_rows is not None and df_disp.shape[0] > max_rows:
        df_disp = df_disp.iloc[:max_rows, :]

    digits_map = {str(k): int(v) for k, v in (float_digits_by_col or {}).items()}
    fmt_map = {str(k): v for k, v in (formatters_by_col or {}).items()}

    def _fmt_cell(x: object, col: str) -> str:
        custom = fmt_map.get(str(col))
        if custom is not None:
            try:
                return str(custom(x))
            except Exception:
                # Fallback to default formatting
                pass
        use_digits = digits_map.get(str(col), int(float_digits))
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return ""
        if isinstance(x, (np.floating, float)):
            return f"{float(x):.{use_digits}f}"
        if isinstance(x, (np.integer, int)):
            return str(int(x))
        return str(x)

    # Keep numeric copy for highlighting before formatting to strings.
    df_num = df_disp.copy()
    df_str = df_disp.copy()
    for col in df_str.columns:
        df_str[col] = df_str[col].map(lambda v, c=col: _fmt_cell(v, c))
    n_rows, n_cols = df_str.shape
    # Heuristic sizing: aim for readable tables without excessive whitespace.
    fig_w = max(8.0, 1.1 * n_cols + 1.5)
    fig_h = max(2.5, 0.35 * (n_rows + 1) + 1.2)
    font_size = 8 if n_rows <= 25 and n_cols <= 12 else 6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, pad=10)

    table = ax.table(
        cellText=df_str.values.tolist(),
        colLabels=list(df_str.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # Style header + zebra rows
    header_color = "#40466e"
    zebra = ["#f2f2f2", "white"]
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(zebra[(r - 1) % 2])
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.6)

    # Optional highlighting: emphasize binary "True/1" cells (e.g., significance tables).
    if bool(highlight_truthy_cells):
        for col_i, col_name in enumerate(df_disp.columns):
            series = df_disp[col_name]
            values = series.dropna().to_numpy()
            if values.size == 0:
                continue

            def _to_bin(v: object) -> Optional[int]:
                if isinstance(v, (bool, np.bool_)):
                    return 1 if bool(v) else 0
                if isinstance(v, (np.integer, int)):
                    return 1 if int(v) == 1 else 0
                if isinstance(v, (np.floating, float)):
                    if not np.isfinite(float(v)):
                        return None
                    return 1 if float(v) == 1.0 else 0 if float(v) == 0.0 else None
                if isinstance(v, str):
                    t = v.strip().lower()
                    if t in {"true", "1"}:
                        return 1
                    if t in {"false", "0"}:
                        return 0
                return None

            mapped: List[int] = []
            for v in values.tolist():
                b = _to_bin(v)
                if b is None:
                    mapped = []
                    break
                mapped.append(int(b))
            if not mapped:
                continue
            uniq = set(mapped)
            if not uniq.issubset({0, 1}) or (1 not in uniq):
                continue

            # Highlight the truthy cells.
            for row_i, v in enumerate(series.tolist()):
                b = _to_bin(v)
                if b != 1:
                    continue
                r = int(row_i) + 1  # table row offset (0 is header)
                c = int(col_i)
                cell = table.get_celld().get((r, c))
                if cell is None:
                    continue
                cell.set_facecolor(truthy_cell_color)
                cell.set_text_props(weight="bold")

    # Optional highlighting: best-3 per column (direction-aware).
    if int(highlight_top_k) > 0:
        k = int(highlight_top_k)
        lib_cols = {str(c) for c in lower_is_better_cols}
        excl_cols = {str(c) for c in exclude_highlight_cols}
        excl_rows_map: Dict[str, set[str]] = {
            str(col): {str(v) for v in vals}
            for col, vals in (exclude_highlight_rows_by_col or {}).items()
        }
        # best1/best2/best3
        highlight_colors = ["#b7f7c1", "#fff3b0", "#ffd6a5"]

        for col_i, col_name in enumerate(df_num.columns):
            col_key = str(col_name)
            if col_key in excl_cols:
                continue
            # Try numeric conversion.
            series = pd.to_numeric(df_num[col_name], errors="coerce")
            finite = series[np.isfinite(series.to_numpy(dtype=float))]
            if finite.empty:
                continue
            # Optional: exclude certain rows (by row_id_col) from the ranking for this column.
            excluded_rows = excl_rows_map.get(col_key)
            if excluded_rows and row_id_col in df_disp.columns:
                try:
                    mask_excl = df_disp[row_id_col].astype(str).isin(excluded_rows)
                    finite = finite[~mask_excl]
                except Exception:
                    pass
            if finite.empty:
                continue
            if col_key in {"n_retrain", "n_invest_steps"}:
                continue

            # Direction:
            # - default: larger is better
            # - for lower_is_better_cols:
            #     if typical values are negative (loss-like), closer to 0 is better => larger is better
            #     else smaller is better
            if col_key in lib_cols:
                med = float(finite.median())
                ascending = bool(med >= 0.0)
            else:
                ascending = False

            best_idx = (
                finite.sort_values(ascending=ascending)
                .head(k)
                .index.tolist()
            )
            for rank, row_idx in enumerate(best_idx, start=1):
                if rank > len(highlight_colors):
                    color = highlight_colors[-1]
                else:
                    color = highlight_colors[rank - 1]
                r = int(row_idx) + 1  # table row offset (0 is header)
                c = int(col_i)
                cell = table.get_celld().get((r, c))
                if cell is None:
                    continue
                cell.set_facecolor(color)
                if rank == 1:
                    cell.set_text_props(weight="bold")

    table.scale(1.0, 1.25)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_csv_table_png(
    csv_path: Path,
    out_path: Path,
    *,
    title: str = "",
    float_digits: int = 3,
    float_digits_by_col: Optional[Dict[str, int]] = None,
    formatters_by_col: Optional[Dict[str, Any]] = None,
    max_rows: int | None = None,
    max_cols: int | None = None,
    highlight_top_k: int = 0,
    lower_is_better_cols: Sequence[str] = (),
    exclude_highlight_cols: Sequence[str] = (),
    exclude_highlight_rows_by_col: Optional[Dict[str, Sequence[str]]] = None,
    row_id_col: str = "model",
    highlight_truthy_cells: bool = False,
    truthy_cell_color: str = "#b7f7c1",
) -> None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    if title == "":
        title = Path(csv_path).name
    export_dataframe_table_png(
        df,
        out_path,
        title=title,
        float_digits=float_digits,
        float_digits_by_col=float_digits_by_col,
        formatters_by_col=formatters_by_col,
        max_rows=max_rows,
        max_cols=max_cols,
        highlight_top_k=highlight_top_k,
        lower_is_better_cols=lower_is_better_cols,
        exclude_highlight_cols=exclude_highlight_cols,
        exclude_highlight_rows_by_col=exclude_highlight_rows_by_col,
        row_id_col=row_id_col,
        highlight_truthy_cells=highlight_truthy_cells,
        truthy_cell_color=truthy_cell_color,
    )


def export_analysis_csv_tables_as_png(analysis_csv_dir: Path, analysis_dir: Path) -> None:
    """Export key analysis CSVs as PNG tables under `analysis/tables/`."""
    analysis_csv_dir = Path(analysis_csv_dir)
    analysis_dir = Path(analysis_dir)
    out_dir = analysis_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        "1-summary.csv",
        "1-summary_cost.csv",
        "2-experiment_config.csv",
        "3-performance_significance_dfl_summary.csv",
    ]
    # In these metrics, "smaller risk/cost is better".
    lower_is_better = [
        # risk metrics
        "ann_volatility",
        "ann_volatility_net",
        "max_drawdown",
        "cvar_95",
        # cost / trading intensity
        "avg_turnover",
        "avg_trading_cost",
        "total_trading_cost",
        # prediction loss
        "mse",
        "rmse",
    ]
    for name in targets:
        src = analysis_csv_dir / name
        if not src.exists():
            continue
        dst = out_dir / f"{Path(name).stem}.png"
        # Config can be wide; allow more columns. Others are small.
        if name == "2-experiment_config.csv":
            export_csv_table_png(src, dst, title=name, float_digits=6, max_rows=200, max_cols=40)
        else:
            highlight = 3 if name in {"1-summary.csv", "1-summary_cost.csv"} else 0
            truthy = bool(name == "3-performance_significance_dfl_summary.csv")
            # Default: 2 decimals. Some columns need higher precision.
            digits_by_col = {
                "r2": 6,
                "rmse": 6,
                "avg_trading_cost": 6,
                "total_trading_cost": 6,
            }
            import math

            def _fmt_sci_scaled(x: object) -> str:
                try:
                    v = float(x)
                except Exception:
                    return ""
                if not np.isfinite(v):
                    return ""
                if v == 0.0:
                    return "0"
                exp = int(math.floor(math.log10(abs(v))))
                mant = v / (10.0 ** exp)
                return f"{mant:.2f}×10^{exp}"

            formatters_by_col = {"r2": _fmt_sci_scaled}
            # Hide columns that are not helpful for quick visual inspection.
            drop_cols = {"n_retrain", "n_invest_steps", "trading_cost_bps", "avg_trading_cost_mean", "avg_trading_cost_means"}
            try:
                df = pd.read_csv(src)
            except Exception:
                df = None
            if isinstance(df, pd.DataFrame) and not df.empty:
                # For table PNGs, omit the ensemble row to keep figures compact.
                if "model" in df.columns:
                    df = df[df["model"].astype(str) != "DFL-QCQP-ens"]
                if "model_dfl" in df.columns:
                    df = df[df["model_dfl"].astype(str) != "DFL-QCQP-ens"]
                # For significance summaries, also omit comparisons *against* the ensemble
                # (e.g., DFL-QCQP-dual/kkt vs DFL-QCQP-ens) to avoid redundant rows in PNGs.
                if "model_other" in df.columns:
                    df = df[df["model_other"].astype(str) != "DFL-QCQP-ens"]
                if "DFLモデル" in df.columns:
                    df = df[df["DFLモデル"].astype(str) != "DFL-QCQP-ens"]
                if "比較モデル" in df.columns:
                    df = df[df["比較モデル"].astype(str) != "DFL-QCQP-ens"]
                keep = [c for c in df.columns if str(c) not in drop_cols]
                df = df[keep]
                # For 1-summary.csv, hide cost-related columns (use 1-summary_cost.csv for those).
                if name == "1-summary.csv":
                    cost_cols = {"avg_turnover", "avg_trading_cost", "avg_trading_cost_mean", "avg_trading_cost_means"}
                    keep2 = [c for c in df.columns if str(c) not in cost_cols]
                    df = df[keep2]
                # Rename for clarity: avg_trading_cost is total trading cost over the full period.
                if "avg_trading_cost" in df.columns:
                    df = df.rename(columns={"avg_trading_cost": "total_trading_cost"})

                bench_models = {"1/N", "TSMOM (SPY)", "TSMOM(SPY)", "Buy&Hold(SPY)", "Buy&Hold SPY"}
                exclude_rows_by_col = {
                    "avg_turnover": bench_models,
                    "total_trading_cost": bench_models,
                }
                export_dataframe_table_png(
                    df,
                    dst,
                    title=name,
                    float_digits=2,
                    float_digits_by_col=digits_by_col,
                    formatters_by_col=formatters_by_col,
                    max_rows=200,
                    max_cols=40,
                    highlight_top_k=highlight,
                    lower_is_better_cols=lower_is_better,
                    highlight_truthy_cells=truthy,
                    exclude_highlight_rows_by_col=exclude_rows_by_col,
                )
                continue
            export_csv_table_png(
                src,
                dst,
                title=name,
                float_digits=2,
                float_digits_by_col=digits_by_col,
                formatters_by_col=formatters_by_col,
                max_rows=200,
                max_cols=40,
                highlight_top_k=highlight,
                lower_is_better_cols=lower_is_better,
                highlight_truthy_cells=truthy,
                exclude_highlight_rows_by_col={"avg_turnover": {"1/N", "TSMOM (SPY)", "TSMOM(SPY)", "Buy&Hold(SPY)", "Buy&Hold SPY"}, "avg_trading_cost": {"1/N", "TSMOM (SPY)", "TSMOM(SPY)", "Buy&Hold(SPY)", "Buy&Hold SPY"}},
            )


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
    title = "資産リターン相関"
    if stats:
        mean_abs = stats.get("mean_abs_corr")
        if mean_abs is not None and not np.isnan(mean_abs):
            title += f" (平均|ρ|={mean_abs:.2f})"
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
    summary_path = analysis_csv_dir / "1-summary.csv"
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
