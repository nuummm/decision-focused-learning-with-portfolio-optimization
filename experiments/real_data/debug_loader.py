from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import numpy as np

try:  # pragma: no cover - optional plotting dependency
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from data.real_data.loader import MarketLoaderConfig, load_market_dataset

# 実データの確認結果を残す出力フォルダ
DEBUG_ROOT = Path(__file__).resolve().parent / "debug_outputs"


def parse_tickers(value: str) -> List[str]:
    """カンマ区切りのティッカー文字列をクリーンアップ。"""
    return [t.strip().upper() for t in value.split(",") if t.strip()]


def save_table(df: pd.DataFrame, path: Path, *, head_rows: int | None = None) -> None:
    """データフレームを CSV に保存（必要なら先頭数行のみ）。"""
    if head_rows is not None:
        df = df.head(head_rows)
    df.to_csv(path)
    print(f"[real-data-debug] saved table: {path}")


def plot_series(df: pd.DataFrame, path: Path, title: str, vline: Optional[pd.Timestamp] = None) -> None:
    """簡易的な時系列プロットを PNG として保存。"""
    if plt is None:
        print("[real-data-debug] matplotlib 未インストールのためプロットをスキップします。")
        return
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.xlabel("date")
    plt.title(title)
    if vline is not None:
        plt.axvline(vline, color="red", linestyle="--", linewidth=1.0, label="start_date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[real-data-debug] saved plot: {path}")


def plot_scalar_series(
    dates: Sequence[pd.Timestamp],
    values: Sequence[float],
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    if plt is None:
        print(f"[real-data-debug] matplotlib 未インストールのため {title} をスキップします。")
        return
    plt.figure(figsize=(12, 4))
    plt.plot(dates, values, label=ylabel)
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[real-data-debug] saved plot: {path}")


def main() -> None:
    # CLI から期間・ティッカーなどを受け取り、即座に可視化する
    parser = argparse.ArgumentParser(description="Real-data loader debug + visualization helper")
    parser.add_argument("--tickers", type=str, default="SPY,TLT,DBC,BIL", help="Comma-separated ticker list")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="1d", help="Yahoo Finance interval (e.g., 1d,1wk,1mo)")
    parser.add_argument("--price-field", type=str, default="Close", help="Price column to use (Close/Open/etc)")
    parser.add_argument("--return-kind", type=str, default="log", choices=["simple", "log"])
    parser.add_argument("--frequency", type=str, default="weekly", choices=["daily", "weekly"])
    parser.add_argument("--resample-rule", type=str, default="W-FRI")
    parser.add_argument("--momentum-window", type=int, default=52)
    parser.add_argument("--return-horizon", type=int, default=1)
    parser.add_argument("--cov-window", type=int, default=60)
    parser.add_argument("--cov-method", type=str, default="diag", choices=["diag", "ledoit_wolf"])
    parser.add_argument("--cov-shrinkage", type=float, default=0.94)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    parser.add_argument("--no-auto-adjust", action="store_true", help="Disable Yahoo auto-adjust prices")
    parser.add_argument("--force-refresh", action="store_true", help="Force re-download even if cache exists")
    parser.add_argument("--outdir", type=Path, default=None, help="Custom output directory (defaults inside debug_outputs)")
    parser.add_argument("--no-debug", action="store_true", help="Silence loader debug prints")

    args = parser.parse_args()
    tickers = parse_tickers(args.tickers)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # デバッグ出力はタイムスタンプ付きディレクトリにまとめる
    outdir = args.outdir or (DEBUG_ROOT / timestamp)
    visuals_dir = outdir / "visuals"
    outdir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # CLI から受け取った条件でローダ設定を構築
    config = MarketLoaderConfig.for_cli(
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

    dataset = load_market_dataset(config)

    print("[real-data-debug] NaN counts (returns):")
    print(dataset.returns.isna().sum())
    print("[real-data-debug] NaN counts (momentum):")
    print(dataset.momentum.isna().sum())
    eig_mins = [stat.eigen_min for stat in dataset.cov_stats]
    print("[real-data-debug] covariance eigen min summary:", {
        "min": float(np.min(eig_mins)),
        "median": float(np.median(eig_mins)),
        "max": float(np.max(eig_mins)),
    })

    # 設定とデータ概要を JSON で残して後から再確認
    summary = {
        "config": config.__dict__,
        "dataset_summary": dataset.summary(),
    }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[real-data-debug] saved summary: {summary_path}")

    # 元データと行列化した特徴量を CSV で確認可能にしておく
    save_table(dataset.prices, outdir / "prices_full.csv")
    save_table(dataset.returns, outdir / "returns_full.csv")
    save_table(dataset.momentum, outdir / "momentum_full.csv")
    save_table(pd.DataFrame(dataset.X, index=dataset.timestamps), outdir / "X_matrix_head.csv", head_rows=10)
    save_table(pd.DataFrame(dataset.Y, index=dataset.timestamps), outdir / "Y_matrix_head.csv", head_rows=10)

    cov_df = pd.DataFrame(
        {
            "date": dataset.cov_times,
            "eig_min": [stat.eigen_min for stat in dataset.cov_stats],
            "eig_max": [stat.eigen_max for stat in dataset.cov_stats],
        }
    )
    cov_df.to_csv(outdir / "covariance_eigenvalues.csv", index=False)
    print(f"[real-data-debug] saved table: {outdir / 'covariance_eigenvalues.csv'}")

    vline = pd.Timestamp(args.start)
    plot_series(dataset.prices, visuals_dir / "prices.png", "Prices timeseries", vline=vline)
    plot_series(dataset.returns, visuals_dir / "returns.png", "Returns timeseries", vline=vline)
    plot_series(dataset.momentum, visuals_dir / "momentum.png", "Momentum (log price diff)", vline=vline)
    plot_scalar_series(
        cov_df["date"],
        cov_df["eig_min"],
        visuals_dir / "covariance_eig_min.png",
        "Covariance minimum eigenvalue",
        "eig_min",
    )

    print(f"[real-data-debug] outputs available in: {outdir}")


if __name__ == "__main__":  # pragma: no cover
    main()
