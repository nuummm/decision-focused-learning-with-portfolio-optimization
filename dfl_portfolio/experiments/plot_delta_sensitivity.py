from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError:  # pragma: no cover
    plt = None
    font_manager = None

try:  # Optional: if available, prefer japanize_matplotlib
    import japanize_matplotlib  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    japanize_matplotlib = None  # type: ignore


DEFAULT_METRICS = [
    "ann_return",
    "total_return",
    "terminal_wealth",
    "sharpe",
    "sortino",
    "ann_volatility",
    "max_drawdown",
    "cvar_95",
    "avg_turnover",
    "avg_trading_cost",
    "r2",
    "rmse",
]


METRIC_LABEL_JP: Dict[str, str] = {
    "ann_return": "年率リターン(%)",
    "total_return": "総リターン(%)",
    "terminal_wealth": "最終資産(倍)",
    "sharpe": "シャープ(年率)",
    "sortino": "ソルティノ(年率)",
    "ann_volatility": "年率ボラティリティ(%)",
    "max_drawdown": "最大ドローダウン(%)",
    "cvar_95": "CVaR(95%)(%)",
    "avg_turnover": "平均ターンオーバー(%)",
    "avg_trading_cost": "取引コスト合計(割合)",
    "r2": "$R^2$",
    "rmse": "RMSE",
}


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def plot_delta_sensitivity(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    metrics: Optional[List[str]] = None,
    title_prefix: str = "Delta sensitivity",
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate plots.")
    # Best-effort Japanese font setup (works even when japanize_matplotlib isn't installed).
    if font_manager is not None:
        candidates = [
            # macOS
            "Hiragino Sans",
            "Hiragino Kaku Gothic ProN",
            "Yu Gothic",
            # Linux / common
            "Noto Sans CJK JP",
            "Noto Sans JP",
            "IPAexGothic",
            "IPAGothic",
            "TakaoGothic",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.family"] = name
                break
        plt.rcParams["axes.unicode_minus"] = False
    if df.empty:
        raise ValueError("input DataFrame is empty.")
    if "delta" not in df.columns:
        raise ValueError("input DataFrame must contain a 'delta' column.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    work["delta"] = _as_numeric(work["delta"])
    work = work.dropna(subset=["delta"]).sort_values("delta", kind="mergesort")

    metrics = metrics or list(DEFAULT_METRICS)
    for metric in metrics:
        if metric not in work.columns:
            continue
        y = _as_numeric(work[metric])
        x = work["delta"]
        sub = pd.DataFrame({"delta": x, metric: y}).dropna()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(sub["delta"], sub[metric], marker="o", linewidth=1.8)
        ax.set_xlabel("delta")
        ax.set_ylabel(METRIC_LABEL_JP.get(metric, metric))
        ax.set_title(f"{title_prefix}: {metric}")
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(sub["delta"].unique().tolist()))
        fig.tight_layout()
        fig.savefig(out_dir / f"delta_{metric}.png", dpi=160)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot delta sensitivity (per-metric line plots).")
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to merged summary CSV (must contain 'delta').",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for PNG figures.",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric columns to plot.",
    )
    p.add_argument("--title-prefix", type=str, default="Delta sensitivity")
    return p


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.csv)
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    plot_delta_sensitivity(df, args.outdir, metrics=metrics, title_prefix=str(args.title_prefix))


if __name__ == "__main__":  # pragma: no cover
    main()
