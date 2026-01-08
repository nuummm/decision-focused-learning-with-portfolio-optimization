from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import math
import os
import shutil
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("KMP_USE_SHM", "0")

try:  # matplotlib is only needed for custom window plots
    import matplotlib

    matplotlib.use("Agg")  # must be set before importing pyplot
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - best effort
    plt = None

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
    PERIOD_WINDOWS,
    MODEL_COLOR_MAP,
    ASSET_COLOR_SEQUENCE,
    compute_correlation_stats,
    compute_period_metrics,
    compute_pairwise_mean_return_tests,
    compute_pairwise_performance_tests,
    compute_sortino_ratio,
    display_model_name,
    export_average_weights,
    export_max_return_winner_counts,
    export_winner_capture_frequency,
    export_weight_threshold_frequency,
    export_weight_variance_correlation,
    format_summary_for_output,
    max_drawdown,
    plot_asset_correlation,
    plot_condition_numbers,
    plot_delta_paths,
    plot_drawdown_curves,
    plot_flex_solver_debug,
    plot_multi_wealth,
    plot_solver_summary_bars,
    plot_wealth_correlation_heatmap,
    plot_wealth_curve,
    plot_wealth_with_events,
    plot_wealth_window_normalized,
    plot_weight_comparison,
    plot_weight_histograms,
    plot_weight_paths,
    plot_time_series,
    run_extended_analysis,
    summarize_dfl_performance_significance,
    update_experiment_ledger,
    build_cost_adjusted_summary,
)
from dfl_portfolio.models.ols_gurobi import solve_mvo_gurobi
from dfl_portfolio.models.ols_multi import train_ols_multi, predict_yhat_multi
from dfl_portfolio.models.dfl_p1_flex_multi import fit_dfl_p1_flex_multi
from dfl_portfolio.models.ipo_closed_form_multi import fit_ipo_closed_form_multi
from dfl_portfolio.models.ipo_grad_multi import fit_ipo_grad_multi
from dfl_portfolio.models.spo_plus_multi import fit_spo_plus_multi
from dfl_portfolio.registry import GUROBI_DEFAULTS, KNITRO_DEFAULTS
from dfl_portfolio.experiments.real_data_common import (
    mvo_cost,
    build_rebalance_schedule,
    build_flex_dual_kkt_ensemble,
    resolve_trading_cost_rates,
)
from dfl_portfolio.experiments.real_data_benchmarks import run_benchmark_suite


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
RESULTS_BASE = PROJECT_ROOT / "results"
RESULTS_ROOT = RESULTS_BASE / "exp_real_data_multi"
DEBUG_ROOT = RESULTS_BASE / "debug_outputs_multi"


logging.getLogger("pyomo").setLevel(logging.ERROR)
logging.getLogger("pyomo.solvers").setLevel(logging.ERROR)

# For window plots (match real_data_run.py ordering).
PREFERRED_MODEL_ORDER = [
    "Buy&Hold SPY",
    "1/N",
    "TSMOM (SPY)",
    "OLS",
    "IPO",
    "IPO-GRAD",
    "SPO+",
    "DFL-CF",
    "DFL-OPT-K",
    "DFL-QCQP-ens",
]


def plot_weight_window_with_connections(
    weight_dict: Dict[str, pd.DataFrame],
    path: Path,
    *,
    returns_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render stacked weights per model and optional winner diagnostics.

    - Saves the stacked weights figure to `path`.
    - If `returns_df` is provided (index=date, columns=assets), it also adds a bottom
      subplot that indicates, per period, which asset achieved the maximum realized
      return (winner=1, others=0).
    - It additionally saves a standalone effective number plot (N_eff = 1/sum(w^2))
      to `path` with the suffix `_neff.png`.
    """

    if plt is None or not weight_dict:
        return
    order_map = {name: idx for idx, name in enumerate(PREFERRED_MODEL_ORDER)}
    entries: List[Tuple[str, pd.DataFrame]] = []
    for model_key, df in weight_dict.items():
        if df.empty or "date" not in df.columns:
            continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.sort_values("date")
        entries.append((str(model_key), tmp))
    # Sort by preferred display order; unknowns last.
    entries.sort(key=lambda pair: order_map.get(display_model_name(pair[0]), len(order_map)))
    if not entries:
        return

    # Window range for axis clamping.
    window_start: Optional[pd.Timestamp] = None
    window_end: Optional[pd.Timestamp] = None
    for _, df_model in entries:
        if df_model.empty:
            continue
        d = pd.to_datetime(df_model["date"])
        if d.empty:
            continue
        d_min = pd.Timestamp(d.min())
        d_max = pd.Timestamp(d.max())
        window_start = d_min if window_start is None else min(window_start, d_min)
        window_end = d_max if window_end is None else max(window_end, d_max)

    n_models = len(entries)
    add_winner = returns_df is not None and isinstance(returns_df, pd.DataFrame) and not returns_df.empty
    n_rows = n_models + (1 if add_winner else 0)
    fig_height = max(2.5 * n_models + (1.8 if add_winner else 0.0), 4)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    axes_list = list(axes)
    winner_ax = axes_list[-1] if add_winner else None
    model_axes = axes_list[:-1] if add_winner else axes_list

    first_dates: Optional[pd.Series] = None
    first_asset_cols: List[str] = []
    for idx, ((model_key, df_model), ax) in enumerate(zip(entries, model_axes)):
        asset_cols = [c for c in df_model.columns if c not in {"date", "portfolio_return_sq"}]
        if not asset_cols:
            continue
        values = df_model[asset_cols].astype(float)
        dates = pd.to_datetime(df_model["date"])
        if first_dates is None:
            first_dates = dates
            first_asset_cols = list(asset_cols)
        bottom = np.zeros(len(values))
        for col in asset_cols:
            label = col if idx == 0 else None
            ax.bar(dates, values[col], bottom=bottom, width=5, label=label)
            bottom += values[col].to_numpy()
        ax.set_ylabel(display_model_name(model_key))
        ax.set_ylim(0.0, 1.0)
        if idx == 0 and asset_cols:
            ax.legend(loc="upper right", fontsize=8)

    if add_winner and winner_ax is not None and first_dates is not None and first_asset_cols:
        try:
            r = returns_df.copy()
            if not isinstance(r.index, pd.DatetimeIndex):
                r.index = pd.to_datetime(r.index)
            # Align to the same date grid as the weights plot (weekly dates)
            r = r.reindex(pd.to_datetime(first_dates))
            r = r.replace([np.inf, -np.inf], np.nan)
            asset_cols = [c for c in first_asset_cols if c in r.columns]
            if asset_cols:
                asset_color_map = {
                    col: ASSET_COLOR_SEQUENCE[i % len(ASSET_COLOR_SEQUENCE)]
                    for i, col in enumerate(asset_cols)
                }
                winners = r[asset_cols].idxmax(axis=1)
                bottom = np.zeros(len(r.index), dtype=float)
                for col in asset_cols:
                    vals = (winners == col).astype(float).to_numpy()
                    winner_ax.bar(
                        r.index,
                        vals,
                        bottom=bottom,
                        width=5,
                        color=asset_color_map.get(col, None),
                        alpha=0.85,
                        edgecolor="black",
                        linewidth=0.3,
                    )
                    bottom += vals
                winner_ax.set_ylim(0.0, 1.0)
                winner_ax.set_yticks([0.0, 1.0])
                winner_ax.set_ylabel("勝者(1/0)")
                winner_ax.set_title("週次：最大リターン資産（勝者=1）")
                winner_ax.grid(axis="y", alpha=0.15)
        except Exception:  # pragma: no cover
            pass

    if window_start is not None and window_end is not None:
        for ax in axes_list:
            ax.set_xlim(window_start, window_end)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    # Standalone N_eff plot (with crisis windows shaded)
    neff_path = path.with_name(f"{path.stem}_neff.png")
    fig2, ax2 = plt.subplots(figsize=(12, 3.6))
    for _, start_s, end_s in PERIOD_WINDOWS:
        start_ts = pd.Timestamp(start_s)
        end_ts = pd.Timestamp(end_s)
        ax2.axvspan(start_ts, end_ts, color="grey", alpha=0.15, zorder=0)

    for _, (model_key, df_model) in enumerate(entries):
        asset_cols = [c for c in df_model.columns if c not in {"date", "portfolio_return_sq"}]
        if not asset_cols:
            continue
        values = df_model[asset_cols].astype(float)
        dates = pd.to_datetime(df_model["date"])
        h_val = (values.pow(2).sum(axis=1)).replace(0.0, np.nan)
        neff = (1.0 / h_val).astype(float)
        color = MODEL_COLOR_MAP.get(str(model_key), None) or MODEL_COLOR_MAP.get(
            display_model_name(model_key), None
        )
        ax2.plot(dates, neff, label=display_model_name(model_key), color=color, linewidth=1.8)

    if window_start is not None and window_end is not None:
        ax2.set_xlim(window_start, window_end)
    ax2.set_ylabel("N_eff")
    ax2.set_xlabel("日付")
    ax2.set_title("有効資産数 (N_eff)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(neff_path)
    plt.close(fig2)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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
            raise ValueError(
                f"Invalid delta specification '{value}'. Expected comma-separated floats."
            ) from exc
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"delta values must be within [0,1]; got {v}")
    if not allow_multiple and len(values) > 1:
        raise ValueError("Multiple values are not allowed for this delta option.")
    return values


def _drop_parser_option(parser, option_string: str) -> None:
    target = None
    for action in list(getattr(parser, "_actions", [])):
        if option_string in getattr(action, "option_strings", []):
            target = action
            break
    if target is None:
        return
    try:
        parser._actions.remove(target)
    except Exception:
        pass
    for opt in getattr(target, "option_strings", []):
        try:
            parser._option_string_actions.pop(opt, None)
        except Exception:
            pass
    for group in getattr(parser, "_action_groups", []):
        try:
            if target in getattr(group, "_group_actions", []):
                group._group_actions.remove(target)
        except Exception:
            pass


def build_multi_parser():
    """build_parser() 互換 + multi-feature 差分 (--momentum-window を複数指定可能にする)."""
    parser = build_parser()
    _drop_parser_option(parser, "--momentum-window")
    parser.add_argument(
        "--momentum-window",
        type=int,
        action="append",
        default=None,
        help=(
            "Momentum window(s) used as features. Repeatable. "
            "e.g. --momentum-window 13 --momentum-window 26 --momentum-window 52. "
            "If omitted, defaults to a single window of 26."
        ),
    )
    parser.add_argument(
        "--multi-vol-window",
        type=int,
        default=0,
        help=(
            "If >0, also add per-asset rolling volatility (std of returns) as an extra feature "
            "using this window length. (0 disables this feature.)"
        ),
    )
    parser.add_argument(
        "--multi-vix",
        action="store_true",
        default=False,
        help="Include VIX index level as an additional feature (broadcasted to all assets).",
    )
    parser.add_argument(
        "--multi-intercept",
        action="store_true",
        default=False,
        help=(
            "Add a constant 1 feature to enable a per-asset intercept term for all models. "
            "With this flag, the prediction model becomes y_hat = alpha + x^T beta."
        ),
    )
    parser.add_argument(
        "--multi-vix-ticker",
        type=str,
        default="^VIX",
        help="Yahoo Finance ticker for the VIX index (default: '^VIX').",
    )
    parser.add_argument(
        "--nlp_algorithm",
        "--nlp-algorithm",
        dest="nlp_algorithm",
        type=int,
        default=1,
        help="Knitro option nlp_algorithm used by flex in multi runs (default: 1).",
    )
    parser.add_argument(
        "--hessopt",
        type=int,
        default=2,
        help="Knitro option hessopt used by flex in multi runs (default: 2).",
    )

    # Multi runner default run: match the documented example below.
    # (Options not explicitly overridden here keep `real_data_run.py` defaults via build_parser().)
    for action in getattr(parser, "_actions", []):
        if getattr(action, "dest", None) == "jobs":
            action.default = 1
        elif getattr(action, "dest", None) == "ipo_grad_seed":
            action.default = 42
        elif getattr(action, "dest", None) == "theta_init_mode":
            action.default = "none"
        elif getattr(action, "dest", None) == "theta_anchor_mode":
            action.default = "ipo"
    return parser


@dataclass(frozen=True)
class MultiFeatureConfig:
    momentum_windows: List[int]
    vol_window: int
    include_vix: bool
    include_intercept: bool
    vix_ticker: str
    vix_zscore_window: int

    def feature_names(self) -> List[str]:
        names: List[str] = []
        if self.include_intercept:
            names.append("const_1")
        names.extend([f"mom_{w}" for w in self.momentum_windows])
        if self.vol_window > 0:
            names.append(f"vol_{self.vol_window}")
        if self.include_vix:
            names.append(f"vix_logz{self.vix_zscore_window}_{self.vix_ticker}")
        return names


def build_multi_features(bundle, cfg: MultiFeatureConfig) -> Tuple[np.ndarray, List[str]]:
    tickers = bundle.dataset.config.tickers
    idx = pd.Index(bundle.dataset.timestamps)
    prices = bundle.dataset.prices.copy()
    returns = bundle.dataset.returns.copy()

    feats: List[np.ndarray] = []
    if cfg.include_intercept:
        feats.append(np.ones((len(idx), len(tickers)), dtype=float))
    for window in cfg.momentum_windows:
        win = int(window)
        if win <= 0:
            raise ValueError(f"momentum_window must be positive, got {window}")
        mom = np.log(prices / prices.shift(win))
        mom_feat = mom.loc[idx, tickers]
        feats.append(mom_feat.to_numpy(dtype=float))
    if int(cfg.vol_window) > 0:
        vol_feat = returns.rolling(int(cfg.vol_window)).std().loc[idx, tickers]
        feats.append(vol_feat.to_numpy(dtype=float))

    if cfg.include_vix:
        from dfl_portfolio.real_data.fetch_yahoo import fetch_yahoo_prices

        cfg_loader = bundle.dataset.config
        vix_ticker = str(cfg.vix_ticker or "^VIX").strip()
        if not vix_ticker:
            raise ValueError("--multi-vix is enabled but --multi-vix-ticker is empty.")
        fetch_start = getattr(bundle.dataset.fetch_result, "start", cfg_loader.start)
        fetch_end = getattr(bundle.dataset.fetch_result, "end", cfg_loader.end)
        # Default 4-ticker runs use a read-only frozen cache for reproducibility.
        # For the optional VIX series, automatically fall back to a writable cache.
        cache_dir = cfg_loader.cache_dir
        cache_readonly = bool(cfg_loader.cache_readonly)
        force_refresh = bool(cfg_loader.force_refresh)
        try:
            from dfl_portfolio.real_data.loader import DEFAULT_FROZEN_CACHE

            if cache_readonly and cache_dir is not None:
                try:
                    is_frozen_cache = Path(cache_dir).resolve() == Path(DEFAULT_FROZEN_CACHE).resolve()
                except Exception:
                    is_frozen_cache = False
                if is_frozen_cache:
                    cache_dir = None
                    cache_readonly = False
                    force_refresh = False
        except Exception:
            # Best-effort: if we can't detect the frozen cache path, keep the original settings.
            pass
        try:
            vix_df_raw, _ = fetch_yahoo_prices(
                [vix_ticker],
                fetch_start,
                fetch_end,
                interval=cfg_loader.interval,
                auto_adjust=bool(cfg_loader.auto_adjust),
                cache_dir=cache_dir,
                cache_readonly=cache_readonly,
                force_refresh=force_refresh,
                debug=bool(cfg_loader.debug),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to fetch VIX series for --multi-vix. "
                "If you are using a read-only/frozen cache, provide a writable --cache-dir "
                "or disable cache-readonly/freeze-default-cache."
            ) from exc

        vix_col = f"{vix_ticker.upper()}_{cfg_loader.price_field}"
        if vix_col not in vix_df_raw.columns:
            raise KeyError(
                f"VIX price column '{vix_col}' not found in Yahoo data. "
                f"Available columns: {list(vix_df_raw.columns)[:10]}..."
            )
        vix_prices = vix_df_raw[[vix_col]].copy()
        vix_prices.columns = [vix_ticker.upper()]

        if cfg_loader.frequency == "daily":
            vix_resampled = vix_prices
        else:
            if cfg_loader.resample_rule:
                rule = cfg_loader.resample_rule
            elif cfg_loader.frequency == "weekly":
                rule = "W-FRI"
            else:
                rule = "M"
            vix_resampled = vix_prices.resample(rule).last().dropna(how="all")

        vix_series = vix_resampled[vix_ticker.upper()].reindex(idx)
        vix_series = vix_series.ffill().bfill()
        if vix_series.isna().any():
            missing = int(vix_series.isna().sum())
            raise ValueError(
                f"VIX feature contains missing values after alignment ({missing} missing). "
                "Try expanding the date range or disabling --cache-readonly so the VIX cache can be built."
            )
        # Scale-align: use log-level and rolling z-score so this feature does not dominate by magnitude.
        # (raw VIX level is O(10~100), while returns/momentum features are O(1e-2~1e-1)).
        vix_level = pd.to_numeric(vix_series, errors="coerce").astype(float)
        if vix_level.isna().any():
            raise ValueError("VIX series contains non-numeric entries after alignment.")
        vix_log = np.log(np.maximum(vix_level.to_numpy(dtype=float), 1e-12))
        win = int(cfg.vix_zscore_window)
        if win <= 1:
            raise ValueError(f"vix_zscore_window must be >=2, got {cfg.vix_zscore_window}")
        vix_log_s = pd.Series(vix_log, index=idx)
        roll_mean = vix_log_s.rolling(win, min_periods=max(2, win // 3)).mean()
        roll_std = vix_log_s.rolling(win, min_periods=max(2, win // 3)).std(ddof=0)
        vix_z = (vix_log_s - roll_mean) / roll_std.replace(0.0, np.nan)
        vix_z = vix_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vix_vec = vix_z.to_numpy(dtype=float)
        vix_feat = np.tile(vix_vec[:, None], (1, len(tickers))).astype(float)
        feats.append(vix_feat)

    if not feats:
        raise ValueError("No features were configured for multi experiment.")
    X_feat = np.stack(feats, axis=-1)
    if not np.all(np.isfinite(X_feat)):
        nan_count = int(np.sum(~np.isfinite(X_feat)))
        raise ValueError(
            f"Multi features contain NaN/inf ({nan_count} entries). "
            "Try increasing the fetch buffer (larger max momentum window) or reducing --multi-vol-window."
        )
    return X_feat, cfg.feature_names()


def run_rolling_experiment_multi(
    model_key: str,
    model_label: str,
    bundle,
    X_feat: np.ndarray,
    delta_up: float,
    delta_down_candidates: Sequence[float],
    trading_cost_enabled: bool,
    trading_cost_default_bps: float,
    asset_cost_overrides: Dict[str, float] | None,
    flex_options: Dict[str, Any] | None,
    spo_plus_options: Dict[str, Any] | None,
    ipo_grad_options: Dict[str, Any] | None,
    train_window: int,
    rebal_interval: int,
    debug_roll: bool,
    debug_dir: Path,
    results_model_dir: Path,
    tee: bool,
    asset_pred_dir: Path | None = None,
    formulation: Optional[str] = None,
    eval_start: Optional[pd.Timestamp] = None,
    ipo_grad_debug_kkt: bool = False,
    base_seed: Optional[int] = None,
    init_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """real_data_run.py の出力仕様を踏襲しつつ、(N,d,K) の多変量特徴量に対応。"""
    schedule = build_rebalance_schedule(
        bundle, train_window, rebal_interval, eval_start=eval_start
    )
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
        default_bps_if_missing=trading_cost_default_bps,
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

    Y = bundle.dataset.Y
    wealth = 1.0
    wealth_net = 1.0
    wealth_dates: List[pd.Timestamp] = []
    wealth_values: List[float] = []
    wealth_net_values: List[float] = []
    wealth_labels: List[str] = []

    step_rows: List[Dict[str, Any]] = []
    rebalance_rows: List[Dict[str, Any]] = []
    asset_rows: List[Dict[str, Any]] = []
    delta_history: List[Dict[str, Any]] = []
    prev_weights: Optional[np.ndarray] = None

    # 訓練時（in-sample）の予測性能: 各リバランスで学習窓の (Y, Yhat) をプールして R^2 を計算
    train_n = 0
    train_sum_y = 0.0
    train_sum_yhat = 0.0
    train_sum_y2 = 0.0
    train_sum_yhat2 = 0.0
    train_sum_y_yhat = 0.0
    train_sum_sqerr = 0.0

    def _update_train_stats(y_mat: np.ndarray, yhat_mat: np.ndarray) -> None:
        nonlocal train_n, train_sum_y, train_sum_yhat, train_sum_y2, train_sum_yhat2, train_sum_y_yhat, train_sum_sqerr
        y = np.asarray(y_mat, dtype=float).reshape(-1)
        yhat = np.asarray(yhat_mat, dtype=float).reshape(-1)
        if y.size == 0 or yhat.size == 0:
            return
        mask = np.isfinite(y) & np.isfinite(yhat)
        if not np.any(mask):
            return
        y = y[mask]
        yhat = yhat[mask]
        train_n += int(y.size)
        train_sum_y += float(np.sum(y))
        train_sum_yhat += float(np.sum(yhat))
        train_sum_y2 += float(np.sum(y * y))
        train_sum_yhat2 += float(np.sum(yhat * yhat))
        train_sum_y_yhat += float(np.sum(y * yhat))
        diff = yhat - y
        train_sum_sqerr += float(np.sum(diff * diff))

    def _finalize_train_metrics() -> tuple[float, float]:
        if train_n < 2:
            return float("nan"), float("nan")
        n = float(train_n)
        mean_y = train_sum_y / n
        mean_yhat = train_sum_yhat / n
        cov = (train_sum_y_yhat / n) - (mean_y * mean_yhat)
        var_y = (train_sum_y2 / n) - (mean_y * mean_y)
        var_yhat = (train_sum_yhat2 / n) - (mean_yhat * mean_yhat)
        denom = float(np.sqrt(max(var_y, 0.0) * max(var_yhat, 0.0)))
        if denom <= 0.0 or not np.isfinite(denom):
            r2_train = float("nan")
        else:
            corr = cov / denom
            r2_train = float(corr * corr) if np.isfinite(corr) else float("nan")
        rmse_train = float(np.sqrt(train_sum_sqerr / n)) if train_sum_sqerr >= 0.0 else float("nan")
        return r2_train, rmse_train

    total_cycles = len(schedule)
    for cycle_id, item in enumerate(schedule):
        seed_event = None
        if base_seed is not None:
            msg = f"{int(base_seed)}|{cycle_id}|{int(item.rebalance_idx)}"
            digest = hashlib.blake2b(msg.encode("utf-8"), digest_size=8).digest()
            seed_event = int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF
        seed_theta_init = None
        if init_seed is not None:
            msg = f"{int(init_seed)}|{cycle_id}|{int(item.rebalance_idx)}|theta"
            digest = hashlib.blake2b(msg.encode("utf-8"), digest_size=8).digest()
            seed_theta_init = int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF

        if debug_roll and (
            cycle_id == 0 or (cycle_id + 1) % 20 == 0 or (cycle_id + 1) == total_cycles
        ):
            progress = (cycle_id + 1) / max(total_cycles, 1)
            bar = "#" * int(progress * 20)
            print(
                f"[roll-debug-multi] model={model_label} cycle={cycle_id+1}/{total_cycles} "
                f"idx={item.rebalance_idx} train=[{item.train_start},{item.train_end}] "
                f"n_eval={len(item.eval_indices)} [{bar:<20}] {progress:.0%}"
            )

        candidate_list = (
            list(delta_down_candidates) if model_key == "flex" else [float(delta_down_candidates[0])]
        )
        selected_delta_down = float(candidate_list[0])
        train_objective = float("nan")
        grid_stats: List[Dict[str, float]] = []

        def _train_once(delta_down_val: float) -> Tuple[np.ndarray, Dict[str, Any], float]:
            delta_for_training = float(delta_down_val) if model_key == "flex" else float(delta_up)

            # flex theta init/anchor (multi-specific; approximates real_data_run.py behavior)
            theta_init_override: Optional[np.ndarray] = None
            theta_anchor_override: Optional[np.ndarray] = None
            if model_key == "flex" and flex_options is not None:
                init_mode = str(flex_options.get("flex_theta_init_mode", "ipo")).lower().strip()
                if init_mode == "ols":
                    theta_init_override = train_ols_multi(
                        X_feat[item.train_start : item.train_end + 1],
                        Y[item.train_start : item.train_end + 1],
                    )
                elif init_mode == "ipo":
                    try:
                        theta_init_override, *_ = fit_ipo_closed_form_multi(
                            X_feat,
                            Y,
                            bundle.covariances,
                            bundle.cov_indices.tolist(),
                            start_index=item.train_start,
                            end_index=item.train_end,
                            delta=float(delta_for_training),
                            mode="budget",
                            psd_eps=1e-12,
                            ridge_theta=1e-10,
                            tee=tee,
                        )
                    except Exception:
                        theta_init_override = None
                elif init_mode in {"none", "zero"}:
                    theta_init_override = np.zeros((X_feat.shape[1], X_feat.shape[2]), dtype=float)

                lam_anchor = float(flex_options.get("flex_lambda_theta_anchor", 0.0))
                if lam_anchor > 0.0:
                    anchor_mode = str(flex_options.get("flex_theta_anchor_mode", "ipo")).lower().strip()
                    if anchor_mode == "ipo":
                        try:
                            theta_anchor_override, *_ = fit_ipo_closed_form_multi(
                                X_feat,
                                Y,
                                bundle.covariances,
                                bundle.cov_indices.tolist(),
                                start_index=item.train_start,
                                end_index=item.train_end,
                                delta=float(delta_for_training),
                                mode="budget",
                                psd_eps=1e-12,
                                ridge_theta=1e-10,
                                tee=tee,
                            )
                        except Exception:
                            theta_anchor_override = np.zeros(
                                (X_feat.shape[1], X_feat.shape[2]), dtype=float
                            )
                    else:
                        theta_anchor_override = np.zeros(
                            (X_feat.shape[1], X_feat.shape[2]), dtype=float
                        )

            start_time = time.perf_counter()
            info: Dict[str, Any] = {}
            if model_key == "ols":
                theta_hat_local = train_ols_multi(
                    X_feat[item.train_start : item.train_end + 1],
                    Y[item.train_start : item.train_end + 1],
                )
            elif model_key == "ipo":
                theta_hat_local, _, _, _, _, info = fit_ipo_closed_form_multi(
                    X_feat,
                    Y,
                    bundle.covariances,
                    bundle.cov_indices.tolist(),
                    start_index=item.train_start,
                    end_index=item.train_end,
                    delta=float(delta_up),
                    mode="budget",
                    psd_eps=1e-12,
                    ridge_theta=1e-10,
                    tee=tee,
                )
            elif model_key == "ipo_grad":
                init_mode = str(
                    ipo_grad_options.get("ipo_grad_init_mode", "none")
                    if ipo_grad_options
                    else "none"
                ).lower()
                theta_init_ig: Optional[np.ndarray] = None
                if init_mode == "ipo":
                    try:
                        theta_init_ig, *_ = fit_ipo_closed_form_multi(
                            X_feat,
                            Y,
                            bundle.covariances,
                            bundle.cov_indices.tolist(),
                            start_index=item.train_start,
                            end_index=item.train_end,
                            delta=float(delta_for_training),
                            mode="budget",
                            psd_eps=1e-12,
                            ridge_theta=1e-10,
                            tee=tee,
                        )
                    except Exception:
                        theta_init_ig = None
                if theta_init_ig is None:
                    theta_init_ig = np.zeros((X_feat.shape[1], X_feat.shape[2]), dtype=float)

                theta_anchor = np.zeros((X_feat.shape[1], X_feat.shape[2]), dtype=float)
                lam_anchor = float(
                    ipo_grad_options.get("ipo_grad_lambda_anchor", 0.0) if ipo_grad_options else 0.0
                )
                if lam_anchor > 0.0 and ipo_grad_options is not None:
                    anchor_mode = str(ipo_grad_options.get("ipo_grad_theta_anchor_mode", "ipo")).lower()
                    if anchor_mode == "ipo":
                        try:
                            theta_anchor, *_ = fit_ipo_closed_form_multi(
                                X_feat,
                                Y,
                                bundle.covariances,
                                bundle.cov_indices.tolist(),
                                start_index=item.train_start,
                                end_index=item.train_end,
                                delta=float(delta_for_training),
                                mode="budget",
                                psd_eps=1e-12,
                                ridge_theta=1e-10,
                                tee=tee,
                            )
                        except Exception:
                            theta_anchor = np.zeros(
                                (X_feat.shape[1], X_feat.shape[2]), dtype=float
                            )

                seed_use = int(seed_event) if seed_event is not None else None
                if ipo_grad_options is not None and ipo_grad_options.get("ipo_grad_seed") is not None:
                    seed_use = int(ipo_grad_options.get("ipo_grad_seed"))
                theta_hat_local, _, _, _, _, info = fit_ipo_grad_multi(
                    X_feat,
                    Y,
                    bundle.covariances,
                    bundle.cov_indices.tolist(),
                    start_index=item.train_start,
                    end_index=item.train_end,
                    delta=float(delta_up),
                    epochs=int(ipo_grad_options.get("ipo_grad_epochs", 500) if ipo_grad_options else 500),
                    lr=float(ipo_grad_options.get("ipo_grad_lr", 1e-3) if ipo_grad_options else 1e-3),
                    batch_size=int(ipo_grad_options.get("ipo_grad_batch_size", 0) if ipo_grad_options else 0),
                    qp_max_iter=int(ipo_grad_options.get("ipo_grad_qp_max_iter", 5000) if ipo_grad_options else 5000),
                    qp_tol=float(ipo_grad_options.get("ipo_grad_qp_tol", 1e-6) if ipo_grad_options else 1e-6),
                    seed=seed_use,
                    theta_init=theta_init_ig,
                    lambda_anchor=float(lam_anchor),
                    theta_anchor=theta_anchor,
                    tee=tee,
                    debug_kkt=bool(ipo_grad_debug_kkt),
                )
            elif model_key == "spo_plus":
                theta_init_spo: Optional[np.ndarray] = None
                init_mode = str(
                    spo_plus_options.get("spo_plus_init_mode", "ipo") if spo_plus_options else "ipo"
                ).lower()
                if init_mode == "ipo":
                    try:
                        theta_init_spo, *_ = fit_ipo_closed_form_multi(
                            X_feat,
                            Y,
                            bundle.covariances,
                            bundle.cov_indices.tolist(),
                            start_index=item.train_start,
                            end_index=item.train_end,
                            delta=float(delta_for_training),
                            mode="budget",
                            psd_eps=1e-12,
                            ridge_theta=1e-10,
                            tee=tee,
                        )
                    except Exception:
                        theta_init_spo = None
                if theta_init_spo is None:
                    theta_init_spo = np.zeros((X_feat.shape[1], X_feat.shape[2]), dtype=float)
                theta_anchor_spo = np.zeros((X_feat.shape[1], X_feat.shape[2]), dtype=float)
                lam_anchor_spo = float(
                    spo_plus_options.get("spo_plus_lambda_anchor", 0.0) if spo_plus_options else 0.0
                )
                if lam_anchor_spo > 0.0 and spo_plus_options is not None:
                    anchor_mode = str(spo_plus_options.get("spo_plus_theta_anchor_mode", "ipo")).lower()
                    if anchor_mode == "ipo":
                        try:
                            theta_anchor_spo, *_ = fit_ipo_closed_form_multi(
                                X_feat,
                                Y,
                                bundle.covariances,
                                bundle.cov_indices.tolist(),
                                start_index=item.train_start,
                                end_index=item.train_end,
                                delta=float(delta_for_training),
                                mode="budget",
                                psd_eps=1e-12,
                                ridge_theta=1e-10,
                                tee=tee,
                            )
                        except Exception:
                            theta_anchor_spo = np.zeros(
                                (X_feat.shape[1], X_feat.shape[2]), dtype=float
                            )
                theta_hat_local, _, _, _, _, info = fit_spo_plus_multi(
                    X_feat,
                    Y,
                    bundle.covariances,
                    bundle.cov_indices.tolist(),
                    start_index=item.train_start,
                    end_index=item.train_end,
                    delta=float(delta_up),
                    epochs=int(spo_plus_options.get("spo_plus_epochs", 500) if spo_plus_options else 500),
                    lr=float(spo_plus_options.get("spo_plus_lr", 1e-3) if spo_plus_options else 1e-3),
                    batch_size=int(spo_plus_options.get("spo_plus_batch_size", 0) if spo_plus_options else 0),
                    lambda_reg=float(spo_plus_options.get("spo_plus_lambda_reg", 0.0) if spo_plus_options else 0.0),
                    lambda_anchor=float(lam_anchor_spo),
                    theta_anchor=theta_anchor_spo,
                    risk_constraint=bool(spo_plus_options.get("spo_plus_risk_constraint", True) if spo_plus_options else True),
                    risk_mult=float(spo_plus_options.get("spo_plus_risk_mult", 2.0) if spo_plus_options else 2.0),
                    psd_eps=1e-9,
                    tee=tee,
                    theta_init=theta_init_spo,
                )
            elif model_key == "flex":
                form = (formulation or "dual").lower()
                solver_name = str(flex_options.get("flex_solver", "knitro") if flex_options else "knitro")
                solver_name_key = solver_name.lower().strip()
                solver_options = None
                if solver_name_key == "knitro":
                    solver_options = dict(KNITRO_DEFAULTS)
                    if flex_options is not None:
                        solver_options["nlp_algorithm"] = int(flex_options.get("flex_knitro_nlp_algorithm", 1))
                        solver_options["hessopt"] = int(flex_options.get("flex_knitro_hessopt", 2))
                elif solver_name_key == "gurobi":
                    solver_options = dict(GUROBI_DEFAULTS)
                theta_hat_local, _, _, _, _, info = fit_dfl_p1_flex_multi(
                    X_feat,
                    Y,
                    bundle.covariances,
                    bundle.cov_indices.tolist(),
                    start_index=item.train_start,
                    end_index=item.train_end,
                    formulation=form,
                    delta=float(delta_for_training),
                    theta_init=theta_init_override,
                    solver=solver_name,
                    solver_options=solver_options,
                    tee=tee,
                    lambda_theta_anchor=float(flex_options.get("flex_lambda_theta_anchor", 0.0) if flex_options else 0.0),
                    theta_anchor=theta_anchor_override,
                    lambda_theta_anchor_l1=0.0,
                    lambda_theta_iso=float(flex_options.get("flex_lambda_theta_iso", 0.0) if flex_options else 0.0),
                )
            else:
                raise ValueError(f"Unsupported model_key for multi experiment: {model_key}")
            elapsed_local = time.perf_counter() - start_time
            return theta_hat_local, info, float(elapsed_local)

        if model_key == "flex" and len(candidate_list) > 1:
            best: Optional[Tuple[np.ndarray, Dict[str, Any], float, float]] = None
            elapsed_total = 0.0
            for cand in candidate_list:
                theta_c, info_c, elapsed_c = _train_once(float(cand))
                elapsed_total += float(elapsed_c)
                obj_val = None
                for key in ("objective_value", "gurobi_obj_val", "ObjVal"):
                    if key in info_c and info_c.get(key) is not None:
                        try:
                            obj_val = float(info_c.get(key))
                        except Exception:
                            obj_val = None
                        break
                if obj_val is None or not np.isfinite(obj_val):
                    obj_val = float("inf")
                grid_stats.append({"delta_down": float(cand), "objective": float(obj_val)})
                if best is None or obj_val < best[2]:
                    best = (theta_c, info_c, float(obj_val), float(cand))
            if best is None:
                raise RuntimeError("Delta grid search failed to produce a valid solution.")
            theta_hat, info, train_objective, selected_delta_down = best
            elapsed = float(elapsed_total)
        else:
            theta_hat, info, elapsed = _train_once(selected_delta_down)

        Yhat_all = predict_yhat_multi(X_feat, theta_hat)
        try:
            _update_train_stats(
                Y[item.train_start : item.train_end + 1],
                Yhat_all[item.train_start : item.train_end + 1],
            )
        except Exception:
            pass

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
                delta=float(delta_up),
                psd_eps=1e-9,
                output=False,
            )
            if z is None or np.isnan(z).any():
                continue
            realized = float(z @ Y[eval_idx])
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

            wealth *= 1.0 + realized
            wealth_net *= 1.0 + net_return
            wealth_dates.append(bundle.dataset.timestamps[eval_idx])
            wealth_values.append(wealth)
            wealth_net_values.append(wealth_net)
            wealth_labels.append("after_step")

            cost = mvo_cost(z, Y[eval_idx], cov, float(delta_up))
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
    returns_net = (
        step_df["net_return"].to_numpy() if "net_return" in step_df.columns else returns.copy()
    )
    avg_turnover = (
        float(step_df["turnover"].mean())
        if "turnover" in step_df.columns and not step_df.empty
        else float("nan")
    )
    avg_trading_cost_mean = (
        float(step_df["trading_cost"].mean())
        if "trading_cost" in step_df.columns and not step_df.empty
        else 0.0
    )
    avg_trading_cost = (
        float(step_df["trading_cost"].sum())
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
    cvar_95_net = compute_cvar(returns_net, alpha=0.05)

    # 決定係数 R^2（資産リターン予測の当てはまり）
    r2: float = float("nan")
    rmse: float = float("nan")
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
                    mse_val = float(np.mean((yhat - y) ** 2))
                    if np.isfinite(mse_val) and mse_val >= 0.0:
                        rmse = float(np.sqrt(mse_val))
    except Exception:
        r2 = float("nan")
        rmse = float("nan")

    r2_train, rmse_train = _finalize_train_metrics()

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
        "max_drawdown": max_drawdown(wealth_values),
        "cvar_95": cvar_95,
        "cvar_95_net": cvar_95_net,
        "r2": r2,
        "r2_train": r2_train,
        "rmse": rmse,
        "rmse_train": rmse_train,
        "ann_return_net": mean_return_net,
        "ann_volatility_net": std_return_net,
        "sharpe_net": sharpe_net,
        "sortino_net": sortino_net,
        "max_drawdown_net": max_drawdown(wealth_net_values),
        "terminal_wealth": terminal_wealth,
        "terminal_wealth_net": terminal_wealth_net,
        "total_return": total_return,
        "total_return_net": total_return_net,
        "train_window": train_window,
        "rebal_interval": rebal_interval,
        "avg_turnover": avg_turnover,
        "avg_trading_cost": avg_trading_cost,
        "avg_trading_cost_mean": avg_trading_cost_mean,
        "trading_cost_bps": trading_cost_label,
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
        "period_metrics": period_metrics,
        "wealth_df": wealth_df,
        "weights_df": weights_df,
        "rebalance_summary": reb_summary,
        "rebalance_df": rebalance_df,
        "delta_history": delta_history,
    }


def main() -> None:
    parser = build_multi_parser()
    args = parser.parse_args()
    # ---------------------------------------------------------------------
    # Shared theta CLI (applies to both flex and IPO-GRAD)
    # ---------------------------------------------------------------------
    if getattr(args, "theta_init_mode", None) is not None:
        shared_init = str(getattr(args, "theta_init_mode")).lower().strip()
        args.flex_theta_init_mode = shared_init
        args.ipo_grad_init_mode = shared_init
        args.spo_plus_init_mode = "ipo" if shared_init == "ipo" else "zero"
    if getattr(args, "lambda_theta_anchor", None) is not None:
        shared_lam = float(getattr(args, "lambda_theta_anchor"))
        args.flex_lambda_theta_anchor = shared_lam
        args.ipo_grad_lambda_anchor = shared_lam
        args.spo_plus_lambda_anchor = shared_lam
    if getattr(args, "theta_anchor_mode", None) is not None:
        shared_anchor = str(getattr(args, "theta_anchor_mode")).lower().strip()
        args.flex_theta_anchor_mode = shared_anchor
        args.ipo_grad_theta_anchor_mode = shared_anchor if shared_anchor != "none" else "zero"
        args.spo_plus_theta_anchor_mode = shared_anchor if shared_anchor != "none" else "zero"

    model_train_windows = parse_model_train_window_spec(getattr(args, "model_train_window", ""))
    benchmark_specs = parse_commalist(getattr(args, "benchmarks", ""))
    if not benchmark_specs:
        legacy_list: List[str] = []
        if (getattr(args, "benchmark_ticker", "") or "").strip():
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
    asset_pred_dir.mkdir(parents=True, exist_ok=True)
    model_outputs_dir = outdir / "model_outputs"
    model_outputs_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = DEBUG_ROOT / f"{outdir.name}_rolling"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Delta and trading cost settings (real_data_run.py と同じ意味)
    # ---------------------------------------------------------------------
    base_delta = float(args.delta)
    delta_up = float(_parse_delta_list(getattr(args, "delta_up", None), base_delta, allow_multiple=False)[0])
    delta_down_list = _parse_delta_list(getattr(args, "delta_down", None), delta_up, allow_multiple=True)
    trading_costs_enabled = float(getattr(args, "trading_cost_bps", 0.0) or 0.0) > 0.0
    trading_cost_default_bps = float(getattr(args, "trading_cost_bps", 0.0) or 0.0)
    raw_asset_costs: Dict[str, float] = getattr(args, "trading_cost_per_asset", {}) or {}
    asset_cost_overrides_dec = {
        ticker.upper(): max(float(rate), 0.0) / 10000.0 for ticker, rate in raw_asset_costs.items()
    }

    # ---------------------------------------------------------------------
    # Multi-feature configuration
    # ---------------------------------------------------------------------
    momentum_windows_raw = getattr(args, "momentum_window", None)
    if momentum_windows_raw is None:
        momentum_windows = [4, 26]
    else:
        momentum_windows = [int(w) for w in momentum_windows_raw if int(w) > 0]
        if not momentum_windows:
            momentum_windows = [4, 26]
    cfg_multi = MultiFeatureConfig(
        momentum_windows=momentum_windows,
        vol_window=int(getattr(args, "multi_vol_window", 0) or 0),
        include_vix=bool(getattr(args, "multi_vix", False)),
        include_intercept=bool(getattr(args, "multi_intercept", False)),
        vix_ticker=str(getattr(args, "multi_vix_ticker", "^VIX")),
        vix_zscore_window=max(26, int(getattr(args, "train_window", 26) or 26)),
    )
    effective_momentum_window = max(
        cfg_multi.momentum_windows + ([cfg_multi.vol_window] if cfg_multi.vol_window > 0 else [])
    )

    loader_cfg = MarketLoaderConfig.for_cli(
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        price_field=args.price_field,
        return_kind=args.return_kind,
        frequency=args.frequency,
        resample_rule=args.resample_rule,
        momentum_window=int(effective_momentum_window),
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
        cache_dir=getattr(args, "cache_dir", None),
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
            "multi_features": {
                "momentum_windows": cfg_multi.momentum_windows,
                "vol_window": cfg_multi.vol_window,
                "include_intercept": cfg_multi.include_intercept,
                "feature_names": cfg_multi.feature_names(),
            },
        }
    )
    summary_path = outdir / "experiment_summary.json"
    summary_path.write_text(json.dumps(bundle_summary, ensure_ascii=False, indent=2))

    # ---------------------------------------------------------------------
    # Data snapshot for reproducibility
    # ---------------------------------------------------------------------
    snapshot_records: List[Dict[str, object]] = []
    try:
        cache_path = Path(bundle.dataset.fetch_result.cache_path)
    except Exception:
        cache_path = None
    if cache_path and cache_path.exists():
        snapshot_dst = analysis_csv_dir / "price_cache_snapshot.csv"
        try:
            shutil.copy2(cache_path, snapshot_dst)
            snapshot_sha256 = _sha256_file(snapshot_dst)
            (analysis_csv_dir / "price_cache_snapshot_sha256.txt").write_text(
                snapshot_sha256 + "\n", encoding="utf-8"
            )
            snapshot_records.extend(
                [
                    {"parameter": "price_cache_path", "value": str(cache_path)},
                    {"parameter": "price_cache_snapshot", "value": str(snapshot_dst)},
                    {"parameter": "price_cache_sha256", "value": snapshot_sha256},
                ]
            )
        except Exception as exc:  # pragma: no cover
            snapshot_records.append({"parameter": "price_cache_snapshot_error", "value": repr(exc)})

    config_records: List[Dict[str, object]] = []
    for key, value in sorted(vars(args).items()):
        config_records.append({"parameter": key, "value": value})
    if snapshot_records:
        config_records.extend(snapshot_records)
    config_csv_path = analysis_csv_dir / "2-experiment_config.csv"
    pd.DataFrame(config_records).to_csv(config_csv_path, index=False)
    try:
        shutil.copy2(config_csv_path, analysis_dir / "2-experiment_config.csv")
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # Data overview plots + correlation diagnostics
    # ---------------------------------------------------------------------
    start_ts = pd.Timestamp(loader_cfg.start)
    data_fig_dir = analysis_fig_dir / "data_overview"
    data_fig_dir.mkdir(parents=True, exist_ok=True)
    plot_time_series(bundle.dataset.prices, "価格時系列", start_ts, data_fig_dir / "data_prices.png")
    plot_time_series(bundle.dataset.returns, "リターン時系列", start_ts, data_fig_dir / "data_returns.png")
    plot_time_series(bundle.dataset.momentum, "モメンタム指標（max window）", start_ts, data_fig_dir / "data_momentum_max.png")
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
        export_max_return_winner_counts(
            returns_matrix,
            analysis_csv_dir / "asset_max_return_wins.csv",
            data_fig_dir / "asset_max_return_wins.png",
        )

    # Multi features
    X_feat, feature_names = build_multi_features(bundle, cfg_multi)
    feature_dir = analysis_csv_dir / "multi_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    try:
        ts_idx = pd.Index(bundle.dataset.timestamps)
        for k_idx, name in enumerate(feature_names):
            df = pd.DataFrame(X_feat[:, :, k_idx], index=ts_idx, columns=tickers)
            df.reset_index(names="date").to_csv(feature_dir / f"{name}.csv", index=False)
            # quick plot
            try:
                plot_time_series(df, f"feature: {name}", start_ts, data_fig_dir / f"feature_{name}.png")
            except Exception:
                pass
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # Model list / flex formulations
    # ---------------------------------------------------------------------
    model_keys = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    supported = {"ols", "ipo", "ipo_grad", "spo_plus", "flex"}
    filtered = [m for m in model_keys if m in supported]
    if not filtered:
        raise ValueError(
            f"Multi-feature experiment supports only models {supported}, got {model_keys}"
        )
    skipped = [m for m in model_keys if m not in supported]
    if skipped:
        print(f"[real-data-multi] skipping unsupported models in multi experiment: {skipped}")

    stats_results: List[Dict[str, Any]] = []
    period_rows: List[Dict[str, Any]] = []
    wealth_dict: Dict[str, pd.DataFrame] = {}
    weight_dict: Dict[str, pd.DataFrame] = {}
    train_window_records: List[Dict[str, Any]] = []
    rebalance_records: List[Dict[str, Any]] = []
    rebalance_log_frames: List[pd.DataFrame] = []
    flex_rebalance_logs: List[pd.DataFrame] = []
    delta_records: List[Dict[str, Any]] = []

    flex_forms_raw = parse_commalist(getattr(args, "flex_formulation", "kkt"))
    if not flex_forms_raw:
        flex_forms_raw = ["dual"]
    ensemble_aliases = {"dual&kkt", "dual_kkt", "dual_kkt_ens", "ensemble", "ens"}
    flex_ensemble_enabled = any(f in ensemble_aliases for f in flex_forms_raw)
    flex_formulations = [f for f in flex_forms_raw if f not in ensemble_aliases]
    if not flex_formulations:
        flex_formulations = ["dual"]
    for form in flex_formulations:
        if form not in {"dual", "kkt"}:
            raise ValueError(f"Unknown flex formulation '{form}'. Use 'dual' or 'kkt'.")

    # Options dictionaries
    flex_options: Dict[str, Any] = {
        "flex_solver": getattr(args, "flex_solver", "knitro"),
        "flex_lambda_theta_anchor": getattr(args, "flex_lambda_theta_anchor", 0.0),
        "flex_lambda_theta_iso": getattr(args, "flex_lambda_theta_iso", 0.0),
        "flex_theta_anchor_mode": getattr(args, "flex_theta_anchor_mode", "ipo"),
        "flex_theta_init_mode": getattr(args, "flex_theta_init_mode", "ipo"),
        "flex_knitro_nlp_algorithm": int(getattr(args, "nlp_algorithm", 1)),
        "flex_knitro_hessopt": int(getattr(args, "hessopt", 2)),
    }
    spo_plus_options: Dict[str, Any] = {
        "spo_plus_epochs": getattr(args, "spo_plus_epochs", 500),
        "spo_plus_lr": getattr(args, "spo_plus_lr", 1e-3),
        "spo_plus_batch_size": getattr(args, "spo_plus_batch_size", 0),
        "spo_plus_lambda_reg": getattr(args, "spo_plus_lambda_reg", 0.0),
        "spo_plus_lambda_anchor": getattr(args, "spo_plus_lambda_anchor", 0.0),
        "spo_plus_theta_anchor_mode": getattr(args, "spo_plus_theta_anchor_mode", "ipo"),
        "spo_plus_risk_constraint": getattr(args, "spo_plus_risk_constraint", True),
        "spo_plus_risk_mult": getattr(args, "spo_plus_risk_mult", 2.0),
        "spo_plus_init_mode": getattr(args, "spo_plus_init_mode", "ipo"),
    }
    ipo_grad_options: Dict[str, Any] = {
        "ipo_grad_epochs": getattr(args, "ipo_grad_epochs", 500),
        "ipo_grad_lr": getattr(args, "ipo_grad_lr", 1e-3),
        "ipo_grad_batch_size": getattr(args, "ipo_grad_batch_size", 0),
        "ipo_grad_qp_max_iter": getattr(args, "ipo_grad_qp_max_iter", 5000),
        "ipo_grad_qp_tol": getattr(args, "ipo_grad_qp_tol", 1e-6),
        "ipo_grad_init_mode": getattr(args, "ipo_grad_init_mode", "ipo"),
        "ipo_grad_lambda_anchor": getattr(args, "ipo_grad_lambda_anchor", 0.0),
        "ipo_grad_theta_anchor_mode": getattr(args, "ipo_grad_theta_anchor_mode", "ipo"),
        "ipo_grad_seed": getattr(args, "ipo_grad_seed", None),
    }

    # ---------------------------------------------------------------------
    # Build run specs + parallelize by model group (same idea as real_data_run.py)
    # ---------------------------------------------------------------------
    @dataclass(frozen=True)
    class _RunSpec:
        order: int
        model_key: str
        label: str
        form: Optional[str]
        train_window: int

    def _group_name(model_key: str) -> str:
        if model_key in {"ols", "ipo"}:
            return "base"
        if model_key == "ipo_grad":
            return "ipo_grad"
        if model_key == "spo_plus":
            return "spo_plus"
        if model_key == "flex":
            return "flex"
        return "other"

    run_specs: List[_RunSpec] = []
    order = 0
    for model_key in filtered:
        if model_key not in supported:
            continue
        formulations = flex_formulations if model_key == "flex" else [None]
        for form in formulations:
            label = model_key
            if model_key == "flex" and (len(formulations) > 1 or (form and form != "dual")):
                label = f"{model_key}_{form}"
            effective_train_window = int(model_train_windows.get(model_key, args.train_window))
            run_specs.append(
                _RunSpec(
                    order=order,
                    model_key=model_key,
                    label=label,
                    form=form,
                    train_window=effective_train_window,
                )
            )
            train_window_records.append(
                {
                    "model": label,
                    "base_model": model_key,
                    "train_window": effective_train_window,
                    "override": "yes" if model_key in model_train_windows else "no",
                }
            )
            order += 1

    jobs: Dict[str, List[_RunSpec]] = {}
    for spec in run_specs:
        jobs.setdefault(_group_name(spec.model_key), []).append(spec)
    job_order = ["base", "ipo_grad", "spo_plus", "flex"]
    job_items: List[Tuple[str, List[_RunSpec]]] = [
        (name, jobs[name]) for name in job_order if name in jobs and jobs[name]
    ]
    auto_workers = len(job_items)
    requested_workers = int(getattr(args, "jobs", 0) or 0)
    max_workers = requested_workers if requested_workers > 0 else auto_workers
    max_workers = max(1, min(max_workers, auto_workers)) if auto_workers > 0 else 1
    if auto_workers > 1:
        print(
            f"[real-data-multi] running model jobs in parallel: {auto_workers} groups, max_workers={max_workers}"
        )

    eval_start_ts = pd.Timestamp(args.start)
    tee = bool(getattr(args, "tee", False))
    base_seed = int(getattr(args, "base_seed", 0))
    init_seed = int(getattr(args, "init_seed", 1))
    ipo_grad_debug_kkt = bool(getattr(args, "ipo_grad_debug_kkt", False))

    def _run_group(group: str, specs: List[_RunSpec]) -> List[Tuple[_RunSpec, Dict[str, object]]]:
        outputs: List[Tuple[_RunSpec, Dict[str, object]]] = []
        for spec in specs:
            if args.debug_roll:
                print(f"[real-data-multi] rolling model={spec.label}")
            results_dir = model_outputs_dir / spec.label
            run_result = run_rolling_experiment_multi(
                model_key=spec.model_key,
                model_label=spec.label,
                bundle=bundle,
                X_feat=X_feat,
                delta_up=delta_up,
                delta_down_candidates=delta_down_list,
                trading_cost_enabled=trading_costs_enabled,
                trading_cost_default_bps=trading_cost_default_bps,
                asset_cost_overrides=asset_cost_overrides_dec,
                flex_options=flex_options,
                spo_plus_options=spo_plus_options,
                ipo_grad_options=ipo_grad_options,
                train_window=spec.train_window,
                rebal_interval=args.rebal_interval,
                debug_roll=args.debug_roll,
                debug_dir=debug_dir,
                results_model_dir=results_dir,
                tee=tee,
                asset_pred_dir=asset_pred_dir,
                formulation=spec.form,
                eval_start=eval_start_ts,
                ipo_grad_debug_kkt=ipo_grad_debug_kkt,
                base_seed=base_seed,
                init_seed=init_seed,
            )
            # condition_numbers はデータ依存でモデル間で共通なので、analysis/figures に 1 枚だけ保存する
            cond_src = results_dir / "condition_numbers.png"
            cond_dst = analysis_fig_dir / "condition_numbers.png"
            if cond_src.exists() and not cond_dst.exists():
                try:
                    shutil.copy2(cond_src, cond_dst)
                except Exception:
                    pass
            outputs.append((spec, run_result))
        return outputs

    group_results: List[Tuple[_RunSpec, Dict[str, object]]] = []
    if len(job_items) > 1 and max_workers > 1:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_group, group, specs): group for group, specs in job_items
            }
            for fut in cf.as_completed(futures):
                group_results.extend(fut.result())
    else:
        for group, specs in job_items:
            group_results.extend(_run_group(group, specs))
    group_results.sort(key=lambda pair: pair[0].order)

    for spec, run_result in group_results:
        stats_results.append(run_result["stats"])
        reb_summary = run_result.get("rebalance_summary", {})
        reb_df = run_result.get("rebalance_df")
        if isinstance(reb_df, pd.DataFrame) and not reb_df.empty:
            rebalance_log_frames.append(reb_df)
            if spec.model_key == "flex":
                flex_rebalance_logs.append(reb_df)
        if reb_summary:
            record = {
                "model": spec.label,
                "base_model": spec.model_key,
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
            period_entry["model"] = display_model_name(spec.label)
            period_entry["train_window"] = run_result["stats"].get("train_window", spec.train_window)
            period_rows.append(period_entry)
        wealth_dict[spec.label] = run_result["wealth_df"][["date", "wealth"]]
        if not run_result["weights_df"].empty:
            weight_dict[spec.label] = run_result["weights_df"]
        for row in run_result.get("delta_history", []) or []:
            delta_records.append(dict(row))

    # flex dual/kkt ensemble (multi) を構成
    if flex_ensemble_enabled:
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

    # Benchmarks (TSMOM needs an int momentum window; use the max)
    from types import SimpleNamespace

    bench_args = SimpleNamespace(**vars(args))
    setattr(bench_args, "momentum_window", int(effective_momentum_window))
    bench_stats, bench_wealth = run_benchmark_suite(
        bundle,
        benchmarks=benchmark_specs,
        args=bench_args,
        trading_costs_enabled=trading_costs_enabled,
        asset_cost_overrides=asset_cost_overrides_dec,
        eval_start=eval_start_ts,
    )
    for entry in bench_stats:
        stats_results.append(entry)
    for key, df in bench_wealth.items():
        wealth_dict[key] = df

    # summary.csv など出力
    summary_raw_df = pd.DataFrame(stats_results)
    summary_df = pd.DataFrame()
    summary_csv_path = analysis_csv_dir / "1-summary.csv"
    summary_cost_csv_path = analysis_csv_dir / "1-summary_cost.csv"
    if not summary_raw_df.empty:
        if "model" in summary_raw_df.columns:
            summary_raw_df["model"] = summary_raw_df["model"].map(display_model_name)
        if "max_drawdown" in summary_raw_df.columns:
            summary_raw_df["max_drawdown"] = summary_raw_df["max_drawdown"].astype(float)
        gross_df = summary_raw_df.drop(
            columns=[c for c in summary_raw_df.columns if c.endswith("_net")],
            errors="ignore",
        ).drop(columns=["trading_cost_bps"], errors="ignore")
        summary_df = format_summary_for_output(gross_df)
        summary_df.to_csv(summary_csv_path, index=False)
        summary_cost_df = format_summary_for_output(build_cost_adjusted_summary(summary_raw_df))
        summary_cost_df.to_csv(summary_cost_csv_path, index=False)
    else:
        summary_csv_path.write_text("")
        summary_cost_csv_path.write_text("")

    if period_rows:
        period_df = pd.DataFrame(period_rows)
        if "model" in period_df.columns:
            period_df["model"] = period_df["model"].map(display_model_name)
        period_df.to_csv(analysis_csv_dir / "period_metrics.csv", index=False)

    if wealth_dict:
        # Solver status events (for plotting annotations in window analyses).
        solver_events_by_model: Dict[str, pd.DataFrame] = {}
        if rebalance_log_frames:
            tmp_events: Dict[str, List[pd.DataFrame]] = {}
            for df in rebalance_log_frames:
                if df is None or df.empty:
                    continue
                statuses = df.get("solver_status", "").astype(str)
                mask = ~statuses.str.contains("optimal", case=False, na=False)
                mask &= ~statuses.str.contains("ok", case=False, na=False)
                df_bad = df.loc[mask]
                if df_bad.empty:
                    continue
                need_cols = {"model", "rebalance_date", "solver_status"}
                if not need_cols.issubset(df_bad.columns):
                    continue
                for model, sub in df_bad[list(need_cols)].groupby("model"):
                    tmp_events.setdefault(model, []).append(sub)
            solver_events_by_model = {
                model: pd.concat(chunks, ignore_index=True).sort_values("rebalance_date")
                for model, chunks in tmp_events.items()
            }

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
            plot_drawdown_curves(
                {m: df for m, df in wealth_dict.items()},
                analysis_fig_dir / "drawdown_curves.png",
            )

            # 1) Wealth events (all models): write primary copy under analysis/ and also under analysis/figures/
            wealth_events_base_path = analysis_dir / "wealth_events.png"
            wealth_events_primary_path = analysis_dir / "2-wealth_events.png"
            plot_wealth_with_events(
                {m: df for m, df in wealth_dict.items()},
                wealth_events_base_path,
            )
            if wealth_events_primary_path.exists():
                try:
                    shutil.copy2(
                        wealth_events_primary_path,
                        analysis_fig_dir / wealth_events_primary_path.name,
                    )
                except Exception:
                    pass

            # 2) Primary learning models only (DFL + IPO + SPO+)
            primary_models = {
                m: df
                for m, df in wealth_dict.items()
                if any(key in str(m) for key in ["flex", "ipo", "spo_plus"])
            }
            if primary_models:
                plot_wealth_with_events(
                    primary_models,
                    analysis_fig_dir / "wealth_events_dfl_only.png",
                )

            # 3) flex_dual vs benchmarks (if present)
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
                    key_str = str(key)
                    if not (key_str.startswith("benchmark_") or key_str == "benchmark_equal_weight"):
                        continue
                    dual_vs_others[key] = df
                if len(dual_vs_others) > 1:
                    plot_wealth_with_events(
                        dual_vs_others,
                        analysis_fig_dir / "wealth_events_flex_dual_vs_baselines.png",
                    )

            # 4) Crisis windows (normalized to 1 at window start)
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

            # 5) 5-year rolling windows
            five_year_dir = analysis_fig_dir / "wealth_windows_5y"
            five_year_dir_all = five_year_dir / "all_models"
            five_year_dir_dfl = five_year_dir / "dfl_only"
            five_year_dir_all.mkdir(parents=True, exist_ok=True)
            five_year_dir_dfl.mkdir(parents=True, exist_ok=True)
            all_dates: List[pd.Timestamp] = []
            for df in wealth_dict.values():
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
                    sub_dict = {}
                    for m, df in wealth_dict.items():
                        if df.empty or "date" not in df.columns:
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
        # Event-window weights comparison (figures/event_windows)
        event_fig_dir = analysis_fig_dir / "event_windows"
        event_fig_dir.mkdir(parents=True, exist_ok=True)
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
                plot_weight_comparison(sub_weights, event_fig_dir / f"weights_comparison_{name}.png")

        # 2-year windows stacked weights (+ winner + N_eff)
        weights_2y_dir = analysis_fig_dir / "weights_windows_2y"
        weights_all_dir = weights_2y_dir / "all_models"
        weights_dfl_dir = weights_2y_dir / "dfl_only"
        weights_all_dir.mkdir(parents=True, exist_ok=True)
        weights_dfl_dir.mkdir(parents=True, exist_ok=True)
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
                sub_weights = {}
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
                        returns_df=bundle.dataset.returns,
                    )
                    flex_only = {k: v for k, v in sub_weights.items() if "flex" in str(k)}
                    if flex_only:
                        plot_weight_window_with_connections(
                            flex_only,
                            weights_dfl_dir / f"weights_window_2y_{window_label}.png",
                            returns_df=bundle.dataset.returns,
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
        # Winner capture analysis: whether each model allocates >=5% weight to the weekly winner asset.
        try:
            winner_dir = analysis_fig_dir / "winner_capture"
            winner_dir.mkdir(parents=True, exist_ok=True)
            export_winner_capture_frequency(
                returns_df=bundle.dataset.returns,
                weight_dict=weight_dict,
                threshold=0.05,
                csv_path=analysis_csv_dir / "winner_capture_rate.csv",
                fig_path=winner_dir / "winner_capture_rate.png",
                by_asset_csv_path=analysis_csv_dir / "winner_capture_by_asset.csv",
                by_asset_fig_path=winner_dir / "winner_capture_by_asset_heatmap.png",
            )
            flex_only = {k: v for k, v in weight_dict.items() if "flex" in str(k)}
            if flex_only:
                export_winner_capture_frequency(
                    returns_df=bundle.dataset.returns,
                    weight_dict=flex_only,
                    threshold=0.05,
                    csv_path=analysis_csv_dir / "winner_capture_rate_dfl_only.csv",
                    fig_path=winner_dir / "winner_capture_rate_dfl_only.png",
                    by_asset_csv_path=analysis_csv_dir / "winner_capture_by_asset_dfl_only.csv",
                    by_asset_fig_path=winner_dir / "winner_capture_by_asset_heatmap_dfl_only.png",
                    title="週次最大リターン資産の捕捉率（DFLのみ, w>=閾値）",
                )
        except Exception as exc:  # pragma: no cover
            print(f"[analysis-multi] winner capture analysis failed: {exc}")

    if train_window_records:
        pd.DataFrame(train_window_records).to_csv(
            analysis_csv_dir / "model_train_windows.csv", index=False
        )
    if rebalance_records:
        rebalance_df = pd.DataFrame(rebalance_records)
        rebalance_df.to_csv(analysis_csv_dir / "rebalance_summary.csv", index=False)

    solver_debug_dir = analysis_fig_dir / "solver_debug"
    if rebalance_log_frames:
        solver_debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            rebalance_log_df = pd.concat(rebalance_log_frames, ignore_index=True)
            plot_solver_summary_bars(rebalance_log_df, solver_debug_dir)
        except Exception:  # pragma: no cover
            pass
    if flex_rebalance_logs:
        solver_debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            flex_debug_df = pd.concat(flex_rebalance_logs, ignore_index=True)
            plot_flex_solver_debug(flex_debug_df, solver_debug_dir / "flex_solver_debug.png")
        except Exception:  # pragma: no cover
            pass

    run_extended_analysis(analysis_csv_dir, analysis_fig_dir, model_outputs_dir, asset_pred_dir)
    if delta_records:
        delta_df = pd.DataFrame(delta_records)
        # Keep legacy/debug file name for multi runs
        delta_df.to_csv(analysis_csv_dir / "flex_delta_history.csv", index=False)
        # Match real_data_run.py artifact names
        delta_df.to_csv(analysis_csv_dir / "delta_trajectory.csv", index=False)
        delta_plot_df = delta_df.copy()
        if "model" in delta_plot_df.columns and "delta_used" in delta_plot_df.columns:
            means = delta_plot_df.groupby("model")["delta_used"].mean().to_dict()
            delta_plot_df["model"] = delta_plot_df["model"].map(
                lambda m: (
                    f"{m} (avg={means.get(m, float('nan')):.3f})"
                    if m in means and np.isfinite(means.get(m, np.nan))
                    else str(m)
                )
            )
        plot_delta_paths(delta_plot_df, analysis_fig_dir / "delta_paths.png")
    if rebalance_log_frames:
        pd.concat(rebalance_log_frames, ignore_index=True).to_csv(
            analysis_csv_dir / "rebalance_log_all_models.csv", index=False
        )
    try:
        update_experiment_ledger(RESULTS_ROOT, outdir, args, summary_df, analysis_csv_dir, bundle_summary)
    except Exception:
        pass

    print(f"[real-data-multi] finished. outputs -> {outdir}")
    print(f"[real-data-multi] debug artifacts -> {debug_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""
source /Users/kensei/Documents/VScode/GraduationResearch/gurobi-env/bin/activate
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.real_data_run_multi \
  --tickers "SPY,GLD,EEM,TLT" \
  --start 2006-01-01 --end 2025-12-31 \
  --multi-vix \
  --momentum-window 26 \
  --cov-window 13 \
  --cov-method oas \
  --train-window 26 \
  --rebal-interval 4 \
  --delta 0.5 \
  --models "ols,ipo,ipo_grad,spo_plus,flex" \
  --flex-solver knitro \
  --flex-formulation 'kkt' \
  --lambda-theta-anchor 0.0 \
  --theta-anchor-mode ipo \
  --theta-init-mode ipo \
  --ipo-grad-seed 42 \
  --benchmarks "spy,1/n" \
  --jobs 1 \
  --debug-roll \
  --nlp_algorithm 3 \
  --hessopt 4 \


--nlp_algorithm 1
--hessopt 2
--momentum-window 4
--multi-vix
--multi-intercept
"""
