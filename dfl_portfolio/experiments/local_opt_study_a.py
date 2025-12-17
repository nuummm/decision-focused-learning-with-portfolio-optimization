from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Avoid GUI backends and OpenMP SHM issues; must be set before importing numpy/scipy.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getenv("TMPDIR", "/tmp"), "mplconfig"))
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np
import pandas as pd

from dfl_portfolio.real_data.loader import MarketLoaderConfig
from dfl_portfolio.real_data.pipeline import PipelineConfig, build_data_bundle
from dfl_portfolio.real_data.cli import make_output_dir, parse_tickers, parse_trading_cost_map


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
RESULTS_BASE = PROJECT_ROOT / "results"
RESULTS_ROOT = RESULTS_BASE / "exp_localopt_A"


def _parse_int_list(value: str) -> List[int]:
    text = (value or "").strip()
    if not text:
        raise ValueError("seeds must be a comma-separated list of integers")
    out: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("seeds must contain at least one integer")
    return out


def _is_ok(status: object) -> bool:
    s = str(status or "").lower()
    return ("optimal" in s) or ("ok" in s) or ("completed" in s)


def _warn_flags_flex(reb_df: pd.DataFrame) -> Tuple[int, int, int]:
    # success / warning / no-solution counts per retrain event
    success = 0
    warning = 0
    no_solution = 0
    for _, row in reb_df.iterrows():
        status = row.get("solver_status", "")
        if not status:
            no_solution += 1
            continue
        if _is_ok(status):
            success += 1
        else:
            warning += 1
    return success, warning, no_solution


def _warn_flags_ipo_grad(reb_df: pd.DataFrame) -> Tuple[int, int, int]:
    success = 0
    warning = 0
    no_solution = 0
    eq_tol = 1e-4
    ineq_tol = 1e-8
    for _, row in reb_df.iterrows():
        status = row.get("solver_status", "")
        if not status:
            no_solution += 1
            continue
        eq = float(row.get("train_eq_viol_max", np.nan))
        ineq = float(row.get("train_ineq_viol_max", np.nan))
        if (np.isfinite(eq) and abs(eq) > eq_tol) or (np.isfinite(ineq) and abs(ineq) > ineq_tol):
            warning += 1
        elif _is_ok(status):
            success += 1
        else:
            warning += 1
    return success, warning, no_solution


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, "n": int(arr.size)}


def _resolve_a_base_mode(value: str) -> Dict[str, Any]:
    """
    Map --a-base-mode to a consistent set of initialization + penalty settings.

    Modes:
      - none:      theta_init=none, theta penalties=0
      - init-ipo:  theta_init=ipo,  theta penalties=0
      - pen-10:    theta_init=none, lambda_theta_anchor=10 with anchor=ipo
    """
    mode = (value or "").strip().lower().replace("_", "-")
    if mode not in {"none", "init-ipo", "pen-10"}:
        raise ValueError("--a-base-mode must be one of: none, init-ipo, pen-10")
    if mode == "none":
        return {
            "a_base_mode": "none",
            "theta_init_mode": "none",
            "flex_lambda_theta_anchor": 0.0,
            "flex_lambda_theta_iso": 0.0,
            "flex_theta_anchor_mode": "none",
            "ipo_grad_init_mode": "none",
            "ipo_grad_lambda_anchor": 0.0,
            "ipo_grad_theta_anchor_mode": "ipo",
        }
    if mode == "init-ipo":
        return {
            "a_base_mode": "init-ipo",
            "theta_init_mode": "ipo",
            "flex_lambda_theta_anchor": 0.0,
            "flex_lambda_theta_iso": 0.0,
            "flex_theta_anchor_mode": "none",
            "ipo_grad_init_mode": "ipo",
            "ipo_grad_lambda_anchor": 0.0,
            "ipo_grad_theta_anchor_mode": "ipo",
        }
    # pen-10
    return {
        "a_base_mode": "pen-10",
        "theta_init_mode": "none",
        "flex_lambda_theta_anchor": 10.0,
        "flex_lambda_theta_iso": 0.0,
        "flex_theta_anchor_mode": "ipo",
        "ipo_grad_init_mode": "none",
        "ipo_grad_lambda_anchor": 10.0,
        "ipo_grad_theta_anchor_mode": "ipo",
    }


def _plot_cumret_overlay(curves: Dict[int, pd.DataFrame], out_path: Path, title: str) -> None:
    if not curves:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    seeds = sorted(curves.keys())
    for seed in seeds:
        df = curves[seed]
        ax.plot(pd.to_datetime(df["date"]), df["wealth"], alpha=0.35, linewidth=1.0, label=None)
    # Mean curve (align by date)
    merged = None
    for seed in seeds:
        tmp = curves[seed][["date", "wealth"]].rename(columns={"wealth": f"w_{seed}"})
        merged = tmp if merged is None else merged.merge(tmp, on="date", how="inner")
    if merged is not None and merged.shape[0] > 0:
        wealth_cols = [c for c in merged.columns if c.startswith("w_")]
        mean_w = merged[wealth_cols].mean(axis=1)
        ax.plot(pd.to_datetime(merged["date"]), mean_w, color="black", linewidth=2.0, label="mean")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("wealth")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def _plot_metric_boxplot(
    *,
    runs_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
    y_label: str,
    model_order: List[str],
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    if metric not in runs_df.columns:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return

    df = runs_df[["model_key", metric]].copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[np.isfinite(df[metric].to_numpy())]
    if df.empty:
        return

    present = sorted(df["model_key"].unique().tolist())
    models = [m for m in model_order if m in present]
    models.extend([m for m in present if m not in models])
    if not models:
        return

    data: List[np.ndarray] = []
    for m in models:
        vals = df[df["model_key"] == m][metric].astype(float).to_numpy()
        if vals.size == 0:
            vals = np.asarray([np.nan], dtype=float)
        data.append(vals)

    fig_w = max(7.0, 1.7 * len(models))
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, m in zip(bp["boxes"], models):
        c = None
        if color_map and m in color_map:
            c = color_map[m]
        if not c:
            c = "#1f77b4"
        patch.set_facecolor(c)
        patch.set_alpha(0.45)

    # Mean markers per model.
    means = [float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan") for arr in data]
    ax.scatter(np.arange(1, len(models) + 1), means, color="black", marker="D", s=24, zorder=4, label="mean")

    ax.set_xticks(np.arange(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_metric_boxplots(runs_df: pd.DataFrame, out_dir: Path, *, color_map: Optional[Dict[str, str]] = None) -> None:
    if runs_df.empty:
        return
    metrics = [
        "ann_return",
        "ann_return_net",
        "sharpe",
        "sharpe_net",
        "sortino",
        "sortino_net",
        "ann_volatility",
        "ann_volatility_net",
        "max_drawdown",
        "cvar_95",
        "terminal_wealth",
        "terminal_wealth_net",
        "total_return",
        "total_return_net",
        "avg_turnover",
        "avg_trading_cost",
        "r2",
        "elapsed_fit_time_mean",
        "elapsed_total_fit_time",
        "elapsed_total_run_sec",
    ]
    model_order = ["flex_dual", "flex_kkt", "ipo_grad"]
    box_dir = out_dir / "metric_boxplots"
    for metric in metrics:
        if metric not in runs_df.columns:
            continue
        _plot_metric_boxplot(
            runs_df=runs_df,
            metric=metric,
            out_path=box_dir / f"{metric}.png",
            title=f"A: metric distribution ({metric})",
            y_label=metric,
            model_order=model_order,
            color_map=color_map,
        )


def _plot_flex_solver_status_bars(runs_df: pd.DataFrame, out_path: Path) -> None:
    """Compare flex_dual vs flex_kkt solver status counts across seeds + overall mean."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return

    needed = {"seed", "model_key", "success_count", "warning_count", "no_solution_count"}
    if not needed.issubset(set(runs_df.columns)):
        return

    models = ["flex_dual", "flex_kkt"]
    parts: Dict[str, pd.DataFrame] = {}
    for m in models:
        part = runs_df[runs_df["model_key"] == m].copy()
        if part.empty:
            continue
        part = part.sort_values("seed")
        mean_row = {
            "seed": "mean",
            "success_count": float(part["success_count"].mean()),
            "warning_count": float(part["warning_count"].mean()),
            "no_solution_count": float(part["no_solution_count"].mean()),
        }
        part2 = pd.concat(
            [part[["seed", "success_count", "warning_count", "no_solution_count"]], pd.DataFrame([mean_row])],
            ignore_index=True,
        )
        part2["seed"] = part2["seed"].astype(str)
        parts[m] = part2

    if not parts:
        return

    fig, axes = plt.subplots(1, len(parts), figsize=(12, 4), sharey=True)
    if len(parts) == 1:
        axes = [axes]

    colors = {
        "success_count": "#2ca02c",     # green
        "warning_count": "#ff7f0e",     # orange
        "no_solution_count": "#d62728",  # red
    }
    labels = {
        "success_count": "OK",
        "warning_count": "Warning",
        "no_solution_count": "No-solution",
    }

    for ax, (model, part) in zip(axes, parts.items()):
        x = np.arange(part.shape[0])
        bottom = np.zeros(part.shape[0], dtype=float)
        for col in ["success_count", "warning_count", "no_solution_count"]:
            vals = part[col].astype(float).to_numpy()
            ax.bar(x, vals, bottom=bottom, color=colors[col], label=labels[col])
            bottom = bottom + vals
        ax.set_title(f"Flex status counts ({model})")
        ax.set_xlabel("seed")
        ax.set_xticks(x)
        ax.set_xticklabels(part["seed"].tolist(), rotation=0)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("count (retrain events)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_flex_elapsed_bars(runs_df: pd.DataFrame, out_path: Path) -> None:
    """Compare flex_dual vs flex_kkt elapsed fit time (mean per retrain) across seeds + overall mean."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return

    needed = {"seed", "model_key", "elapsed_fit_time_mean"}
    if not needed.issubset(set(runs_df.columns)):
        return

    part = runs_df[runs_df["model_key"].isin(["flex_dual", "flex_kkt"])].copy()
    if part.empty:
        return

    pivot = part.pivot_table(index="seed", columns="model_key", values="elapsed_fit_time_mean", aggfunc="mean").sort_index()
    mean_row = pivot.mean(axis=0).to_frame().T
    mean_row.index = ["mean"]
    pivot2 = pd.concat([pivot, mean_row], axis=0)

    x_labels = [str(i) for i in pivot2.index.tolist()]
    x = np.arange(len(x_labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 4))
    dual = pivot2.get("flex_dual", pd.Series(index=pivot2.index, dtype=float)).astype(float).to_numpy()
    kkt = pivot2.get("flex_kkt", pd.Series(index=pivot2.index, dtype=float)).astype(float).to_numpy()
    ax.bar(x - width / 2, dual, width=width, label="flex_dual", color="#1f77b4")
    ax.bar(x + width / 2, kkt, width=width, label="flex_kkt", color="#9467bd")
    ax.set_title("Flex elapsed fit time comparison")
    ax.set_xlabel("seed")
    ax.set_ylabel("elapsed_fit_time_mean (sec)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    formulation: Optional[str] = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment A: local-optimality robustness (multi-seed)")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--tickers", type=str, default="SPY,GLD,EEM,TLT")
    p.add_argument("--start", type=str, default="2006-01-01")
    p.add_argument("--end", type=str, default="2025-12-01")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--price-field", type=str, default="Close")
    p.add_argument("--return-kind", type=str, default="log", choices=["simple", "log"])
    p.add_argument("--frequency", type=str, default="weekly", choices=["daily", "weekly", "monthly"])
    p.add_argument("--resample-rule", type=str, default="W-FRI")
    p.add_argument("--momentum-window", type=int, default=26)
    p.add_argument("--return-horizon", type=int, default=1)
    p.add_argument("--cov-window", type=int, default=13)
    p.add_argument("--cov-method", type=str, default="oas", choices=["diag", "oas", "robust_lw", "mini_factor"])
    p.add_argument("--cov-shrinkage", type=float, default=0.94)
    p.add_argument("--cov-eps", type=float, default=1e-6)
    p.add_argument("--cov-robust-huber-k", type=float, default=1.5)
    p.add_argument("--cov-factor-rank", type=int, default=1)
    p.add_argument("--cov-factor-shrinkage", type=float, default=0.5)
    p.add_argument("--cov-ewma-alpha", type=float, default=0.97)
    p.add_argument("--no-auto-adjust", action="store_true")
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--train-window", type=int, default=26)
    p.add_argument("--rebal-interval", type=int, default=4)
    p.add_argument("--delta-up", type=float, default=0.5)
    p.add_argument("--delta-down", type=float, default=0.5)
    p.add_argument(
        "--trading-cost-per-asset",
        type=str,
        default="",
        help="Optional overrides like 'SPY:5,GLD:8' (basis points) applied per ticker.",
    )
    p.add_argument("--outdir", type=Path, default=None)

    p.add_argument(
        "--w-warmstart",
        "--w-warm-start",
        dest="w_warmstart",
        action="store_true",
        default=False,
        help="Enable warm-start for decision variables w when available (default: disabled).",
    )
    p.add_argument(
        "--no-w-warmstart",
        "--no-w-warm-start",
        dest="w_warmstart",
        action="store_false",
        help="Disable warm-start for decision variables w (flex warm-start init). This is the default.",
    )

    # Flex solver
    p.add_argument("--flex-solver", type=str, default="knitro")
    p.add_argument("--flex-maxtime", type=float, default=180.0)
    p.add_argument(
        "--flex-aux-init-radius-w",
        type=float,
        default=None,
        help=(
            "Experiment A (flex only): L2 'radius' for w initialization noise. "
            "If set, overrides --flex-aux-init-sigma-w via sigma_w = radius_w / sqrt(d), "
            "where d is the number of assets (simplex dimension). "
            "Note: actual ||w0-w_base|| depends on simplex projection; diagnostics are logged."
        ),
    )
    p.add_argument("--flex-aux-init-sigma-w", type=float, default=0.05)
    p.add_argument("--flex-aux-init-sigma-lam", type=float, default=1e-2)
    p.add_argument("--flex-aux-init-sigma-mu", type=float, default=1e-2)

    # A-base preset (theta init + theta penalty)
    p.add_argument(
        "--a-base-mode",
        type=str,
        default="init-ipo",
        choices=["none", "init-ipo", "pen-10"],
        help="Base preset: none (no init, no theta penalty), init-ipo (IPO init only), pen-10 (anchor penalty=10 with IPO anchor, init none).",
    )

    # IPO-GRAD
    p.add_argument("--ipo-grad-epochs", type=int, default=250)
    p.add_argument("--ipo-grad-lr", type=float, default=1e-3)
    p.add_argument("--ipo-grad-batch-size", type=int, default=32)
    p.add_argument("--ipo-grad-qp-max-iter", type=int, default=1500)
    p.add_argument("--ipo-grad-qp-tol", type=float, default=1e-6)
    p.add_argument("--tee", action="store_true")
    p.add_argument("--debug-roll", action="store_true", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    # Heavy imports after parsing so `--help` stays lightweight.
    from dfl_portfolio.real_data.reporting import MODEL_COLOR_MAP, display_model_name
    from dfl_portfolio.experiments.real_data_run import run_rolling_experiment
    from dfl_portfolio.registry import SolverSpec, KNITRO_DEFAULTS
    seeds = _parse_int_list(args.seeds)

    raw_asset_costs = parse_trading_cost_map(args.trading_cost_per_asset) if args.trading_cost_per_asset else {}
    asset_cost_overrides_dec = {k.upper(): float(v) / 10000.0 for k, v in raw_asset_costs.items()}
    trading_costs_enabled = bool(asset_cost_overrides_dec)

    tickers = parse_tickers(args.tickers)
    loader_cfg = MarketLoaderConfig.for_cli(
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
        cov_shrinkage=args.cov_shrinkage,
        cov_eps=args.cov_eps,
        cov_robust_huber_k=args.cov_robust_huber_k,
        cov_factor_rank=args.cov_factor_rank,
        cov_factor_shrinkage=args.cov_factor_shrinkage,
        cov_ewma_alpha=args.cov_ewma_alpha,
        auto_adjust=not args.no_auto_adjust,
        cache_dir=None,
        force_refresh=args.force_refresh,
        debug=False,
        train_window=args.train_window,
    )
    bundle = build_data_bundle(PipelineConfig(loader=loader_cfg, debug=False))

    outdir = make_output_dir(RESULTS_ROOT, args.outdir)
    analysis_dir = outdir / "analysis"
    analysis_csv = analysis_dir / "csv"
    analysis_fig = analysis_dir / "figures"
    analysis_csv.mkdir(parents=True, exist_ok=True)
    analysis_fig.mkdir(parents=True, exist_ok=True)

    # Model specs for A
    models = [
        ModelSpec(key="flex", label="flex_dual", formulation="dual"),
        ModelSpec(key="flex", label="flex_kkt", formulation="kkt"),
        ModelSpec(key="ipo_grad", label="ipo_grad"),
    ]

    # Solver specs (flex uses knitro; ipo_grad uses analytic wrapper)
    knitro_opts = dict(KNITRO_DEFAULTS)
    knitro_opts["maxtime_real"] = float(args.flex_maxtime)
    flex_solver_spec = SolverSpec(name=args.flex_solver, options=knitro_opts, tee=bool(args.tee))
    analytic_spec = SolverSpec(name="analytic", tee=bool(args.tee))

    base_cfg = _resolve_a_base_mode(getattr(args, "a_base_mode", "init-ipo"))
    if getattr(args, "tee", False):
        print(f"[local-opt-A] a_base_mode={base_cfg['a_base_mode']}")

    # Resolve aux init noise scale for w in a more interpretable way (radius overrides sigma).
    # radius_w is defined on the *pre-projection* Gaussian noise:
    #   eta ~ N(0, sigma_w^2 I_d)  =>  E||eta||_2 ≈ sigma_w * sqrt(d)
    # so we map: sigma_w := radius_w / sqrt(d).
    n_assets = int(getattr(bundle.dataset.Y, "shape", (0, 0))[1] or 0)
    sigma_w = float(args.flex_aux_init_sigma_w)
    if args.flex_aux_init_radius_w is not None:
        radius = float(args.flex_aux_init_radius_w)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("--flex-aux-init-radius-w must be a finite non-negative float.")
        if n_assets <= 0:
            raise ValueError("Cannot resolve --flex-aux-init-radius-w: n_assets is unknown/invalid.")
        sigma_w = radius / float(np.sqrt(n_assets))

    # Options
    flex_base_options: Dict[str, Any] = {
        # keep theta fixed across seeds; controlled by --a-base-mode
        "theta_init_mode": str(base_cfg["theta_init_mode"]),
        "theta_anchor_mode": str(base_cfg["flex_theta_anchor_mode"]),
        "lambda_theta_anchor": float(base_cfg["flex_lambda_theta_anchor"]),
        "lambda_theta_iso": float(base_cfg["flex_lambda_theta_iso"]),
        "w_warmstart": bool(getattr(args, "w_warmstart", True)),
        "aux_init_mode": "random",
        "aux_init_sigma_w": float(sigma_w),
        "aux_init_sigma_lam": float(args.flex_aux_init_sigma_lam),
        "aux_init_sigma_mu": float(args.flex_aux_init_sigma_mu),
        # Preserve randomized (w/lam/mu) initial values to probe local optima in Experiment A.
        # Default behavior in flex is unchanged because this flag is only set here.
        "aux_init_keep": True,
    }
    ipo_grad_options: Dict[str, Any] = {
        "ipo_grad_epochs": int(args.ipo_grad_epochs),
        "ipo_grad_lr": float(args.ipo_grad_lr),
        "ipo_grad_batch_size": int(args.ipo_grad_batch_size),
        "ipo_grad_qp_max_iter": int(args.ipo_grad_qp_max_iter),
        "ipo_grad_qp_tol": float(args.ipo_grad_qp_tol),
        "ipo_grad_init_mode": str(base_cfg["ipo_grad_init_mode"]),
        "ipo_grad_lambda_anchor": float(base_cfg["ipo_grad_lambda_anchor"]),
        "ipo_grad_theta_anchor_mode": str(base_cfg["ipo_grad_theta_anchor_mode"]),
    }

    # Aggregate outputs
    run_rows: List[Dict[str, Any]] = []
    retrain_frames: List[pd.DataFrame] = []
    wealth_curves: Dict[str, Dict[int, pd.DataFrame]] = {m.label: {} for m in models}

    total_jobs = len(seeds) * len(models)
    job_idx = 0
    job_times: List[float] = []
    print(f"[local-opt-A] start: seeds={len(seeds)} models={len(models)} jobs={total_jobs}")
    print(f"[local-opt-A] outdir={outdir}")

    for seed_i, seed in enumerate(seeds, start=1):
        print(f"[local-opt-A] seed {seed_i}/{len(seeds)} = {seed}")
        seed_dir = outdir / f"seed_{seed}"
        seed_debug = seed_dir / "debug"
        seed_models = seed_dir / "models"
        seed_debug.mkdir(parents=True, exist_ok=True)
        seed_models.mkdir(parents=True, exist_ok=True)

        for spec in models:
            job_idx += 1
            avg = (sum(job_times) / len(job_times)) if job_times else float("nan")
            eta_sec = (total_jobs - job_idx + 1) * avg if job_times else float("nan")
            eta_str = f"{eta_sec/60:.1f}m" if job_times else "n/a"
            print(f"[local-opt-A] job {job_idx}/{total_jobs} seed={seed} model={spec.label} (eta={eta_str})")
            solver_spec = flex_solver_spec if spec.key == "flex" else analytic_spec
            flex_options = None
            if spec.key == "flex":
                flex_options = dict(flex_base_options)
                flex_options["formulation"] = spec.formulation
            results_model_dir = seed_models / spec.label
            start_time = time.perf_counter()
            run_result = run_rolling_experiment(
                model_key=spec.key,
                model_label=spec.label,
                bundle=bundle,
                delta_up=float(args.delta_up),
                delta_down_candidates=[float(args.delta_down)],
                trading_cost_enabled=trading_costs_enabled,
                asset_cost_overrides=asset_cost_overrides_dec,
                solver_spec=solver_spec,
                flex_options=flex_options,
                spo_plus_options=None,
                ipo_grad_options=ipo_grad_options if spec.key == "ipo_grad" else None,
                train_window=int(args.train_window),
                rebal_interval=int(args.rebal_interval),
                debug_roll=bool(args.debug_roll),
                debug_dir=seed_debug,
                results_model_dir=results_model_dir,
                tee=bool(args.tee),
                asset_pred_dir=None,
                eval_start=pd.Timestamp(args.start),
                ipo_grad_debug_kkt=False,
                base_seed=int(seed),
            )
            elapsed_total = time.perf_counter() - start_time
            job_times.append(float(elapsed_total))

            stats = dict(run_result["stats"])
            stats["seed"] = int(seed)
            stats["model_key"] = spec.label
            stats["elapsed_total_run_sec"] = float(elapsed_total)

            reb_df = run_result.get("rebalance_df", pd.DataFrame())
            if isinstance(reb_df, pd.DataFrame) and not reb_df.empty:
                reb_df = reb_df.copy()
                reb_df["seed"] = int(seed)
                reb_df["model_key"] = spec.label
                retrain_frames.append(reb_df)
                fit_times = reb_df["elapsed_sec"].astype(float).to_numpy()
                stats["elapsed_total_fit_time"] = float(np.nansum(fit_times))
                stats["elapsed_fit_time_mean"] = float(np.nanmean(fit_times))
                stats["elapsed_fit_time_min"] = float(np.nanmin(fit_times))
                stats["elapsed_fit_time_max"] = float(np.nanmax(fit_times))
                if spec.key == "ipo_grad":
                    succ, warn, fail = _warn_flags_ipo_grad(reb_df)
                else:
                    succ, warn, fail = _warn_flags_flex(reb_df)
                stats["success_count"] = int(succ)
                stats["warning_count"] = int(warn)
                stats["no_solution_count"] = int(fail)
            else:
                stats["elapsed_total_fit_time"] = float("nan")
                stats["elapsed_fit_time_mean"] = float("nan")
                stats["elapsed_fit_time_min"] = float("nan")
                stats["elapsed_fit_time_max"] = float("nan")
                stats["success_count"] = 0
                stats["warning_count"] = 0
                stats["no_solution_count"] = 0

            run_rows.append(stats)
            wealth_df = run_result["wealth_df"][["date", "wealth"]].copy()
            wealth_curves[spec.label][int(seed)] = wealth_df
            print(
                f"[local-opt-A] done job {job_idx}/{total_jobs} seed={seed} model={spec.label} "
                f"elapsed={elapsed_total:.1f}s warn={stats.get('warning_count','?')} fail={stats.get('no_solution_count','?')}"
            )

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(analysis_csv / "runs.csv", index=False)
    if retrain_frames:
        retrain_df = pd.concat(retrain_frames, ignore_index=True)
        retrain_df.to_csv(analysis_csv / "retrain_log.csv", index=False)
    else:
        retrain_df = pd.DataFrame()

    # Summary table (mean/std over seeds)
    summary_rows: List[Dict[str, Any]] = []
    metrics = [
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
        "elapsed_total_fit_time",
        "elapsed_fit_time_mean",
        "warning_count",
        "no_solution_count",
    ]
    for model in sorted(runs_df["model_key"].unique()):
        part = runs_df[runs_df["model_key"] == model]
        row: Dict[str, Any] = {"model_key": model, "model": display_model_name(model)}
        for col in metrics:
            stats = _summary_stats(part[col].astype(float).to_numpy()) if col in part.columns else {"mean": np.nan, "std": np.nan, "n": 0}
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"] = stats["std"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(analysis_csv / "summary_table.csv", index=False)

    # Plot cumret overlays per model
    for model_label, curves in wealth_curves.items():
        if not curves:
            continue
        title = f"A: cumulative wealth overlay ({model_label})"
        color = MODEL_COLOR_MAP.get(model_label, None)
        _plot_cumret_overlay(curves, analysis_fig / f"cumret_{model_label}.png", title=title)

    # Boxplots: metric distributions across seeds (compare models)
    _plot_metric_boxplots(runs_df, analysis_fig, color_map=MODEL_COLOR_MAP)

    # Flex dual/kkt comparisons (status + time)
    if not runs_df.empty:
        _plot_flex_solver_status_bars(runs_df, analysis_fig / "flex_solver_status_counts.png")
        _plot_flex_elapsed_bars(runs_df, analysis_fig / "flex_elapsed_fit_time_mean.png")

    # Save config (include derived sigma used for w-aux init)
    cfg = vars(args).copy()
    cfg["seeds"] = seeds
    cfg["flex_aux_init_sigma_w_used"] = float(sigma_w)
    cfg["n_assets"] = int(n_assets)
    (analysis_csv / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[local-opt-A] finished. outputs -> {outdir}")


if __name__ == "__main__":  # pragma: no cover
    main()


"""
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.local_opt_study_a \
  --a-base-mode "init-ipo" \
  --flex-aux-init-radius-w 0.1 \
  --seeds "0,1,2,3,4,5,6,7,8,9" \

# 補助変数（w）の揺らぎ強度は、次のどちらでも指定できます（radius が優先）：
# - --flex-aux-init-radius-w R : 事前（射影前）ガウスノイズの「目安半径」R（L2）
# - --flex-aux-init-sigma-w σ : 各成分の標準偏差 σ
# 変換は R ≈ σ * sqrt(d)（d=資産数）を用います。
# ※ 実際の ||w0-w_base|| は simplex 射影の影響で変わるため、
#    run のメタ情報（aux_w0_minus_base_l1/l2_mean/std）で実測値を保存しています。
"""
