from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Avoid GUI backends and OpenMP SHM issues; must be set before importing numpy/scipy.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getenv("TMPDIR", "/tmp"), "mplconfig"))
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np
import pandas as pd

from dfl_portfolio.real_data.cli import build_parser, make_output_dir, parse_commalist, parse_tickers
from dfl_portfolio.real_data.loader import MarketLoaderConfig
from dfl_portfolio.real_data.pipeline import PipelineConfig, build_data_bundle

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
RESULTS_BASE = PROJECT_ROOT / "results"


RESULTS_ROOT = RESULTS_BASE / "exp_localopt_B"


def _parse_int_list(value: str) -> List[int]:
    text = (value or "").strip()
    if not text:
        return []
    out: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_b_targets(value: str) -> Tuple[List[str], bool]:
    """
    Parse --b-target as a comma-separated list.

    Supported tokens:
      - dual
      - kkt
      - ipo_grad (alias: ipo-grad)
      - both (back-compat; expands to dual,kkt)
    """
    raw = (value or "").strip()
    tokens = parse_commalist(raw) if raw else ["dual", "kkt", "ipo_grad"]

    expanded: List[str] = []
    for tok in tokens:
        t = (tok or "").strip().lower().replace("-", "_")
        if not t:
            continue
        if t == "both":
            expanded.extend(["dual", "kkt"])
        else:
            expanded.append(t)

    # preserve order, unique
    seen: set[str] = set()
    norm: List[str] = []
    for t in expanded:
        if t in seen:
            continue
        seen.add(t)
        norm.append(t)

    allowed = {"dual", "kkt", "ipo_grad"}
    unknown = set(norm) - allowed
    if unknown:
        raise ValueError(f"--b-target has unknown entries: {sorted(unknown)} (allowed: {sorted(allowed)})")

    formulations = [t for t in norm if t in ("dual", "kkt")]
    include_ipo_grad = "ipo_grad" in norm
    if not formulations and not include_ipo_grad:
        raise ValueError("--b-target must include at least one of: dual,kkt,ipo_grad")
    return formulations, include_ipo_grad


def _resolve_b_base_mode(value: str) -> Dict[str, Any]:
    """
    Map --b-base-mode to a consistent set of initialization + penalty settings.

    Modes:
      - none:      theta_init=none, theta penalties=0
      - init-ipo:  theta_init=ipo,  theta penalties=0
      - pen-10:    theta_init=none, lambda_theta_anchor=10 with anchor=ipo
    """
    mode = (value or "").strip().lower().replace("_", "-")
    if mode not in {"none", "init-ipo", "pen-10"}:
        raise ValueError("--b-base-mode must be one of: none, init-ipo, pen-10")

    if mode == "none":
        return {
            "b_base_mode": "none",
            "theta_base_init_mode": "none",
            "flex_lambda_theta_anchor": 0.0,
            "flex_lambda_theta_iso": 0.0,
            "flex_theta_anchor_mode": "none",
            "ipo_grad_lambda_anchor": 0.0,
            "ipo_grad_theta_anchor_mode": "ipo",
        }
    if mode == "init-ipo":
        return {
            "b_base_mode": "init-ipo",
            "theta_base_init_mode": "ipo",
            "flex_lambda_theta_anchor": 0.0,
            "flex_lambda_theta_iso": 0.0,
            "flex_theta_anchor_mode": "none",
            "ipo_grad_lambda_anchor": 0.0,
            "ipo_grad_theta_anchor_mode": "ipo",
        }
    # pen-10
    return {
        "b_base_mode": "pen-10",
        "theta_base_init_mode": "none",
        "flex_lambda_theta_anchor": 10.0,
        "flex_lambda_theta_iso": 0.0,
        "flex_theta_anchor_mode": "ipo",
        "ipo_grad_lambda_anchor": 10.0,
        "ipo_grad_theta_anchor_mode": "ipo",
    }


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _merge_on_date(curves: Dict[int, pd.DataFrame]) -> Optional[pd.DataFrame]:
    merged = None
    for seed, df in curves.items():
        tmp = df[["date", "wealth"]].rename(columns={"wealth": f"w_{seed}"})
        merged = tmp if merged is None else merged.merge(tmp, on="date", how="inner")
    return merged


def _get_plt():  # type: ignore
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:  # pragma: no cover
        return None


def _plot_family_plus_base(
    *,
    family_curves: Dict[int, pd.DataFrame],
    base_curve: pd.DataFrame,
    out_path: Path,
    title: str,
    family_name: str,
    family_color: str,
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for seed in sorted(family_curves.keys()):
        df = family_curves[seed]
        ax.plot(pd.to_datetime(df["date"]), df["wealth"], alpha=0.22, linewidth=1.0, color=family_color)

    merged = _merge_on_date(family_curves)
    if merged is not None and merged.shape[0] > 0:
        wealth_cols = [c for c in merged.columns if c.startswith("w_")]
        mean_w = merged[wealth_cols].mean(axis=1)
        ax.plot(pd.to_datetime(merged["date"]), mean_w, color=family_color, linewidth=2.0, label=f"{family_name} mean")

    ax.plot(
        pd.to_datetime(base_curve["date"]),
        base_curve["wealth"],
        color="black",
        linewidth=2.5,
        label="base",
    )
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("wealth")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_all_families(
    *,
    local_curves: Dict[int, pd.DataFrame],
    global_curves: Dict[int, pd.DataFrame],
    base_curve: pd.DataFrame,
    out_path: Path,
    title: str,
    local_color: str,
    global_color: str,
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for seed in sorted(local_curves.keys()):
        df = local_curves[seed]
        ax.plot(pd.to_datetime(df["date"]), df["wealth"], alpha=0.18, linewidth=1.0, color=local_color)
    for seed in sorted(global_curves.keys()):
        df = global_curves[seed]
        ax.plot(pd.to_datetime(df["date"]), df["wealth"], alpha=0.18, linewidth=1.0, color=global_color)

    merged_local = _merge_on_date(local_curves)
    if merged_local is not None and merged_local.shape[0] > 0:
        wealth_cols = [c for c in merged_local.columns if c.startswith("w_")]
        mean_w = merged_local[wealth_cols].mean(axis=1)
        ax.plot(pd.to_datetime(merged_local["date"]), mean_w, color=local_color, linewidth=2.0, label="local mean")

    merged_global = _merge_on_date(global_curves)
    if merged_global is not None and merged_global.shape[0] > 0:
        wealth_cols = [c for c in merged_global.columns if c.startswith("w_")]
        mean_w = merged_global[wealth_cols].mean(axis=1)
        ax.plot(pd.to_datetime(merged_global["date"]), mean_w, color=global_color, linewidth=2.0, label="global mean")

    ax.plot(
        pd.to_datetime(base_curve["date"]),
        base_curve["wealth"],
        color="black",
        linewidth=2.5,
        label="base",
    )
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("wealth")
    ax.legend(loc="best", fontsize=9)
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
    model_order: Optional[List[str]] = None,
) -> None:
    plt = _get_plt()
    if plt is None:
        return
    if metric not in runs_df.columns:
        return
    needed = {"model_label", "init_family", metric}
    if not needed.issubset(set(runs_df.columns)):
        return

    df = runs_df[["model_label", "init_family", metric]].copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[np.isfinite(df[metric].to_numpy())]
    if df.empty:
        return

    # Use only multi-sample families for boxplots; show base as a marker.
    fam_candidates = ["local", "global"]
    families = [f for f in fam_candidates if (df["init_family"] == f).any()]
    if not families:
        families = sorted([f for f in df["init_family"].unique().tolist() if f != "base"])
    base_df = df[df["init_family"] == "base"]
    dist_df = df[df["init_family"] != "base"]
    if dist_df.empty:
        return

    models_present = sorted(dist_df["model_label"].unique().tolist())
    if model_order:
        models = [m for m in model_order if m in models_present]
        models.extend([m for m in models_present if m not in models])
    else:
        models = models_present

    # Collect series for each (model, family)
    data: List[np.ndarray] = []
    positions: List[float] = []
    box_colors: List[str] = []
    fam_color = {
        "local": "#1f77b4",
        "global": "#ff7f0e",
    }
    width = 0.28 if len(families) > 1 else 0.5
    group_gap = 0.55
    pos = 0.0
    group_centers: Dict[str, float] = {}
    for model in models:
        start = pos
        for j, fam in enumerate(families):
            vals = dist_df[(dist_df["model_label"] == model) & (dist_df["init_family"] == fam)][metric].astype(float).to_numpy()
            if vals.size == 0:
                vals = np.asarray([np.nan], dtype=float)
            data.append(vals)
            positions.append(pos)
            box_colors.append(fam_color.get(fam, "#7f7f7f"))
            pos += 1.0
        end = pos - 1.0 if families else pos
        group_centers[model] = (start + end) / 2.0
        pos += group_gap

    fig, ax = plt.subplots(figsize=(max(8.0, 1.7 * len(models)), 4.2))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)

    # Base point overlay (single value per model, if present).
    for model in models:
        base_vals = base_df[base_df["model_label"] == model][metric].astype(float).to_numpy()
        if base_vals.size == 0:
            continue
        ax.scatter(
            [group_centers[model]],
            [float(base_vals[0])],
            color="black",
            marker="D",
            s=28,
            zorder=5,
            label="base" if model == models[0] else None,
        )

    # X ticks at group centers.
    ax.set_xticks([group_centers[m] for m in models])
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.2)

    # Legend: families + base
    handles: List[Any] = []
    labels: List[str] = []
    for fam in families:
        handles.append(plt.Line2D([0], [0], color=fam_color.get(fam, "#7f7f7f"), linewidth=6, alpha=0.6))
        labels.append(fam)
    if not base_df.empty:
        handles.append(plt.Line2D([0], [0], color="black", marker="D", linestyle="", markersize=6))
        labels.append("base")
    if handles:
        ax.legend(handles, labels, loc="best", frameon=True, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_metric_boxplots(runs_df: pd.DataFrame, out_dir: Path) -> None:
    if runs_df.empty:
        return
    metrics = [
        # core performance
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
        # trading / cost
        "avg_turnover",
        "avg_trading_cost",
        # predictive fit
        "r2",
        # runtime
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
            title=f"B: metric distribution ({metric})",
            y_label=metric,
            model_order=model_order,
        )


@dataclass(frozen=True)
class _RunSpec:
    formulation: str
    init_family: str  # base | local | global
    init_seed: Optional[int]


def main() -> None:
    parser = build_parser()
    parser.description = "Experiment B: initialization sensitivity"
    # B-specific options
    parser.add_argument(
        "--b-target",
        type=str,
        default="dual,kkt,ipo_grad",
        help='Comma-separated targets to run (e.g., "dual,kkt,ipo-grad"). Back-compat: "both" = dual,kkt.',
    )
    parser.add_argument("--b-process-seed", type=int, default=0)
    parser.add_argument("--b-init-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--b-base-mode",
        type=str,
        default="init-ipo",
        choices=["none", "init-ipo", "pen-10"],
        help="Base preset: none (no init, no theta penalty), init-ipo (IPO init only), pen-10 (anchor penalty=10 with IPO anchor, init none).",
    )
    parser.add_argument("--b-theta-init-clip", type=float, default=5.0)
    parser.add_argument("--b-theta-init-sigma-local", type=float, default=0.1)
    parser.add_argument("--b-theta-init-sigma-global", type=float, default=0.3)
    parser.add_argument("--b-init-families", type=str, default="local,global")

    args = parser.parse_args()
    tickers = parse_tickers(args.tickers)

    base_delta = float(args.delta)
    delta_up = float(getattr(args, "delta_up", None) or base_delta)
    delta_down = float(getattr(args, "delta_down", None) or delta_up)

    init_seeds = _parse_int_list(getattr(args, "b_init_seeds", ""))
    if not init_seeds:
        raise ValueError("--b-init-seeds must contain at least one integer seed")
    init_families = {s.strip().lower() for s in parse_commalist(getattr(args, "b_init_families", ""))}
    if not init_families:
        init_families = {"local", "global"}
    unknown = init_families - {"local", "global"}
    if unknown:
        raise ValueError(f"Unknown init families: {sorted(unknown)}")

    outdir = make_output_dir(RESULTS_ROOT, args.outdir)
    analysis_dir = outdir / "analysis"
    analysis_csv = analysis_dir / "csv"
    analysis_fig = analysis_dir / "figures"
    analysis_csv.mkdir(parents=True, exist_ok=True)
    analysis_fig.mkdir(parents=True, exist_ok=True)

    # Data bundle (same pipeline as real_data_run)
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

    # Heavy imports (matplotlib, reporting, etc.) after parsing so `--help` stays lightweight.
    from dfl_portfolio.experiments.real_data_run import run_rolling_experiment
    from dfl_portfolio.registry import KNITRO_DEFAULTS, SolverSpec

    # Solver spec (Flex only)
    knitro_opts = dict(KNITRO_DEFAULTS)
    # Use flex-maxtime if available (real_data_run sets maxtime_real)
    flex_maxtime = float(getattr(args, "flex_maxtime", getattr(args, "maxtime_real", 180.0)) or 180.0)
    knitro_opts["maxtime_real"] = flex_maxtime
    solver_spec = SolverSpec(name=str(getattr(args, "flex_solver", "knitro")), options=knitro_opts, tee=bool(args.tee))
    ipo_grad_solver_spec = SolverSpec(name="ipo_grad", tee=bool(args.tee))

    # Trading cost overrides are already parsed into a dict by build_parser.
    raw_asset_costs: Dict[str, float] = getattr(args, "trading_cost_per_asset", {}) or {}
    asset_cost_overrides_dec = {
        ticker.upper(): max(float(rate), 0.0) / 10000.0 for ticker, rate in raw_asset_costs.items()
    }
    trading_costs_enabled = float(getattr(args, "trading_cost_bps", 0.0)) > 0.0 or bool(asset_cost_overrides_dec)

    # Formulations to test
    formulations, include_ipo_grad = _parse_b_targets(getattr(args, "b_target", "dual,kkt,ipo_grad"))

    base_mode_cfg = _resolve_b_base_mode(getattr(args, "b_base_mode", "init-ipo"))
    base_mode = str(base_mode_cfg["b_base_mode"])
    theta_base_init_mode = str(base_mode_cfg["theta_base_init_mode"])
    clip = float(getattr(args, "b_theta_init_clip", 5.0))
    sigma_local = float(getattr(args, "b_theta_init_sigma_local", 0.1))
    sigma_global = float(getattr(args, "b_theta_init_sigma_global", 0.3))
    process_seed = int(getattr(args, "b_process_seed", 0))

    # Base flex options (same granularity as real_data_run)
    # Note: theta_init_mode is set per run spec below.
    flex_base_options: Dict[str, Any] = dict(
        lambda_theta_anchor=float(base_mode_cfg["flex_lambda_theta_anchor"]),
        lambda_theta_iso=float(base_mode_cfg["flex_lambda_theta_iso"]),
        theta_anchor_mode=str(base_mode_cfg["flex_theta_anchor_mode"]),
        # Keep auxiliary init deterministic in study B.
        aux_init_mode="none",
    )
    ipo_grad_options: Dict[str, Any] = dict(
        ipo_grad_epochs=int(getattr(args, "ipo_grad_epochs", 500)),
        ipo_grad_lr=float(getattr(args, "ipo_grad_lr", 1e-3)),
        ipo_grad_batch_size=int(getattr(args, "ipo_grad_batch_size", 0)),
        ipo_grad_qp_max_iter=int(getattr(args, "ipo_grad_qp_max_iter", 5000)),
        ipo_grad_qp_tol=float(getattr(args, "ipo_grad_qp_tol", 1e-6)),
        ipo_grad_lambda_anchor=float(base_mode_cfg["ipo_grad_lambda_anchor"]),
        ipo_grad_theta_anchor_mode=str(base_mode_cfg["ipo_grad_theta_anchor_mode"]),
    )

    # Runs
    run_specs: List[_RunSpec] = []
    for form in formulations:
        run_specs.append(_RunSpec(formulation=form, init_family="base", init_seed=None))
        for fam in ["local", "global"]:
            if fam not in init_families:
                continue
            for s in init_seeds:
                run_specs.append(_RunSpec(formulation=form, init_family=fam, init_seed=int(s)))
    if include_ipo_grad:
        # IPO-GRAD (same init seeds/families; no dual/kkt formulation)
        run_specs.append(_RunSpec(formulation="ipo_grad", init_family="base", init_seed=None))
        for fam in ["local", "global"]:
            if fam not in init_families:
                continue
            for s in init_seeds:
                run_specs.append(_RunSpec(formulation="ipo_grad", init_family=fam, init_seed=int(s)))

    total_jobs = len(run_specs)
    targets_str = ",".join(formulations + (["ipo_grad"] if include_ipo_grad else []))
    print(
        f"[local-opt-B] start: targets={targets_str} "
        f"base_mode={base_mode} init_seeds={len(init_seeds)} jobs={total_jobs}"
    )
    print(f"[local-opt-B] outdir={outdir}")

    run_rows: List[Dict[str, Any]] = []
    retrain_frames: List[pd.DataFrame] = []

    base_curves: Dict[str, pd.DataFrame] = {}
    plot_forms = list(formulations) + (["ipo_grad"] if include_ipo_grad else [])
    local_curves: Dict[str, Dict[int, pd.DataFrame]] = {f: {} for f in plot_forms}
    global_curves: Dict[str, Dict[int, pd.DataFrame]] = {f: {} for f in plot_forms}

    job_times: List[float] = []
    for job_i, spec in enumerate(run_specs, start=1):
        avg = (sum(job_times) / len(job_times)) if job_times else float("nan")
        eta_sec = (total_jobs - job_i + 1) * avg if job_times else float("nan")
        eta_str = f"{eta_sec/60:.1f}m" if job_times else "n/a"
        print(f"[local-opt-B] job {job_i}/{total_jobs} form={spec.formulation} init={spec.init_family} seed={spec.init_seed} (eta={eta_str})")

        is_ipo_grad = spec.formulation == "ipo_grad"
        theta_init_spec: Dict[str, Any] = {}
        flex_options = None
        if not is_ipo_grad:
            flex_options = dict(flex_base_options)
            flex_options["formulation"] = spec.formulation
        if spec.init_family == "base":
            theta_init_spec["theta_init_mode"] = theta_base_init_mode
        elif spec.init_family == "local":
            theta_init_spec["theta_init_mode"] = "rand_local"
            theta_init_spec["theta_init_base_mode"] = theta_base_init_mode
            theta_init_spec["theta_init_sigma"] = sigma_local
            theta_init_spec["theta_init_clip"] = clip
        elif spec.init_family == "global":
            theta_init_spec["theta_init_mode"] = "rand_zero"
            theta_init_spec["theta_init_sigma"] = sigma_global
            theta_init_spec["theta_init_clip"] = clip
        else:  # pragma: no cover
            raise ValueError(f"Unexpected init_family={spec.init_family}")
        if flex_options is not None:
            flex_options.update(theta_init_spec)

        # Keep model_label stable across families; encode family in our own CSV columns.
        model_label = "ipo_grad" if is_ipo_grad else f"flex_{spec.formulation}"
        results_model_dir = outdir / "models" / model_label / spec.init_family
        debug_dir = outdir / "debug" / model_label / spec.init_family
        results_model_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        run_result = run_rolling_experiment(
            model_key="ipo_grad" if is_ipo_grad else "flex",
            model_label=model_label,
            bundle=bundle,
            delta_up=delta_up,
            delta_down_candidates=[delta_down],
            trading_cost_enabled=trading_costs_enabled,
            asset_cost_overrides=asset_cost_overrides_dec,
            solver_spec=ipo_grad_solver_spec if is_ipo_grad else solver_spec,
            flex_options=flex_options,
            spo_plus_options=None,
            ipo_grad_options=ipo_grad_options if is_ipo_grad else None,
            train_window=int(args.train_window),
            rebal_interval=int(args.rebal_interval),
            debug_roll=bool(getattr(args, "debug_roll", True)),
            debug_dir=debug_dir,
            results_model_dir=results_model_dir,
            tee=bool(args.tee),
            asset_pred_dir=None,
            eval_start=pd.Timestamp(args.start),
            ipo_grad_debug_kkt=False,
            base_seed=process_seed,
            init_seed=spec.init_seed,
            theta_init_spec=theta_init_spec if is_ipo_grad else None,
            theta_init_delta=float(delta_down),
        )
        elapsed_total = time.perf_counter() - start_time
        job_times.append(float(elapsed_total))

        stats = dict(run_result["stats"])
        stats["formulation"] = spec.formulation
        stats["init_family"] = spec.init_family
        stats["init_seed"] = spec.init_seed if spec.init_seed is not None else ""
        stats["model_key"] = "ipo_grad" if is_ipo_grad else "flex"
        stats["model_label"] = model_label
        stats["b_base_mode"] = base_mode
        stats["theta_base_init_mode"] = theta_base_init_mode
        stats["lambda_theta_anchor"] = float(flex_base_options.get("lambda_theta_anchor", 0.0))
        stats["theta_anchor_mode"] = str(flex_base_options.get("theta_anchor_mode", ""))
        stats["ipo_grad_lambda_anchor"] = float(ipo_grad_options.get("ipo_grad_lambda_anchor", 0.0))
        stats["ipo_grad_theta_anchor_mode"] = str(ipo_grad_options.get("ipo_grad_theta_anchor_mode", ""))
        stats["theta_init_sigma_local"] = sigma_local
        stats["theta_init_sigma_global"] = sigma_global
        stats["theta_init_clip"] = clip
        stats["process_seed"] = process_seed
        stats["elapsed_total_run_sec"] = float(elapsed_total)
        run_rows.append(stats)

        reb_df = run_result.get("rebalance_df", pd.DataFrame())
        if isinstance(reb_df, pd.DataFrame) and not reb_df.empty:
            reb_df = reb_df.copy()
            reb_df["formulation"] = spec.formulation
            reb_df["init_family"] = spec.init_family
            reb_df["init_seed"] = spec.init_seed if spec.init_seed is not None else ""
            reb_df["process_seed"] = process_seed
            reb_df["model_key"] = "ipo_grad" if is_ipo_grad else "flex"
            reb_df["model_label"] = model_label
            retrain_frames.append(reb_df)

        wealth_df = run_result["wealth_df"][["date", "wealth"]].copy()
        if spec.init_family == "base":
            base_curves[spec.formulation] = wealth_df
        elif spec.init_family == "local" and spec.init_seed is not None:
            local_curves[spec.formulation][int(spec.init_seed)] = wealth_df
        elif spec.init_family == "global" and spec.init_seed is not None:
            global_curves[spec.formulation][int(spec.init_seed)] = wealth_df

        print(f"[local-opt-B] done job {job_i}/{total_jobs} elapsed={elapsed_total:.1f}s")

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(analysis_csv / "runs.csv", index=False)
    if retrain_frames:
        retrain_df = pd.concat(retrain_frames, ignore_index=True)
        retrain_df.to_csv(analysis_csv / "retrain_log.csv", index=False)
    else:
        retrain_df = pd.DataFrame()

    # Summary table (mean/std over init seeds; base is a single point)
    summary_rows: List[Dict[str, Any]] = []
    metrics = [
        "ann_return",
        "total_return",
        "terminal_wealth",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "avg_trading_cost",
        "elapsed_total_run_sec",
    ]
    for (label, fam), part in runs_df.groupby(["model_label", "init_family"], dropna=False):
        row: Dict[str, Any] = {
            "model_label": label,
            "init_family": fam,
            "model": str(label),
        }
        for col in metrics:
            stats = _summary_stats(part[col].astype(float).to_numpy()) if col in part.columns else {"mean": np.nan, "std": np.nan, "n": 0}
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"] = stats["std"]
            row[f"{col}_n"] = stats["n"]
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(analysis_csv / "summary_table.csv", index=False)

    # Boxplots: metric distributions across init seeds (compare models; split by local/global when available)
    _plot_metric_boxplots(runs_df, analysis_fig)

    # Plots: three figures per model (local+base, global+base, all)
    for form in plot_forms:
        base_curve = base_curves.get(form)
        if base_curve is None:
            continue
        local_color = "#1f77b4"
        global_color = "#ff7f0e"

        if local_curves.get(form):
            _plot_family_plus_base(
                family_curves=local_curves[form],
                base_curve=base_curve,
                out_path=analysis_fig / f"cumret_overlay_{form}_local_plus_base.png",
                title=f"B: local init vs base ({form})",
                family_name="local",
                family_color=local_color,
            )
        if global_curves.get(form):
            _plot_family_plus_base(
                family_curves=global_curves[form],
                base_curve=base_curve,
                out_path=analysis_fig / f"cumret_overlay_{form}_global_plus_base.png",
                title=f"B: global init vs base ({form})",
                family_name="global",
                family_color=global_color,
            )
        if local_curves.get(form) or global_curves.get(form):
            _plot_all_families(
                local_curves=local_curves.get(form, {}),
                global_curves=global_curves.get(form, {}),
                base_curve=base_curve,
                out_path=analysis_fig / f"cumret_overlay_{form}_all.png",
                title=f"B: local+global init vs base ({form})",
                local_color=local_color,
                global_color=global_color,
            )

    # Save config
    cfg = vars(args).copy()
    cfg["tickers"] = tickers
    cfg["init_seeds"] = init_seeds
    (analysis_csv / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[local-opt-B] finished. outputs -> {outdir}")


if __name__ == "__main__":  # pragma: no cover
    main()


"""
実行例
------
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.local_opt_study_b \
  --b-target "dual,kkt,ipo-grad" \
  --b-process-seed 0 \
  --b-base-mode init-ipo \
  --b-init-seeds "0,1,2,3,4,5,6,7,8,9" \
  --b-init-families "local,global" \
  --b-theta-init-sigma-local 0.1 \
  --b-theta-init-sigma-global 1 \
  --b-theta-init-clip 5.0


概要（何を変えて何を固定するか）
------------------------------
- --b-target "dual,kkt,ipo-grad"
  - flex_dual / flex_kkt / ipo_grad を実行
  - 同じ init_seed を各ターゲットで共有して「初期θの比較」を公平にする
- --b-process-seed 0
  - プロセス側の乱数（補助変数初期化など）を固定
- --b-base-mode（base run の「初期θ＋θペナルティ」プリセット）
  - none: theta_init=none、θペナルティ=0
  - init-ipo: theta_init=ipo（IPO解析解）、θペナルティ=0
  - pen-10: theta_init=none、θペナルティ=10（アンカー=ipo）
- --b-init-seeds "0..9"
  - ランダム初期θを10種類（同じ seed を全ターゲットで共有）
- --b-init-families "local,global"
  - local:  theta_init = theta_base + sigma_local * N(0,I)（base近傍ノイズ）
  - global: theta_init = sigma_global * N(0,I)（ゼロ近傍ランダム）
  - ipo_grad も同じ初期θ（base/local/global + init_seed）を与えて学習・評価（指定時）


sigma / clip の見方（各要素 θ_j の範囲感）
---------------------------------------
- クリップ前:
  - global: θ_j ~ N(0, sigma_global^2)
  - local:  (θ_j - θ_base,j) ~ N(0, sigma_local^2)
- 典型レンジ: 約95%が ±1.96*sigma、約99.7%が ±3*sigma
- clip: 各要素を [-clip,+clip] に制限（発散防止）
- 目安: clip >= 4*sigma だと clip はほぼ効かず「sigma 主導」で広がる

具体例（1要素あたり）
-------------------
- sigma=0.1, clip=5.0 → 95%: ±0.196 / 99.7%: ±0.3（clip無関係）
- sigma=0.3, clip=5.0 → 95%: ±0.588 / 99.7%: ±0.9（clip無関係）
- sigma=1.0, clip=10.0 → 95%: ±1.96 / 99.7%: ±3.0（clipほぼ無関係）
- sigma=1.0, clip=2.0 → 一部が ±2 に張り付く（clipが効く）
- sigma=2.0, clip=10.0 → 95%: ±3.92 / 99.7%: ±6.0（かなり広い）


ジョブ数の目安（init_seeds=10, families=local,global）
----------------------------------------------
- 各ターゲット: base 1回 + local 10回 + global 10回 = 21 run
- 例: dual,kkt,ipo_grad の3ターゲットなら 21*3=63 run
"""
