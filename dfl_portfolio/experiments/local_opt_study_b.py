from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Avoid GUI backends (safe on macOS and when users run with threads).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getenv("TMPDIR", "/tmp"), "mplconfig"))

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


@dataclass(frozen=True)
class _RunSpec:
    formulation: str
    init_family: str  # base | local | global
    init_seed: Optional[int]


def main() -> None:
    parser = build_parser()
    parser.description = "Experiment B: initialization sensitivity (Flex only)"
    # B-specific options
    parser.add_argument("--b-target", type=str, default="both", choices=["dual", "kkt", "both"])
    parser.add_argument("--b-process-seed", type=int, default=0)
    parser.add_argument("--b-init-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--b-theta-base-mode", type=str, default="ipo", choices=["ipo", "ols", "none"])
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
    from dfl_portfolio.real_data.reporting import MODEL_COLOR_MAP, display_model_name
    from dfl_portfolio.experiments.real_data_run import run_rolling_experiment
    from dfl_portfolio.registry import KNITRO_DEFAULTS, SolverSpec

    # Solver spec (Flex only)
    knitro_opts = dict(KNITRO_DEFAULTS)
    # Use flex-maxtime if available (real_data_run sets maxtime_real)
    flex_maxtime = float(getattr(args, "flex_maxtime", getattr(args, "maxtime_real", 180.0)) or 180.0)
    knitro_opts["maxtime_real"] = flex_maxtime
    solver_spec = SolverSpec(name=str(getattr(args, "flex_solver", "knitro")), options=knitro_opts, tee=bool(args.tee))

    # Trading cost overrides are already parsed into a dict by build_parser.
    raw_asset_costs: Dict[str, float] = getattr(args, "trading_cost_per_asset", {}) or {}
    asset_cost_overrides_dec = {
        ticker.upper(): max(float(rate), 0.0) / 10000.0 for ticker, rate in raw_asset_costs.items()
    }
    trading_costs_enabled = float(getattr(args, "trading_cost_bps", 0.0)) > 0.0 or bool(asset_cost_overrides_dec)

    # Formulations to test
    target = str(getattr(args, "b_target", "both")).lower()
    formulations = ["dual", "kkt"] if target == "both" else [target]

    theta_base_mode = str(getattr(args, "b_theta_base_mode", "ipo")).lower()
    clip = float(getattr(args, "b_theta_init_clip", 5.0))
    sigma_local = float(getattr(args, "b_theta_init_sigma_local", 0.1))
    sigma_global = float(getattr(args, "b_theta_init_sigma_global", 0.3))
    process_seed = int(getattr(args, "b_process_seed", 0))

    # Base flex options (same granularity as real_data_run)
    # Note: theta_init_mode is set per run spec below.
    flex_base_options: Dict[str, Any] = dict(
        lambda_theta_anchor=float(getattr(args, "flex_lambda_theta_anchor", 0.0)),
        lambda_theta_iso=float(getattr(args, "flex_lambda_theta_iso", 0.0)),
        theta_anchor_mode=str(getattr(args, "flex_theta_anchor_mode", "none")),
        # Keep auxiliary init deterministic in study B.
        aux_init_mode="none",
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

    total_jobs = len(run_specs)
    print(f"[local-opt-B] start: formulations={formulations} base={theta_base_mode} init_seeds={len(init_seeds)} jobs={total_jobs}")
    print(f"[local-opt-B] outdir={outdir}")

    run_rows: List[Dict[str, Any]] = []
    retrain_frames: List[pd.DataFrame] = []

    base_curves: Dict[str, pd.DataFrame] = {}
    local_curves: Dict[str, Dict[int, pd.DataFrame]] = {f: {} for f in formulations}
    global_curves: Dict[str, Dict[int, pd.DataFrame]] = {f: {} for f in formulations}

    job_times: List[float] = []
    for job_i, spec in enumerate(run_specs, start=1):
        avg = (sum(job_times) / len(job_times)) if job_times else float("nan")
        eta_sec = (total_jobs - job_i + 1) * avg if job_times else float("nan")
        eta_str = f"{eta_sec/60:.1f}m" if job_times else "n/a"
        print(f"[local-opt-B] job {job_i}/{total_jobs} form={spec.formulation} init={spec.init_family} seed={spec.init_seed} (eta={eta_str})")

        flex_options = dict(flex_base_options)
        flex_options["formulation"] = spec.formulation
        if spec.init_family == "base":
            flex_options["theta_init_mode"] = theta_base_mode
        elif spec.init_family == "local":
            flex_options["theta_init_mode"] = "rand_local"
            flex_options["theta_init_base_mode"] = theta_base_mode
            flex_options["theta_init_sigma"] = sigma_local
            flex_options["theta_init_clip"] = clip
        elif spec.init_family == "global":
            flex_options["theta_init_mode"] = "rand_zero"
            flex_options["theta_init_sigma"] = sigma_global
            flex_options["theta_init_clip"] = clip
        else:  # pragma: no cover
            raise ValueError(f"Unexpected init_family={spec.init_family}")

        # Keep model_label stable across families; encode family in our own CSV columns.
        model_label = f"flex_{spec.formulation}"
        results_model_dir = outdir / "models" / model_label / spec.init_family
        debug_dir = outdir / "debug" / model_label / spec.init_family
        results_model_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        run_result = run_rolling_experiment(
            model_key="flex",
            model_label=model_label,
            bundle=bundle,
            delta_up=delta_up,
            delta_down_candidates=[delta_down],
            trading_cost_enabled=trading_costs_enabled,
            asset_cost_overrides=asset_cost_overrides_dec,
            solver_spec=solver_spec,
            flex_options=flex_options,
            spo_plus_options=None,
            ipo_grad_options=None,
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
        )
        elapsed_total = time.perf_counter() - start_time
        job_times.append(float(elapsed_total))

        stats = dict(run_result["stats"])
        stats["formulation"] = spec.formulation
        stats["init_family"] = spec.init_family
        stats["init_seed"] = spec.init_seed if spec.init_seed is not None else ""
        stats["theta_base_mode"] = theta_base_mode
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
    for (form, fam), part in runs_df.groupby(["formulation", "init_family"], dropna=False):
        row: Dict[str, Any] = {
            "formulation": form,
            "init_family": fam,
            "model": display_model_name(f"flex_{form}"),
        }
        for col in metrics:
            stats = _summary_stats(part[col].astype(float).to_numpy()) if col in part.columns else {"mean": np.nan, "std": np.nan, "n": 0}
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"] = stats["std"]
            row[f"{col}_n"] = stats["n"]
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(analysis_csv / "summary_table.csv", index=False)

    # Plots: three figures per formulation (local+base, global+base, all)
    for form in formulations:
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
cd "/Users/kensei/VScode/卒業研究2/Decision-Focused-Learning with Portfolio Optimization"

python -m dfl_portfolio.experiments.local_opt_study_b \
  --tickers "SPY,GLD,EEM,TLT" \
  --start 2006-01-01 --end 2025-12-01 \
  --momentum-window 26 \
  --cov-window 13 \
  --cov-method oas \
  --train-window 26 \
  --rebal-interval 4 \
  --delta-up 0.5 \
  --delta-down 0.5 \
  --flex-solver knitro \
  --flex-lambda-theta-anchor 10.0 \
  --flex-theta-anchor-mode ipo \
  --b-target both \
  --b-process-seed 0 \
  --b-theta-base-mode ipo \
  --b-init-seeds "0,1,2,3,4,5,6,7,8,9" \
  --b-theta-init-sigma-local 0.1 \
  --b-theta-init-sigma-global 0.3 \
  --b-theta-init-clip 5.0 \
  --b-init-families "local,global" \
  --trading-cost-per-asset "SPY:5,GLD:10,EEM:10,TLT:5" \
  --debug-roll

"""