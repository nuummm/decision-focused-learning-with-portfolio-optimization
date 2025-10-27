#!/usr/bin/env python3
"""Run QCQP experiments on explicit seed lists while accepting partial Gurobi solutions.

This helper mirrors the ``run_small_global_eval`` workflow but specialises it for
Gurobi investigations: we iterate over user-provided seeds, optionally accept
time-limit terminations, track the reported MIP gap, and organise outputs in a
dedicated results tree.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

import run_small_global_eval as small


HERE = small.HERE
RUNPY = small.RUNPY
RESULTS_ROOT = small.RESULTS_ROOT
DEFAULT_OUTDIR = RESULTS_ROOT / "exp_gurobi_partial"
DEFAULT_REPORT = DEFAULT_OUTDIR / "gurobi_partial_summary.csv"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dual/kkt models on fixed seeds while accepting partial Gurobi solutions."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        required=True,
        help="Seeds to evaluate. Accepts space-separated integers or comma-separated lists, e.g. '100 105' or '100,101,102'.",
    )
    parser.add_argument("--runs", type=int, default=None, help="Legacy compatibility; ignored when seeds are provided.")
    parser.add_argument("--seed0", type=int, default=100, help="Fallback initial seed (unused when --seeds supplied).")
    parser.add_argument("--N", type=int, default=50, help="Number of samples.")
    parser.add_argument("--d", type=int, default=3, help="Asset dimension.")
    parser.add_argument("--snr", type=float, default=0.1, help="Signal-to-noise ratio.")
    parser.add_argument("--rho", type=float, default=0.5, help="Correlation parameter.")
    parser.add_argument("--sigma", type=float, default=0.0125, help="Marginal stdev of returns.")
    parser.add_argument("--res", type=int, default=10, help="Rolling window length (burn-in).")
    parser.add_argument("--delta", type=float, default=1.0, help="Risk-aversion parameter.")
    parser.add_argument("--lambda-theta", type=float, default=0.0, help="L2 regularisation for theta passed to run.py.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=DEFAULT_REPORT,
        help="Summary CSV name written into each experiment directory.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python interpreter to invoke run.py.")
    parser.add_argument("--tee", action="store_true", help="Pass --tee to run.py.")
    parser.add_argument("--estimate-cov", action="store_true", help="Use rolling covariance estimates (default: true covariance).")
    parser.add_argument("--no-plots", action="store_true", help="Forward --no-plots to run.py.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw run.py CSV outputs instead of deleting them.")
    parser.add_argument("--fig-format", default="png", help="Image format for metric plots.")
    parser.add_argument("--dryrun", action="store_true", help="Print commands only.")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble models (average/weighted/normalized).")
    parser.add_argument(
        "--gurobi-accept-time-limit",
        action="store_true",
        help="Allow Gurobi runs that end due to time limits to be treated as successes.",
    )
    parser.add_argument(
        "--gurobi-max-gap",
        type=float,
        default=None,
        help="Accept runs whose reported Gurobi relative MIP gap is at or below this value.",
    )
    parser.add_argument("--log-gaps", action="store_true", help="Print per-run Gurobi MIP gap summaries.")
    return parser.parse_args(argv)


def build_cmd(
    model: str,
    solver: str,
    args: argparse.Namespace,
    *,
    seed: int,
    outdir: Path,
    fixed_theta: Optional[Path] = None,
) -> List[str]:
    cmd = [
        str(args.python),
        str(RUNPY),
        "--model",
        model,
        "--solver",
        solver,
        "--N",
        str(args.N),
        "--d",
        str(args.d),
        "--snr",
        str(args.snr),
        "--rho",
        str(args.rho),
        "--sigma",
        str(args.sigma),
        "--res",
        str(args.res),
        "--delta",
        str(args.delta),
        "--runs",
        "1",
        "--seed0",
        str(seed),
        "--lambda-theta",
        str(args.lambda_theta),
        "--outdir",
        str(outdir),
        "--log-every",
        "1",
    ]
    if args.gurobi_max_gap is not None:
        cmd.extend(["--gurobi-max-gap", str(args.gurobi_max_gap)])
    if args.no_plots:
        cmd.append("--no-plots")
    if not args.estimate_cov:
        cmd.append("--use-true-cov")
    if args.tee:
        cmd.append("--tee")
    if args.gurobi_accept_time_limit:
        cmd.append("--allow-gurobi-partial")
    if fixed_theta is not None:
        cmd.append("--fixed-theta")
        cmd.append(str(fixed_theta))
    return cmd


def run_model(
    model: str,
    solver: str,
    args: argparse.Namespace,
    *,
    seed: int,
    outdir: Path,
    log_handle,
    keep_raw: bool,
    fixed_theta: Optional[Path] = None,
) -> small.RunOutcome:
    cmd = build_cmd(model, solver, args, seed=seed, outdir=outdir, fixed_theta=fixed_theta)
    if args.dryrun:
        print("[DRYRUN]", " ".join(cmd))
        raise SystemExit(0)

    print(f"[INFO] Running {model} with solver={solver} seed={seed}")

    stdout_lines: List[str] = []

    if log_handle is not None:
        log_handle.write("=" * 80 + "\n")
        log_handle.write(f"CMD: {' '.join(cmd)}\n")
        log_handle.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_lines.append(line)
        print(line, end="")
        if log_handle is not None:
            log_handle.write(line)
            log_handle.flush()

    proc.wait()

    if log_handle is not None:
        log_handle.write("\n")
        log_handle.flush()

    if proc.returncode != 0:
        msg = "".join(stdout_lines).strip() or "run.py terminated with non-zero status"
        raise RuntimeError(f"run.py failed for model={model}, solver={solver}: {msg}")

    stdout_text = "".join(stdout_lines)

    summary_path = None
    runs_path = None
    for line in stdout_text.splitlines():
        match_summary = small.SUMMARY_PATH_RE.search(line)
        if match_summary:
            summary_path = Path(match_summary.group("path")).resolve()
        match_runs = small.RUNS_PATH_RE.search(line)
        if match_runs:
            runs_path = Path(match_runs.group("path")).resolve()

    if summary_path is None or runs_path is None:
        raise RuntimeError("Failed to locate summary/per-run CSV paths in run.py output")

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    if not runs_path.exists():
        raise FileNotFoundError(f"Per-run CSV not found: {runs_path}")

    with summary_path.open(newline="") as f_sum, runs_path.open(newline="") as f_runs:
        summary_reader = csv.DictReader(f_sum)
        runs_reader = csv.DictReader(f_runs)
        try:
            summary_row = next(summary_reader)
        except StopIteration as exc:
            raise RuntimeError(f"Summary CSV empty: {summary_path}") from exc
        try:
            run_row = next(runs_reader)
        except StopIteration as exc:
            raise RuntimeError(f"Per-run CSV empty: {runs_path}") from exc

    actual_seed = int(run_row.get("seed", seed))

    outcome = small.RunOutcome(
        model=model,
        solver=solver,
        seed=actual_seed,
        summary_csv=summary_path,
        runs_csv=runs_path,
        summary_row=summary_row,
        run_row=run_row,
    )

    if not keep_raw:
        small.cleanup_file(summary_path, stop_dir=outdir)
        small.cleanup_file(runs_path, stop_dir=outdir)

    return outcome


def float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if isinstance(args.python, str):
        args.python = Path(args.python)

    raw_seed_tokens: List[str] = []
    for entry in args.seeds:
        parts = str(entry).split(",")
        raw_seed_tokens.extend(part.strip() for part in parts if part.strip())
    if not raw_seed_tokens:
        print("[WARN] No seeds provided; nothing to do.")
        return 0
    try:
        seeds = sorted(set(int(token) for token in raw_seed_tokens))
    except ValueError as exc:
        raise SystemExit(f"[ERROR] Failed to parse seeds: {exc}")
    if not seeds:
        print("[WARN] No seeds provided; nothing to do.")
        return 0

    enabled_solvers: Dict[str, str] = dict(small.DEFAULT_SOLVERS)
    if args.no_ensemble:
        for key in ["ensemble_avg", "ensemble_weighted", "ensemble_normalized"]:
            enabled_solvers.pop(key, None)
    run_ensemble = all(k in enabled_solvers for k in ["ensemble_avg", "ensemble_weighted", "ensemble_normalized"])

    base_outdir: Path = args.outdir
    base_outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = base_outdir / timestamp
    raw_dir = exp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "experiment.log"

    shared_seeds: List[int] = []
    per_model_rows: Dict[str, List[Dict[str, str]]] = {key: [] for key in enabled_solvers}

    with open(log_path, "w", encoding="utf-8") as log_handle:
        for requested_seed in seeds:
            current_seed = int(requested_seed)
            try:
                dual_outcome = run_model(
                    "dual",
                    enabled_solvers["dual"],
                    args,
                    seed=current_seed,
                    outdir=raw_dir,
                    log_handle=log_handle,
                    keep_raw=args.keep_raw,
                )
            except RuntimeError as exc:
                print(f"[WARN] dual failed for seed={current_seed}: {exc}")
                continue
            dual_seed = dual_outcome.seed
            if dual_seed != current_seed:
                print(f"[WARN] dual deviated to seed={dual_seed} (expected {current_seed}); skipping")
                continue
            per_model_rows["dual"].append({**dual_outcome.run_row, "model": "dual", "solver": dual_outcome.solver})

            try:
                kkt_outcome = run_model(
                    "kkt",
                    enabled_solvers["kkt"],
                    args,
                    seed=current_seed,
                    outdir=raw_dir,
                    log_handle=log_handle,
                    keep_raw=args.keep_raw,
                )
            except RuntimeError as exc:
                print(f"[WARN] kkt failed for seed={current_seed}: {exc}")
                per_model_rows["dual"].pop()
                continue
            if kkt_outcome.seed != current_seed:
                print(f"[WARN] kkt deviated to seed={kkt_outcome.seed} (expected {current_seed}); skipping")
                per_model_rows["dual"].pop()
                continue
            per_model_rows["kkt"].append({**kkt_outcome.run_row, "model": "kkt", "solver": kkt_outcome.solver})

            try:
                ols_outcome = run_model(
                    "ols",
                    enabled_solvers["ols"],
                    args,
                    seed=current_seed,
                    outdir=raw_dir,
                    log_handle=log_handle,
                    keep_raw=args.keep_raw,
                )
            except RuntimeError as exc:
                print(f"[WARN] ols failed for seed={current_seed}: {exc}")
                per_model_rows["dual"].pop()
                per_model_rows["kkt"].pop()
                continue
            per_model_rows["ols"].append({**ols_outcome.run_row, "model": "ols", "solver": ols_outcome.solver})

            try:
                ipo_outcome = run_model(
                    "ipo",
                    enabled_solvers["ipo"],
                    args,
                    seed=current_seed,
                    outdir=raw_dir,
                    log_handle=log_handle,
                    keep_raw=args.keep_raw,
                )
            except RuntimeError as exc:
                print(f"[WARN] ipo failed for seed={current_seed}: {exc}")
                per_model_rows["dual"].pop()
                per_model_rows["kkt"].pop()
                per_model_rows["ols"].pop()
                continue
            if ipo_outcome.seed != current_seed:
                print(f"[WARN] ipo deviated to seed={ipo_outcome.seed} (expected {current_seed}); skipping")
                per_model_rows["dual"].pop()
                per_model_rows["kkt"].pop()
                per_model_rows["ols"].pop()
                continue
            per_model_rows["ipo"].append({**ipo_outcome.run_row, "model": "ipo", "solver": ipo_outcome.solver})

            if args.log_gaps:
                for model_name in ["dual", "kkt"]:
                    gap_str = per_model_rows[model_name][-1].get("gurobi_mip_gap", "")
                    print(f"[INFO] seed={current_seed} model={model_name} gap={gap_str}")

            if not run_ensemble:
                shared_seeds.append(current_seed)
                print(f"[INFO] Accepted seed {current_seed}; total collected {len(shared_seeds)}/{len(seeds)}")
                continue

            theta_dual_vec = small.parse_theta(per_model_rows["dual"][-1])
            theta_kkt_vec = small.parse_theta(per_model_rows["kkt"][-1])
            if theta_dual_vec is None or theta_kkt_vec is None:
                print(f"[WARN] Failed to parse theta for ensemble on seed={current_seed}; skipping")
                per_model_rows["dual"].pop()
                per_model_rows["kkt"].pop()
                per_model_rows["ols"].pop()
                per_model_rows["ipo"].pop()
                continue

            ensembles = {
                "ensemble_avg": 0.5 * (theta_dual_vec + theta_kkt_vec),
                "ensemble_weighted": (
                    np.linalg.norm(theta_dual_vec) * theta_dual_vec
                    + np.linalg.norm(theta_kkt_vec) * theta_kkt_vec
                )
                / (np.linalg.norm(theta_dual_vec) + np.linalg.norm(theta_kkt_vec) + 1e-12),
                "ensemble_normalized": 0.5
                * (small.normalize_theta(theta_dual_vec) + small.normalize_theta(theta_kkt_vec)),
            }

            ensemble_results = {}
            ensemble_fail = False
            for ens_model, theta_vec in ensembles.items():
                theta_tmp = raw_dir / f"{ens_model}_seed{current_seed}.npy"
                np.save(theta_tmp, theta_vec)
                try:
                    outcome = run_model(
                        ens_model,
                        enabled_solvers[ens_model],
                        args,
                        seed=current_seed,
                        outdir=raw_dir,
                        log_handle=log_handle,
                        keep_raw=args.keep_raw,
                        fixed_theta=theta_tmp,
                    )
                except RuntimeError as exc:
                    print(f"[WARN] {ens_model} failed for seed={current_seed}: {exc}; skipping")
                    ensemble_fail = True
                    break
                finally:
                    if not args.keep_raw and theta_tmp.exists():
                        theta_tmp.unlink()
                ensemble_results[ens_model] = outcome

            if ensemble_fail:
                per_model_rows["dual"].pop()
                per_model_rows["kkt"].pop()
                per_model_rows["ols"].pop()
                per_model_rows["ipo"].pop()
                continue

            for ens_model, outcome in ensemble_results.items():
                per_model_rows[ens_model].append({**outcome.run_row, "model": ens_model, "solver": outcome.solver})

            shared_seeds.append(current_seed)
            print(f"[INFO] Accepted seed {current_seed}; total collected {len(shared_seeds)}/{len(seeds)}")

    detail_rows: List[Dict[str, str]] = []
    for model, rows in per_model_rows.items():
        for row in rows:
            row = dict(row)
            row.setdefault("seed", "")
            row.setdefault("model", model)
            detail_rows.append(row)

    detail_csv = exp_dir / "global_details.csv"
    small.write_csv(detail_csv, detail_rows)

    summary_metrics = list(small.SUMMARY_METRIC_MAP.keys())
    summary_rows: List[Dict[str, str]] = []
    for model, rows in per_model_rows.items():
        if not rows:
            continue
        metrics = small.summarise_metrics(rows, small.SUMMARY_METRIC_MAP.keys())
        gap_values = [float_or_nan(row.get("gurobi_mip_gap", "nan")) for row in rows]
        summary_row = {
            "model": model,
            "solver": enabled_solvers.get(model, small.DEFAULT_SOLVERS.get(model, "")),
            "n_seeds": str(len(rows)),
            "seeds": small.seeds_to_string(rows),
        }
        for src, dst in small.SUMMARY_METRIC_MAP.items():
            value = metrics.get(src, math.nan)
            summary_row[dst] = f"{value:.10g}" if not math.isnan(value) else ""
        summary_row["theta"] = small.average_theta(rows)
        if any(np.isfinite(g) for g in gap_values):
            summary_row["gurobi_mip_gap"] = f"{np.nanmean(gap_values):.6g}"
        summary_rows.append(summary_row)

    summary_csv = exp_dir / args.report_csv.name
    small.write_csv(summary_csv, summary_rows)

    small.plot_metrics(per_model_rows, summary_metrics, exp_dir, args.fig_format)

    for row in summary_rows:
        model = row["model"]
        mean_cost = row.get("mean_cost_test", "")
        dec_err = row.get("decision_error_test", "")
        gap_avg = row.get("gurobi_mip_gap", "")
        extra = f" gap={gap_avg}" if gap_avg else ""
        print(
            f"[RESULT] model={model} n={row['n_seeds']} "
            f"mean_cost_test={mean_cost} decision_error_test={dec_err}{extra}"
        )

    print(f"[INFO] Experiment outputs stored in {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python GraduationResearch/DFL_Portfolio_Optimization2/experiments/run_gurobi_partial_eval.py \
#   --seeds 100,101,102,103,106,107,108,109,111,113,116,118,119,120,121,123,124,126,127,128,129,130,131,133,134,137,138,139,140,141 \
#   --gurobi-accept-time-limit \
#   --gurobi-max-gap 0.01 \
#   --no-ensemble --tee
