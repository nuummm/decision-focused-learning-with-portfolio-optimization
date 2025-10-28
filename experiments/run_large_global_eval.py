#!/usr/bin/env python3
"""Sweep wrapper around run_small_global_eval for large experiment batches.

This helper iterates over Cartesian products of selected hyperparameters and
solver assignments, delegating each concrete configuration to the existing
``run_small_global_eval`` entry point.  Results are organised into a timestamped
directory with one sub-folder per configuration to keep outputs tidy.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import re
import sys
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import run_small_global_eval as small_eval


HERE = Path(__file__).resolve()
RESULTS_ROOT = HERE.parents[2] / "results"
DEFAULT_SWEEP_OUTDIR = RESULTS_ROOT / "exp_large_global"


def unique(seq: Sequence) -> List:
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_solver_options(entries: Sequence[str]) -> Dict[str, List[str]]:
    options: Dict[str, List[str]] = {}
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"--solver expects model=solver[,solver...] (got {raw!r})")
        model, values_raw = raw.split("=", 1)
        model_key = model.strip().lower()
        values = [v.strip() for v in values_raw.split(",") if v.strip()]
        if not values:
            raise ValueError(f"No solver values supplied for model {model!r}")
        options[model_key] = unique(values)
    return options


def sanitise_value(value) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    text = text.replace("+", "")
    text = text.replace("-", "neg")
    text = text.replace(".", "p")
    text = text.replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "val"


def format_config_name(param_items: Iterable[Tuple[str, object]], solver_items: Dict[str, str]) -> str:
    parts = []
    for key, value in param_items:
        parts.append(f"{key}{sanitise_value(value)}")
    solver_bits = [f"{model}{sanitise_value(solver)}" for model, solver in sorted(solver_items.items())]
    if solver_bits:
        parts.append("solver_" + "-".join(solver_bits))
    return "_".join(parts)


def locate_latest_summary(combo_dir: Path, report_name: str) -> Path | None:
    if not combo_dir.exists():
        return None
    subdirs = [p for p in combo_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    candidate = latest / report_name
    return candidate if candidate.exists() else None


@contextmanager
def override_solvers(mapping: Dict[str, str]):
    original = dict(small_eval.DEFAULT_SOLVERS)
    updated = dict(original)
    updated.update(mapping)
    try:
        small_eval.DEFAULT_SOLVERS.clear()
        small_eval.DEFAULT_SOLVERS.update(updated)
        yield
    finally:
        small_eval.DEFAULT_SOLVERS.clear()
        small_eval.DEFAULT_SOLVERS.update(original)


@dataclass
class SweepConfig:
    runs: int
    seed0: int
    lambda_theta: float
    sigma: float
    python: Path
    tee: bool
    estimate_cov: bool
    no_plots: bool
    keep_raw: bool
    fig_format: str
    dryrun: bool
    no_ensemble: bool
    continue_on_error: bool
    report_name: str
    info_only: bool


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand run_small_global_eval over parameter grids and solver options."
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of globally optimal seeds per run_small invocation.")
    parser.add_argument("--seed0", type=int, default=100, help="Initial seed forwarded to run_small_global_eval.")
    parser.add_argument("--N", type=int, nargs="+", default=[50], help="Sample sizes to evaluate.", metavar="N")
    parser.add_argument("--d", type=int, nargs="+", default=[3], help="Asset dimensions to evaluate.", metavar="D")
    parser.add_argument("--res", type=int, nargs="+", default=[0], help="Rolling window lengths to evaluate.", metavar="RES")
    parser.add_argument("--snr", type=float, nargs="+", default=[0.1], help="Signal-to-noise ratios to evaluate.", metavar="SNR")
    parser.add_argument("--rho", type=float, nargs="+", default=[0.5], help="Correlation parameters to evaluate.", metavar="RHO")
    parser.add_argument("--delta", type=float, nargs="+", default=[1.0], help="Risk-aversion parameters to evaluate.", metavar="DELTA")
    parser.add_argument("--solver", action="append", default=[], help="Model solver sweep in model=solver[,solver...] format.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_SWEEP_OUTDIR,
        help="Root directory where the sweep folder will be created.",
    )
    parser.add_argument(
        "--lambda-theta",
        type=float,
        default=0.0,
        help="L2 regularisation for theta passed to run_small_global_eval.",
    )
    parser.add_argument("--sigma", type=float, default=0.0125, help="Marginal stdev of returns.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python interpreter for downstream runs.")
    parser.add_argument("--tee", action="store_true", help="Forward --tee to run_small_global_eval.")
    parser.add_argument("--estimate-cov", action="store_true", help="Forward --estimate-cov to run_small_global_eval.")
    parser.add_argument("--no-plots", action="store_true", help="Forward --no-plots to run_small_global_eval.")
    parser.add_argument("--keep-raw", action="store_true", help="Preserve raw CSVs emitted by run_small_global_eval.")
    parser.add_argument("--fig-format", default="png", help="Plot format for downstream metric charts.")
    parser.add_argument("--dryrun", action="store_true", help="Show planned commands without executing the sweep.")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble models in downstream runs.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue the sweep even if a configuration fails.")
    parser.add_argument(
        "--report-name",
        default="global_summary.csv",
        help="Expected report file name to capture in the sweep index.",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only echo stdout lines containing status tags (INFO/WARN/ERROR) from child runs.",
    )
    return parser.parse_args(argv)


def build_sweep_config(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        runs=args.runs,
        seed0=args.seed0,
        lambda_theta=args.lambda_theta,
        sigma=args.sigma,
        python=args.python,
        tee=args.tee,
        estimate_cov=args.estimate_cov,
        no_plots=args.no_plots,
        keep_raw=args.keep_raw,
        fig_format=args.fig_format,
        dryrun=args.dryrun,
        no_ensemble=args.no_ensemble,
        continue_on_error=args.continue_on_error,
        report_name=args.report_name,
        info_only=args.info_only,
    )


def enumerate_param_grid(args: argparse.Namespace) -> Tuple[List[str], List[Tuple]]:
    grid_keys = ["N", "d", "res", "snr", "rho", "delta"]
    grid_values: List[List] = []
    for key in grid_keys:
        values = unique(getattr(args, key))
        grid_values.append(values)
    combos = list(itertools.product(*grid_values))
    return grid_keys, combos


def enumerate_solver_combos(solver_args: Sequence[str], base_defaults: Dict[str, str]) -> List[Dict[str, str]]:
    if not solver_args:
        return [dict(base_defaults)]

    overrides = parse_solver_options(solver_args)
    solver_lists: List[List[Tuple[str, str]]] = []
    for model, default_solver in base_defaults.items():
        candidates = overrides.get(model, [default_solver])
        solver_lists.append([(model, solver) for solver in candidates])

    combos = []
    for items in itertools.product(*solver_lists):
        combo = dict(base_defaults)
        combo.update(dict(items))
        combos.append(combo)
    return combos


def run_configuration(
    sweep_cfg: SweepConfig,
    param_map: Dict[str, object],
    solver_map: Dict[str, str],
    combo_dir: Path,
) -> int:
    run_args = [
        "--runs",
        str(sweep_cfg.runs),
        "--seed0",
        str(sweep_cfg.seed0),
        "--N",
        str(param_map["N"]),
        "--d",
        str(param_map["d"]),
        "--res",
        str(param_map["res"]),
        "--snr",
        str(param_map["snr"]),
        "--rho",
        str(param_map["rho"]),
        "--delta",
        str(param_map["delta"]),
        "--lambda-theta",
        str(sweep_cfg.lambda_theta),
        "--sigma",
        str(sweep_cfg.sigma),
        "--python",
        str(sweep_cfg.python),
        "--fig-format",
        sweep_cfg.fig_format,
        "--outdir",
        str(combo_dir),
    ]
    if sweep_cfg.tee:
        run_args.append("--tee")
    if sweep_cfg.estimate_cov:
        run_args.append("--estimate-cov")
    if sweep_cfg.no_plots:
        run_args.append("--no-plots")
    if sweep_cfg.keep_raw:
        run_args.append("--keep-raw")
    if sweep_cfg.dryrun:
        run_args.append("--dryrun")
    if sweep_cfg.no_ensemble:
        run_args.append("--no-ensemble")

    output_buffer = io.StringIO()
    system_exit_exc: SystemExit | None = None
    result: int | None = None

    with override_solvers(solver_map):
        try:
            with redirect_stdout(output_buffer):
                result = small_eval.main(run_args)
        except SystemExit as exc:
            system_exit_exc = exc
            result = exc.code if isinstance(exc.code, int) else None

    captured = output_buffer.getvalue()
    if sweep_cfg.info_only:
        for line in captured.splitlines():
            if any(tag in line for tag in ("[INFO]", "[WARN]", "[ERROR]")):
                print(line)
    else:
        if captured:
            sys.stdout.write(captured)

    if system_exit_exc is not None:
        if sweep_cfg.dryrun and (system_exit_exc.code is None or system_exit_exc.code == 0):
            return 0
        raise system_exit_exc

    return int(result) if isinstance(result, int) else 0


def write_index(
    sweep_dir: Path,
    rows: List[Dict[str, str]],
) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    index_path = sweep_dir / "sweep_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote sweep index to {index_path}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    sweep_cfg = build_sweep_config(args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    sweep_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = args.outdir / sweep_timestamp
    sweep_dir.mkdir(parents=True, exist_ok=False)

    baseline_solvers = dict(small_eval.DEFAULT_SOLVERS)
    param_keys, param_combos = enumerate_param_grid(args)
    solver_combos = enumerate_solver_combos(args.solver, baseline_solvers)

    total_jobs = len(param_combos) * len(solver_combos)
    print(f"[INFO] Prepared {total_jobs} configurations (params={len(param_combos)}, solvers={len(solver_combos)})")

    index_rows: List[Dict[str, str]] = []
    job_counter = 0

    for param_values in param_combos:
        param_pairs = list(zip(param_keys, param_values))
        param_map = dict(param_pairs)
        for solver_map in solver_combos:
            job_counter += 1
            changed_solver_entries = {
                model: solver for model, solver in solver_map.items() if solver != baseline_solvers.get(model)
            }
            combo_name = format_config_name(param_pairs, changed_solver_entries)
            combo_dir = sweep_dir / combo_name
            combo_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] ({job_counter}/{total_jobs}) Running configuration {combo_name}")
            try:
                ret = run_configuration(sweep_cfg, param_map, solver_map, combo_dir)
            except Exception as exc:
                print(f"[ERROR] Configuration {combo_name} failed: {exc}")
                if not sweep_cfg.continue_on_error:
                    raise
                ret = -1

            summary_path = locate_latest_summary(combo_dir, sweep_cfg.report_name) if ret == 0 else None
            row = {
                "combo_name": combo_name,
                "status": "ok" if ret == 0 else "failed",
                "runs": str(sweep_cfg.runs),
                "seed0": str(sweep_cfg.seed0),
            }
            for key, value in param_pairs:
                row[key] = str(value)
            for model_key, solver_name in sorted(solver_map.items()):
                row[f"solver_{model_key}"] = solver_name
            row["output_dir"] = str(combo_dir)
            row["summary_csv"] = str(summary_path) if summary_path is not None else ""
            index_rows.append(row)

    write_index(sweep_dir, index_rows)
    print(f"[INFO] Sweep directory: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python GraduationResearch/DFL_Portfolio_Optimization2/experiments/run_large_global_eval.py \
#   --runs 30 --seed0 100 \
#   --N 50 100 --d 3 5 10 --res 0 \
#   --snr 0.1 0.01 0.001 --rho 0.5 --delta 1.0 \
#   --solver dual=knitro,ipopt --solver kkt=knitro,ipopt \
#   --no-ensemble \
#   --info-only
