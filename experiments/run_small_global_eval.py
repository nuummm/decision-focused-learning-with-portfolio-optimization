#!/usr/bin/env python3
"""Evaluate dual/kkt/ols on shared globally-optimal seeds (small QCQP).

The goal is to study the "clean" regime where both decision-focused
formulations (dual / kkt) solve the non-convex QCQP to proven optimality,
and to compare them against the OLS baseline under identical randomness.

Workflow implemented here:

1. Repeatedly call ``run.py`` for the *dual* model with a single requested
   seed.  ``run.py`` itself skips any run that does not end with an optimal
   solver termination, so the per-run CSV it emits contains *only* seeds
   with globally optimal solutions.  We collect these seeds one by one.
2. For each accepted seed, we run ``kkt`` and ``ols`` by forwarding the same
   seed to ``run.py`` (again one at a time).  If ``kkt`` fails to prove
   optimality for that seed (and hence ``run.py`` falls forward to a
   different seed), the seed is discarded and the process continues until the
   requested number of shared seeds is obtained.
3. Per-seed metrics are aggregated and written to ``global_details.csv``
   (one row per model & seed) as well as ``global_summary.csv`` (per-model
   averages) so that any seed dependence is explicitly removed.

No code outside this helper is modified; we orchestrate everything via the
existing ``run.py`` entry point.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None


HERE = Path(__file__).resolve()
RUNPY = HERE.with_name("run.py")
RESULTS_ROOT = HERE.parents[2] / "results"
DEFAULT_OUTDIR = RESULTS_ROOT / "exp_small_global"
DEFAULT_REPORT = DEFAULT_OUTDIR / "global_summary.csv"

SUMMARY_PATH_RE = re.compile(r"summary CSV\s*:\s*(?P<path>.+)")
RUNS_PATH_RE = re.compile(r"per-run CSV\s*:\s*(?P<path>.+)")


def comma_split(text: str) -> List[str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [p.lower() for p in parts]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run small-scale QCQP experiments and aggregate optimal summaries."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of successful (globally optimal) seeds required per model",
    )
    parser.add_argument("--seed0", type=int, default=100, help="Initial seed")
    parser.add_argument("--N", type=int, default=50, help="Number of samples")
    parser.add_argument("--d", type=int, default=3, help="Asset dimension")
    parser.add_argument("--snr", type=float, default=0.1, help="Signal-to-noise ratio")
    parser.add_argument("--rho", type=float, default=0.5, help="Correlation parameter")
    parser.add_argument("--sigma", type=float, default=0.0125, help="Marginal stdev of returns")
    parser.add_argument("--res", type=int, default=10, help="Rolling window length (burn-in)")
    parser.add_argument("--delta", type=float, default=1.0, help="Risk-aversion parameter")
    parser.add_argument(
        "--lambda-theta",
        type=float,
        default=0.0,
        help="L2 regularisation for theta passed to run.py",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Root directory for experiment outputs",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=DEFAULT_REPORT,
        help="Summary CSV name (written inside each experiment directory)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter to invoke run.py",
    )
    parser.add_argument("--tee", action="store_true", help="Pass --tee to run.py")
    parser.add_argument(
        "--estimate-cov",
        action="store_true",
        help="Use rolling covariance estimates (default: true covariance)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Forward --no-plots")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw run.py CSV outputs instead of deleting them after aggregation",
    )
    parser.add_argument(
        "--fig-format",
        default="png",
        help="Image format for metric plots (default: png)",
    )
    parser.add_argument("--dryrun", action="store_true", help="Print commands only")
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble models (average/weighted/normalized).",
    )
    parser.add_argument(
        "--disable-dual",
        action="store_true",
        help="Skip the dual model in the shared-seed evaluation.",
    )
    parser.add_argument(
        "--disable-kkt",
        action="store_true",
        help="Skip the kkt model in the shared-seed evaluation.",
    )
    parser.add_argument(
        "--enable-flex",
        action="store_true",
        help="Include the flex model in the shared-seed evaluation pipeline.",
    )
    parser.add_argument(
        "--flex-solver",
        type=str,
        default="gurobi",
        help="Solver to use for the flex model when enabled.",
    )
    parser.add_argument(
        "--flex-formulation",
        type=str,
        default="dual",
        help="Flex formulation to forward to run.py (dual or kkt).",
    )
    parser.add_argument(
        "--flex-lambda-theta-anchor",
        type=float,
        default=0.0,
        help="Flex lambda for L2 anchoring on theta.",
    )
    parser.add_argument(
        "--flex-lambda-theta-anchor-l1",
        type=float,
        default=0.0,
        help="Flex lambda for L1 anchoring on theta.",
    )
    parser.add_argument(
        "--flex-lambda-theta-iso",
        type=float,
        default=0.0,
        help="Flex lambda for isotropic L2 regularisation on theta.",
    )
    parser.add_argument(
        "--flex-lambda-w-anchor",
        type=float,
        default=0.0,
        help="Flex lambda for L2 anchoring on portfolio weights.",
    )
    parser.add_argument(
        "--flex-lambda-w-anchor-l1",
        type=float,
        default=0.0,
        help="Flex lambda for L1 anchoring on portfolio weights.",
    )
    parser.add_argument(
        "--flex-lambda-w-iso",
        type=float,
        default=0.0,
        help="Flex lambda for isotropic L2 regularisation on portfolio weights.",
    )
    parser.add_argument(
        "--flex-theta-anchor-mode",
        type=str,
        default="none",
        help="Automatic theta anchor selection strategy for flex (e.g., none or ols).",
    )
    parser.add_argument(
        "--flex-w-anchor-mode",
        type=str,
        default="ols",
        help="Automatic weight anchor selection strategy for flex (e.g., ols).",
    )
    return parser.parse_args(argv)


@dataclass
class RunOutcome:
    model: str
    solver: str
    seed: int
    summary_csv: Path
    runs_csv: Path
    summary_row: Dict[str, str]
    run_row: Dict[str, str]


DEFAULT_SOLVERS: Dict[str, str] = {
    "dual": "knitro",
    "kkt": "knitro",
    "flex": "gurobi",
    "ols": "analytic",
    "ipo": "analytic",
    "ensemble_avg": "analytic",
    "ensemble_weighted": "analytic",
    "ensemble_normalized": "analytic",
}

SUMMARY_METRIC_MAP = {
    "mean_cost_test_vhat": "mean_cost_test",
    "train_cost_vhat": "mean_cost_train",
    "decision_error_test": "decision_error_test",
    "decision_error_train": "decision_error_train",
    "mean_corr2_test": "mean_corr2_test",
    "mean_r2_test": "mean_r2_test",
    "mse_test": "mse_test",
    "mean_corr2_train": "mean_corr2_train",
    "mean_r2_train": "mean_r2_train",
    "mse_train": "mse_train",
    "mean_return_test": "mean_return_test",
    "std_return_test": "std_return_test",
    "sharpe_ratio_test": "sharpe_ratio_test",
    "mean_return_train": "mean_return_train",
    "std_return_train": "std_return_train",
    "sharpe_ratio_train": "sharpe_ratio_train",
}


def build_cmd(
    model_key: str,
    solver: str,
    args: argparse.Namespace,
    *,
    seed: int,
    outdir: Path,
    fixed_theta: Optional[Path] = None,
) -> List[str]:
    base_model = base_model_key(model_key)
    cmd = [
        str(args.python),
        str(RUNPY),
        "--model",
        base_model,
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
    if base_model in {"dual", "kkt", "flex"}:
        cmd.append("--no-auto-baseline")
    if base_model == "flex":
        flex_form = flex_variant_form(model_key, args)
        cmd.extend(
            [
                "--flex-formulation",
                flex_form,
                "--flex-lambda-theta-anchor",
                str(getattr(args, "flex_lambda_theta_anchor", 0.0)),
                "--flex-lambda-theta-anchor-l1",
                str(getattr(args, "flex_lambda_theta_anchor_l1", 0.0)),
                "--flex-lambda-theta-iso",
                str(getattr(args, "flex_lambda_theta_iso", 0.0)),
                "--flex-lambda-w-anchor",
                str(getattr(args, "flex_lambda_w_anchor", 0.0)),
                "--flex-lambda-w-anchor-l1",
                str(getattr(args, "flex_lambda_w_anchor_l1", 0.0)),
                "--flex-lambda-w-iso",
                str(getattr(args, "flex_lambda_w_iso", 0.0)),
                "--flex-theta-anchor-mode",
                str(getattr(args, "flex_theta_anchor_mode", "none")).lower(),
                "--flex-w-anchor-mode",
                str(getattr(args, "flex_w_anchor_mode", "ols")).lower(),
            ]
        )
    if args.no_plots:
        cmd.append("--no-plots")
    if not args.estimate_cov:
        cmd.append("--use-true-cov")
    if args.tee:
        cmd.append("--tee")
    if fixed_theta is not None:
        cmd.append("--fixed-theta")
        cmd.append(str(fixed_theta))
    return cmd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def cleanup_file(path: Path, stop_dir: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return

    parent = path.parent
    stop_dir = stop_dir.resolve()
    while parent.resolve() != parent.parent.resolve():
        if parent.resolve() == stop_dir:
            if not any(parent.iterdir()):
                parent.rmdir()
            break
        if not any(parent.iterdir()):
            tmp = parent
            parent = parent.parent
            tmp.rmdir()
        else:
            break


def run_model(
    model_key: str,
    solver: str,
    args: argparse.Namespace,
    *,
    seed: int,
    outdir: Path,
    log_handle,
    keep_raw: bool,
    fixed_theta: Optional[Path] = None,
) -> RunOutcome:
    cmd = build_cmd(model_key, solver, args, seed=seed, outdir=outdir, fixed_theta=fixed_theta)
    if args.dryrun:
        print("[DRYRUN]", " ".join(cmd))
        raise SystemExit(0)

    display_model = model_display_name(model_key, args)
    print(f"[INFO] Running {display_model} (key={model_key}) with solver={solver} seed={seed}")

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
        raise RuntimeError(f"run.py failed for model={model_key}, solver={solver}: {msg}")

    stdout_text = "".join(stdout_lines)

    summary_path = None
    runs_path = None
    for line in stdout_text.splitlines():
        match_summary = SUMMARY_PATH_RE.search(line)
        if match_summary:
            summary_path = Path(match_summary.group("path")).resolve()
        match_runs = RUNS_PATH_RE.search(line)
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

    outcome = RunOutcome(
        model=model_key,
        solver=solver,
        seed=actual_seed,
        summary_csv=summary_path,
        runs_csv=runs_path,
        summary_row=summary_row,
        run_row=run_row,
    )

    if not keep_raw:
        cleanup_file(summary_path, stop_dir=outdir)
        cleanup_file(runs_path, stop_dir=outdir)

    return outcome


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        print(f"[WARN] No data to write for {path}")
        return
    ensure_parent(path)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    else:
        extra: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames and key not in extra:
                    extra.append(key)
        fieldnames = list(fieldnames) + extra
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[INFO] Wrote {path}")


def float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def summarise_metrics(rows: List[Dict[str, str]], metrics: List[str]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for metric in metrics:
        values = [float_or_nan(row.get(metric, "nan")) for row in rows]
        clean = [v for v in values if not math.isnan(v)]
        summary[metric] = float(sum(clean) / len(clean)) if clean else math.nan
    return summary


def seeds_to_string(rows: List[Dict[str, str]]) -> str:
    seeds: List[str] = []
    for row in rows:
        seed_val = row.get("seed")
        if seed_val is None or seed_val == "":
            continue
        try:
            seeds.append(str(int(round(float(seed_val)))))
        except (TypeError, ValueError):
            continue
    return ",".join(seeds)


def parse_vector_string(text: str) -> Optional[np.ndarray]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if not cleaned.strip():
        return None
    vec = np.fromstring(cleaned, sep=",")
    if vec.size == 0:
        return None
    return vec


def parse_theta(row: Dict[str, str]) -> Optional[np.ndarray]:
    return parse_vector_string(row.get("theta", ""))


def normalize_theta(vec: np.ndarray) -> np.ndarray:
    total = np.sum(vec)
    if not np.isfinite(total) or abs(total) < 1e-12:
        return vec.copy()
    return vec / total


def average_vector(rows: List[Dict[str, str]], key: str) -> str:
    vectors: List[np.ndarray] = []
    for row in rows:
        vec = parse_vector_string(row.get(key, ""))
        if vec is not None:
            vectors.append(vec)
    if not vectors:
        return ""
    lengths = {vec.size for vec in vectors}
    ref_len = max(lengths)
    aligned = [vec for vec in vectors if vec.size == ref_len]
    if not aligned:
        return ""
    mean_vec = np.mean(np.vstack(aligned), axis=0)
    return "[" + ", ".join(f"{val:.6g}" for val in mean_vec) + "]"


def average_theta(rows: List[Dict[str, str]]) -> str:
    return average_vector(rows, "theta")


def base_model_key(model_key: str) -> str:
    if model_key.startswith("flex:"):
        return "flex"
    return model_key


def flex_variant_form(model_key: str, args: argparse.Namespace) -> str:
    if model_key.startswith("flex:"):
        _, form = model_key.split(":", 1)
        return form.strip().lower() or "dual"
    value = str(getattr(args, "flex_formulation", "dual") or "dual")
    return value.strip().lower() or "dual"


def model_display_name(model_key: str, args: argparse.Namespace) -> str:
    base = base_model_key(model_key)
    if base == "flex":
        formulation = flex_variant_form(model_key, args)
        return f"{formulation}(flex)"
    return model_key


def plot_metrics(
    per_model_rows: Dict[str, List[Dict[str, str]]],
    metrics: List[str],
    exp_dir: Path,
    fig_format: str,
    args: argparse.Namespace,
) -> None:
    if plt is None:
        print("[WARN] matplotlib unavailable; skipping metric plots")
        return
    seeds = sorted(
        {int(round(float(row["seed"]))) for rows in per_model_rows.values() for row in rows if row.get("seed")}
    )
    if not seeds:
        print("[WARN] No seeds available for plotting")
        return

    for metric in metrics:
        plt.figure(figsize=(10, 4))
        for model, rows in per_model_rows.items():
            values_by_seed = {}
            for row in rows:
                seed_val = row.get("seed")
                if seed_val is None or seed_val == "":
                    continue
                try:
                    seed_int = int(round(float(seed_val)))
                    values_by_seed[seed_int] = float_or_nan(row.get(metric, "nan"))
                except (TypeError, ValueError):
                    continue
            series = [values_by_seed.get(seed, math.nan) for seed in seeds]
            plt.scatter(seeds, series, label=model_display_name(model, args))

        plt.xlabel("seed")
        plt.ylabel(metric)
        plt.title(f"{metric} by seed")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        fig_path = exp_dir / f"{metric}_by_seed.{fig_format}"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    enabled_solvers: Dict[str, str] = dict(DEFAULT_SOLVERS)
    if args.disable_dual:
        enabled_solvers.pop("dual", None)
    if args.disable_kkt:
        enabled_solvers.pop("kkt", None)
    flex_variant_keys: List[str] = []
    flex_forms: List[str] = []
    if args.enable_flex:
        raw_spec = str(getattr(args, "flex_formulation", "dual") or "dual")
        flex_forms = comma_split(raw_spec)
        if not flex_forms:
            cleaned = raw_spec.strip().lower()
            flex_forms = [cleaned] if cleaned else ["dual"]
        # remove duplicates while preserving order
        flex_forms = list(dict.fromkeys(flex_forms))
        if not flex_forms:
            flex_forms = ["dual"]
        enabled_solvers.pop("flex", None)
        for form in flex_forms:
            key = f"flex:{form}"
            flex_variant_keys.append(key)
            enabled_solvers[key] = str(args.flex_solver or "gurobi")
    else:
        enabled_solvers.pop("flex", None)
    setattr(args, "_flex_forms", flex_forms)
    ensemble_keys = ["ensemble_avg", "ensemble_weighted", "ensemble_normalized"]
    if args.no_ensemble:
        for key in ensemble_keys:
            enabled_solvers.pop(key, None)
    has_ensemble = all(key in enabled_solvers for key in ensemble_keys)
    can_ensemble = "dual" in enabled_solvers and "kkt" in enabled_solvers
    run_ensemble = has_ensemble and can_ensemble
    if not run_ensemble:
        for key in ensemble_keys:
            enabled_solvers.pop(key, None)
    else:
        run_ensemble = True

    target = int(args.runs)
    if target <= 0:
        print("[WARN] runs must be positive; nothing to do")
        return 0

    base_outdir: Path = args.outdir
    base_outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = base_outdir / timestamp
    raw_dir = exp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "experiment.log"

    shared_seeds: List[int] = []
    per_model_rows: Dict[str, List[Dict[str, str]]] = {key: [] for key in enabled_solvers}
    next_seed = int(args.seed0)

    core_sequence: List[str] = []
    for base in ["dual", "kkt"]:
        if base in enabled_solvers:
            core_sequence.append(base)
    core_sequence.extend(flex_variant_keys)
    for base in ["ols", "ipo"]:
        if base in enabled_solvers:
            core_sequence.append(base)
    if not core_sequence:
        print("[ERROR] No predictive model enabled (dual/kkt/flex/ols/ipo); aborting.")
        return 1

    def pop_attempt_rows(models: List[str]) -> None:
        for model in models:
            rows = per_model_rows.get(model)
            if rows:
                rows.pop()

    with open(log_path, "w", encoding="utf-8") as log_handle:
        while len(shared_seeds) < target:
            attempt_models: List[str] = []
            anchor_seed: Optional[int] = None
            failed = False

            for model in core_sequence:
                request_seed = next_seed if anchor_seed is None else anchor_seed
                outcome = run_model(
                    model,
                    enabled_solvers[model],
                    args,
                    seed=request_seed,
                    outdir=raw_dir,
                    log_handle=log_handle,
                    keep_raw=args.keep_raw,
                )
                model_seed = outcome.seed
                if anchor_seed is None:
                    anchor_seed = model_seed
                elif model_seed != anchor_seed:
                    print(
                        f"[WARN] {model} deviated to seed={model_seed} (expected {anchor_seed}); discarding"
                    )
                    pop_attempt_rows(attempt_models)
                    next_seed = max(model_seed, anchor_seed) + 1
                    failed = True
                    break

                display_model = model_display_name(model, args)
                per_model_rows[model].append({**outcome.run_row, "model": display_model, "solver": outcome.solver})
                attempt_models.append(model)

            if failed:
                continue

            if anchor_seed is None:
                print("[WARN] No valid seed collected; advancing seed cursor")
                next_seed += 1
                continue

            if not run_ensemble:
                shared_seeds.append(anchor_seed)
                next_seed = anchor_seed + 1
                print(f"[INFO] Accepted seed {anchor_seed}; total collected {len(shared_seeds)}/{target}")
                continue

            theta_dual_vec = parse_theta(per_model_rows["dual"][-1]) if "dual" in per_model_rows else None
            theta_kkt_vec = parse_theta(per_model_rows["kkt"][-1]) if "kkt" in per_model_rows else None
            if theta_dual_vec is None or theta_kkt_vec is None:
                print("[WARN] Failed to parse theta for ensemble; discarding seed")
                pop_attempt_rows(attempt_models)
                next_seed = anchor_seed + 1
                continue

            ensembles = {
                "ensemble_avg": 0.5 * (theta_dual_vec + theta_kkt_vec),
                "ensemble_weighted": (np.linalg.norm(theta_dual_vec) * theta_dual_vec + np.linalg.norm(theta_kkt_vec) * theta_kkt_vec) / (np.linalg.norm(theta_dual_vec) + np.linalg.norm(theta_kkt_vec) + 1e-12),
                "ensemble_normalized": 0.5 * (normalize_theta(theta_dual_vec) + normalize_theta(theta_kkt_vec)),
            }

            ensemble_results = {}
            ensemble_fail = False
            for ens_model, theta_vec in ensembles.items():
                theta_tmp = raw_dir / f"{ens_model}_seed{anchor_seed}.npy"
                np.save(theta_tmp, theta_vec)
                try:
                    outcome = run_model(
                        ens_model,
                        enabled_solvers[ens_model],
                        args,
                        seed=anchor_seed,
                        outdir=raw_dir,
                        log_handle=log_handle,
                        keep_raw=args.keep_raw,
                        fixed_theta=theta_tmp,
                    )
                except RuntimeError as exc:
                    print(f"[WARN] {ens_model} failed for seed={anchor_seed}: {exc}; discarding")
                    ensemble_fail = True
                    break
                finally:
                    if not args.keep_raw and theta_tmp.exists():
                        theta_tmp.unlink()
                if ensemble_fail:
                    break
                ensemble_results[ens_model] = outcome

            if ensemble_fail:
                pop_attempt_rows(attempt_models)
                next_seed = anchor_seed + 1
                continue

            for ens_model, outcome in ensemble_results.items():
                per_model_rows[ens_model].append({**outcome.run_row, "model": ens_model, "solver": outcome.solver})

            shared_seeds.append(anchor_seed)
            next_seed = anchor_seed + 1
            print(f"[INFO] Accepted seed {anchor_seed}; total collected {len(shared_seeds)}/{target}")

    # --- Prepare CSV outputs ---
    detail_rows: List[Dict[str, str]] = []
    for model, rows in per_model_rows.items():
        display_model = model_display_name(model, args)
        for row in rows:
            row = dict(row)
            row["model"] = row.get("model") or display_model
            row["solver"] = row.get("solver") or enabled_solvers.get(model, DEFAULT_SOLVERS.get(model, ""))
            if "seed" in row:
                try:
                    row["seed"] = str(int(round(float(row["seed"]))))
                except Exception:
                    row["seed"] = str(row.get("seed", ""))
            else:
                row["seed"] = ""
            detail_rows.append(row)

    detail_field_order = [
        "model",
        "solver",
        "runtime_sec",
        "eval_rows",
        "mean_cost_test_vtrue",
        "mean_cost_test_vhat",
        "train_cost_vtrue",
        "train_cost_vhat",
        "optimal_cost_test_vtrue",
        "optimal_cost_train_vtrue",
        "regret_test",
        "decision_error_test",
        "decision_error_train",
        "mean_r2_test",
        "mean_r2_train",
        "mean_corr2_test",
        "mean_corr2_train",
        "mse_test",
        "mse_train",
        "mean_return_test",
        "std_return_test",
        "sharpe_ratio_test",
        "mean_return_train",
        "std_return_train",
        "sharpe_ratio_train",
        "avg_weight_test",
        "avg_weight_train",
        "theta",
        "solver_status",
        "solver_termination",
        "solver_time",
        "solver_message",
        "theta_source",
        "gurobi_mip_gap",
        "budget_violation_train",
        "nonneg_violation_train",
        "stationarity_violation_train",
        "complementarity_violation_train",
        "strong_duality_violation_train",
        "seed",
    ]

    detail_csv = exp_dir / "global_details.csv"
    write_csv(detail_csv, detail_rows, fieldnames=detail_field_order)

    summary_metrics = list(SUMMARY_METRIC_MAP.keys())
    summary_rows: List[Dict[str, str]] = []
    for model, rows in per_model_rows.items():
        if not rows:
            continue
        metrics = summarise_metrics(rows, SUMMARY_METRIC_MAP.keys())
        summary_row = {
            "model": model_display_name(model, args),
            "solver": enabled_solvers.get(model, DEFAULT_SOLVERS.get(model, "")),
            "n_seeds": str(len(rows)),
            "seeds": seeds_to_string(rows),
        }
        for src, dst in SUMMARY_METRIC_MAP.items():
            value = metrics.get(src, math.nan)
            summary_row[dst] = f"{value:.10g}" if not math.isnan(value) else ""
        summary_row["theta"] = average_theta(rows)
        summary_row["avg_weight_test"] = average_vector(rows, "avg_weight_test")
        summary_row["avg_weight_train"] = average_vector(rows, "avg_weight_train")
        summary_rows.append(summary_row)

    summary_csv = exp_dir / args.report_csv.name
    write_csv(summary_csv, summary_rows)

    plot_metrics(per_model_rows, summary_metrics, exp_dir, args.fig_format, args)

    for row in summary_rows:
        model = row["model"]
        mean_cost = row.get("mean_cost_test", "")
        dec_err = row.get("decision_error_test", "")
        print(
            f"[RESULT] model={model} n={row['n_seeds']} "
            f"mean_cost_test={mean_cost} decision_error_test={dec_err}"
        )

    print(f"[INFO] Experiment outputs stored in {exp_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# python GraduationResearch/DFL_Portfolio_Optimization2/experiments/run_small_global_eval.py \
#   --runs 1 --seed0 100 --N 50 --d 3 --res 10 --snr 1e+6 --rho 0.5 --delta 1.0 --no-ensemble

'''
python /Users/kensei/VScode/GraduationResearch/DFL_Portfolio_Optimization2/experiments/run_small_global_eval.py \
  --N 50 \
  --d 3 \
  --snr 0.1 \
  --rho 0.5 \
  --sigma 0.0125 \
  --res 0 \
  --delta 1.0 \
  --runs 1 \
  --seed0 200 \
  --enable-flex \
  --flex-solver gurobi \
  --flex-formulation 'kkt,dual' \
  --flex-lambda-theta-anchor 0.0 \
  --flex-lambda-w-anchor 0.0 \
  --flex-lambda-theta-iso 0.0 \
  --flex-lambda-w-iso 0.0 \
  --flex-theta-anchor-mode ols \
  --flex-w-anchor-mode ols \
  --no-ensemble \
  --disable-dual \
  --disable-kkt

'''
