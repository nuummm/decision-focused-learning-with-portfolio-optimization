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
from itertools import product
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


def float_list(text: str) -> List[float]:
    items = [p.strip() for p in text.split(",") if p.strip()]
    values: List[float] = []
    for item in items:
        values.append(float(item))
    return values


def int_list(text: str) -> List[int]:
    items = [p.strip() for p in text.split(",") if p.strip()]
    values: List[int] = []
    for item in items:
        values.append(int(item))
    return values


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
    parser.add_argument(
        "--seed-list",
        type=str,
        default="",
        help="Comma-separated list of explicit seeds to evaluate (overrides runs/seed0).",
    )
    parser.add_argument(
        "--N",
        type=str,
        default="50",
        help="Number of samples (either a single integer or comma-separated list)",
    )
    parser.add_argument(
        "--d",
        type=str,
        default="3",
        help="Asset dimension (single integer or comma-separated list)",
    )
    parser.add_argument("--snr", type=float, default=0.1, help="Signal-to-noise ratio")
    parser.add_argument("--rho", type=float, default=0.5, help="Correlation parameter")
    parser.add_argument("--sigma", type=float, default=0.0125, help="Marginal stdev of returns")
    parser.add_argument("--res", type=int, default=10, help="Rolling window length (burn-in)")
    parser.add_argument(
        "--delta",
        type=str,
        default="1.0",
        help="Risk-aversion parameter (single value or comma-separated list)",
    )
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
        help="Comma-separated list of solvers for flex (e.g., 'gurobi,knitro').",
    )
    parser.add_argument(
        "--flex-formulation",
        type=str,
        default="dual",
        help="Flex formulation to forward to run.py (dual or kkt).",
    )
    parser.add_argument(
        "--flex-lambda-theta-anchor",
        type=str,
        default="0.0",
        help="Flex lambda for L2 anchoring on theta (comma-separated for multiple).",
    )
    parser.add_argument(
        "--flex-lambda-theta-anchor-l1",
        type=float,
        default=0.0,
        help="Flex lambda for L1 anchoring on theta.",
    )
    parser.add_argument(
        "--flex-lambda-theta-iso",
        type=str,
        default="0.0",
        help="Flex lambda for isotropic L2 regularisation on theta (comma-separated for multiple).",
    )
    parser.add_argument(
        "--flex-lambda-w-anchor",
        type=str,
        default="0.0",
        help="Flex lambda for L2 anchoring on portfolio weights (comma-separated for multiple).",
    )
    parser.add_argument(
        "--flex-lambda-w-anchor-l1",
        type=float,
        default=0.0,
        help="Flex lambda for L1 anchoring on portfolio weights.",
    )
    parser.add_argument(
        "--flex-lambda-w-iso",
        type=str,
        default="0.0",
        help="Flex lambda for isotropic L2 regularisation on portfolio weights (comma-separated for multiple).",
    )
    parser.add_argument(
        "--flex-theta-anchor-mode",
        type=str,
        default="none",
        help="Automatic theta anchor selection strategy for flex (comma-separated: none, ols, ipo).",
    )
    parser.add_argument(
        "--flex-w-anchor-mode",
        type=str,
        default="ols",
        help="Automatic weight anchor selection strategy for flex (comma-separated: ols, ipo).",
    )
    parser.add_argument(
        "--flex-theta-init-mode",
        type=str,
        default="ols",
        help="Initial theta strategy for flex (comma-separated: none, ols, ipo).",
    )
    parser.add_argument(
        "--flex-theta-clamp-enable",
        action="store_true",
        help="Enable element-wise clamp around a reference theta (OLS or IPO).",
    )
    parser.add_argument(
        "--flex-theta-clamp-source",
        type=str,
        default="none",
        help="Anchor source for theta clamp (none, ols, ipo).",
    )
    parser.add_argument(
        "--flex-w-clamp-enable",
        action="store_true",
        help="Enable element-wise clamp around a reference w (OLS or IPO).",
    )
    parser.add_argument(
        "--flex-w-clamp-source",
        type=str,
        default="none",
        help="Anchor source for w clamp (none, ols, ipo).",
    )
    parser.add_argument(
        "--flex-anchor-clamp-tol",
        type=str,
        default="0.0",
        help="Relative tolerance (e.g., 0.05 ⇒ ±5%) for theta/w clamp constraints (comma-separated).",
    )
    parser.add_argument(
        "--flex-anchor-clamp-floor",
        type=float,
        default=0.0,
        help="Minimum absolute tolerance for theta/w clamp constraints.",
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


@dataclass(frozen=True)
class FlexConfig:
    lambda_theta_anchor: float
    lambda_w_anchor: float
    lambda_theta_iso: float
    lambda_w_iso: float
    theta_anchor_mode: str
    w_anchor_mode: str
    theta_init_mode: str
    anchor_clamp_tol: float
    anchor_clamp_floor: float

    def metadata_pairs(self) -> List[Tuple[str, str]]:
        return [
            ("lambda_theta_anchor", f"{self.lambda_theta_anchor:g}"),
            ("lambda_w_anchor", f"{self.lambda_w_anchor:g}"),
            ("lambda_theta_iso", f"{self.lambda_theta_iso:g}"),
            ("lambda_w_iso", f"{self.lambda_w_iso:g}"),
            ("theta_anchor_mode", self.theta_anchor_mode),
            ("w_anchor_mode", self.w_anchor_mode),
            ("theta_init_mode", self.theta_init_mode),
            ("anchor_clamp_tol", f"{self.anchor_clamp_tol:g}"),
            ("anchor_clamp_floor", f"{self.anchor_clamp_floor:g}"),
        ]

    def metadata_dict(self) -> Dict[str, str]:
        return dict(self.metadata_pairs())

    def key_suffix(self) -> str:
        parts = []
        for key, value in self.metadata_pairs():
            if key == "lambda_theta_anchor":
                parts.append(f"lambda={value}")
            parts.append(f"{key}={value}")
        return "|".join(parts)

    def raw_dir_name(self, index: int) -> str:
        def sanitize_token(text: str) -> str:
            return text.replace("-", "m").replace(".", "p")

        parts = [
            f"lam{sanitize_token(f'{self.lambda_theta_anchor:g}')}",
            f"lwa{sanitize_token(f'{self.lambda_w_anchor:g}')}",
            f"lti{sanitize_token(f'{self.lambda_theta_iso:g}')}",
            f"lwi{sanitize_token(f'{self.lambda_w_iso:g}')}",
            f"tam_{self.theta_anchor_mode}",
            f"wam_{self.w_anchor_mode}",
            f"tim_{self.theta_init_mode}",
            f"clt{sanitize_token(f'{self.anchor_clamp_tol:g}')}",
            f"clf{sanitize_token(f'{self.anchor_clamp_floor:g}')}",
        ]
        return f"combo_{index:03d}_" + "_".join(parts)

    def describe(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in self.metadata_pairs())


def parse_float_choices(spec: object, default: float) -> List[float]:
    text = str(spec).strip()
    if not text:
        return [default]
    values = float_list(text)
    return values if values else [default]


def parse_int_choices(spec: object, default: int) -> List[int]:
    text = str(spec).strip()
    if not text:
        return [default]
    values = int_list(text)
    return values if values else [default]


def parse_str_choices(spec: object, default: str) -> List[str]:
    text = str(spec or "").strip()
    if not text:
        return [default.lower()]
    values = comma_split(text)
    return values if values else [default.lower()]


DEFAULT_SOLVERS: Dict[str, str] = {
    "dual": "gurobi",
    "kkt": "gurobi",
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

FLEX_METADATA_KEYS = [
    "lambda_theta_anchor",
    "lambda_w_anchor",
    "lambda_theta_iso",
    "lambda_w_iso",
    "theta_anchor_mode",
    "w_anchor_mode",
    "theta_init_mode",
    "anchor_clamp_tol",
    "anchor_clamp_floor",
    "theta_clamp_enable",
    "theta_clamp_source",
    "w_clamp_enable",
    "w_clamp_source",
]


def _is_zero_value(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        return abs(float(text)) < 1e-12
    except (TypeError, ValueError):
        return False


def _format_meta_params(meta: Dict[str, str]) -> List[str]:
    parts: List[str] = []
    for key in FLEX_METADATA_KEYS:
        value = meta.get(key)
        if value is None or value == "":
            continue
        if key.startswith("lambda") and _is_zero_value(value):
            continue
        parts.append(f"{key}={value}")
    return parts


def build_cmd(
    model_key: str,
    solver: str,
    args: argparse.Namespace,
    *,
    seed: int,
    outdir: Path,
    fixed_theta: Optional[Path] = None,
    flex_config: Optional[FlexConfig] = None,
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
        if flex_config is None:
            flex_config = FlexConfig(
                lambda_theta_anchor=parse_float_choices(getattr(args, "flex_lambda_theta_anchor", "0"), 0.0)[0],
                lambda_w_anchor=parse_float_choices(getattr(args, "flex_lambda_w_anchor", "0"), 0.0)[0],
                lambda_theta_iso=parse_float_choices(getattr(args, "flex_lambda_theta_iso", "0"), 0.0)[0],
                lambda_w_iso=parse_float_choices(getattr(args, "flex_lambda_w_iso", "0"), 0.0)[0],
                theta_anchor_mode=parse_str_choices(getattr(args, "flex_theta_anchor_mode", "none"), "none")[0],
                w_anchor_mode=parse_str_choices(getattr(args, "flex_w_anchor_mode", "ols"), "ols")[0],
                theta_init_mode=parse_str_choices(getattr(args, "flex_theta_init_mode", "ols"), "ols")[0],
                anchor_clamp_tol=parse_float_choices(getattr(args, "flex_anchor_clamp_tol", "0"), 0.0)[0],
                anchor_clamp_floor=float(getattr(args, "flex_anchor_clamp_floor", 0.0) or 0.0),
            )
        cmd.extend(
            [
                "--flex-formulation",
                flex_form,
                "--flex-lambda-theta-anchor",
                str(flex_config.lambda_theta_anchor),
                "--flex-lambda-theta-anchor-l1",
                str(getattr(args, "flex_lambda_theta_anchor_l1", 0.0)),
                "--flex-lambda-theta-iso",
                str(flex_config.lambda_theta_iso),
                "--flex-lambda-w-anchor",
                str(flex_config.lambda_w_anchor),
                "--flex-lambda-w-anchor-l1",
                str(getattr(args, "flex_lambda_w_anchor_l1", 0.0)),
                "--flex-lambda-w-iso",
                str(flex_config.lambda_w_iso),
                "--flex-theta-anchor-mode",
                str(flex_config.theta_anchor_mode).lower(),
                "--flex-w-anchor-mode",
                str(flex_config.w_anchor_mode).lower(),
                "--flex-theta-init-mode",
                str(flex_config.theta_init_mode).lower(),
            ]
        )
        if getattr(args, "flex_theta_clamp_enable", False):
            cmd.append("--flex-theta-clamp-enable")
        cmd.extend(
            [
                "--flex-theta-clamp-source",
                str(getattr(args, "flex_theta_clamp_source", "none")),
            ]
        )
        if getattr(args, "flex_w_clamp_enable", False):
            cmd.append("--flex-w-clamp-enable")
        cmd.extend(
            [
                "--flex-w-clamp-source",
                str(getattr(args, "flex_w_clamp_source", "none")),
            ]
        )
        cmd.extend(
            [
                "--flex-anchor-clamp-tol",
                str(flex_config.anchor_clamp_tol),
                "--flex-anchor-clamp-floor",
                str(flex_config.anchor_clamp_floor),
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
    flex_config: Optional[FlexConfig] = None,
) -> RunOutcome:
    cmd = build_cmd(
        model_key,
        solver,
        args,
        seed=seed,
        outdir=outdir,
        fixed_theta=fixed_theta,
        flex_config=flex_config,
    )
    if args.dryrun:
        print("[DRYRUN]", " ".join(cmd))
        raise SystemExit(0)

    display_model = model_display_name(model_key, args)
    if flex_config is not None:
        display_model = f"{display_model} [{flex_config.describe()}]"
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


def strip_lambda_suffix(model_key: str) -> str:
    if "|lambda=" in model_key:
        return model_key.split("|lambda=", 1)[0]
    return model_key


def model_key_metadata(model_key: str) -> Dict[str, str]:
    if "|lambda=" not in model_key:
        return {}
    suffix = model_key.split("|lambda=", 1)[1]
    parts = suffix.split("|")
    metadata: Dict[str, str] = {}
    for idx, part in enumerate(parts):
        if "=" not in part:
            if idx == 0 and part:
                metadata.setdefault("lambda_theta_anchor", part)
            continue
        key, value = part.split("=", 1)
        if key == "lambda":
            metadata.setdefault("lambda_theta_anchor", value)
        else:
            metadata[key] = value
    ordered_metadata: Dict[str, str] = {}
    for key in FLEX_METADATA_KEYS:
        if key in metadata:
            ordered_metadata[key] = metadata[key]
    return ordered_metadata


def base_model_key(model_key: str) -> str:
    base_key = strip_lambda_suffix(model_key)
    if base_key.startswith("flex:"):
        return "flex"
    return base_key


def flex_variant_form(model_key: str, args: argparse.Namespace) -> str:
    key = strip_lambda_suffix(model_key)
    if key.startswith("flex:"):
        parts = key.split(":")
        if len(parts) >= 3:
            return parts[2].strip().lower() or "dual"
        if len(parts) == 2:
            return parts[1].strip().lower() or "dual"
    value = str(getattr(args, "flex_formulation", "dual") or "dual")
    return value.strip().lower() or "dual"


def flex_solver_from_key(model_key: str, args: argparse.Namespace) -> str:
    key = strip_lambda_suffix(model_key)
    if key.startswith("flex:"):
        parts = key.split(":")
        if len(parts) >= 3:
            return parts[1].strip() or DEFAULT_SOLVERS.get("flex", "gurobi")
        if len(parts) == 2:
            return DEFAULT_SOLVERS.get("flex", "gurobi")
    solvers = getattr(args, "_flex_solvers", None)
    if solvers:
        return str(solvers[0])
    return DEFAULT_SOLVERS.get("flex", "gurobi")


def model_display_name(model_key: str, args: argparse.Namespace) -> str:
    base = base_model_key(model_key)
    meta = model_key_metadata(model_key)
    param_display = ""
    if meta:
        param_bits = _format_meta_params(meta)
        if param_bits:
            param_display = " (" + ", ".join(param_bits) + ")"
    if base == "flex":
        formulation = flex_variant_form(model_key, args)
        solver = flex_solver_from_key(model_key, args)
        return f"{formulation}(flex,{solver}){param_display}"
    return f"{strip_lambda_suffix(model_key)}{param_display}"


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


def run_single_configuration(args: argparse.Namespace) -> int:
    enabled_solvers: Dict[str, str] = dict(DEFAULT_SOLVERS)
    if args.disable_dual:
        enabled_solvers.pop("dual", None)
    if args.disable_kkt:
        enabled_solvers.pop("kkt", None)
    flex_variant_keys: List[str] = []
    flex_forms: List[str] = []
    flex_solvers: List[str] = []
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
        raw_solver_spec = str(getattr(args, "flex_solver", "gurobi") or "gurobi")
        flex_solvers = comma_split(raw_solver_spec)
        if not flex_solvers:
            cleaned_solver = raw_solver_spec.strip().lower()
            flex_solvers = [cleaned_solver] if cleaned_solver else [DEFAULT_SOLVERS.get("flex", "gurobi")]
        flex_solvers = list(dict.fromkeys(flex_solvers)) or [DEFAULT_SOLVERS.get("flex", "gurobi")]
        enabled_solvers.pop("flex", None)
        for solver_name in flex_solvers:
            solver_clean = solver_name.strip() or DEFAULT_SOLVERS.get("flex", "gurobi")
            for form in flex_forms:
                key = f"flex:{solver_clean}:{form}"
                flex_variant_keys.append(key)
                enabled_solvers[key] = solver_clean
    else:
        enabled_solvers.pop("flex", None)
    setattr(args, "_flex_forms", flex_forms)
    setattr(args, "_flex_solvers", flex_solvers)
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

    lambda_theta_anchor_values = parse_float_choices(getattr(args, "flex_lambda_theta_anchor", "0"), 0.0)
    lambda_w_anchor_values = parse_float_choices(getattr(args, "flex_lambda_w_anchor", "0"), 0.0)
    lambda_theta_iso_values = parse_float_choices(getattr(args, "flex_lambda_theta_iso", "0"), 0.0)
    lambda_w_iso_values = parse_float_choices(getattr(args, "flex_lambda_w_iso", "0"), 0.0)
    theta_anchor_modes = parse_str_choices(getattr(args, "flex_theta_anchor_mode", "none"), "none")
    w_anchor_modes = parse_str_choices(getattr(args, "flex_w_anchor_mode", "ols"), "ols")
    theta_init_modes = parse_str_choices(getattr(args, "flex_theta_init_mode", "ols"), "ols")
    anchor_clamp_tol_values = parse_float_choices(getattr(args, "flex_anchor_clamp_tol", "0"), 0.0)
    anchor_clamp_floor_value = float(getattr(args, "flex_anchor_clamp_floor", 0.0) or 0.0)

    flex_configs: List[FlexConfig] = [
        FlexConfig(
            lambda_theta_anchor=lam_theta_anchor,
            lambda_w_anchor=lam_w_anchor,
            lambda_theta_iso=lam_theta_iso,
            lambda_w_iso=lam_w_iso,
            theta_anchor_mode=theta_anchor_mode,
            w_anchor_mode=w_anchor_mode,
            theta_init_mode=theta_init_mode,
            anchor_clamp_tol=clamp_tol,
            anchor_clamp_floor=anchor_clamp_floor_value,
        )
        for (
            lam_theta_anchor,
            lam_w_anchor,
            lam_theta_iso,
            lam_w_iso,
            theta_anchor_mode,
            w_anchor_mode,
            theta_init_mode,
            clamp_tol,
        ) in product(
            lambda_theta_anchor_values,
            lambda_w_anchor_values,
            lambda_theta_iso_values,
            lambda_w_iso_values,
            theta_anchor_modes,
            w_anchor_modes,
            theta_init_modes,
            anchor_clamp_tol_values,
        )
    ]

    seed_list_raw = str(getattr(args, "seed_list", "") or "").strip()
    seed_list_values = int_list(seed_list_raw) if seed_list_raw else []
    use_seed_list = bool(seed_list_values)

    if use_seed_list:
        seed_pool = seed_list_values
    else:
        target = int(args.runs)
        if target <= 0:
            print("[WARN] runs must be positive; nothing to do")
            return 0
        seed0 = int(args.seed0)
        seed_pool = [seed0 + i for i in range(target)]
    target = len(seed_pool)
    if target <= 0:
        print("[WARN] No seeds available; nothing to do")
        return 0

    base_outdir: Path = args.outdir
    base_outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = base_outdir / timestamp
    raw_dir = exp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "experiment.log"

    per_model_rows: Dict[str, List[Dict[str, str]]] = {}
    model_metadata: Dict[str, Dict[str, object]] = {}

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

    def pop_attempt_rows(model_keys: List[str]) -> None:
        for key in model_keys:
            rows = per_model_rows.get(key)
            if rows:
                rows.pop()

    with open(log_path, "w", encoding="utf-8") as log_handle:
        for combo_idx, flex_config in enumerate(flex_configs):
            config_pairs = flex_config.metadata_pairs()
            combo_metadata = dict(config_pairs)
            config_desc = flex_config.describe()
            lam_str = combo_metadata["lambda_theta_anchor"]
            key_suffix = flex_config.key_suffix()

            theta_clamp_enabled = bool(getattr(args, "flex_theta_clamp_enable", False))
            theta_clamp_source = (getattr(args, "flex_theta_clamp_source", "none") or "none").lower()
            combo_metadata["theta_clamp_enable"] = "true" if theta_clamp_enabled else ""
            combo_metadata["theta_clamp_source"] = theta_clamp_source if theta_clamp_enabled else ""

            w_clamp_enabled = bool(getattr(args, "flex_w_clamp_enable", False))
            w_clamp_source = (getattr(args, "flex_w_clamp_source", "none") or "none").lower()
            combo_metadata["w_clamp_enable"] = "true" if w_clamp_enabled else ""
            combo_metadata["w_clamp_source"] = w_clamp_source if w_clamp_enabled else ""

            log_handle.write(f"\n=== flex_config[{combo_idx}] {config_desc} ===\n")
            lam_raw_dir = raw_dir / flex_config.raw_dir_name(combo_idx)
            lam_raw_dir.mkdir(parents=True, exist_ok=True)

            shared_seeds: List[int] = []
            for current_seed in seed_pool:
                attempt_keys: List[str] = []
                anchor_seed: Optional[int] = None
                failed = False

                for model in core_sequence:
                    request_seed = current_seed if anchor_seed is None else anchor_seed
                    current_flex_config = flex_config if model.startswith("flex") else None
                    outcome = run_model(
                        model,
                        enabled_solvers[model],
                        args,
                        seed=request_seed,
                        outdir=lam_raw_dir,
                        log_handle=log_handle,
                        keep_raw=args.keep_raw,
                        flex_config=current_flex_config,
                    )
                    model_seed = outcome.seed
                    if anchor_seed is None:
                        anchor_seed = model_seed
                    elif model_seed != anchor_seed:
                        print(
                            f"[WARN] {model} deviated to seed={model_seed} (expected {anchor_seed}); discarding"
                        )
                        pop_attempt_rows(attempt_keys)
                        failed = True
                        break

                    display_model = model_display_name(model, args)
                    key = f"{model}|{key_suffix}"
                    row_data = {
                        **outcome.run_row,
                        "model": display_model,
                        "solver": outcome.solver,
                        **combo_metadata,
                    }
                    per_model_rows.setdefault(key, []).append(row_data)
                    model_metadata.setdefault(
                        key,
                        {
                            "model": display_model,
                            "solver": outcome.solver,
                            **combo_metadata,
                        },
                    )
                    attempt_keys.append(key)

                if failed:
                    continue

                if anchor_seed is None:
                    print("[WARN] No valid seed collected; advancing seed cursor")
                    continue

                if not run_ensemble:
                    shared_seeds.append(anchor_seed)
                    print(
                        f"[INFO] Accepted seed {anchor_seed} ({config_desc}); total collected {len(shared_seeds)}/{target}"
                    )
                    continue

                dual_key = f"dual|{key_suffix}"
                kkt_key = f"kkt|{key_suffix}"
                theta_dual_vec = parse_theta(per_model_rows.get(dual_key, [{}])[-1]) if dual_key in per_model_rows else None
                theta_kkt_vec = parse_theta(per_model_rows.get(kkt_key, [{}])[-1]) if kkt_key in per_model_rows else None
                if theta_dual_vec is None or theta_kkt_vec is None:
                    print("[WARN] Failed to parse theta for ensemble; discarding seed")
                    pop_attempt_rows(attempt_keys)
                    continue

                ensembles = {
                    "ensemble_avg": 0.5 * (theta_dual_vec + theta_kkt_vec),
                    "ensemble_weighted": (np.linalg.norm(theta_dual_vec) * theta_dual_vec + np.linalg.norm(theta_kkt_vec) * theta_kkt_vec) / (np.linalg.norm(theta_dual_vec) + np.linalg.norm(theta_kkt_vec) + 1e-12),
                    "ensemble_normalized": 0.5 * (normalize_theta(theta_dual_vec) + normalize_theta(theta_kkt_vec)),
                }

                ensemble_fail = False
                for ens_model, theta_vec in ensembles.items():
                    theta_tmp = lam_raw_dir / f"{ens_model}_seed{anchor_seed}.npy"
                    np.save(theta_tmp, theta_vec)
                    try:
                        outcome = run_model(
                            ens_model,
                            enabled_solvers[ens_model],
                            args,
                            seed=anchor_seed,
                            outdir=lam_raw_dir,
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
                    ensemble_key = f"{ens_model}|{key_suffix}"
                    per_model_rows.setdefault(ensemble_key, []).append(
                        {
                            **outcome.run_row,
                            "model": ens_model,
                            "solver": outcome.solver,
                            **combo_metadata,
                        }
                    )
                    model_metadata.setdefault(
                        ensemble_key,
                        {"model": ens_model, "solver": outcome.solver, **combo_metadata},
                    )
                    attempt_keys.append(ensemble_key)

                if ensemble_fail:
                    pop_attempt_rows(attempt_keys)
                    continue

                shared_seeds.append(anchor_seed)
                print(
                    f"[INFO] Accepted seed {anchor_seed} ({config_desc}); total collected {len(shared_seeds)}/{target}"
                )

            if len(shared_seeds) < target:
                reason = "seed list exhausted" if use_seed_list else "some seeds failed"
                print(
                    f"[WARN] {reason} for config {config_desc}; collected {len(shared_seeds)}/{target} seeds"
                )

    # --- Prepare CSV outputs ---
    detail_rows: List[Dict[str, str]] = []
    for model_key, rows in per_model_rows.items():
        meta = model_metadata.get(model_key, {})
        for row in rows:
            row = dict(row)
            row.setdefault("model", meta.get("model", model_key))
            row.setdefault("solver", meta.get("solver", ""))
            if "seed" in row:
                try:
                    row["seed"] = str(int(round(float(row["seed"]))))
                except Exception:
                    row["seed"] = str(row.get("seed", ""))
            else:
                row["seed"] = ""
            for key in FLEX_METADATA_KEYS:
                row.setdefault(key, meta.get(key, ""))
            row.setdefault("N", str(args.N))
            row.setdefault("d", str(args.d))
            row.setdefault("snr", str(args.snr))
            row.setdefault("rho", str(args.rho))
            row.setdefault("sigma", str(args.sigma))
            row.setdefault("res", str(args.res))
            row.setdefault("delta", str(args.delta))
            detail_rows.append(row)

    detail_field_order = [
        "model",
        "solver",
        "N",
        "d",
        "snr",
        "rho",
        "sigma",
        "res",
        "delta",
    ] + FLEX_METADATA_KEYS + [
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
    for model_key, rows in per_model_rows.items():
        if not rows:
            continue
        metrics = summarise_metrics(rows, SUMMARY_METRIC_MAP.keys())
        meta = model_metadata.get(model_key, {})
        summary_row = {
            "model": meta.get("model", model_key),
            "solver": meta.get("solver", ""),
            "N": str(args.N),
            "d": str(args.d),
            "snr": str(args.snr),
            "rho": str(args.rho),
            "sigma": str(args.sigma),
            "res": str(args.res),
            "delta": str(args.delta),
        }
        for key in FLEX_METADATA_KEYS:
            summary_row[key] = meta.get(key, "")
        summary_row["n_seeds"] = str(len(rows))
        summary_row["seeds"] = seeds_to_string(rows)
        for src, dst in SUMMARY_METRIC_MAP.items():
            value = metrics.get(src, math.nan)
            summary_row[dst] = f"{value:.10g}" if not math.isnan(value) else ""
        summary_row["theta"] = average_theta(rows)
        summary_row["avg_weight_test"] = average_vector(rows, "avg_weight_test")
        summary_row["avg_weight_train"] = average_vector(rows, "avg_weight_train")
        summary_rows.append(summary_row)

    summary_field_order = [
        "model",
        "solver",
        "N",
        "d",
        "snr",
        "rho",
        "sigma",
        "res",
        "delta",
    ] + FLEX_METADATA_KEYS + [
        "n_seeds",
        "seeds",
    ] + list(SUMMARY_METRIC_MAP.values()) + ["theta", "avg_weight_test", "avg_weight_train"]

    summary_csv = exp_dir / args.report_csv.name
    write_csv(summary_csv, summary_rows, fieldnames=summary_field_order)

    plot_metrics(per_model_rows, summary_metrics, exp_dir, args.fig_format, args)

    for row in summary_rows:
        model = row["model"]
        meta = {name: row.get(name, "") for name in FLEX_METADATA_KEYS}
        param_bits = _format_meta_params(meta)
        param_display = ", ".join(param_bits)
        mean_cost = row.get("mean_cost_test", "")
        dec_err = row.get("decision_error_test", "")
        print(
            f"[RESULT] model={model} params=({param_display}) n={row['n_seeds']} "
            f"mean_cost_test={mean_cost} decision_error_test={dec_err}"
        )

    print(f"[INFO] Experiment outputs stored in {exp_dir}")

    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parsed_args = parse_args(argv)
    N_values = parse_int_choices(getattr(parsed_args, "N", "50"), 50)
    d_values = parse_int_choices(getattr(parsed_args, "d", "3"), 3)
    delta_values = parse_float_choices(getattr(parsed_args, "delta", "1.0"), 1.0)

    combos = list(product(N_values, d_values, delta_values))
    total = len(combos)
    if total == 0:
        print("[WARN] No (N, d, delta) combinations specified; exiting.")
        return 0

    base_kwargs = vars(parsed_args).copy()
    for idx, (N_val, d_val, delta_val) in enumerate(combos, start=1):
        combo_args = argparse.Namespace(**base_kwargs)
        combo_args.N = int(N_val)
        combo_args.d = int(d_val)
        combo_args.delta = float(delta_val)
        print(
            f"[CONFIG] Combination {idx}/{total} -> "
            f"N={combo_args.N}, d={combo_args.d}, delta={combo_args.delta}"
        )
        code = run_single_configuration(combo_args)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# python GraduationResearch/DFL_Portfolio_Optimization2/experiments/run_small_global_eval.py \
#   --runs 1 --seed0 100 --N 50 --d 3 --res 10 --snr 1e+6 --rho 0.5 --delta 1.0 --no-ensemble

'''
python /Users/kensei/VScode/GraduationResearch/DFL_Portfolio_Optimization2/experiments/run_small_global_eval.py \
  --N 50 \
  --d 3 \
  --snr 0.01 \
  --rho 0.5 \
  --sigma 0.0125 \
  --res 0 \
  --delta 1.0 \
  --runs 100 \
  --seed0 0 \
  --tee \
  --enable-flex \
  --flex-solver 'knitro' \
  --flex-formulation 'dual,kkt' \
  --flex-lambda-theta-anchor '0' \
  --flex-lambda-w-anchor 0 \
  --flex-lambda-theta-iso 0.0 \
  --flex-lambda-w-iso 0.0 \
  --flex-theta-anchor-mode ols \
  --flex-w-anchor-mode ols \
  --flex-theta-init-mode none \
  --no-ensemble \
  --disable-dual \
  --disable-kkt \
  --flex-w-clamp-enable --flex-w-clamp-source ols \
  --flex-anchor-clamp-tol 1,0.5,0.25,0.1,0.05,0.01 --flex-anchor-clamp-floor 0.0 \
  --flex-theta-clamp-enable --flex-theta-clamp-source ols \
  --flex-anchor-clamp-tol 1,0.5,0.25,0.1,0.05,0.01 --flex-anchor-clamp-floor 0.0

  




  --seed-list '1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31' \
  
  flex-theta-init-mode：ols,ipo,none
'''
