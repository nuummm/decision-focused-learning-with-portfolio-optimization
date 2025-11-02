#!/usr/bin/env python3
"""Benchmark solver configurations for flex model on a small problem.

Runs a fixed experimental setup for IPOPT and Knitro with multiple solver
parameterisations, measures wall-clock time per run, and reports whether each
configuration satisfies the desired time budget.
"""
from __future__ import annotations

import copy
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULT_ROOT = ROOT.parents[0] / "results"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

from experiments import registry, run_small_global_eval  # noqa: E402

RUNS = 5
TIME_LIMIT_PER_RUN = 60.0
SEEDS = [200 + i for i in range(RUNS)]
BASE_ARGS = [
    "--N", "50",
    "--d", "3",
    "--snr", "0.1",
    "--rho", "0.5",
    "--sigma", "0.0125",
    "--res", "0",
    "--delta", "1.0",
    "--enable-flex",
    "--flex-formulation", "kkt,dual",
    "--flex-lambda-theta-anchor", "0",
    "--flex-lambda-w-anchor", "0.0",
    "--flex-lambda-theta-iso", "0.0",
    "--flex-lambda-w-iso", "0.0",
    "--flex-theta-anchor-mode", "ols",
    "--flex-w-anchor-mode", "ols",
    "--no-ensemble",
    "--disable-dual",
    "--disable-kkt",
    "--flex-theta-init-mode", "none",
]

SCENARIOS = [
    {
        "name": "ipopt_adaptive_exact",
        "solver": "ipopt",
        "options": {
            "tol": 1e-8,
            "acceptable_tol": 1e-6,
            "max_iter": 60000,
            "mu_strategy": "adaptive",
            "hessian_approximation": "exact",
            "linear_solver": "ma57",
            "max_cpu_time": 240,
            "print_level": 4,
        },
    },
    {
        "name": "ipopt_monotone_lbfgs",
        "solver": "ipopt",
        "options": {
            "tol": 1e-6,
            "acceptable_tol": 1e-4,
            "hessian_approximation": "limited-memory",
            "limited_memory_max_history": 20,
            "linear_solver": "mumps",
            "mu_strategy": "monotone",
            "max_iter": 40000,
            "max_cpu_time": 180,
            "print_level": 3,
        },
    },
    {
        "name": "ipopt_adaptive_lbfgs",
        "solver": "ipopt",
        "options": {
            "tol": 5e-7,
            "acceptable_tol": 1e-4,
            "hessian_approximation": "limited-memory",
            "limited_memory_max_history": 15,
            "linear_solver": "mumps",
            "mu_strategy": "adaptive",
            "max_iter": 50000,
            "max_cpu_time": 200,
            "print_level": 3,
        },
    },
    {
        "name": "knitro_sqp_lbfgs",
        "solver": "knitro",
        "options": {
            "outlev": 3,
            "nlp_algorithm": 2,
            "hessopt": 6,
            "lmsize": 30,
            "feastol": 1e-6,
            "opttol": 1e-6,
            "maxit": 7000,
            "maxtime_real": 200,
            "ms_enable": 1,
            "ms_maxsolves": 5,
            "ms_maxtime": 35,
        },
    },
    {
        "name": "knitro_active_sr1",
        "solver": "knitro",
        "options": {
            "outlev": 3,
            "nlp_algorithm": 3,
            "hessopt": 4,
            "feastol": 1e-6,
            "opttol": 1e-6,
            "maxit": 6000,
            "maxtime_real": 180,
            "ms_enable": 0,
        },
    },
    {
        "name": "knitro_ip_light",
        "solver": "knitro",
        "options": {
            "outlev": 3,
            "nlp_algorithm": 1,
            "hessopt": 6,
            "lmsize": 25,
            "feastol": 5e-6,
            "opttol": 5e-6,
            "maxit": 5000,
            "maxtime_real": 180,
            "ms_enable": 0,
            "bar_murule": 4,
        },
    },
]


def apply_solver_options(scenario: Dict[str, object], originals: Dict[str, Dict[str, object]]) -> None:
    solver = scenario["solver"]
    options = scenario["options"]
    if solver == "ipopt":
        registry.IPOPT_DEFAULTS.clear()
        registry.IPOPT_DEFAULTS.update(copy.deepcopy(originals["ipopt"]))
        registry.IPOPT_DEFAULTS.update(options)
    elif solver == "knitro":
        registry.KNITRO_DEFAULTS.clear()
        registry.KNITRO_DEFAULTS.update(copy.deepcopy(originals["knitro"]))
        registry.KNITRO_DEFAULTS.update(options)
    else:
        raise ValueError(f"Unsupported solver: {solver}")


def restore_defaults(originals: Dict[str, Dict[str, object]]) -> None:
    registry.IPOPT_DEFAULTS.clear()
    registry.IPOPT_DEFAULTS.update(copy.deepcopy(originals["ipopt"]))
    registry.KNITRO_DEFAULTS.clear()
    registry.KNITRO_DEFAULTS.update(copy.deepcopy(originals["knitro"]))


def run_scenario(
    scenario: Dict[str, object], originals: Dict[str, Dict[str, object]], base_dir: Path
) -> Dict[str, object]:
    name = scenario["name"]
    solver = scenario["solver"]
    scenario_dir = base_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    total_elapsed = 0.0
    completed = 0
    status = "ok"
    note = ""

    per_run_records: List[Dict[str, object]] = []
    metrics_acc: Dict[str, List[float]] = {
        "kkt_mean_cost_test": [],
        "kkt_mean_cost_train": [],
        "dual_mean_cost_test": [],
        "dual_mean_cost_train": [],
    }

    apply_solver_options(scenario, originals)
    try:
        for seed in SEEDS:
            seed_outdir = scenario_dir / f"seed{seed}"
            args = BASE_ARGS + [
                "--runs", "1",
                "--seed0", str(seed),
                "--flex-solver", solver,
                "--outdir", str(seed_outdir),
            ]
            start = time.perf_counter()
            try:
                rc = run_small_global_eval.main(args)
            except Exception as exc:  # pylint: disable=broad-except
                status = "error"
                note = f"seed {seed}: {exc}"
                break
            run_time = time.perf_counter() - start
            total_elapsed += run_time
            completed += 1
            if rc != 0:
                status = "error"
                note = f"seed {seed}: run_small_global_eval exited with code {rc}"
                break

            try:
                latest_dir = max(
                    [p for p in seed_outdir.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                )
                summary_path = latest_dir / "global_summary.csv"
                with summary_path.open("r", newline="") as f_sum:
                    reader = csv.DictReader(f_sum)
                    for row in reader:
                        model = row.get("model", "")
                        if not model:
                            continue
                        if model.startswith("kkt") or model.startswith("dual"):
                            try:
                                test_cost = float(row.get("mean_cost_test", "nan"))
                            except ValueError:
                                test_cost = math.nan
                            try:
                                train_cost = float(row.get("mean_cost_train", "nan"))
                            except ValueError:
                                train_cost = math.nan
                            key_prefix = "kkt" if model.startswith("kkt") else "dual"
                            metrics_acc[f"{key_prefix}_mean_cost_test"].append(test_cost)
                            metrics_acc[f"{key_prefix}_mean_cost_train"].append(train_cost)
                            per_run_records.append(
                                {
                                    "seed": seed,
                                    "model": model,
                                    "run_time_sec": run_time,
                                    "mean_cost_test": test_cost,
                                    "mean_cost_train": train_cost,
                                }
                            )
            except Exception as exc:  # pylint: disable=broad-except
                per_run_records.append(
                    {
                        "seed": seed,
                        "model": "parse_error",
                        "run_time_sec": run_time,
                        "mean_cost_test": math.nan,
                        "mean_cost_train": math.nan,
                        "note": str(exc),
                    }
                )

            if run_time > TIME_LIMIT_PER_RUN:
                status = "timeout"
                note = f"seed {seed} took {run_time:.1f}s (> {TIME_LIMIT_PER_RUN:.0f}s)"
                break
    finally:
        restore_defaults(originals)

    avg_per_run = total_elapsed / completed if completed else 0.0

    if per_run_records:
        per_run_csv = scenario_dir / "per_run_metrics.csv"
        fieldnames = ["seed", "model", "run_time_sec", "mean_cost_test", "mean_cost_train", "note"]
        with per_run_csv.open("w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for rec in per_run_records:
                if "note" not in rec:
                    rec = {**rec, "note": ""}
                writer.writerow(rec)

    def avg_metric(key: str) -> float:
        vals = [v for v in metrics_acc[key] if v is not None and math.isfinite(v)]
        return float(sum(vals) / len(vals)) if vals else math.nan

    return {
        "name": name,
        "solver": solver,
        "status": status,
        "elapsed_sec": total_elapsed,
        "avg_per_run_sec": avg_per_run,
        "completed_runs": completed,
        "note": note,
        "kkt_mean_cost_test": avg_metric("kkt_mean_cost_test"),
        "kkt_mean_cost_train": avg_metric("kkt_mean_cost_train"),
        "dual_mean_cost_test": avg_metric("dual_mean_cost_test"),
        "dual_mean_cost_train": avg_metric("dual_mean_cost_train"),
    }


def main() -> int:
    originals = {
        "ipopt": copy.deepcopy(registry.IPOPT_DEFAULTS),
        "knitro": copy.deepcopy(registry.KNITRO_DEFAULTS),
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_dir = RESULT_ROOT / "solver_speed" / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        print(f"\n[INFO] Benchmarking scenario: {scenario['name']} (solver={scenario['solver']})")
        res = run_scenario(scenario, originals, summary_dir)
        results.append(res)
        print(
            f"  status={res['status']} runs={res['completed_runs']}/{RUNS} "
            f"total={res['elapsed_sec']:.2f}s avg/run={res['avg_per_run_sec']:.2f}s note={res['note']}"
        )
    print("\n=== Summary ===")
    for res in results:
        print(
            f"{res['name']:<20} solver={res['solver']:<6} status={res['status']:<8} "
            f"runs={res['completed_runs']}/{RUNS} total={res['elapsed_sec']:.2f}s avg/run={res['avg_per_run_sec']:.2f}s "
            f"kkt_test={res['kkt_mean_cost_test']:.6g} dual_test={res['dual_mean_cost_test']:.6g} {res['note']}"
        )

    summary_csv = summary_dir / "solver_speed_summary.csv"
    fieldnames = [
        "name",
        "solver",
        "status",
        "completed_runs",
        "elapsed_sec",
        "avg_per_run_sec",
        "kkt_mean_cost_test",
        "kkt_mean_cost_train",
        "dual_mean_cost_test",
        "dual_mean_cost_train",
        "note",
    ]
    with summary_csv.open("w", newline="") as f_sum:
        writer = csv.DictWriter(f_sum, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({field: res.get(field, "") for field in fieldnames})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
