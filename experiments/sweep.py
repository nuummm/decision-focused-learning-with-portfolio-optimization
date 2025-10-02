# DFL_Portfolio_Optimization2/experiments/sweep.py
from __future__ import annotations
import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import csv
import shlex 
import re
import os

# プロジェクトルートと run.py
ROOT = Path(__file__).resolve().parents[1]
RUNPY = ROOT / "experiments" / "run.py"

# ---------- 入力パーサ（空白混じりOK） ----------
def _split_commalist(s: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,\s]+", s.strip()) if x.strip()]

def comma_floats(s: str) -> list[float]:
    return [float(x) for x in _split_commalist(s)]

def comma_ints(s: str) -> list[int]:
    return [int(x) for x in _split_commalist(s)]

def comma_strs(s: str) -> list[str]:
    return [x.lower() for x in _split_commalist(s)]

# ---------- コマンド生成（YAMLは使わない） ----------
def cmd_for(params: dict, python_exec: str) -> list[str]:
    cmd = [
        python_exec, str(RUNPY),
        "--model",  params["model"],
        "--solver", params["solver"],
        "--N", str(params["N"]), "--d", str(params["d"]),
        "--res", str(params["res"]), "--snr", str(params["snr"]),
        "--rho", str(params["rho"]), "--sigma", str(params["sigma"]),
        "--delta", str(params["delta"]), "--runs", str(params["runs"]),
        "--seed0", str(params["seed0"]), "--outdir", str(params["outdir"]),
        "--log-every", str(params["log_every"]),
        "--lambda-theta", str(params["lambda_theta"]),
    ]
    if params.get("tee", False): cmd.append("--tee")
    if params.get("no_plots", False): cmd.append("--no-plots")
    return cmd

def tag_for(params: dict) -> str:
    def f(x: float) -> str: return f"{x:g}"
    return (
        f"{params['model']}_{params['solver']}"
        f"_N{params['N']}_d{params['d']}"
        f"_res{params['res']}_snr{f(params['snr'])}_rho{f(params['rho'])}"
        f"_lambda{f(params['lambda_theta'])}"
    )

def main():
    ap = argparse.ArgumentParser(
        description="Grid sweep runner for DFL experiments (direct CLI; no YAML)."
    )

    # 掃引グリッド（必須）
    ap.add_argument("--snr_list", type=comma_floats, required=True,
                    help='e.g. "0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 1"')
    ap.add_argument("--N_list",   type=comma_ints,   required=True, help='"500, 1000" など')
    ap.add_argument("--res_list", type=comma_ints,   required=True, help='"50, 100" など')
    ap.add_argument("--rho_list", type=comma_floats, required=True, help='"0.25, 0.5, 0.75" など')

    # モデル×ソルバー（集合）
    ap.add_argument("--models",  type=comma_strs, default=comma_strs("dual,ols"),
                    help='例: "dual, ols"（kkt を含める場合は追加）')
    ap.add_argument("--solvers", type=comma_strs, default=comma_strs("ipopt"),
                    help='例: "ipopt"（dual/kkt 用）。gurobi/knitro を加えてもOK')
    ap.add_argument("--ols_solver", type=str, default="ipopt",
                    help="OLS 実行時にタグ用として付けるソルバー名（見かけ上の表示）")

    # 実験の固定パラメータ（全ジョブ共通）
    ap.add_argument("--d",      type=int,   default=10)
    ap.add_argument("--sigma",  type=float, default=0.0125)
    ap.add_argument("--delta",  type=float, default=1.0)
    ap.add_argument("--runs",   type=int,   default=10)
    ap.add_argument("--seed0",  type=int,   default=100)
    ap.add_argument("--outdir", type=str,   default="results/exp_unified_yaml")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--tee", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--lambda_list", type=comma_floats, default=comma_floats("0.0"),
                    help='theta L2 list, e.g. "0, 1e-6, 1e-4"')

    # 実行制御
    ap.add_argument("--workers", type=int, default=4, help="並列実行スレッド数（-1 or 0 で全コア）")
    ap.add_argument("--python",  type=str, default=sys.executable, help="python 実行ファイルのパス（venv使用時に指定）")
    ap.add_argument("--logdir",  type=str, default="results/sweep_logs",
                    help="各ジョブの stdout/stderr を保存するルート")
    ap.add_argument("--dryrun",  action="store_true", help="コマンドだけ表示して実行しない")
    args = ap.parse_args()

    # ---------- クエリチェック（入力正規化 & 検証） ----------
    def _uniq_sorted(xs):  # 数値/文字どちらもOK
        try:
            return sorted(set(xs))
        except TypeError:
            # 混在型にならない想定だが保険で
            return list(dict.fromkeys(xs))

    args.snr_list = _uniq_sorted(args.snr_list)
    args.N_list   = _uniq_sorted(args.N_list)
    args.res_list = _uniq_sorted(args.res_list)
    args.rho_list = _uniq_sorted(args.rho_list)
    args.models   = _uniq_sorted(args.models)
    args.solvers  = _uniq_sorted(args.solvers)
    args.lambda_list = _uniq_sorted(args.lambda_list)

    for name in ["snr_list","N_list","res_list","rho_list","models","solvers"]:
        if not getattr(args, name):
            print(f"[ERR] {name} is empty.")
            sys.exit(2)

    allowed_models  = {"dual","kkt","ols"}
    unknown_models  = set(args.models) - allowed_models
    if unknown_models:
        print(f"[ERR] unknown models: {sorted(unknown_models)}")
        sys.exit(2)

    allowed_solvers = {"ipopt","knitro","gurobi"}
    unknown_solvers = set(args.solvers) - allowed_solvers
    if unknown_solvers:
        print(f"[ERR] unknown solvers: {sorted(unknown_solvers)}")
        sys.exit(2)

    # workers を正規化
    workers = args.workers
    if workers is None or workers <= 0:  # -1, 0, None → 全コア
        workers = max(1, os.cpu_count() or 1)

    # 出力（ログ）ディレクトリ
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = Path(args.logdir) / ts
    log_root.mkdir(parents=True, exist_ok=True)

    # インデックスCSV
    index_csv = log_root / "sweep_index.csv"
    with open(index_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tag", "model", "solver", "snr", "N", "res", "rho", "lambda_theta",
            "d", "sigma", "delta", "runs", "seed0",
            "returncode", "log_path", "cmd"
        ])

    # ---------- ジョブ列挙 ----------
    jobs = []
    # OLS は正則化を掃引しない（lambda=0 の 1 回だけ）
    for snr, N, res, rho in itertools.product(args.snr_list, args.N_list, args.res_list, args.rho_list):
        if "ols" in args.models:
            jobs.append({
                "model": "ols",
                "solver": args.ols_solver,      # 表示用
                "snr": snr, "N": N, "res": res, "rho": rho,
                "lambda_theta": 0.0,            # 固定
                "d": args.d, "sigma": args.sigma, "delta": args.delta,
                "runs": args.runs, "seed0": args.seed0,
                "outdir": args.outdir, "log_every": args.log_every,
                "tee": args.tee, "no_plots": args.no_plots,
            })
        # dual/kkt は lambda を掃引
        for model in args.models:
            if model == "ols":
                continue
            for solver, lam in itertools.product(args.solvers, args.lambda_list):
                jobs.append({
                    "model": model,
                    "solver": solver,
                    "snr": snr, "N": N, "res": res, "rho": rho,
                    "lambda_theta": lam,
                    "d": args.d, "sigma": args.sigma, "delta": args.delta,
                    "runs": args.runs, "seed0": args.seed0,
                    "outdir": args.outdir, "log_every": args.log_every,
                    "tee": args.tee, "no_plots": args.no_plots,
                })

    total = len(jobs)
    print(f"[INFO] total jobs: {total}")

    def run_one(params: dict):
        tag = tag_for(params)
        log_path = log_root / f"{tag}.log"
        cmd = cmd_for(params, args.python)

        if args.dryrun:
            print(f"[DRY] {shlex.join(cmd)}")
            rc = 0
            cmd_str = " ".join(cmd)
            return (tag, params, rc, str(log_path), cmd_str)

        with open(log_path, "w") as lf:
            lf.write(f"# CMD: {shlex.join(cmd)}\n")
            lf.write(f"# START: {datetime.now().isoformat()}\n\n")
            lf.flush()
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            rc = proc.returncode
            lf.write(f"\n# END: {datetime.now().isoformat()}\n")
        cmd_str = " ".join(cmd)
        print(f"[DONE] tag={tag} -> rc={rc} | log={log_path}")
        return (tag, params, rc, str(log_path), cmd_str)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, p) for p in jobs]
        for fut in as_completed(futs):
            tag, params, rc, log_path, cmd_str = fut.result()
            results.append((tag, params, rc, log_path, cmd_str))
            with open(index_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    tag, params["model"], params["solver"],
                    params["snr"], params["N"], params["res"], params["rho"], params["lambda_theta"],
                    params["d"], params["sigma"], params["delta"], params["runs"], params["seed0"],
                    rc, log_path, cmd_str
                ])

    n_fail = sum(1 for _, _, rc, _, _ in results if rc != 0)
    print(f"\n[SUMMARY] done={len(results)} fail={n_fail} logs_dir={log_root}")
    print(f"[SUMMARY] index CSV: {index_csv}")

if __name__ == "__main__":
    main()

"""
cd ~/VScode/GraduationResearch

python DFL_Portfolio_Optimization2/experiments/sweep.py \
  --snr_list "0.001, 0.002, 0.005, 0.01" \
  --N_list "500" \
  --res_list "100" \
  --rho_list "0.5" \
  --models "dual, ols" \
  --solvers "ipopt" \
  --d 10 \
  --runs 50 \
  --seed0 100 \
  --lambda_list "0" \
  --outdir results/exp_unified_yaml \
  --logdir results/sweep_logs \
  --workers -1
"""