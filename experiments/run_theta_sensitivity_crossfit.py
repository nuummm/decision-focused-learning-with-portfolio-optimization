#!/usr/bin/env python3
"""
Theta感度分析と交差適合（seed／初期値スイープ）をまとめて実行する補助スクリプト。

主な機能
--------
1. ベースラインとなる `theta*` を学習し、各成分方向に ±ε の摂動を加えて
   - テスト区間での MVO コスト（V_true 評価）の分散
   - MVO 解ベクトル w(θ) の変動ノルム
   を記録・可視化。
2. seed や flex の初期化モードを切り替えて複数回学習し、テスト性能分布と
   「良い局所解」の頻度を集計。

出力
----
- `/Users/kensei/VScode/GraduationResearch/results/<tag>/<timestamp>/` に
  CSV・図・設定ファイル・学習済みtheta等を保存。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional
    plt = None

from experiments.registry import SolverSpec
from experiments.run import mvo_cost, run_once
from data.synthetic import generate_simulation1_dataset
from models.covariance import estimate_epscov_rolling
from models.ols import predict_yhat
from models.ols_gurobi import solve_series_mvo_gurobi


# ---------------------------------------------------------------------------
# 汎用ヘルパー
# ---------------------------------------------------------------------------


def comma_split(text: str) -> List[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


def float_from_text(text: str | float | None, default: float) -> float:
    if text is None:
        return float(default)
    try:
        return float(text)
    except (TypeError, ValueError):
        parts = comma_split(str(text))
        for part in parts:
            try:
                return float(part)
            except ValueError:
                continue
        return float(default)


def int_list_from_text(text: str | int | None, default: Sequence[int]) -> List[int]:
    if text is None:
        return list(default)
    if isinstance(text, int):
        return [int(text)]
    vals = []
    for part in comma_split(str(text)):
        try:
            vals.append(int(part))
        except ValueError:
            continue
    return vals or list(default)


def str_list_from_text(text: str | None, default: Sequence[str]) -> List[str]:
    if text is None:
        return list(default)
    vals = comma_split(str(text))
    return vals or list(default)


def vector_to_string(vec: np.ndarray) -> str:
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0:
        return ""
    return "[" + ", ".join(f"{val:.6g}" for val in vec) + "]"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentParams:
    N: int
    d: int
    snr: float
    rho: float
    sigma: float
    res: int
    delta: float


@dataclass
class DatasetBundle:
    X: np.ndarray
    Y: np.ndarray
    V_true: np.ndarray
    theta0: np.ndarray
    tau: np.ndarray
    idx: List[int]
    Vhats: List[np.ndarray]
    train_indices: List[int]
    test_indices: List[int]
    Vhats_train: List[np.ndarray]
    Vhats_test: List[np.ndarray]
    use_true_cov: bool


@dataclass
class EvaluationResult:
    theta: np.ndarray
    mean_cost_true: float
    mean_cost_vhat: float
    var_cost_true: float
    costs_true: np.ndarray
    costs_vhat: np.ndarray
    weights: np.ndarray
    mean_return: float
    std_return: float
    sharpe: float


# ---------------------------------------------------------------------------
# データ準備 & 評価
# ---------------------------------------------------------------------------


def prepare_dataset(
    params: ExperimentParams,
    seed: int,
    *,
    use_true_cov: bool,
) -> DatasetBundle:
    X, Y, V_true, theta0, tau = generate_simulation1_dataset(
        n_samples=params.N,
        n_assets=params.d,
        snr=params.snr,
        rho=params.rho,
        sigma=params.sigma,
        seed=seed,
    )
    if use_true_cov:
        idx_list = list(range(params.res, params.N))
        Vhats_list = [np.asarray(V_true, dtype=float) for _ in idx_list]
    else:
        Vhats, idx = estimate_epscov_rolling(
            Y,
            X,
            theta0,
            tau,
            res=params.res,
            include_current=False,
        )
        idx_list = list(idx)
        Vhats_list = [np.asarray(V, dtype=float) for V in Vhats]

    burn_in = params.res
    if burn_in >= params.N:
        raise ValueError("res (burn-in) must be smaller than N")
    n_eff = params.N - burn_in
    n_tr = n_eff // 2

    train_pairs = [
        (i, V) for i, V in zip(idx_list, Vhats_list) if burn_in <= i < burn_in + n_tr
    ]
    test_pairs = [
        (i, V) for i, V in zip(idx_list, Vhats_list) if burn_in + n_tr <= i < params.N
    ]

    train_idx = [i for i, _ in train_pairs]
    test_idx = [i for i, _ in test_pairs]
    Vhats_train = [V for _, V in train_pairs]
    Vhats_test = [V for _, V in test_pairs]

    if not test_idx:
        raise ValueError("No test indices available; check (N, res) configuration.")

    return DatasetBundle(
        X=np.asarray(X, dtype=float),
        Y=np.asarray(Y, dtype=float),
        V_true=np.asarray(V_true, dtype=float),
        theta0=np.asarray(theta0, dtype=float),
        tau=np.asarray(tau, dtype=float),
        idx=train_idx + test_idx,
        Vhats=Vhats_list,
        train_indices=train_idx,
        test_indices=test_idx,
        Vhats_train=Vhats_train,
        Vhats_test=Vhats_test,
        use_true_cov=use_true_cov,
    )


def evaluate_theta(
    theta: np.ndarray,
    dataset: DatasetBundle,
    params: ExperimentParams,
) -> EvaluationResult:
    theta = np.asarray(theta, dtype=float)
    Yhat_all = predict_yhat(dataset.X, theta)

    weights = solve_series_mvo_gurobi(
        Yhat_all=Yhat_all,
        Vhats=dataset.Vhats_test,
        idx=dataset.test_indices,
        delta=params.delta,
        psd_eps=1e-12,
        output=False,
        start_index=None,
    )

    if weights.size == 0:
        raise ValueError("solve_series_mvo_gurobi returned empty weight matrix.")

    Y_eval = dataset.Y[dataset.test_indices]
    costs_true = np.array(
        [mvo_cost(z, y, dataset.V_true, params.delta) for z, y in zip(weights, Y_eval)],
        dtype=float,
    )
    costs_vhat = np.array(
        [
            mvo_cost(z, y, V_hat, params.delta)
            for z, y, V_hat in zip(weights, Y_eval, dataset.Vhats_test)
        ],
        dtype=float,
    )

    mean_return = float(np.mean(np.sum(weights * Y_eval, axis=1)))
    std_return = float(np.std(np.sum(weights * Y_eval, axis=1), ddof=1))
    sharpe = float(mean_return / std_return) if std_return > 1e-12 else float("nan")

    var_cost_true = float(np.var(costs_true, ddof=1)) if costs_true.size > 1 else 0.0

    return EvaluationResult(
        theta=theta,
        mean_cost_true=float(np.mean(costs_true)),
        mean_cost_vhat=float(np.mean(costs_vhat)),
        var_cost_true=var_cost_true,
        costs_true=costs_true,
        costs_vhat=costs_vhat,
        weights=weights,
        mean_return=mean_return,
        std_return=std_return,
        sharpe=sharpe,
    )


# ---------------------------------------------------------------------------
# 感度解析
# ---------------------------------------------------------------------------


def run_theta_sensitivity(
    base_theta: np.ndarray,
    dataset: DatasetBundle,
    params: ExperimentParams,
    epsilons: Sequence[float],
    max_directions: Optional[int] = None,
    random_directions: int = 0,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    base_eval = evaluate_theta(base_theta, dataset, params)
    rows = []

    dims = base_theta.size
    limit = dims if max_directions is None else max(1, min(dims, int(max_directions)))
    rng = np.random.default_rng(random_seed)

    directions: List[Tuple[str, str, np.ndarray]] = []
    for j in range(limit):
        direction = np.zeros(dims, dtype=float)
        direction[j] = 1.0
        directions.append(("axis", f"axis_{j}", direction))

    for idx in range(max(0, int(random_directions))):
        vec = rng.normal(size=dims)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            continue
        direction = vec / norm
        directions.append(("random", f"random_{idx}", direction))

    for eps in epsilons:
        eps = float(eps)
        for kind, name, direction in directions:
            for sign in (-1.0, 1.0):
                theta_pert = base_theta + sign * eps * direction
                eval_pert = evaluate_theta(theta_pert, dataset, params)
                weight_diff = eval_pert.weights - base_eval.weights
                weight_diff_fro = float(np.linalg.norm(weight_diff))
                weight_diff_mean = float(
                    np.linalg.norm(np.mean(weight_diff, axis=0))
                )
                theta_diff = theta_pert - base_theta
                cost_delta = eval_pert.mean_cost_true - base_eval.mean_cost_true
                rows.append(
                    {
                        "epsilon": eps,
                        "direction_type": kind,
                        "direction_label": name,
                        "direction_index": int(name.split("_")[-1]) if kind == "axis" else "",
                        "sign": sign,
                        "theta_diff_norm": float(np.linalg.norm(theta_diff)),
                        "theta_diff_max": float(np.max(np.abs(theta_diff))),
                        "mean_cost_true": eval_pert.mean_cost_true,
                        "mean_cost_vhat": eval_pert.mean_cost_vhat,
                        "var_cost_true": eval_pert.var_cost_true,
                        "mean_return": eval_pert.mean_return,
                        "std_return": eval_pert.std_return,
                        "sharpe": eval_pert.sharpe,
                        "weight_diff_fro": weight_diff_fro,
                        "weight_diff_mean_norm": weight_diff_mean,
                        "cost_delta": cost_delta,
                        "cost_improvement": -cost_delta,
                    }
                )

    df = pd.DataFrame(rows)
    summary = {
        "base_mean_cost_true": base_eval.mean_cost_true,
        "base_mean_cost_vhat": base_eval.mean_cost_vhat,
        "base_var_cost_true": base_eval.var_cost_true,
        "base_mean_return": base_eval.mean_return,
        "base_std_return": base_eval.std_return,
        "base_sharpe": base_eval.sharpe,
    }
    return df, summary


def plot_sensitivity(df: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib が見つからなかったため、感度図をスキップしました。")
        return
    if df.empty:
        print("[WARN] 感度データが空のため、図を作成しません。")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        df["weight_diff_fro"],
        df["mean_cost_true"],
        c=df["epsilon"],
        cmap="viridis",
        alpha=0.75,
    )
    ax.set_xlabel(r"$\|w(\theta)-w(\theta^\star)\|_F$")
    ax.set_ylabel("Mean MVO Cost (Vtrue)")
    ax.set_title("Theta Sensitivity (per-dimension perturbations)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("epsilon")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 交差適合
# ---------------------------------------------------------------------------


def unpack_run_once(result: Tuple) -> Dict[str, object]:
    """
    run_once の戻り値（tuple）を dict に整形。
    """
    theta = np.asarray(result[16], dtype=float)
    avg_weight_test = np.asarray(result[34], dtype=float)
    avg_weight_train = np.asarray(result[35], dtype=float)
    return {
        "mean_cost_true": float(result[0]),
        "mean_cost_vhat": float(result[5]),
        "elapsed": float(result[6]),
        "decision_error_test": float(result[14]) if result[14] is not None else math.nan,
        "decision_error_train": float(result[15]) if result[15] is not None else math.nan,
        "mean_r2_test": float(result[1]),
        "mean_r2_train": float(result[7]),
        "mean_r2_sklearn": float(result[8]),
        "train_r2_sklearn": float(result[9]),
        "mse_test": float(result[10]),
        "mse_train": float(result[11]),
        "best_cost_true": float(result[12]),
        "train_best_cost_true": float(result[13]),
        "solver_status": str(result[17]),
        "solver_term": str(result[18]),
        "solver_time": float(result[19]) if _is_float_like(result[19]) else math.nan,
        "solver_message": str(result[20]),
        "theta_source": str(result[21]),
        "budget_violation": float(result[22]) if _is_float_like(result[22]) else math.nan,
        "nonneg_violation": float(result[23]) if _is_float_like(result[23]) else math.nan,
        "stationarity_violation": float(result[24]) if _is_float_like(result[24]) else math.nan,
        "complementarity_violation": float(result[25]) if _is_float_like(result[25]) else math.nan,
        "strong_duality_violation": float(result[26]) if _is_float_like(result[26]) else math.nan,
        "mean_return_test": float(result[27]) if _is_float_like(result[27]) else math.nan,
        "std_return_test": float(result[28]) if _is_float_like(result[28]) else math.nan,
        "sharpe_test": float(result[29]) if _is_float_like(result[29]) else math.nan,
        "mean_return_train": float(result[30]) if _is_float_like(result[30]) else math.nan,
        "std_return_train": float(result[31]) if _is_float_like(result[31]) else math.nan,
        "sharpe_train": float(result[32]) if _is_float_like(result[32]) else math.nan,
        "gurobi_gap": float(result[33]) if _is_float_like(result[33]) else math.nan,
        "theta": theta,
        "avg_weight_test": avg_weight_test,
        "avg_weight_train": avg_weight_train,
    }


def _is_float_like(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def run_cross_fit(
    params: ExperimentParams,
    solver_spec: SolverSpec,
    base_theta: np.ndarray,
    flex_options: Optional[Dict[str, object]],
    seeds: Sequence[int],
    theta_init_modes: Sequence[str],
    tee: bool,
    lambda_theta: float,
    use_true_cov: bool,
    allow_gurobi_partial: bool,
    gurobi_max_gap: Optional[float],
    model_key: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for init_mode in theta_init_modes:
            try:
                result = run_once(
                    model_key=model_key,
                    solver_spec=solver_spec,
                    seed=int(seed),
                    N=params.N,
                    d=params.d,
                    snr=params.snr,
                    rho=params.rho,
                    sigma=params.sigma,
                    res=params.res,
                    delta=params.delta,
                    tee=tee,
                    reg_theta_l2=lambda_theta,
                    use_true_cov=use_true_cov,
                    flex_options=flex_options,
                    theta_fixed=None,
                    allow_gurobi_partial=allow_gurobi_partial,
                    gurobi_max_gap=gurobi_max_gap,
                    flex_theta_init_mode=init_mode,
                )
                unpacked = unpack_run_once(result)
                theta_hat = np.asarray(unpacked.pop("theta"))
                avg_weight_test = np.asarray(unpacked.pop("avg_weight_test"))
                rows.append(
                    {
                        **unpacked,
                        "seed": int(seed),
                        "theta_init_mode": init_mode,
                        "theta_norm": float(np.linalg.norm(theta_hat)),
                        "theta_diff_norm": float(np.linalg.norm(theta_hat - base_theta)),
                        "avg_weight_test": vector_to_string(avg_weight_test),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # pragma: no cover - solver may fail sporadically
                rows.append(
                    {
                        "seed": int(seed),
                        "theta_init_mode": init_mode,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def summarise_cross_fit(
    df: pd.DataFrame,
    base_cost_true: float,
    good_cost_ratio: float,
) -> Dict[str, float]:
    ok_df = df[df["status"] == "ok"].copy()
    summary: Dict[str, float] = {
        "n_total": float(len(df)),
        "n_success": float(len(ok_df)),
    }
    if ok_df.empty:
        return summary
    summary.update(
        {
            "mean_cost_true_mean": float(ok_df["mean_cost_true"].mean()),
            "mean_cost_true_std": float(ok_df["mean_cost_true"].std(ddof=1)),
            "decision_error_test_mean": float(ok_df["decision_error_test"].mean()),
            "runs_per_mode": float(ok_df.groupby("theta_init_mode").size().mean()),
        }
    )
    threshold = base_cost_true * (1.0 + good_cost_ratio)
    good_ratio = float(
        (ok_df["mean_cost_true"] <= threshold).sum() / len(ok_df)
    )
    summary["good_solution_ratio"] = good_ratio
    return summary


def plot_crossfit_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib が見つからなかったため、交差適合図をスキップしました。")
        return
    ok_df = df[df["status"] == "ok"]
    if ok_df.empty:
        print("[WARN] 正常終了した run がないため、交差適合図を作成しません。")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [
        ok_df[ok_df["theta_init_mode"] == mode]["mean_cost_true"].values
        for mode in sorted(ok_df["theta_init_mode"].unique())
    ]
    labels = sorted(ok_df["theta_init_mode"].unique())
    ax.boxplot(data, labels=labels, vert=True, patch_artist=True)
    ax.set_ylabel("Mean MVO Cost (Vtrue)")
    ax.set_xlabel("theta_init_mode")
    ax.set_title("Cross-fit Cost Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Theta感度解析と交差適合検証をまとめて実行するユーティリティ"
    )
    parser.add_argument("--model", type=str, default="flex", help="使用するモデルキー")
    parser.add_argument(
        "--solver",
        type=str,
        default="ipopt",
        choices=["ipopt", "knitro", "gurobi", "analytic"],
        help="使用するソルバー",
    )
    parser.add_argument("--tee", action="store_true", help="ソルバーログを表示する")
    parser.add_argument("--N", type=int, default=50, help="サンプル数")
    parser.add_argument("--d", type=int, default=3, help="資産次元")
    parser.add_argument("--snr", type=float, default=0.01, help="信号対雑音比")
    parser.add_argument("--rho", type=float, default=0.5, help="リターン相関")
    parser.add_argument("--sigma", type=float, default=0.0125, help="周辺標準偏差")
    parser.add_argument("--res", type=int, default=0, help="ローリング窓 (burn-in を兼ねる)")
    parser.add_argument("--delta", type=float, default=1.0, help="MVO のリスク回避パラメータ")
    parser.add_argument("--lambda-theta", dest="lambda_theta", type=float, default=0.0)
    parser.add_argument(
        "--base-seed", type=int, default=200, help="ベースライン学習と感度解析で用いる seed"
    )
    parser.add_argument(
        "--base-theta-init-mode",
        type=str,
        default="ols",
        help="ベース run の flex theta_init_mode",
    )
    parser.add_argument(
        "--sensitivity-epsilons",
        type=str,
        default="0.01",
        help="θ感度で用いる ε のリスト（カンマ区切り）",
    )
    parser.add_argument(
        "--sensitivity-max-directions",
        type=int,
        default=None,
        help="感度解析で評価する軸方向数の上限（未指定なら全成分）",
    )
    parser.add_argument(
        "--sensitivity-random-directions",
        type=int,
        default=0,
        help="追加で評価するランダム方向の本数",
    )
    parser.add_argument(
        "--sensitivity-random-seed",
        type=int,
        default=0,
        help="ランダム方向生成に用いる乱数シード",
    )
    parser.add_argument(
        "--theta-init-modes",
        type=str,
        default="ols,ipo,none",
        help="交差適合で試す flex theta_init_mode（カンマ区切り）",
    )
    parser.add_argument(
        "--crossfit-seeds",
        type=str,
        default="200,201,202,203,204",
        help="交差適合で使用する seed 群（カンマ区切り）",
    )
    parser.add_argument(
        "--good-cost-ratio",
        type=float,
        default=0.05,
        help="良好解とみなす許容比率（mean_cost_true <= base*(1+ratio)）",
    )
    parser.add_argument(
        "--use-true-cov",
        dest="use_true_cov",
        action="store_true",
        help="V_true をそのまま使用して評価する（デフォルト）",
    )
    parser.add_argument(
        "--use-estimated-cov",
        dest="use_true_cov",
        action="store_false",
        help="推定共分散 V̂ を使用する",
    )
    parser.set_defaults(use_true_cov=True)
    parser.add_argument(
        "--allow-gurobi-partial",
        action="store_true",
        help="Gurobi がタイムリミット停止しても結果を受け入れる",
    )
    parser.add_argument(
        "--gurobi-max-gap",
        type=float,
        default=None,
        help="Gurobi の MIP ギャップ許容値（指定時のみ有効）",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="theta_sensitivity_crossfit",
        help="results 配下に作成するサブフォルダ名",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="出力先のルート（未指定時は /Users/kensei/VScode/GraduationResearch/results）",
    )
    # flex 用オプション（単一値想定）
    parser.add_argument("--flex-formulation", type=str, default="dual")
    parser.add_argument("--flex-lambda-theta-anchor", type=str, default="0.0")
    parser.add_argument("--flex-lambda-theta-anchor-l1", type=float, default=0.0)
    parser.add_argument("--flex-lambda-theta-iso", type=str, default="0.0")
    parser.add_argument("--flex-lambda-w-anchor", type=str, default="0.0")
    parser.add_argument("--flex-lambda-w-anchor-l1", type=float, default=0.0)
    parser.add_argument("--flex-lambda-w-iso", type=str, default="0.0")
    parser.add_argument("--flex-theta-anchor-mode", type=str, default="none")
    parser.add_argument("--flex-w-anchor-mode", type=str, default="ols")
    return parser


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    params = ExperimentParams(
        N=int(args.N),
        d=int(args.d),
        snr=float(args.snr),
        rho=float(args.rho),
        sigma=float(args.sigma),
        res=int(args.res),
        delta=float(args.delta),
    )

    epsilon_list = [float(val) for val in comma_split(args.sensitivity_epsilons) or ["0.01"]]
    theta_init_modes = str_list_from_text(args.theta_init_modes, ["ols", "ipo", "none"])
    crossfit_seeds = int_list_from_text(args.crossfit_seeds, [args.base_seed])

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path(__file__).resolve().parents[2] / "results"
    )
    tag_dir = ensure_dir(output_root / args.tag)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = ensure_dir(tag_dir / timestamp)
    figures_dir = ensure_dir(exp_dir / "figures")

    solver_spec = SolverSpec(name=args.solver, options={}, tee=args.tee)
    lambda_theta = float(args.lambda_theta)

    flex_options = None
    if args.model == "flex":
        flex_options = {
            "formulation": (args.flex_formulation or "dual").lower(),
            "lambda_theta_anchor": float_from_text(args.flex_lambda_theta_anchor, 0.0),
            "lambda_theta_anchor_l1": float(args.flex_lambda_theta_anchor_l1),
            "lambda_theta_iso": float_from_text(args.flex_lambda_theta_iso, 0.0),
            "lambda_w_anchor": float_from_text(args.flex_lambda_w_anchor, 0.0),
            "lambda_w_anchor_l1": float(args.flex_lambda_w_anchor_l1),
            "lambda_w_iso": float_from_text(args.flex_lambda_w_iso, 0.0),
            "theta_anchor_mode": (args.flex_theta_anchor_mode or "none").lower(),
            "w_anchor_mode": (args.flex_w_anchor_mode or "ols").lower(),
        }

    print("[INFO] === Baseline training (theta*) ===")
    baseline_result = run_once(
        model_key=args.model,
        solver_spec=solver_spec,
        seed=int(args.base_seed),
        N=params.N,
        d=params.d,
        snr=params.snr,
        rho=params.rho,
        sigma=params.sigma,
        res=params.res,
        delta=params.delta,
        tee=args.tee,
        reg_theta_l2=lambda_theta,
        use_true_cov=args.use_true_cov,
        flex_options=flex_options,
        theta_fixed=None,
        allow_gurobi_partial=args.allow_gurobi_partial,
        gurobi_max_gap=args.gurobi_max_gap,
        flex_theta_init_mode=args.base_theta_init_mode,
    )
    baseline_unpacked = unpack_run_once(baseline_result)
    theta_star = np.asarray(baseline_unpacked.pop("theta"))
    np.save(exp_dir / "theta_star.npy", theta_star)

    dataset = prepare_dataset(
        params,
        seed=int(args.base_seed),
        use_true_cov=bool(args.use_true_cov),
    )
    sensitivity_df, sensitivity_summary = run_theta_sensitivity(
        theta_star,
        dataset,
        params,
        epsilon_list,
        max_directions=args.sensitivity_max_directions,
        random_directions=int(args.sensitivity_random_directions),
        random_seed=int(args.sensitivity_random_seed) if args.sensitivity_random_seed is not None else None,
    )
    sensitivity_csv = exp_dir / "theta_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_csv, index=False)
    print(f"[INFO] 感度解析結果を {sensitivity_csv} に保存しました。")
    plot_sensitivity(
        sensitivity_df,
        figures_dir / "theta_sensitivity_scatter.png",
    )

    crossfit_df = run_cross_fit(
        params=params,
        solver_spec=solver_spec,
        base_theta=theta_star,
        flex_options=flex_options,
        seeds=crossfit_seeds,
        theta_init_modes=theta_init_modes,
        tee=args.tee,
        lambda_theta=lambda_theta,
        use_true_cov=args.use_true_cov,
        allow_gurobi_partial=args.allow_gurobi_partial,
        gurobi_max_gap=args.gurobi_max_gap,
        model_key=args.model,
    )
    crossfit_csv = exp_dir / "crossfit_results.csv"
    crossfit_df.to_csv(crossfit_csv, index=False)
    print(f"[INFO] 交差適合結果を {crossfit_csv} に保存しました。")
    plot_crossfit_distribution(
        crossfit_df,
        figures_dir / "crossfit_cost_distribution.png",
    )

    crossfit_summary = summarise_cross_fit(
        crossfit_df,
        base_cost_true=sensitivity_summary["base_mean_cost_true"],
        good_cost_ratio=float(args.good_cost_ratio),
    )

    meta = {
        "timestamp": timestamp,
        "output_dir": str(exp_dir),
        "params": params.__dict__,
        "baseline_seed": int(args.base_seed),
        "epsilon_list": epsilon_list,
        "theta_dim": int(theta_star.size),
        "baseline_metrics": sensitivity_summary,
        "crossfit_summary": crossfit_summary,
        "crossfit_seeds": crossfit_seeds,
        "theta_init_modes": theta_init_modes,
        "solver": args.solver,
        "model": args.model,
        "lambda_theta": lambda_theta,
        "use_true_cov": bool(args.use_true_cov),
        "sensitivity_max_directions": None if args.sensitivity_max_directions is None else int(args.sensitivity_max_directions),
        "sensitivity_random_directions": int(args.sensitivity_random_directions),
        "sensitivity_random_seed": int(args.sensitivity_random_seed)
        if args.sensitivity_random_seed is not None
        else None,
        "allow_gurobi_partial": bool(args.allow_gurobi_partial),
        "gurobi_max_gap": float(args.gurobi_max_gap) if args.gurobi_max_gap is not None else None,
        "flex_options": flex_options,
        "sensitivity_rows": int(len(sensitivity_df)),
        "crossfit_rows": int(len(crossfit_df)),
    }
    with (exp_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] サマリを {exp_dir / 'summary.json'} に保存しました。")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
'''
python /Users/kensei/VScode/GraduationResearch/DFL_Portfolio_Optimization2/experiments/run_theta_sensitivity_crossfit.py \
  --model flex --solver ipopt --tee \
  --N 50 --d 3 --snr 0.01 --rho 0.5 --sigma 0.0125 --res 0 --delta 1.0 \
  --base-seed 200 --sensitivity-epsilons 0.005,0.01 --sensitivity-max-directions 40 \
  --theta-init-modes none --crossfit-seeds 200,201,202,203,204 \
  --flex-formulation dual --flex-lambda-theta-anchor 0.0 \
  --flex-theta-anchor-mode ols --flex-w-anchor-mode ols

'''
