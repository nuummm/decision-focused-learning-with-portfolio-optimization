from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# from data.synthetic import generate_simulation1_dataset
from data_stress.synthetic import generate_simulation1_dataset
from experiments.registry import SolverSpec, get_trainer
from models.covariance import estimate_epscov_rolling
from models.ols import train_ols, predict_yhat


@dataclass
class SelMetrics:
    seed: int
    model: str
    mean_ret: float
    std_ret: float
    sharpe: float
    hit_rate: float
    k_select: int
    T_eval: int

    def to_row(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "model": self.model,
            "mean_ret": self.mean_ret,
            "std_ret": self.std_ret,
            "sharpe": self.sharpe,
            "hit_rate": self.hit_rate,
            "k_select": int(self.k_select),
            "T_eval": int(self.T_eval),
        }


def summarize(xs: List[float]) -> float:
    arr = np.array(xs, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def split_indices(idx: Sequence[int], burn_in: int, n_tr: int, n_val: int) -> Tuple[List[int], List[int], List[int]]:
    test_start = burn_in + n_tr + n_val
    train_idx = [i for i in idx if burn_in <= i < burn_in + n_tr]
    val_idx = [i for i in idx if burn_in + n_tr <= i < test_start]
    test_idx = [i for i in idx if i >= test_start]
    return train_idx, val_idx, test_idx


def equal_weight_topk(scores: np.ndarray, k: int) -> np.ndarray:
    # scores: shape (d,)
    d = scores.shape[0]
    w = np.zeros(d, dtype=float)
    if k <= 0:
        return w
    k = min(k, d)
    topk_idx = np.argsort(-scores)[:k]
    w[topk_idx] = 1.0 / k
    return w


def run_one(model: str,
            solver: str,
            tee: bool,
            seed: int,
            N: int,
            d: int,
            snr: float,
            rho: float,
            sigma: float,
            res: int,
            k_select: int,
            train_frac: float,
            val_frac: float) -> SelMetrics:

    X, Y, V_true, theta0, tau = generate_simulation1_dataset(
        n_samples=N, n_assets=d, snr=snr, rho=rho, sigma=sigma, seed=seed
    )

    burn_in = int(res)
    n_eff = N - burn_in
    if n_eff <= 3:
        raise ValueError("Not enough samples after burn-in")
    n_tr = max(1, int(n_eff * train_frac))
    n_val = max(1, int(n_eff * val_frac))

    Vhats, idx = estimate_epscov_rolling(Y, X, theta0, tau, res=res, include_current=False)
    if not Vhats:
        raise RuntimeError("empty Vhats")
    train_idx, val_idx, test_idx = split_indices(idx, burn_in, n_tr, n_val)
    if not test_idx:
        raise RuntimeError("empty test indices")

    X_train, Y_train = X[train_idx], Y[train_idx]
    theta_init = train_ols(X_train, Y_train)

    if model == "ols":
        theta_hat = theta_init
    else:
        trainer = get_trainer(model, SolverSpec(name=solver, options={}, tee=tee))
        ret = trainer(
            X, Y, Vhats, idx,
            start_index=train_idx[0], end_index=train_idx[-1],
            delta=1.0, theta_init=theta_init, tee=tee,
            reg_theta_l2=0.0,
        )
        theta_hat = ret[0]

    Yhat = predict_yhat(X, theta_hat)

    rets: List[float] = []
    hits: List[float] = []
    for t in test_idx:
        scores = Yhat[t]
        w = equal_weight_topk(scores, k_select)
        r = float(w @ Y[t])  # 実現リターン
        rets.append(r)
        # ヒット率: 予測上位の平均 vs 全資産平均の超過符号
        hits.append(1.0 if r > float(Y[t].mean()) else 0.0)

    rets_arr = np.array(rets, dtype=float)
    mean_ret = float(np.mean(rets_arr))
    std_ret = float(np.std(rets_arr, ddof=1)) if len(rets_arr) > 1 else 0.0
    sharpe = float(mean_ret / std_ret) if std_ret > 0 else float("nan")
    hit_rate = float(np.mean(hits)) if hits else float("nan")

    return SelMetrics(
        seed=seed, model=model, mean_ret=mean_ret, std_ret=std_ret, sharpe=sharpe,
        hit_rate=hit_rate, k_select=k_select, T_eval=len(test_idx)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-K equal-weight selection evaluation (directional test)")
    ap.add_argument("--models", type=str, default="dual,ols")
    ap.add_argument("--solver", type=str, default="ipopt")
    ap.add_argument("--tee", action="store_true")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--d", type=int, default=50, help="number of candidates")
    ap.add_argument("--snr", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.0125)
    ap.add_argument("--res", type=int, default=100)
    ap.add_argument("--k-select", type=int, default=10)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    models = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    rows: List[SelMetrics] = []

    for model in models:
        print(f"\n=== Model: {model} (Top-{args.k_select}, equal-weight) ===")
        for k in range(args.runs):
            seed = args.seed0 + k
            m = run_one(
                model=model, solver=args.solver, tee=args.tee,
                seed=seed, N=args.N, d=args.d, snr=args.snr, rho=args.rho, sigma=args.sigma,
                res=args.res, k_select=args.k_select, train_frac=args.train_frac, val_frac=args.val_frac,
            )
            rows.append(m)
            print(f" seed={seed}  mean={m.mean_ret:.5f}  std={m.std_ret:.5f}  sharpe={m.sharpe:.3f}  hit={m.hit_rate:.3f}  T={m.T_eval}")

    # 集計
    def sel(vals: List[SelMetrics], attr: str) -> float:
        return summarize([getattr(v, attr) for v in vals])

    print("\n=== Summary ===")
    for model in models:
        ms = [r for r in rows if r.model == model]
        if not ms: continue
        print(f" {model}: mean={sel(ms,'mean_ret'):.5f}, std={sel(ms,'std_ret'):.5f}, sharpe={sel(ms,'sharpe'):.3f}, hit={sel(ms,'hit_rate'):.3f}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w') as f:
            f.write("seed,model,mean_ret,std_ret,sharpe,hit_rate,k_select,T_eval\n")
            for r in rows:
                vals = r.to_row()
                f.write(
                    f"{int(vals['seed'])},{vals['model']},{vals['mean_ret']},{vals['std_ret']},{vals['sharpe']},{vals['hit_rate']},{vals['k_select']},{vals['T_eval']}\n"
                )
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
