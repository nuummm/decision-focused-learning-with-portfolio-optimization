# -*- coding: utf-8 -*-
"""
特定パラメータ群 (N,d,res,rho,sigma,delta) かつ特定SNRのみを集計して、
SNR×手法（指定セット）の箱ひげ図を保存するツール。

出力:
  boxplot_train_obj_vhat.png    （学習目的: VhatでのMVOコスト）
  boxplot_test_cost_vtrue.png   （テスト: VtrueでのMVOコスト）
  boxplot_r2.png                （R^2）

例:
cd ~/VScode/GraduationResearch
python DFL_Portfolio_Optimization2/viz/aggregate_plots_filtered.py \
    --root /Users/kensei/VScode/GraduationResearch/results/exp_unified_yaml/20250910 \
    --N 1000 --d 5 --res 100 --rho 0.5 --sigma 0.0125 --delta 1.0 \
    --snrs 0.1 0.05 0.025 0.01 0.001
"""
from __future__ import annotations
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple, DefaultDict, Iterable
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- ここを必要に応じてコメントアウト可（日本語フォント） ----
try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    pass


# ====== 手法の選択（この順で表示） ======
# ファイル名の先頭（model_solver）を “メソッド表示名” に正規化
# OLS は solver 名を無視してすべて "OLS" に寄せる
WANTED_METHODS = [
    "kkt_knitro",
    "dual_knitro",
    "dual_ipopt",
    "ols",           # ← OLS は 'ols_*' を全部まとめて "OLS"
]

DISPLAY_LABELS = {
    "kkt_knitro":  "KKT+KNITRO",
    "dual_knitro": "DUAL+KNITRO",
    "dual_ipopt":  "DUAL+IPOPT",
    "ols":         "OLS",
}

# run.csv 側の列名（run.py の出力準拠）
COL_TEST_VTRUE = "mean_cost_test_vtrue"
COL_TRAIN_VHAT = "train_cost_vhat"
COL_R2_TEST    = "mean_r2_test"   # ★変更
COL_R2_TRAIN   = "mean_r2_train"  # ★追加


# ========= ユーティリティ =========
def _float_in_name(name: str, key: str) -> float | None:
    """
    ファイル名から _{key}{number} を抜く（例: _N1000, _d5, _res100, _snr0.01 ...）
    見つからなければ None
    """
    m = re.search(rf"_{re.escape(key)}(-?\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def parse_fields_from_filename(csv_path: Path):
    """
    ファイル名の例:
      dual_ipopt_N1000_d5_res100_snr0.1_rho0.5_sigma0.0125_delta1.0_20250910-192206_runs.csv
    から、(method_key, params_dict) を返す
    """
    name = csv_path.stem
    # 先頭の method_key は "_N" の前
    method_key = name.split("_N", 1)[0] if "_N" in name else name.split("_runs", 1)[0]

    snr   = _float_in_name(name, "snr")
    N     = _float_in_name(name, "N")
    d     = _float_in_name(name, "d")
    res   = _float_in_name(name, "res")
    rho   = _float_in_name(name, "rho")
    sigma = _float_in_name(name, "sigma")
    delta = _float_in_name(name, "delta")

    params = dict(snr=snr, N=N, d=d, res=res, rho=rho, sigma=sigma, delta=delta)
    return method_key, params


def normalize_method(method_key: str) -> str | None:
    """
    使いたい手法のみを表示名に正規化。
    - OLS は 'ols_*' でも 'ols' でも "ols" に統一
    - それ以外は WANTED_METHODS に完全一致したものだけ採用
    """
    k = method_key.lower()
    if k.startswith("ols"):
        return "ols" if "ols" in WANTED_METHODS else None
    return k if k in WANTED_METHODS else None


def collect_runs_filtered(
    root: Path,
    target_snrs: Iterable[float],
    target_params: dict,
):
    """
    条件に合う *_runs.csv を集めて
    dict[(snr, label)] -> {
        'train_vhat': [...],
        'test_vtrue': [...],
        'r2_test': [...],
        'r2_train': [...],
        'r2_gap': [...]
    } を返す
    """
    # r2 系も含めたバケツを用意
    buckets: DefaultDict[Tuple[float, str], Dict[str, List[float]]] = defaultdict(lambda: {
        "train_vhat": [],
        "test_vtrue": [],
        "r2_test": [],
        "r2_train": [],
        "r2_gap": [],
    })

    snr_set = {float(s) for s in target_snrs}

    for csv_path in root.rglob("*_runs.csv"):
        try:
            method_key, params = parse_fields_from_filename(csv_path)
        except Exception as e:
            print(f"[WARN] 名前解析に失敗: {csv_path.name}: {e}")
            continue

        # ---- パラメータ一致チェック ----
        ok = True
        for k, v in target_params.items():
            if v is None:
                continue
            if params.get(k) is None or float(params[k]) != float(v):
                ok = False
                break
        if not ok:
            continue

        # ---- SNR フィルタ ----
        snr = params.get("snr")
        if snr is None or float(snr) not in snr_set:
            continue

        # ---- 手法名を正規化 ----
        norm = normalize_method(method_key)
        if norm is None:
            continue
        label = DISPLAY_LABELS.get(norm, norm.upper())
        key = (float(snr), label)

        # ---- CSV を読み込む ----
        df = pd.read_csv(csv_path)

        # 互換対応
        if COL_TRAIN_VHAT not in df.columns and "train_cost_vhat" in df.columns:
            df.rename(columns={"train_cost_vhat": COL_TRAIN_VHAT}, inplace=True)
        if COL_R2_TEST not in df.columns and "mean_r2" in df.columns:
            df.rename(columns={"mean_r2": COL_R2_TEST}, inplace=True)

        # 必須列のチェック
        missing = [c for c in [COL_TEST_VTRUE, COL_TRAIN_VHAT, COL_R2_TEST] if c not in df.columns]
        if missing:
            print(f"[WARN] 欠損列 {missing} -> スキップ: {csv_path.name}")
            continue

        # ---- 値の取り込み ----
        buckets[key]["train_vhat"].extend(df[COL_TRAIN_VHAT].astype(float).tolist())
        buckets[key]["test_vtrue"].extend(df[COL_TEST_VTRUE].astype(float).tolist())
        buckets[key]["r2_test"].extend(df[COL_R2_TEST].astype(float).tolist())

        # 訓練R²があれば gap も計算
        if COL_R2_TRAIN in df.columns:
            buckets[key]["r2_train"].extend(df[COL_R2_TRAIN].astype(float).tolist())
            a = np.array(df[COL_R2_TRAIN].astype(float).tolist())
            b = np.array(df[COL_R2_TEST].astype(float).tolist())
            L = min(len(a), len(b))
            if L > 0:
                buckets[key]["r2_gap"].extend((a[:L] - b[:L]).tolist())

    return buckets


def grouped_boxplot(
    buckets: Dict[Tuple[float, str], Dict[str, List[float]]],
    metric_key: str,
    save_path: Path,
    title: str,
    ylabel: str,
):
    if not buckets:
        print("[INFO] 該当データが見つかりません。")
        return

    snrs = sorted({k[0] for k in buckets.keys()})
    methods_present = sorted({k[1] for k in buckets.keys()},
                             key=lambda x: list(DISPLAY_LABELS.values()).index(x) if x in DISPLAY_LABELS.values() else 999)

    plt.figure(figsize=(max(6, 2 + 1.2*len(snrs)), 5))

    group_width = 0.8
    n_methods = len(methods_present)
    box_width = group_width / max(1, n_methods)
    centers = np.arange(len(snrs))

    legend_entries = []
    for m_idx, method in enumerate(methods_present):
        data_per_snr = []
        for snr in snrs:
            arr = buckets.get((snr, method), {}).get(metric_key, [])
            data_per_snr.append(np.array(arr, dtype=float))

        positions = centers - group_width/2 + box_width*(m_idx + 0.5)

        bp = plt.boxplot(
            data_per_snr,
            positions=positions,
            widths=box_width*0.9,
            patch_artist=True,
            showfliers=False,
        )
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][m_idx % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        legend_entries.append((bp['boxes'][0], method))

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(centers, [str(s) for s in snrs])
    plt.xlabel("SNR")
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=entry[0].get_facecolor(), edgecolor='black', alpha=0.6, label=entry[1])
               for entry in legend_entries]
    plt.legend(handles=patches, loc="best", frameon=True)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SAVED] {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="*_runs.csv が入っている日付フォルダ（例: results/exp_unified_yaml/20250910）")
    # 既定値は今回の指定に合わせておく
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--res", type=int, default=100)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.0125)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0.1, 0.05, 0.025, 0.01, 0.001],
                    help="可視化対象のSNR群")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    target_params = dict(N=args.N, d=args.d, res=args.res, rho=args.rho, sigma=args.sigma, delta=args.delta)
    buckets = collect_runs_filtered(root, target_snrs=args.snrs, target_params=target_params)

    # main() 最後に3枚の出力を追加
    # 1) 学習目的（VhatでのMVOコスト）
    grouped_boxplot(
        buckets,
        metric_key="train_vhat",
        save_path=root / "boxplot_train_obj_vhat.filtered.png",
        title="訓練時の目的関数値 (MVO cost with V̂)",
        ylabel="訓練時の目的関数値 (V̂)",
    )

    # 2) テスト（Vtrue評価のMVOコスト）
    grouped_boxplot(
        buckets,
        metric_key="test_vtrue",
        save_path=root / "boxplot_test_cost_vtrue.filtered.png",
        title="評価指標 MVOコスト関数 (evaluated with V_true)",
        ylabel="評価指標 MVOコスト関数 (V_true)",
    )

    # 3) 決定係数（テスト／訓練／ギャップ）
    grouped_boxplot(
        buckets,
        metric_key="r2_test",
        save_path=root / "boxplot_r2_test.filtered.png",
        title="決定係数 R² (Test)",
        ylabel="R² (test)",
    )
    grouped_boxplot(
        buckets,
        metric_key="r2_train",
        save_path=root / "boxplot_r2_train.filtered.png",
        title="決定係数 R² (Train)",
        ylabel="R² (train)",
    )
    grouped_boxplot(
        buckets,
        metric_key="r2_gap",
        save_path=root / "boxplot_r2_gap.filtered.png",
        title="R² ギャップ (Train − Test)",
        ylabel="ΔR² (train − test)",
    )

if __name__ == "__main__":
    main()