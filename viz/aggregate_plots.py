# -*- coding: utf-8 -*-
"""
SNR×手法の箱ひげ図を自動生成する可視化ツール
- 入力: 指定ディレクトリ配下の *_runs.csv （run.py が吐く per-run ファイル）
- 出力: 3枚のPNG
    boxplot_train_obj_vhat.png    （学習目的: VhatでのMVOコスト）
    boxplot_test_cost_vtrue.png   （テスト: VtrueでのMVOコスト）
    boxplot_r2.png                （R^2）
使い方:

  python viz/aggregate_plots.py --root /Users/kensei/VScode/GraduationResearch/results/exp_unified_yaml/20250910
"""

from __future__ import annotations
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib


# ========= 設定 =========

# 手法ラベルの統一（ファイル名の先頭 "model_solver" を人が読める表記に）
METHOD_LABELS = {
    "ols": "OLS",
    "dual_ipopt": "DUAL+IPOPT",
    "dual_knitro": "DUAL+KNITRO",
    "kkt_knitro": "KKT+KNITRO",
    "kkt_gurobi": "KKT+GUROBI",
    "kkt_ipopt": "KKT+IPOPT",  # もし将来あれば
}

# 図での表示順
METHOD_ORDER = ["OLS", "DUAL+IPOPT", "DUAL+KNITRO", "KKT+KNITRO", "KKT+GUROBI"]

# 収集する列名（run.py が出す per-run CSV 準拠）
COL_TEST_VTRUE = "mean_cost_test_vtrue"
COL_TRAIN_VHAT = "train_cost_vhat"
COL_R2         = "mean_r2"

# ========= ユーティリティ =========

def parse_method_and_snr_from_name(csv_path: Path) -> Tuple[str, float]:
    """
    例: dual_ipopt_N300_d4_res100_snr0.01_rho0.5_sigma0.0125_delta1.0_20250910-174646_runs.csv
        -> ("dual_ipopt", 0.01)
    """
    name = csv_path.stem  # 拡張子除き
    # method part は "_N" の前
    if "_N" in name:
        method_key = name.split("_N", 1)[0]
    else:
        # 念のため先頭2トークンを model + solver とみなす
        toks = name.split("_")
        method_key = "_".join(toks[:2]) if len(toks) >= 2 else toks[0]

    # snr の抽出
    m = re.search(r"_snr([0-9.]+)", name)
    if not m:
        raise ValueError(f"SNR がファイル名から読めません: {name}")
    snr = float(m.group(1))

    return method_key, snr


def method_key_to_label(method_key: str) -> str:
    # "ols" / "dual_ipopt" / "dual_knitro" / "kkt_knitro" / "kkt_gurobi" など
    return METHOD_LABELS.get(method_key, method_key.upper())


def collect_runs(root: Path):
    """
    ディレクトリ配下の *_runs.csv を探索し、
    dict[(snr, method_label)] -> {'train_vhat': [...], 'test_vtrue': [...], 'r2': [...]}
    を返す
    """
    buckets: DefaultDict[Tuple[float, str], Dict[str, List[float]]] = defaultdict(lambda: {
        "train_vhat": [],
        "test_vtrue": [],
        "r2": [],
    })

    for csv_path in root.rglob("*_runs.csv"):
        try:
            method_key, snr = parse_method_and_snr_from_name(csv_path)
            label = method_key_to_label(method_key)
        except Exception as e:
            print(f"[WARN] スキップ: {csv_path.name}: {e}")
            continue

        df = pd.read_csv(csv_path)
        if COL_TRAIN_VHAT not in df.columns and "train_cost_vhat" in df.columns:
            # 念のため別名にも対応（古い書式）
            df.rename(columns={"train_cost_vhat": COL_TRAIN_VHAT}, inplace=True)

        # 必要列の存在チェック
        missing = [c for c in [COL_TEST_VTRUE, COL_TRAIN_VHAT, COL_R2] if c not in df.columns]
        if missing:
            print(f"[WARN] 欠損列 {missing} -> スキップ: {csv_path.name}")
            continue

        key = (snr, label)
        buckets[key]["train_vhat"].extend(df[COL_TRAIN_VHAT].astype(float).tolist())
        buckets[key]["test_vtrue"].extend(df[COL_TEST_VTRUE].astype(float).tolist())
        buckets[key]["r2"].extend(df[COL_R2].astype(float).tolist())

    return buckets


def grouped_boxplot(
    buckets: Dict[Tuple[float, str], Dict[str, List[float]]],
    metric_key: str,
    save_path: Path,
    title: str,
    ylabel: str,
):
    """
    SNRごとのグループ×手法の箱ひげ図を描画して保存
    """
    if not buckets:
        print("[INFO] データが見つかりませんでした。")
        return

    # SNR を昇順に、手法は既定順に並べる
    snrs = sorted({k[0] for k in buckets.keys()})
    methods_present = sorted({k[1] for k in buckets.keys()}, key=lambda x: (METHOD_ORDER.index(x) if x in METHOD_ORDER else 999, x))

    # 図の下地
    plt.figure(figsize=(max(6, 2 + 1.5*len(snrs)), 5))

    # 箱ひげの配置計算
    group_width = 0.8
    n_methods = len(methods_present)
    box_width = group_width / max(1, n_methods)
    centers = np.arange(len(snrs))

    legend_handles = []
    for m_idx, method in enumerate(methods_present):
        # それぞれのSNRでデータ収集
        data_per_snr = []
        for snr in snrs:
            arr = buckets.get((snr, method), {}).get(metric_key, [])
            data_per_snr.append(np.array(arr, dtype=float))

        # x位置（SNRグループの中でずらす）
        positions = centers - group_width/2 + box_width*(m_idx + 0.5)

        # 箱ひげ
        bp = plt.boxplot(
            data_per_snr,
            positions=positions,
            widths=box_width*0.9,
            patch_artist=True,
            showfliers=False,
        )
        # 見やすい色（matplotlib既定の色サイクル）
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][m_idx % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        legend_handles.append((bp["boxes"][0], method))

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(centers, [str(s) for s in snrs])
    plt.xlabel("SNR")
    # 凡例
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=h[0].get_facecolor(), edgecolor='black', alpha=0.6, label=h[1]) for h in legend_handles]
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
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    buckets = collect_runs(root)

    # 1) 学習目的（VhatでのMVOコスト）
    grouped_boxplot(
        buckets,
        metric_key="train_vhat",
        save_path=root / "boxplot_train_obj_vhat.png",
        title="訓練時の目的関数値 (MVO cost with V̂)",
        ylabel="訓練時の目的関数値 (V̂)",
    )

    # 2) テスト（Vtrue評価のMVOコスト）
    grouped_boxplot(
        buckets,
        metric_key="test_vtrue",
        save_path=root / "boxplot_test_cost_vtrue.png",
        title="評価指標 MVOコスト関数 (evaluated with V_true)",
        ylabel="評価指標 MVOコスト関数 (V_true)",
    )

    # 3) R^2
    grouped_boxplot(
        buckets,
        metric_key="r2",
        save_path=root / "boxplot_r2.png",
        title="R^2 by SNR and method",
        ylabel="R^2",
    )


if __name__ == "__main__":
    main()