# -*- coding: utf-8 -*-
"""
各実験の *_summary.csv を集約し、手法×ソルバーごとに
SNR を行に持つ1枚の表（CSV と PNG）を出力するスクリプト。

使い方:
  python DFL_Portfolio_Optimization2/viz/make_summary_tables.py \
    --root /Users/kensei/VScode/GraduationResearch/results/exp_unified_yaml/20250910

出力:
  /.../20250910/table_kkt_gurobi.csv
  /.../20250910/table_kkt_gurobi.png
  ...（model×solver ごと）
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib


# ---------- ヘルパ ----------

FNAME_RE = re.compile(
    r".*_(?P<ts>\d{8}-\d{6})_summary$"
)

def parse_key_from_name(stem: str) -> Dict[str, str]:
    """
    ファイル名の stem から model / solver / snr などを推定する。
    例:
      kkt_gurobi_N300_d4_res100_snr0.01_rho0.5_sigma0.0125_delta1.0_20250910-171519_summary
    """
    parts = stem.split("_")
    out: Dict[str, str | float] = {}
    # model / solver は先頭2要素想定（run.py の tag と同じ命名）
    # 例: kkt_gurobi_... / dual_ipopt_... / ols_gurobi_...
    if len(parts) >= 2:
        out["model"]  = parts[0]
        out["solver"] = parts[1]
    # パラメータを拾う
    for p in parts:
        if p.startswith("snr"):
            try:
                out["snr"] = float(p.replace("snr", ""))
            except Exception:
                pass
        elif p.startswith("N"):
            out["N"] = p.replace("N", "")
        elif p.startswith("d"):
            out["d"] = p.replace("d", "")
        elif p.startswith("res"):
            out["res"] = p.replace("res", "")
        elif p.startswith("rho"):
            out["rho"] = p.replace("rho", "")
        elif p.startswith("sigma"):
            out["sigma"] = p.replace("sigma", "")
        elif p.startswith("delta"):
            out["delta"] = p.replace("delta", "")
    # タイムスタンプ（末尾）
    m = FNAME_RE.match(stem)
    if m:
        out["ts"] = m.group("ts")
    return out  # type: ignore[return-value]


JP_HEADERS = [
    "SNR",
    "テストMVOコスト (Vtrue)",
    "テストMVOコスト (Vhat)",
    "訓練MVOコスト (Vtrue)",
    "訓練MVOコスト (Vhat)",
    "決定係数 R^2",
    "計算時間 (秒)",
]

# run.py の summary_csv で採番しているキー名に対応
SUMMARY_TO_TABLE = {
    "SNR": "snr",
    "テストMVOコスト (Vtrue)": "mean_cost_test_vtrue",
    "テストMVOコスト (Vhat)": "mean_cost_test_vhat",
    "訓練MVOコスト (Vtrue)": "train_cost_vtrue",
    "訓練MVOコスト (Vhat)": "train_cost_vhat",
    "決定係数 R^2": "mean_r2",
    "計算時間 (秒)": "avg_time_per_run_sec",
}


def load_summaries(root: Path) -> pd.DataFrame:
    """root 直下の *_summary.csv を全て読み込んで縦結合。"""
    files = sorted(root.glob("*_summary.csv"))
    if not files:
        raise FileNotFoundError(f"No *_summary.csv in {root}")
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
        except Exception as e:
            print(f"[WARN] failed to read {f}: {e}")
            continue

        meta = parse_key_from_name(f.stem)
        row.update(meta)
        row["file"] = str(f.name)
        rows.append(row)
    if not rows:
        raise RuntimeError("No readable summary rows.")
    return pd.DataFrame(rows)


def make_table_per_group(df: pd.DataFrame, model: str, solver: str) -> pd.DataFrame:
    """特定の model×solver について、SNRごとの1枚表を返す。"""
    sub = df[(df["model"] == model) & (df["solver"] == solver)].copy()
    if sub.empty:
        return pd.DataFrame(columns=JP_HEADERS)

    # 欲しい列をマッピング
    out = pd.DataFrame({
        "SNR": sub.get("snr", np.nan),
        "テストMVOコスト (Vtrue)": sub.get("mean_cost_test_vtrue", np.nan),
        "テストMVOコスト (Vhat)":  sub.get("mean_cost_test_vhat", np.nan),
        "訓練MVOコスト (Vtrue)":   sub.get("train_cost_vtrue", np.nan),
        "訓練MVOコスト (Vhat)":    sub.get("train_cost_vhat", np.nan),
        "決定係数 R^2":            sub.get("mean_r2", np.nan),
        "計算時間 (秒)":           sub.get("avg_time_per_run_sec", np.nan),
    })

    # SNR 昇順（好みに応じて並び替え）
    out = out.sort_values(by="SNR", kind="mergesort").reset_index(drop=True)

    # 小数表示を整える（見やすさ）
    def fmt(col: str, n=6):
        if col in out:
            out[col] = out[col].astype(float).map(lambda v: f"{v:.6f}")

    fmt("テストMVOコスト (Vtrue)")
    fmt("テストMVOコスト (Vhat)")
    fmt("訓練MVOコスト (Vtrue)")
    fmt("訓練MVOコスト (Vhat)")
    fmt("決定係数 R^2", n=6)
    if "計算時間 (秒)" in out:
        out["計算時間 (秒)"] = out["計算時間 (秒)"].astype(float).map(lambda v: f"{v:.3f}")
    # SNR は整数 or 小数のまま
    return out


def save_table_png(df_table: pd.DataFrame, title: str, out_png: Path) -> None:
    """DataFrame を PNG の“表”画像として保存。"""
    # フォントサイズやレイアウトを広めに
    n_rows, n_cols = df_table.shape
    fig_w = max(8, n_cols * 1.8)
    fig_h = max(2.5, 1.0 + 0.45 * (n_rows + 1))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    the_table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.0, 1.3)

    ax.set_title(title, y=1.05, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- メイン ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="実験結果フォルダ（例: results/exp_unified_yaml/20250910）")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    df = load_summaries(root)

    # どの model×solver があるかを列挙
    groups: List[Tuple[str, str]] = sorted(
        {(str(m), str(s)) for m, s in zip(df["model"], df["solver"])}
    )

    if not groups:
        print("No (model, solver) groups detected.")
        return

    for model, solver in groups:
        table = make_table_per_group(df, model, solver)
        if table.empty:
            continue

        base = f"table_{model}_{solver}"
        csv_path = root / f"{base}.csv"
        png_path = root / f"{base}.png"

        # CSV 保存
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # PNG 保存（タイトルは “MODEL  SOLVER” を太字っぽく）
        title = f"{model.upper()}   {solver.capitalize()}"
        save_table_png(table, title, png_path)

        print(f"[Saved] {csv_path.name} / {png_path.name}")


if __name__ == "__main__":
    main()