# -*- coding: utf-8 -*-
"""
固定ハイパラと手法を絞って *_summary.csv を集約し、
手法×ソルバーごとに SNR を行にもつ表（CSV/PNG）を出力するスクリプト（フィルタ版）

使い方（例）:
  python DFL_Portfolio_Optimization2/viz/make_summary_tables_filtered.py \
    --root /Users/kensei/VScode/GraduationResearch/results/exp_unified_yaml/20250910 \
    --N 1000 --d 5 --res 100 --rho 0.5 --sigma 0.0125 --delta 1.0 \
    --snrs 0.1 0.05 0.025 0.01 0.001 \
    --methods kkt+knitro dual+knitro dual+ipopt ols

出力例:
  /.../20250910/table_kkt_knitro.csv
  /.../20250910/table_kkt_knitro.png
  /.../20250910/table_dual_ipopt.csv
  ...
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# （日本語フォントが不要なら次の行は削除可）
try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    pass

# ---------- 既存スクリプト相当のパース ----------

FNAME_RE = re.compile(r".*_(?P<ts>\d{8}-\d{6})_summary$")

def parse_key_from_name(stem: str) -> Dict[str, str]:
    """
    ファイル名の stem から model / solver / snr などを推定する。
    例:
      kkt_knitro_N1000_d5_res100_snr0.01_rho0.5_sigma0.0125_delta1.0_20250910-171519_summary
    """
    parts = stem.split("_")
    out: Dict[str, str] = {}
    if len(parts) >= 2:
        out["model"]  = parts[0]
        out["solver"] = parts[1]

    for p in parts:
        if p.startswith("snr"):
            out["snr"] = p.replace("snr", "")
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
    m = FNAME_RE.match(stem)
    if m:
        out["ts"] = m.group("ts")
    return out

JP_HEADERS = [
    "SNR",
    "テストMVOコスト (Vtrue)",
    "テストMVOコスト (Vhat)",
    "訓練MVOコスト (Vtrue)",
    "訓練MVOコスト (Vhat)",
    "決定係数 R^2 (Test)",   # ★名称変更
    "決定係数 R^2 (Train)",  # ★追加
    "計算時間 (秒)",
]

def make_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "SNR": "snr",
        "テストMVOコスト (Vtrue)": "mean_cost_test_vtrue",
        "テストMVOコスト (Vhat)" : "mean_cost_test_vhat",
        "訓練MVOコスト (Vtrue)"  : "train_cost_vtrue",
        "訓練MVOコスト (Vhat)"   : "train_cost_vhat",
        "決定係数 R^2 (Test)"    : "mean_r2",         # ← summaryでは test 側を mean_r2 に入れている
        "決定係数 R^2 (Train)"   : "mean_r2_train",   # ← run.py で追加済み
        "計算時間 (秒)"          : "avg_time_per_run_sec",
    }
    # あとは既存の整形処理でOK

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

# ---------- 表の生成・保存 ----------

def make_table(df: pd.DataFrame) -> pd.DataFrame:
    """SNRごとの1枚表を返す（日本語ヘッダで）。"""
    cols = {
        "SNR": "snr",
        "テストMVOコスト (Vtrue)": "mean_cost_test_vtrue",
        "テストMVOコスト (Vhat)" : "mean_cost_test_vhat",
        "訓練MVOコスト (Vtrue)"  : "train_cost_vtrue",
        "訓練MVOコスト (Vhat)"   : "train_cost_vhat",
        "決定係数 R^2"           : "mean_r2",
        "計算時間 (秒)"          : "avg_time_per_run_sec",
    }
    out = pd.DataFrame({ jp: df.get(col, np.nan) for jp, col in cols.items() })
    # SNR 昇順
    out = out.sort_values(by="SNR", kind="mergesort").reset_index(drop=True)

    # 表示整形
    def fmt(col: str, nd=6):
        if col in out:
            out[col] = out[col].astype(float).map(lambda v: f"{v:.{nd}f}")
    fmt("テストMVOコスト (Vtrue)")
    fmt("テストMVOコスト (Vhat)")
    fmt("訓練MVOコスト (Vtrue)")
    fmt("訓練MVOコスト (Vhat)")
    fmt("決定係数 R^2", nd=6)
    if "計算時間 (秒)" in out:
        out["計算時間 (秒)"] = out["計算時間 (秒)"].astype(float).map(lambda v: f"{v:.3f}")
    return out

def save_table_png(df_table: pd.DataFrame, title: str, out_png: Path) -> None:
    """DataFrame を PNG の“表”画像として保存。"""
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

# ---------- フィルタロジック ----------

def method_token(model: str, solver: str) -> str:
    """('kkt','knitro')->'kkt+knitro' のようにまとめる。"""
    m = (model or "").lower()
    s = (solver or "").lower()
    if m == "ols":
        return "ols"  # OLS はソルバー無視
    return f"{m}+{s}"

def filter_df(
    df: pd.DataFrame,
    N: int, d: int, res: int, rho: float, sigma: float, delta: float,
    snrs: Set[float],
    allowed_methods: Set[str],
) -> pd.DataFrame:
    """固定ハイパラ、SNR、手法のフィルタを適用。"""
    # 数値化（ファイル名パースは文字列のことがある）
    for k in ["N","d","res"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    for k in ["snr","rho","sigma","delta"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    # ハイパラ固定
    mask = (
        (df.get("N") == N) &
        (df.get("d") == d) &
        (df.get("res") == res) &
        (np.isclose(df.get("rho"), rho, rtol=0, atol=1e-12)) &
        (np.isclose(df.get("sigma"), sigma, rtol=0, atol=1e-12)) &
        (np.isclose(df.get("delta"), delta, rtol=0, atol=1e-12))
    )
    df1 = df[mask].copy()
    if df1.empty:
        return df1

    # SNR フィルタ
    df1 = df1[df1["snr"].apply(lambda x: any(np.isclose(x, s, rtol=0, atol=1e-12) for s in snrs))].copy()
    if df1.empty:
        return df1

    # 手法フィルタ（OLS は solver 無視、それ以外は model+solver が一致）
    df1["method_key"] = df1.apply(lambda r: method_token(str(r.get("model","")), str(r.get("solver",""))), axis=1)
    keep_rows = []
    for _, r in df1.iterrows():
        mk = str(r["method_key"])
        m = str(r.get("model","")).lower()
        if m == "ols":
            # OLS は 'ols' として許す（solver名は問わない）
            if "ols" in allowed_methods:
                keep_rows.append(True)
            else:
                keep_rows.append(False)
        else:
            keep_rows.append(mk in allowed_methods)
    df2 = df1[keep_rows].copy()
    return df2

# ---------- メイン ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="結果フォルダ（例: results/exp_unified_yaml/20250910）")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--snrs", type=float, nargs="+", required=True,
                    help="採用する SNR 値の列（例: 0.1 0.05 0.025 0.01 0.001）")
    ap.add_argument("--methods", type=str, nargs="+", required=True,
                    help="採用する手法キー（例: kkt+knitro dual+knitro dual+ipopt ols）")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    df_all = load_summaries(root)

    allowed_methods = set([m.lower() for m in args.methods])
    snr_set = set(args.snrs)

    df_f = filter_df(
        df_all,
        N=args.N, d=args.d, res=args.res, rho=args.rho, sigma=args.sigma, delta=args.delta,
        snrs=snr_set,
        allowed_methods=allowed_methods,
    )
    if df_f.empty:
        raise SystemExit("指定条件に一致する summary が見つかりませんでした。")

    # 出力対象の method 群（データに存在するもののみ）
    # OLS は solver 不問なので 'ols' として 1 つにまとめる
    present_methods: List[str] = []
    for m in allowed_methods:
        if m == "ols":
            if any(df_f["model"].str.lower() == "ols"):
                present_methods.append("ols")
        else:
            # 例えば 'kkt+knitro' → model='kkt', solver='knitro'
            model, solver = m.split("+", 1)
            if any((df_f["model"].str.lower() == model) & (df_f["solver"].str.lower() == solver)):
                present_methods.append(m)
    present_methods = sorted(set(present_methods))

    for m in present_methods:
        if m == "ols":
            sub = df_f[df_f["model"].str.lower() == "ols"].copy()
            model, solver = "ols", "any"
            title = "OLS"
            base = "table_ols"
        else:
            model, solver = m.split("+", 1)
            sub = df_f[(df_f["model"].str.lower() == model) & (df_f["solver"].str.lower() == solver)].copy()
            title = f"{model.upper()}   {solver.capitalize()}"
            base = f"table_{model}_{solver}"

        if sub.empty:
            continue

        table = make_table(sub)
        # CSV
        csv_path = root / f"{base}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        # PNG
        png_path = root / f"{base}.png"
        save_table_png(table, title, png_path)
        print(f"[Saved] {csv_path.name} / {png_path.name}")

if __name__ == "__main__":
    main()