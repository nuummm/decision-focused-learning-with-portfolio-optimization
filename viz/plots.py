# DFL_Portfolio_Optimization2/viz/plots.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def save_hist(arr: Sequence[float], path: Path, title: str, xlabel: str) -> None:
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    _ensure_dir(path)
    plt.figure()
    plt.hist(arr, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_scatter(x: Sequence[float], y: Sequence[float], path: Path, title: str, xlabel: str, ylabel: str) -> None:
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    _ensure_dir(path)
    plt.figure()
    plt.scatter(x[mask], y[mask], s=16)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_bar_with_ci(labels: Sequence[str], means: Sequence[float], ci_los: Sequence[float], ci_his: Sequence[float], path: Path, title: str) -> None:
    _ensure_dir(path)
    means = np.array(means, float)
    lo = np.array(ci_los, float)
    hi = np.array(ci_his, float)
    yerr = np.vstack([means - lo, hi - means])
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, means, yerr=yerr, capsize=4)
    plt.xticks(x, labels, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_summary_plots(
    outdir: Path,
    tag: str,
    arrays: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    """
    arrays: {
      'cost_list': [...], 'r2_list': [...], 'test_vhat_list': [...],
      'train_true_list': [...], 'train_vhat_list': [...]
    }
    summary: {
      'cost_mean': float, 'cost_ci': (lo,hi),
      'r2_mean': float, 'r2_ci': (lo,hi),
      ...
    }
    """
    outdir = Path(outdir)
    # 1) 分布: test の Vtrue コスト
    save_hist(
        arrays.get("cost_list", []),
        outdir / f"{tag}_hist_cost_test_vtrue.png",
        "Histogram: Test MVO Cost (Vtrue eval)",
        "cost",
    )

    # 2) 分布: test の Vhat コスト
    save_hist(
        arrays.get("test_vhat_list", []),
        outdir / f"{tag}_hist_cost_test_vhat.png",
        "Histogram: Test MVO Cost (Vhat eval)",
        "cost_vhat",
    )

    # 3) 分布: R^2 (Test)
    save_hist(
        arrays.get("r2_list", []),
        outdir / f"{tag}_hist_r2_test.png",
        "Histogram: R^2 (Test)",
        "R^2 (Test)",
    )

    # 3b) 分布: R^2 (Train) — あれば
    if arrays.get("train_r2_list", None) is not None:
        save_hist(
            arrays.get("train_r2_list", []),
            outdir / f"{tag}_hist_r2_train.png",
            "Histogram: R^2 (Train)",
            "R^2 (Train)",
        )
        # gap
        tr = np.array(arrays.get("train_r2_list", []), float)
        te = np.array(arrays.get("r2_list", []), float)
        L = min(len(tr), len(te))
        if L > 0:
            save_hist(
                tr[:L] - te[:L],
                outdir / f"{tag}_hist_r2_gap.png",
                "Histogram: R^2 gap (Train - Test)",
                "gap",
            )

    # 4) 散布図: R^2 vs Test Cost(Vtrue)
    save_scatter(
        arrays.get("r2_list", []),
        arrays.get("cost_list", []),
        outdir / f"{tag}_scatter_r2_vs_cost.png",
        "R^2 vs Test MVO Cost (Vtrue)",
        "R^2",
        "Test Cost (Vtrue)",
    )

    # 5) バー(平均+CI): Test Vtrue / Test Vhat / Train Vtrue / Train Vhat
    labels = ["Test Vtrue", "Test Vhat", "Train Vtrue", "Train Vhat"]
    means = [
        summary.get("cost_mean", np.nan),
        summary.get("te_vhat_mean", np.nan),
        summary.get("tr_true_mean", np.nan),
        summary.get("tr_vhat_mean", np.nan),
    ]
    ci_los = [
        summary.get("cost_ci", (np.nan, np.nan))[0],
        np.nan,  # 簡易: 個別CIが無いので NaN
        np.nan,
        np.nan,
    ]
    ci_his = [
        summary.get("cost_ci", (np.nan, np.nan))[1],
        np.nan,
        np.nan,
        np.nan,
    ]
    save_bar_with_ci(
        labels, means, ci_los, ci_his,
        outdir / f"{tag}_bar_summary.png",
        "Summary (mean ± 95% CI where available)",
    )
