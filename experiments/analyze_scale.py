from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def parse_csv(path: Path) -> Dict[str, List[Dict[str, float]]]:
    data: Dict[str, List[Dict[str, float]]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"].strip().lower()
            rec = {
                "seed": float(row["seed"]),
                "std_ratio_mean_test": float(row["std_ratio_mean_test"]),
                "std_ratio_mean_train": float(row["std_ratio_mean_train"]),
                "corr2_test": float(row["corr2_test"]),
                "corr2_train": float(row["corr2_train"]),
                "r2_test": float(row["r2_test"]),
                "r2_train": float(row["r2_train"]),
            }
            data.setdefault(model, []).append(rec)
    return data


def describe(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    return (
        statistics.mean(values),
        statistics.median(values),
        statistics.pstdev(values) if len(values) > 1 else 0.0,
    )


def maybe_plot(data: Dict[str, List[Dict[str, float]]], outdir: Path, filename: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[WARN] matplotlib not available; skipping plots.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for idx, (model, rows) in enumerate(sorted(data.items())):
        xs = [r["corr2_test"] for r in rows]
        ys = [r["std_ratio_mean_test"] for r in rows]
        ax.scatter(xs, ys, label=model, alpha=0.6, color=colors[idx % len(colors)], edgecolors="none")

    ax.set_xlabel("corr² (test)")
    ax.set_ylabel("std_ratio (test)")
    ax.set_title("Scale collapse diagnostics")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] saved scatter plot to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze scale diagnostics CSV created by test_scale.py")
    ap.add_argument("--csv", type=str, required=True, help="Path to results/scale_diag.csv")
    ap.add_argument("--outdir", type=str, default=None, help="Optional directory to save plots")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = parse_csv(csv_path)
    if not data:
        raise ValueError("No rows found in CSV.")

    print("=== Scale Diagnostic Summary ===")
    for model, rows in sorted(data.items()):
        std_ratios = [r["std_ratio_mean_test"] for r in rows]
        corr2s = [r["corr2_test"] for r in rows]
        r2s = [r["r2_test"] for r in rows]

        mean_ratio, median_ratio, std_ratio = describe(std_ratios)
        mean_corr2, median_corr2, _ = describe(corr2s)
        mean_r2, median_r2, _ = describe(r2s)

        print(
            f" model={model:>5s} | std_ratio mean/median={mean_ratio:.4e}/{median_ratio:.4e} "
            f"(σ={std_ratio:.4e}), corr² mean/median={mean_corr2:.4f}/{median_corr2:.4f}, "
            f"R² mean/median={mean_r2:.4f}/{median_r2:.4f}"
        )

    if args.outdir:
        maybe_plot(data, Path(args.outdir), "scale_scatter.png")


if __name__ == "__main__":
    main()
