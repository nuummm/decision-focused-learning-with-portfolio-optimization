from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
RESULTS_BASE = PROJECT_ROOT / "results"
# Default output root for this study (a new folder under /Users/.../卒業研究2/results).
RESULTS_ROOT = RESULTS_BASE / "formulation_choice"


@dataclass(frozen=True)
class ModelPaths:
    label: str
    outputs_dir: Path
    debug_dir: Path


def _is_ok(status: object) -> bool:
    s = str(status).lower()
    return ("optimal" in s) or ("ok" in s)


def _percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": _percentile(arr, 90.0),
        "n": int(arr.size),
    }


def _ensure_outdir(exp_dir: Optional[Path]) -> Tuple[Path, Path, Path]:
    if exp_dir is None:
        from dfl_portfolio.real_data.cli import make_output_dir

        outdir = make_output_dir(RESULTS_ROOT, None)
        analysis_dir = outdir / "analysis"
    else:
        analysis_dir = Path(exp_dir) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        outdir = analysis_dir.parent
    analysis_csv = analysis_dir / "csv"
    analysis_fig = analysis_dir / "figures"
    analysis_csv.mkdir(parents=True, exist_ok=True)
    analysis_fig.mkdir(parents=True, exist_ok=True)
    return outdir, analysis_csv, analysis_fig


def _resolve_model_paths(debug_dir: Path, *, label: str) -> ModelPaths:
    debug_dir = Path(debug_dir)
    outputs_dir = debug_dir / label
    per_model_debug = debug_dir / f"model_{label}"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Missing model outputs dir: {outputs_dir}")
    if not per_model_debug.exists():
        raise FileNotFoundError(f"Missing model debug dir: {per_model_debug}")
    return ModelPaths(label=label, outputs_dir=outputs_dir, debug_dir=per_model_debug)


def _read_rebalance_log(paths: ModelPaths) -> pd.DataFrame:
    csv_path = paths.debug_dir / "rebalance_log.csv"
    df = pd.read_csv(csv_path)
    if "rebalance_date" in df.columns:
        df["rebalance_date"] = pd.to_datetime(df["rebalance_date"], errors="coerce")
    return df


def _read_weights(paths: ModelPaths) -> pd.DataFrame:
    csv_path = paths.outputs_dir / "weights.csv"
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _filter_period(df: pd.DataFrame, date_col: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is None or end is None:
        return df
    out = df.copy()
    if date_col not in out.columns:
        return out
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    mask = (out[date_col] >= pd.Timestamp(start)) & (out[date_col] <= pd.Timestamp(end))
    return out.loc[mask].copy()


def _period_windows() -> List[Tuple[str, str, str]]:
    # Reuse the same event windows used in reporting.
    from dfl_portfolio.real_data.reporting import PERIOD_WINDOWS

    out: List[Tuple[str, str, str]] = []
    for name, start, end in PERIOD_WINDOWS:
        out.append((str(name), str(start), str(end)))
    return out


def _export_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _plot_status_counts(df_dual: pd.DataFrame, df_kkt: pd.DataFrame, out_path: Path) -> None:
    if plt is None:
        return
    if "solver_status" not in df_dual.columns or "solver_status" not in df_kkt.columns:
        return
    c_dual = df_dual["solver_status"].astype(str).value_counts()
    c_kkt = df_kkt["solver_status"].astype(str).value_counts()
    statuses = sorted(set(c_dual.index).union(set(c_kkt.index)))
    x = np.arange(len(statuses))
    w = 0.38
    dual_vals = np.array([int(c_dual.get(s, 0)) for s in statuses], dtype=float)
    kkt_vals = np.array([int(c_kkt.get(s, 0)) for s in statuses], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, dual_vals, width=w, label="dual", color="#2ca02c", alpha=0.85)
    ax.bar(x + w / 2, kkt_vals, width=w, label="kkt", color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(statuses, rotation=30, ha="right")
    ax.set_ylabel("件数")
    ax.set_title("ソルバステータス内訳（dual vs kkt）")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_elapsed_box(df_dual: pd.DataFrame, df_kkt: pd.DataFrame, out_path: Path) -> None:
    if plt is None:
        return
    if "elapsed_sec" not in df_dual.columns or "elapsed_sec" not in df_kkt.columns:
        return
    dual = pd.to_numeric(df_dual["elapsed_sec"], errors="coerce").dropna().to_numpy()
    kkt = pd.to_numeric(df_kkt["elapsed_sec"], errors="coerce").dropna().to_numpy()
    if dual.size == 0 and kkt.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.boxplot(
        [dual, kkt],
        tick_labels=["dual", "kkt"],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="#333333"),
        medianprops=dict(color="#333333"),
    )
    ax.set_ylabel("elapsed_sec")
    ax.set_title("計算時間の分布（dual vs kkt）")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _export_warning_breakdown(df: pd.DataFrame, out_path: Path, *, formulation: str) -> None:
    """Export a breakdown of non-ok solver outcomes using termination condition + message.

    This relies on `rebalance_log.csv` containing:
        - solver_status
        - solver_term (termination_condition_str)
        - solver_message
    """
    if df is None or df.empty:
        return
    if "solver_status" not in df.columns:
        return
    if "solver_term" not in df.columns and "solver_message" not in df.columns:
        return

    sub = df.copy()
    sub["solver_status"] = sub["solver_status"].astype(str)
    sub = sub.loc[~sub["solver_status"].map(_is_ok)].copy()
    if sub.empty:
        return

    def _col_series(col: str) -> pd.Series:
        if col in sub.columns:
            return sub[col].astype(str).fillna("")
        return pd.Series([""] * int(len(sub)), index=sub.index, dtype=str)

    term = _col_series("solver_term")
    msg = _col_series("solver_message")
    key = (term.str.strip() + " | " + msg.str.strip()).str.strip(" |")
    counts = key.value_counts().head(30)
    out = pd.DataFrame(
        {
            "formulation": [formulation] * int(counts.shape[0]),
            "count": counts.to_numpy(),
            "termination_and_message": counts.index.to_numpy(),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def _compute_weight_agreement(w_dual: pd.DataFrame, w_kkt: pd.DataFrame) -> pd.DataFrame:
    if "date" not in w_dual.columns or "date" not in w_kkt.columns:
        return pd.DataFrame()
    assets = [c for c in w_dual.columns if c != "date" and c in w_kkt.columns]
    # Exclude non-asset diagnostics columns.
    assets = [c for c in assets if c.lower() not in {"portfolio_return_sq"}]
    if not assets:
        return pd.DataFrame()
    a = w_dual[["date"] + assets].copy()
    b = w_kkt[["date"] + assets].copy()
    for col in assets:
        a[col] = pd.to_numeric(a[col], errors="coerce")
        b[col] = pd.to_numeric(b[col], errors="coerce")
    m = a.merge(b, on="date", suffixes=("_dual", "_kkt"), how="inner").sort_values("date")
    if m.empty:
        return pd.DataFrame()
    diff = np.stack([m[f"{c}_dual"].to_numpy() - m[f"{c}_kkt"].to_numpy() for c in assets], axis=1)
    l1 = np.sum(np.abs(diff), axis=1)
    l2 = np.sqrt(np.sum(diff * diff, axis=1))
    out = pd.DataFrame(
        {
            "date": m["date"].to_numpy(),
            "l1": l1,
            "l2": l2,
        }
    )
    return out


def _plot_agreement_ts(agree: pd.DataFrame, out_path: Path, *, metric: str) -> None:
    if plt is None or agree.empty:
        return
    if "date" not in agree.columns or metric not in agree.columns:
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(pd.to_datetime(agree["date"]), pd.to_numeric(agree[metric], errors="coerce"), color="tab:blue", lw=1.2)
    ax.set_xlabel("日付")
    ax.set_ylabel(metric)
    title_map = {"l1": "||w_dual - w_kkt||_1", "l2": "||w_dual - w_kkt||_2"}
    ax.set_title(f"解の一致度推移（{title_map.get(metric, metric)}）")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_agreement_box(agree: pd.DataFrame, out_path: Path) -> None:
    if plt is None or agree.empty:
        return
    l1 = pd.to_numeric(agree.get("l1", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
    l2 = pd.to_numeric(agree.get("l2", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
    if l1.size == 0 and l2.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.boxplot(
        [l1, l2],
        tick_labels=["L1", "L2"],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="#333333"),
        medianprops=dict(color="#333333"),
    )
    ax.set_ylabel("distance")
    ax.set_title("解の一致度（dual vs kkt）")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Formulation choice study: dual vs kkt (1〜3)")
    p.add_argument(
        "--debug-dir",
        type=Path,
        required=True,
        help="Path to results/debug_outputs/<timestamp>_rolling (contains flex_dual/flex_kkt and model_* logs).",
    )
    p.add_argument(
        "--exp-dir",
        type=Path,
        default=None,
        help="If provided, write outputs under <exp-dir>/analysis/{csv,figures}. Otherwise create a new results dir.",
    )
    p.add_argument(
        "--dual-label",
        type=str,
        default="flex_dual",
        help="Model label folder name for dual (default: flex_dual).",
    )
    p.add_argument(
        "--kkt-label",
        type=str,
        default="flex_kkt",
        help="Model label folder name for kkt (default: flex_kkt).",
    )
    p.add_argument(
        "--no-period-split",
        action="store_true",
        help="Disable period-window split outputs (lehman/covid/inflation).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    debug_dir = Path(args.debug_dir)
    outdir, analysis_csv, analysis_fig = _ensure_outdir(args.exp_dir)
    fig_dir = analysis_fig / "formulation_choice"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dual = _resolve_model_paths(debug_dir, label=str(args.dual_label))
    kkt = _resolve_model_paths(debug_dir, label=str(args.kkt_label))

    reb_dual = _read_rebalance_log(dual)
    reb_kkt = _read_rebalance_log(kkt)
    reb_dual["formulation"] = "dual"
    reb_kkt["formulation"] = "kkt"

    # Overall summaries
    def _status_summary(df: pd.DataFrame) -> Dict[str, object]:
        out: Dict[str, object] = {}
        if "solver_status" in df.columns:
            s = df["solver_status"].astype(str)
            out["n_cycles"] = int(len(s))
            out["ok_rate"] = float(s.map(_is_ok).mean()) if len(s) else float("nan")
        if "elapsed_sec" in df.columns:
            vals = pd.to_numeric(df["elapsed_sec"], errors="coerce").dropna().to_numpy()
            out.update({f"elapsed_{k}": v for k, v in _summary_stats(vals).items()})
        return out

    rows = []
    for name, df in [("dual", reb_dual), ("kkt", reb_kkt)]:
        rec = {"period": "all", "formulation": name}
        rec.update(_status_summary(df))
        rows.append(rec)
    status_summary_df = pd.DataFrame(rows)
    _export_csv(status_summary_df, analysis_csv / "formulation_choice_rebalance_summary.csv")

    # Period splits (案B: no extra runs)
    if not bool(getattr(args, "no_period_split", False)):
        for pname, start, end in _period_windows():
            for name, df in [("dual", reb_dual), ("kkt", reb_kkt)]:
                sub = _filter_period(df, "rebalance_date", start, end)
                rec = {"period": pname, "start": start, "end": end, "formulation": name}
                rec.update(_status_summary(sub))
                rows.append(rec)
        period_df = pd.DataFrame(rows)
        _export_csv(period_df, analysis_csv / "formulation_choice_rebalance_summary_by_period.csv")

    # Plots: status and time
    _plot_status_counts(reb_dual, reb_kkt, fig_dir / "solver_status_counts.png")
    _plot_elapsed_box(reb_dual, reb_kkt, fig_dir / "elapsed_sec_boxplot.png")

    # Optional: detailed breakdown (if the runner logged termination/message)
    _export_warning_breakdown(reb_dual, analysis_csv / "warning_breakdown_dual.csv", formulation="dual")
    _export_warning_breakdown(reb_kkt, analysis_csv / "warning_breakdown_kkt.csv", formulation="kkt")

    # Agreement on weights
    w_dual = _read_weights(dual)
    w_kkt = _read_weights(kkt)
    agree = _compute_weight_agreement(w_dual, w_kkt)
    if not agree.empty:
        _export_csv(agree, analysis_csv / "formulation_choice_weight_agreement.csv")
        _plot_agreement_ts(agree, fig_dir / "weight_distance_l1_timeseries.png", metric="l1")
        _plot_agreement_ts(agree, fig_dir / "weight_distance_l2_timeseries.png", metric="l2")
        _plot_agreement_box(agree, fig_dir / "weight_distance_boxplot.png")

        # Period splits for agreement
        if not bool(getattr(args, "no_period_split", False)):
            rows_ag = []
            # overall
            s_all_l1 = _summary_stats(agree["l1"].to_numpy(dtype=float))
            s_all_l2 = _summary_stats(agree["l2"].to_numpy(dtype=float))
            rows_ag.append({"period": "all", **{f"l1_{k}": v for k, v in s_all_l1.items()}, **{f"l2_{k}": v for k, v in s_all_l2.items()}})
            for pname, start, end in _period_windows():
                sub = _filter_period(agree, "date", start, end)
                if sub.empty:
                    continue
                s_l1 = _summary_stats(sub["l1"].to_numpy(dtype=float))
                s_l2 = _summary_stats(sub["l2"].to_numpy(dtype=float))
                rows_ag.append(
                    {
                        "period": pname,
                        "start": start,
                        "end": end,
                        **{f"l1_{k}": v for k, v in s_l1.items()},
                        **{f"l2_{k}": v for k, v in s_l2.items()},
                    }
                )
            _export_csv(pd.DataFrame(rows_ag), analysis_csv / "formulation_choice_weight_agreement_summary.csv")

    # Save config for reproducibility
    cfg = {
        "debug_dir": str(debug_dir),
        "exp_dir": str(args.exp_dir) if args.exp_dir is not None else None,
        "dual_label": str(args.dual_label),
        "kkt_label": str(args.kkt_label),
        "period_split": not bool(getattr(args, "no_period_split", False)),
    }
    (analysis_csv / "formulation_choice_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[formulation-choice] finished. outputs -> {outdir}")


if __name__ == "__main__":  # pragma: no cover
    main()
