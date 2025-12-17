from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

def _setup_matplotlib_japanese() -> None:
    """Enable Japanese labels if japanize_matplotlib is available."""
    if plt is None:
        return
    try:
        import japanize_matplotlib  # noqa: F401
    except Exception:
        # Fall back to default fonts (labels may not render on some environments).
        pass


def _add_mean_p90_legend(ax) -> None:
    """Add a small legend for mean and p90 guide lines."""
    if plt is None:
        return
    try:
        from matplotlib.lines import Line2D
    except Exception:
        return
    handles, labels = ax.get_legend_handles_labels()
    if "mean" in labels and "p90" in labels:
        return
    mean_proxy = Line2D([0], [0], color="#2ca02c", linestyle="--", linewidth=1.5)
    p90_proxy = Line2D([0], [0], color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax.legend(handles + [mean_proxy, p90_proxy], labels + ["mean", "p90"], loc="upper right", frameon=True)


def _draw_p90_lines(ax, series_list: List[np.ndarray], *, width: float = 0.32, logy: bool = False) -> None:
    """Overlay p90 horizontal lines on a Matplotlib boxplot axis."""
    if plt is None:
        return
    color = "#ff7f0e"
    for i, arr in enumerate(series_list, start=1):
        v = np.asarray(arr, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        p90 = float(np.percentile(v, 90.0))
        if logy and p90 <= 0:
            continue
        ax.hlines(p90, i - width, i + width, colors=color, linestyles="--", linewidth=1.5, alpha=0.95)
    _add_mean_p90_legend(ax)


def _apply_loglike_yscale(ax, series_list: List[np.ndarray], *, linthresh: float = 1e-6) -> None:
    """Use log scale when possible; fall back to symlog when zeros/nonpositive exist."""
    if plt is None:
        return
    vals = []
    for arr in series_list:
        v = np.asarray(arr, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return
    all_v = np.concatenate(vals, axis=0)
    if all_v.size == 0:
        return
    if np.nanmin(all_v) > 0:
        ax.set_yscale("log")
    else:
        # Distances/abs-diffs can be exactly 0; symlog keeps readability without dropping 0.
        ax.set_yscale("symlog", linthresh=linthresh)


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


def _read_step_metrics(paths: ModelPaths) -> pd.DataFrame:
    csv_path = paths.outputs_dir / "step_metrics.csv"
    if not csv_path.exists():
        return pd.DataFrame()
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
    kw = dict(
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="#333333"),
        medianprops=dict(color="#333333"),
    )
    # Matplotlib <=3.8 uses `labels`, >=3.9 uses `tick_labels`.
    try:
        ax.boxplot([dual, kkt], tick_labels=["dual", "kkt"], **kw)
    except TypeError:
        ax.boxplot([dual, kkt], labels=["dual", "kkt"], **kw)
    _draw_p90_lines(ax, [dual, kkt])
    _apply_loglike_yscale(ax, [dual, kkt])
    ax.set_ylabel("elapsed_sec")
    ax.set_title("計算時間の分布（dual vs kkt）")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_elapsed_box_subset(
    df_dual: pd.DataFrame,
    df_kkt: pd.DataFrame,
    out_path: Path,
    *,
    subset_label: str,
    mask_dual: Optional[pd.Series],
    mask_kkt: Optional[pd.Series],
) -> None:
    """Boxplot elapsed_sec for a subset (e.g., ok-only / warning-only), comparing dual vs kkt."""
    if plt is None:
        return
    if "elapsed_sec" not in df_dual.columns or "elapsed_sec" not in df_kkt.columns:
        return

    dual_s = pd.to_numeric(df_dual["elapsed_sec"], errors="coerce")
    kkt_s = pd.to_numeric(df_kkt["elapsed_sec"], errors="coerce")
    if mask_dual is not None:
        dual_s = dual_s[mask_dual]
    if mask_kkt is not None:
        kkt_s = kkt_s[mask_kkt]
    dual = dual_s.dropna().to_numpy()
    kkt = kkt_s.dropna().to_numpy()
    if dual.size == 0 and kkt.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    kw = dict(
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="#333333"),
        medianprops=dict(color="#333333"),
    )
    try:
        ax.boxplot([dual, kkt], tick_labels=["dual", "kkt"], **kw)
    except TypeError:
        ax.boxplot([dual, kkt], labels=["dual", "kkt"], **kw)
    _draw_p90_lines(ax, [dual, kkt])
    _apply_loglike_yscale(ax, [dual, kkt])

    ax.set_ylabel("elapsed_sec")
    ax.set_title(f"計算時間の分布（{subset_label}）")
    ax.grid(axis="y", alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_elapsed_box_ok_vs_warning(reb_dual: pd.DataFrame, reb_kkt: pd.DataFrame, out_path: Path) -> None:
    """Combine ok-only and warning-only elapsed boxplots into a single PNG."""
    if plt is None:
        return
    if "solver_status" not in reb_dual.columns or "solver_status" not in reb_kkt.columns:
        return
    if "elapsed_sec" not in reb_dual.columns or "elapsed_sec" not in reb_kkt.columns:
        return

    ok_dual = reb_dual["solver_status"].astype(str).map(_is_ok)
    ok_kkt = reb_kkt["solver_status"].astype(str).map(_is_ok)

    def _vals(df: pd.DataFrame, mask: pd.Series) -> np.ndarray:
        return pd.to_numeric(df["elapsed_sec"], errors="coerce")[mask].dropna().to_numpy(dtype=float)

    dual_ok, kkt_ok = _vals(reb_dual, ok_dual), _vals(reb_kkt, ok_kkt)
    dual_ng, kkt_ng = _vals(reb_dual, ~ok_dual), _vals(reb_kkt, ~ok_kkt)
    if (dual_ok.size + kkt_ok.size + dual_ng.size + kkt_ng.size) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    kw = dict(
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="#333333"),
        medianprops=dict(color="#333333"),
    )
    for ax, data, title in [
        (axes[0], [dual_ok, kkt_ok], "成功（okのみ）"),
        (axes[1], [dual_ng, kkt_ng], "warning/非成功のみ"),
    ]:
        try:
            ax.boxplot(data, tick_labels=["dual", "kkt"], **kw)
        except TypeError:
            ax.boxplot(data, labels=["dual", "kkt"], **kw)
        _draw_p90_lines(ax, data)
        _apply_loglike_yscale(ax, data)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("elapsed_sec")
    fig.suptitle("計算時間の分布（dual vs kkt）", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _export_warning_breakdown(df: pd.DataFrame, out_path: Path, *, formulation: str) -> None:
    """Export a breakdown of non-ok solver outcomes using termination condition + message.

    This uses whatever is available in `rebalance_log.csv`:
        - `solver_term` (preferred; typically termination_condition_str)
        - `solver_message` (optional)
        - fallback: `solver_status_str` (optional)
    """
    if df is None or df.empty:
        return
    if "solver_status" not in df.columns:
        return
    if not (("solver_term" in df.columns) or ("solver_message" in df.columns) or ("solver_status_str" in df.columns)):
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

    term_s = _col_series("solver_term")
    msg_s = _col_series("solver_message")
    status_str_s = _col_series("solver_status_str")
    reasons = [
        _reason_code_and_short(term_s.iloc[i], msg_s.iloc[i], status_str_s.iloc[i])
        for i in range(int(len(sub)))
    ]
    reason_code = pd.Series([r[0] for r in reasons], index=sub.index, dtype=str)
    reason_short = pd.Series([r[1] for r in reasons], index=sub.index, dtype=str)
    key = (reason_code + ": " + reason_short).astype(str)
    counts = key.value_counts().head(30)
    total = float(len(sub))
    out = pd.DataFrame(
        {
            "formulation": [formulation] * int(counts.shape[0]),
            "reason": counts.index.to_numpy(),
            "count": counts.to_numpy(),
            "rate": (counts.to_numpy(dtype=float) / total) if total > 0 else np.nan,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def _clean_solver_message(msg: str) -> str:
    """Normalize solver message strings for grouping.

    We mostly see Knitro messages via Pyomo/ASL, often prefixed like:
        'Knitro 15.0.0\\x3a <reason...>'
    """
    s = str(msg or "")
    s = s.replace("\\x3a", ":").replace("\u0000", " ").strip()
    s = " ".join(s.split())
    # Drop 'Knitro <ver>:' prefix if present
    if s.lower().startswith("knitro "):
        parts = s.split(":", 1)
        if len(parts) == 2:
            s = parts[1].strip()
    return s


def _reason_code_and_short(term: str, msg: str, status_str: str) -> tuple[str, str]:
    """Map verbose messages to compact reason codes for readability."""
    m = _clean_solver_message(msg).lower()
    t = str(term or "").strip().lower()
    s = str(status_str or "").strip().lower()
    blob = " ".join([x for x in [m, t, s] if x])

    def _has(sub: str) -> bool:
        return sub in blob

    if _has("relative change in feasible solution estimate") and _has("xtol"):
        return "XTOL_ITERS", "相対変化が xtol 未満（xtol_iters）"
    if _has("cannot be improved") and _has("nearly optimal"):
        return "NEARLY_OPT", "改善不可（Nearly optimal）"
    if _has("cannot be improved"):
        return "NO_IMPROVE", "改善不可"
    if _has("maxit") or _has("max iterations") or _has("iteration limit"):
        return "MAXIT", "反復上限"
    if _has("maxtime") or _has("time limit"):
        return "MAXTIME", "時間上限"
    if _has("infeasible"):
        return "INFEASIBLE", "不 feasible"
    if _has("numerical") or _has("ill-conditioned") or _has("singular"):
        return "NUMERICAL", "数値問題"
    if _has("optimal"):
        return "OPTIMAL", "optimal（ただし warning）"
    return "OTHER", "その他"


_RE_OBJ = re.compile(r"objective\s+([^;]+);", re.IGNORECASE)
_RE_FEAS = re.compile(r"feasibility\s+error\s+([^;]+);", re.IGNORECASE)
_RE_ITERS = re.compile(r"(\d+)\s+iterations", re.IGNORECASE)
_RE_FEVAL = re.compile(r"(\d+)\s+function\s+evaluations", re.IGNORECASE)


def _safe_float(x: object) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return float("nan")


def _safe_int(x: object) -> float:
    try:
        return float(int(str(x).strip()))
    except Exception:
        return float("nan")


def _parse_knitro_message(msg: str) -> dict:
    """Parse Knitro message (as seen via Pyomo/ASL) into compact numeric fields.

    Example:
        'Current feasible solution estimate cannot be improved.; objective -0.0026; feasibility error 2.22e-16; 42 iterations; 0 function evaluations'
    """
    s = _clean_solver_message(msg)
    if not s:
        return {
            "reason_text": "",
            "objective": float("nan"),
            "feasibility_error": float("nan"),
            "iterations": float("nan"),
            "function_evals": float("nan"),
        }
    reason_text = s.split("; objective", 1)[0].strip()
    m_obj = _RE_OBJ.search(s)
    m_feas = _RE_FEAS.search(s)
    m_it = _RE_ITERS.search(s)
    m_fe = _RE_FEVAL.search(s)
    return {
        "reason_text": reason_text,
        "objective": _safe_float(m_obj.group(1)) if m_obj else float("nan"),
        "feasibility_error": _safe_float(m_feas.group(1)) if m_feas else float("nan"),
        "iterations": _safe_int(m_it.group(1)) if m_it else float("nan"),
        "function_evals": _safe_int(m_fe.group(1)) if m_fe else float("nan"),
    }


def _plot_warning_breakdown(df: pd.DataFrame, out_path: Path, *, formulation: str, top_n: int = 12) -> None:
    """Plot the top-N warning reasons (requires the same columns as _export_warning_breakdown)."""
    if plt is None or df is None or df.empty:
        return
    if "solver_status" not in df.columns:
        return
    if not (("solver_term" in df.columns) or ("solver_message" in df.columns) or ("solver_status_str" in df.columns)):
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

    term_s = _col_series("solver_term")
    msg_s = _col_series("solver_message")
    status_str_s = _col_series("solver_status_str")
    reasons = [
        _reason_code_and_short(term_s.iloc[i], msg_s.iloc[i], status_str_s.iloc[i])
        for i in range(int(len(sub)))
    ]
    reason_code = pd.Series([r[0] for r in reasons], index=sub.index, dtype=str)
    reason_short = pd.Series([r[1] for r in reasons], index=sub.index, dtype=str)
    key = (reason_code + ": " + reason_short).astype(str)
    counts = key.value_counts().head(int(top_n))
    if counts.empty:
        return

    fig_h = max(3.0, 0.35 * float(len(counts)) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.to_numpy()[::-1], color="#d62728" if formulation == "kkt" else "#2ca02c", alpha=0.85)
    ax.set_yticks(y)
    labels = [str(s) for s in counts.index.to_numpy()[::-1]]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("件数")
    ax.set_title(f"warning 内訳（{formulation}）")
    ax.grid(axis="x", alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use bbox_inches='tight' to avoid clipped labels without calling tight_layout (which can warn).
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_warning_breakdown_dual_kkt(
    df_dual: pd.DataFrame,
    df_kkt: pd.DataFrame,
    out_path: Path,
    *,
    top_n: int = 12,
) -> None:
    """Combine dual/kkt warning breakdown into a single PNG."""
    if plt is None:
        return
    if df_dual is None or df_kkt is None:
        return
    if "solver_status" not in df_dual.columns or "solver_status" not in df_kkt.columns:
        return

    def _counts(df: pd.DataFrame) -> pd.Series:
        sub = df.copy()
        sub["solver_status"] = sub["solver_status"].astype(str)
        sub = sub.loc[~sub["solver_status"].map(_is_ok)].copy()
        if sub.empty:
            return pd.Series(dtype=int)
        term_s = sub.get("solver_term", "").astype(str) if "solver_term" in sub.columns else pd.Series([""] * len(sub))
        msg_s = sub.get("solver_message", "").astype(str) if "solver_message" in sub.columns else pd.Series([""] * len(sub))
        status_str_s = sub.get("solver_status_str", "").astype(str) if "solver_status_str" in sub.columns else pd.Series([""] * len(sub))
        labels = [
            f"{code}: {short}"
            for code, short in (
                _reason_code_and_short(term_s.iloc[i], msg_s.iloc[i], status_str_s.iloc[i])
                for i in range(int(len(sub)))
            )
        ]
        return pd.Series(labels, dtype=str).value_counts()

    c_dual = _counts(df_dual)
    c_kkt = _counts(df_kkt)
    if c_dual.empty and c_kkt.empty:
        return
    combined = c_dual.add(c_kkt, fill_value=0).sort_values(ascending=False).head(int(top_n))
    reasons = list(combined.index)
    dual_vals = np.array([float(c_dual.get(r, 0)) for r in reasons], dtype=float)
    kkt_vals = np.array([float(c_kkt.get(r, 0)) for r in reasons], dtype=float)

    y = np.arange(len(reasons))
    h = 0.38
    fig_h = max(3.0, 0.42 * float(len(reasons)) + 1.6)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    ax.barh(y - h / 2, dual_vals, height=h, color="#2ca02c", alpha=0.85, label="dual")
    ax.barh(y + h / 2, kkt_vals, height=h, color="#d62728", alpha=0.85, label="kkt")
    ax.set_yticks(y)
    ax.set_yticklabels(reasons, fontsize=9)
    ax.set_xlabel("件数")
    ax.set_title("warning 内訳（dual vs kkt）")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _status_counts_and_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "solver_status" not in df.columns:
        return pd.DataFrame()
    s = df["solver_status"].astype(str).fillna("")
    counts = s.value_counts(dropna=False)
    total = float(len(s)) if len(s) else 0.0
    out = pd.DataFrame({"solver_status": counts.index.to_numpy(), "count": counts.to_numpy()})
    out["rate"] = out["count"] / total if total > 0 else np.nan
    return out


def _elapsed_summary(df: pd.DataFrame, *, ok_mask: Optional[pd.Series] = None) -> Dict[str, float]:
    if df is None or df.empty or "elapsed_sec" not in df.columns:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "n": 0.0}
    vals = pd.to_numeric(df["elapsed_sec"], errors="coerce")
    if ok_mask is not None:
        vals = vals[ok_mask]
    vals = vals.dropna().to_numpy(dtype=float)
    stats = _summary_stats(vals)
    return {"mean": stats["mean"], "median": stats["median"], "p90": stats["p90"], "n": float(stats["n"])}


def _merge_train_objective_diff(reb_dual: pd.DataFrame, reb_kkt: pd.DataFrame) -> pd.DataFrame:
    need = {"rebalance_idx", "rebalance_date", "train_objective"}
    if not need.issubset(reb_dual.columns) or not need.issubset(reb_kkt.columns):
        return pd.DataFrame()
    join_cols = ["rebalance_idx", "rebalance_date"]
    if "seed" in reb_dual.columns and "seed" in reb_kkt.columns:
        join_cols = ["seed"] + join_cols
        need = set(need) | {"seed"}
    a = reb_dual[list(need)].copy()
    b = reb_kkt[list(need)].copy()
    a = a.rename(columns={"train_objective": "train_objective_dual"})
    b = b.rename(columns={"train_objective": "train_objective_kkt"})
    a["rebalance_date"] = pd.to_datetime(a["rebalance_date"], errors="coerce")
    b["rebalance_date"] = pd.to_datetime(b["rebalance_date"], errors="coerce")
    m = a.merge(b, on=join_cols, how="inner")
    if m.empty:
        return pd.DataFrame()
    m["train_objective_dual"] = pd.to_numeric(m["train_objective_dual"], errors="coerce")
    m["train_objective_kkt"] = pd.to_numeric(m["train_objective_kkt"], errors="coerce")
    m = m.dropna(subset=["train_objective_dual", "train_objective_kkt"]).sort_values("rebalance_date")
    if m.empty:
        return pd.DataFrame()
    diff = m["train_objective_dual"] - m["train_objective_kkt"]
    m["abs_diff"] = diff.abs()
    m["diff"] = diff
    return m


def _merge_mvo_cost_diff(step_dual: pd.DataFrame, step_kkt: pd.DataFrame) -> pd.DataFrame:
    need = {"date", "mvo_cost"}
    if not need.issubset(step_dual.columns) or not need.issubset(step_kkt.columns):
        return pd.DataFrame()
    join_cols = ["date"]
    cols_a = ["date", "mvo_cost"]
    cols_b = ["date", "mvo_cost"]
    if "seed" in step_dual.columns and "seed" in step_kkt.columns:
        join_cols = ["seed", "date"]
        cols_a = ["seed"] + cols_a
        cols_b = ["seed"] + cols_b
    a = step_dual[cols_a].copy().rename(columns={"mvo_cost": "mvo_cost_dual"})
    b = step_kkt[cols_b].copy().rename(columns={"mvo_cost": "mvo_cost_kkt"})
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    b["date"] = pd.to_datetime(b["date"], errors="coerce")
    a["mvo_cost_dual"] = pd.to_numeric(a["mvo_cost_dual"], errors="coerce")
    b["mvo_cost_kkt"] = pd.to_numeric(b["mvo_cost_kkt"], errors="coerce")
    m = a.merge(b, on=join_cols, how="inner").dropna().sort_values("date")
    if m.empty:
        return pd.DataFrame()
    diff = m["mvo_cost_dual"] - m["mvo_cost_kkt"]
    m["abs_diff"] = diff.abs()
    m["diff"] = diff
    return m


def _plot_diff_timeseries(df: pd.DataFrame, out_path: Path, *, date_col: str, val_col: str, title: str, ylab: str) -> None:
    if plt is None or df is None or df.empty:
        return
    if date_col not in df.columns or val_col not in df.columns:
        return
    x = pd.to_datetime(df[date_col], errors="coerce")
    y = pd.to_numeric(df[val_col], errors="coerce")
    m = pd.DataFrame({date_col: x, val_col: y}).dropna()
    # If multiple seeds are concatenated, aggregate by date for a readable line plot.
    if m.duplicated(subset=[date_col]).any():
        m = m.groupby(date_col, as_index=False)[val_col].median()
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(m[date_col], m[val_col], color="tab:blue", lw=1.2)
    ax.set_xlabel("日付")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_absdiff_box(df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    if plt is None or df is None or df.empty or "abs_diff" not in df.columns:
        return
    vals = pd.to_numeric(df["abs_diff"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    kw = dict(
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="#333333"),
        medianprops=dict(color="#333333"),
    )
    try:
        ax.boxplot([vals], tick_labels=["abs_diff"], **kw)
    except TypeError:
        ax.boxplot([vals], labels=["abs_diff"], **kw)
    _draw_p90_lines(ax, [vals])
    _apply_loglike_yscale(ax, [vals])
    ax.set_ylabel("abs diff")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_two_absdiff_boxplots(obj_diff: pd.DataFrame, mvo_diff: pd.DataFrame, out_path: Path) -> None:
    """Combine train-objective and mvo-cost abs-diff boxplots into a single PNG."""
    if plt is None:
        return

    def _vals(df: pd.DataFrame) -> np.ndarray:
        if df is None or df.empty or "abs_diff" not in df.columns:
            return np.asarray([], dtype=float)
        return pd.to_numeric(df["abs_diff"], errors="coerce").dropna().to_numpy(dtype=float)

    v1 = _vals(obj_diff)
    v2 = _vals(mvo_diff)
    if v1.size == 0 and v2.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    kw = dict(
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="#333333"),
        medianprops=dict(color="#333333"),
    )
    for ax, vals, title in [
        (axes[0], v1, "train objective |dual-kkt|"),
        (axes[1], v2, "mvo_cost |dual-kkt|"),
    ]:
        if vals.size == 0:
            ax.axis("off")
            continue
        try:
            ax.boxplot([vals], tick_labels=["abs_diff"], **kw)
        except TypeError:
            ax.boxplot([vals], labels=["abs_diff"], **kw)
        _draw_p90_lines(ax, [vals])
        _apply_loglike_yscale(ax, [vals])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("abs diff")
    fig.suptitle("目的関数差（dual vs kkt）", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_numeric_box_dual_kkt(
    df_dual: pd.DataFrame,
    df_kkt: pd.DataFrame,
    out_path: Path,
    *,
    col: str,
    title: str,
    ylabel: str,
    logy: bool = False,
    mask_dual: Optional[pd.Series] = None,
    mask_kkt: Optional[pd.Series] = None,
) -> None:
    """Boxplot a numeric column comparing dual vs kkt (optionally with subset masks)."""
    if plt is None:
        return
    if col not in df_dual.columns or col not in df_kkt.columns:
        return
    a = pd.to_numeric(df_dual[col], errors="coerce")
    b = pd.to_numeric(df_kkt[col], errors="coerce")
    if mask_dual is not None:
        a = a[mask_dual]
    if mask_kkt is not None:
        b = b[mask_kkt]
    a = a.dropna().to_numpy(dtype=float)
    b = b.dropna().to_numpy(dtype=float)
    if a.size == 0 and b.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    kw = dict(
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="#333333"),
        medianprops=dict(color="#333333"),
    )
    try:
        ax.boxplot([a, b], tick_labels=["dual", "kkt"], **kw)
    except TypeError:
        ax.boxplot([a, b], labels=["dual", "kkt"], **kw)
    _draw_p90_lines(ax, [a, b], logy=logy)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        _apply_loglike_yscale(ax, [a, b])
    ax.grid(axis="y", alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _compute_weight_agreement(w_dual: pd.DataFrame, w_kkt: pd.DataFrame) -> pd.DataFrame:
    if "date" not in w_dual.columns or "date" not in w_kkt.columns:
        return pd.DataFrame()
    join_cols = ["date"]
    if "seed" in w_dual.columns and "seed" in w_kkt.columns:
        join_cols = ["seed", "date"]
    join_set = set(join_cols)
    assets = [c for c in w_dual.columns if c in w_kkt.columns and c not in join_set]
    # Exclude non-asset diagnostics columns.
    assets = [c for c in assets if c.lower() not in {"portfolio_return_sq"}]
    # Deduplicate while preserving order (important when concatenating pooled runs).
    assets = list(dict.fromkeys(assets))
    if not assets:
        return pd.DataFrame()
    a_cols = join_cols + assets
    b_cols = join_cols + assets
    a = w_dual[a_cols].copy()
    b = w_kkt[b_cols].copy()
    for col in assets:
        a[col] = pd.to_numeric(a[col], errors="coerce")
        b[col] = pd.to_numeric(b[col], errors="coerce")
    m = a.merge(b, on=join_cols, suffixes=("_dual", "_kkt"), how="inner").sort_values("date")
    if m.empty:
        return pd.DataFrame()
    diff = np.stack([m[f"{c}_dual"].to_numpy() - m[f"{c}_kkt"].to_numpy() for c in assets], axis=1)
    l1 = np.sum(np.abs(diff), axis=1)
    l2 = np.sqrt(np.sum(diff * diff, axis=1))
    out = pd.DataFrame({"date": m["date"].to_numpy(), "l1": l1, "l2": l2})
    if "seed" in join_cols:
        out["seed"] = m["seed"].to_numpy()
    return out


def _plot_agreement_ts(agree: pd.DataFrame, out_path: Path, *, metric: str) -> None:
    if plt is None or agree.empty:
        return
    if "date" not in agree.columns or metric not in agree.columns:
        return
    plot_df = agree.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=["date", metric])
    if plot_df.empty:
        return
    # If multiple seeds are present, plot median distance per date.
    if plot_df.duplicated(subset=["date"]).any():
        plot_df = plot_df.groupby("date", as_index=False)[metric].median()
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(plot_df["date"], plot_df[metric], color="tab:blue", lw=1.2)
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
    kw = dict(
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="#333333"),
        medianprops=dict(color="#333333"),
    )
    try:
        ax.boxplot([l1, l2], tick_labels=["L1", "L2"], **kw)
    except TypeError:
        ax.boxplot([l1, l2], labels=["L1", "L2"], **kw)
    _draw_p90_lines(ax, [l1, l2])
    _apply_loglike_yscale(ax, [l1, l2])
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
        default=None,
        help="Path to results/debug_outputs/<timestamp>_rolling (contains flex_dual/flex_kkt and model_* logs).",
    )
    p.add_argument(
        "--localopt-a-dir",
        type=Path,
        default=None,
        help="If set, aggregate across all seeds under an exp_localopt_A run dir (contains seed_*/{models,debug}).",
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
    _setup_matplotlib_japanese()
    if args.debug_dir is None and args.localopt_a_dir is None:
        raise SystemExit("Either --debug-dir or --localopt-a-dir must be provided.")
    debug_dir = Path(args.debug_dir) if args.debug_dir is not None else None
    outdir, analysis_csv, analysis_fig = _ensure_outdir(args.exp_dir)
    fig_dir = analysis_fig / "formulation_choice"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_localopt_seed(seed_dir: Path, *, label: str) -> ModelPaths:
        outputs_dir = seed_dir / "models" / label
        debug_subdir = seed_dir / "debug" / f"model_{label}"
        if not outputs_dir.exists():
            raise FileNotFoundError(f"Missing model outputs dir: {outputs_dir}")
        if not debug_subdir.exists():
            raise FileNotFoundError(f"Missing model debug dir: {debug_subdir}")
        return ModelPaths(label=label, outputs_dir=outputs_dir, debug_dir=debug_subdir)

    # --- Load data either from a single real-data debug dir or pooled across local-opt-A seeds ---
    if args.localopt_a_dir is None:
        assert debug_dir is not None
        dual = _resolve_model_paths(debug_dir, label=str(args.dual_label))
        kkt = _resolve_model_paths(debug_dir, label=str(args.kkt_label))

        reb_dual = _read_rebalance_log(dual)
        reb_kkt = _read_rebalance_log(kkt)
        reb_dual["formulation"] = "dual"
        reb_kkt["formulation"] = "kkt"

        w_dual = _read_weights(dual)
        w_kkt = _read_weights(kkt)
        step_dual = _read_step_metrics(dual)
        step_kkt = _read_step_metrics(kkt)
    else:
        base = Path(args.localopt_a_dir)
        seed_dirs = sorted([p for p in base.glob("seed_*") if p.is_dir()])
        if not seed_dirs:
            raise SystemExit(f"No seed_* directories found under: {base}")

        reb_dual_parts = []
        reb_kkt_parts = []
        w_dual_parts = []
        w_kkt_parts = []
        step_dual_parts = []
        step_kkt_parts = []

        for seed_dir in seed_dirs:
            # seed id from folder name (seed_0 ...)
            try:
                seed_id = int(str(seed_dir.name).split("_", 1)[1])
            except Exception:
                seed_id = -1
            dual = _resolve_localopt_seed(seed_dir, label=str(args.dual_label))
            kkt = _resolve_localopt_seed(seed_dir, label=str(args.kkt_label))

            r_dual = _read_rebalance_log(dual)
            r_kkt = _read_rebalance_log(kkt)
            r_dual["seed"] = seed_id
            r_kkt["seed"] = seed_id
            reb_dual_parts.append(r_dual)
            reb_kkt_parts.append(r_kkt)

            wd = _read_weights(dual)
            wk = _read_weights(kkt)
            if not wd.empty:
                wd["seed"] = seed_id
                w_dual_parts.append(wd)
            if not wk.empty:
                wk["seed"] = seed_id
                w_kkt_parts.append(wk)

            sd = _read_step_metrics(dual)
            sk = _read_step_metrics(kkt)
            if not sd.empty:
                sd["seed"] = seed_id
                step_dual_parts.append(sd)
            if not sk.empty:
                sk["seed"] = seed_id
                step_kkt_parts.append(sk)

        reb_dual = pd.concat(reb_dual_parts, ignore_index=True) if reb_dual_parts else pd.DataFrame()
        reb_kkt = pd.concat(reb_kkt_parts, ignore_index=True) if reb_kkt_parts else pd.DataFrame()
        reb_dual["formulation"] = "dual"
        reb_kkt["formulation"] = "kkt"

        w_dual = pd.concat(w_dual_parts, ignore_index=True) if w_dual_parts else pd.DataFrame()
        w_kkt = pd.concat(w_kkt_parts, ignore_index=True) if w_kkt_parts else pd.DataFrame()
        step_dual = pd.concat(step_dual_parts, ignore_index=True) if step_dual_parts else pd.DataFrame()
        step_kkt = pd.concat(step_kkt_parts, ignore_index=True) if step_kkt_parts else pd.DataFrame()

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

    # Paper-friendly: status proportions + elapsed summaries (all vs ok-only)
    paper_rows = []
    for name, df in [("dual", reb_dual), ("kkt", reb_kkt)]:
        status_tbl = _status_counts_and_rates(df)
        if not status_tbl.empty:
            status_tbl = status_tbl.copy()
            status_tbl["formulation"] = name
            _export_csv(status_tbl, analysis_csv / f"formulation_choice_status_breakdown_{name}.csv")
        ok_mask = df["solver_status"].astype(str).map(_is_ok) if "solver_status" in df.columns else None
        all_s = _elapsed_summary(df, ok_mask=None)
        ok_s = _elapsed_summary(df, ok_mask=ok_mask) if ok_mask is not None else {"mean": np.nan, "median": np.nan, "p90": np.nan, "n": 0.0}
        non_ok_s = _elapsed_summary(df, ok_mask=~ok_mask) if ok_mask is not None else {"mean": np.nan, "median": np.nan, "p90": np.nan, "n": 0.0}
        paper_rows.append(
            {
                "formulation": name,
                "n_all": int(all_s["n"]),
                "elapsed_all_mean": float(all_s["mean"]),
                "elapsed_all_median": float(all_s["median"]),
                "elapsed_all_p90": float(all_s["p90"]),
                "n_ok": int(ok_s["n"]),
                "elapsed_ok_mean": float(ok_s["mean"]),
                "elapsed_ok_median": float(ok_s["median"]),
                "elapsed_ok_p90": float(ok_s["p90"]),
                "n_non_ok": int(non_ok_s["n"]),
                "elapsed_non_ok_mean": float(non_ok_s["mean"]),
                "elapsed_non_ok_median": float(non_ok_s["median"]),
                "elapsed_non_ok_p90": float(non_ok_s["p90"]),
            }
        )
    _export_csv(pd.DataFrame(paper_rows), analysis_csv / "formulation_choice_elapsed_summary_split.csv")

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

    # Plots: elapsed split by success vs warning/non-ok
    if "solver_status" in reb_dual.columns and "solver_status" in reb_kkt.columns:
        _plot_elapsed_box_ok_vs_warning(reb_dual, reb_kkt, fig_dir / "elapsed_sec_boxplot_ok_vs_warning.png")

    # Optional: detailed breakdown (if the runner logged termination/message)
    _export_warning_breakdown(reb_dual, analysis_csv / "warning_breakdown_dual.csv", formulation="dual")
    _export_warning_breakdown(reb_kkt, analysis_csv / "warning_breakdown_kkt.csv", formulation="kkt")
    _plot_warning_breakdown_dual_kkt(reb_dual, reb_kkt, fig_dir / "warning_breakdown_dual_vs_kkt.png")

    # Extra: parse Knitro messages into compact numeric fields for easier inspection in papers.
    def _export_warning_details(df: pd.DataFrame, *, formulation: str) -> None:
        if df is None or df.empty or "solver_status" not in df.columns:
            return
        sub = df.copy()
        sub["solver_status"] = sub["solver_status"].astype(str)
        sub = sub.loc[~sub["solver_status"].map(_is_ok)].copy()
        if sub.empty:
            return
        term_s = sub.get("solver_term", "").astype(str) if "solver_term" in sub.columns else pd.Series([""] * len(sub))
        msg_s = sub.get("solver_message", "").astype(str) if "solver_message" in sub.columns else pd.Series([""] * len(sub))
        status_str_s = sub.get("solver_status_str", "").astype(str) if "solver_status_str" in sub.columns else pd.Series([""] * len(sub))
        codes = []
        shorts = []
        parsed = []
        for i in range(len(sub)):
            code, short = _reason_code_and_short(term_s.iloc[i], msg_s.iloc[i], status_str_s.iloc[i])
            codes.append(code)
            shorts.append(short)
            parsed.append(_parse_knitro_message(msg_s.iloc[i]))
        det = sub.copy()
        det["reason_code"] = codes
        det["reason_short"] = shorts
        det["reason_text"] = [p["reason_text"] for p in parsed]
        det["msg_objective"] = [p["objective"] for p in parsed]
        det["msg_feasibility_error"] = [p["feasibility_error"] for p in parsed]
        det["msg_iterations"] = [p["iterations"] for p in parsed]
        det["msg_function_evals"] = [p["function_evals"] for p in parsed]
        keep_cols = [
            c
            for c in [
                "cycle",
                "rebalance_idx",
                "rebalance_date",
                "solver_status",
                "solver_term",
                "reason_code",
                "reason_short",
                "reason_text",
                "msg_objective",
                "msg_feasibility_error",
                "msg_iterations",
                "msg_function_evals",
                "elapsed_sec",
                "train_objective",
            ]
            if c in det.columns
        ]
        _export_csv(det[keep_cols], analysis_csv / f"warning_details_{formulation}.csv")

        # Aggregate by reason_code for paper-friendly summary (counts + numeric medians)
        g = det.groupby(["reason_code", "reason_short"], dropna=False)
        rows = []
        total = float(len(det))
        for (code, short), part in g:
            it = pd.to_numeric(part.get("msg_iterations", pd.Series(dtype=float)), errors="coerce").dropna()
            feas = pd.to_numeric(part.get("msg_feasibility_error", pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append(
                {
                    "formulation": formulation,
                    "reason_code": str(code),
                    "reason_short": str(short),
                    "count": int(len(part)),
                    "rate": float(len(part) / total) if total > 0 else float("nan"),
                    "median_iterations": float(it.median()) if len(it) else float("nan"),
                    "median_feasibility_error": float(feas.median()) if len(feas) else float("nan"),
                }
            )
        if rows:
            _export_csv(pd.DataFrame(rows).sort_values(["count", "reason_code"], ascending=[False, True]), analysis_csv / f"warning_reason_stats_{formulation}.csv")

    _export_warning_details(reb_dual, formulation="dual")
    _export_warning_details(reb_kkt, formulation="kkt")

    # Violation diagnostics (if available): eq/ineq max violations from training
    if "solver_status" in reb_dual.columns and "solver_status" in reb_kkt.columns:
        ok_dual = reb_dual["solver_status"].astype(str).map(_is_ok)
        ok_kkt = reb_kkt["solver_status"].astype(str).map(_is_ok)
        _plot_numeric_box_dual_kkt(
            reb_dual,
            reb_kkt,
            fig_dir / "train_eq_viol_max_boxplot.png",
            col="train_eq_viol_max",
            title="制約違反（等式, max）",
            ylabel="eq_viol_max",
            logy=True,
        )
        _plot_numeric_box_dual_kkt(
            reb_dual,
            reb_kkt,
            fig_dir / "train_ineq_viol_max_boxplot.png",
            col="train_ineq_viol_max",
            title="制約違反（不等式, max）",
            ylabel="ineq_viol_max",
            logy=True,
        )
        _plot_numeric_box_dual_kkt(
            reb_dual,
            reb_kkt,
            fig_dir / "train_eq_viol_max_boxplot_ok_only.png",
            col="train_eq_viol_max",
            title="制約違反（等式, max）(okのみ)",
            ylabel="eq_viol_max",
            logy=True,
            mask_dual=ok_dual,
            mask_kkt=ok_kkt,
        )
        _plot_numeric_box_dual_kkt(
            reb_dual,
            reb_kkt,
            fig_dir / "train_ineq_viol_max_boxplot_ok_only.png",
            col="train_ineq_viol_max",
            title="制約違反（不等式, max）(okのみ)",
            ylabel="ineq_viol_max",
            logy=True,
            mask_dual=ok_dual,
            mask_kkt=ok_kkt,
        )

    # Knitro message diagnostics (warning/non-ok only): iterations / feasibility error
    def _warning_parsed_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "solver_status" not in df.columns:
            return pd.DataFrame()
        sub = df.copy()
        sub["solver_status"] = sub["solver_status"].astype(str)
        sub = sub.loc[~sub["solver_status"].map(_is_ok)].copy()
        if sub.empty or "solver_message" not in sub.columns:
            return pd.DataFrame()
        parsed = [_parse_knitro_message(m) for m in sub["solver_message"].astype(str).tolist()]
        sub["msg_feasibility_error"] = [p["feasibility_error"] for p in parsed]
        sub["msg_iterations"] = [p["iterations"] for p in parsed]
        sub["msg_function_evals"] = [p["function_evals"] for p in parsed]
        return sub

    warn_dual = _warning_parsed_df(reb_dual)
    warn_kkt = _warning_parsed_df(reb_kkt)
    _plot_numeric_box_dual_kkt(
        warn_dual,
        warn_kkt,
        fig_dir / "warning_iterations_boxplot.png",
        col="msg_iterations",
        title="イテレーション数（warningのみ）",
        ylabel="iterations",
        logy=False,
    )
    _plot_numeric_box_dual_kkt(
        warn_dual,
        warn_kkt,
        fig_dir / "warning_feasibility_error_boxplot.png",
        col="msg_feasibility_error",
        title="feasibility error（warningのみ）",
        ylabel="feasibility_error",
        logy=True,
    )
    _plot_numeric_box_dual_kkt(
        warn_dual,
        warn_kkt,
        fig_dir / "warning_function_evals_boxplot.png",
        col="msg_function_evals",
        title="関数評価回数（warningのみ）",
        ylabel="function_evals",
        logy=True,
    )

    # Agreement on weights
    # For local-opt-A pooled runs, w_dual/w_kkt include `seed` and are pooled on `date` only.
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

    # Objective difference: training objective (per rebalance)
    obj_diff = _merge_train_objective_diff(reb_dual, reb_kkt)
    if not obj_diff.empty:
        _export_csv(obj_diff, analysis_csv / "formulation_choice_train_objective_diff.csv")
        s_abs = _summary_stats(obj_diff["abs_diff"].to_numpy(dtype=float))
        _export_csv(pd.DataFrame([{"metric": "abs_diff", **s_abs}]), analysis_csv / "formulation_choice_train_objective_diff_summary.csv")
        _plot_diff_timeseries(
            obj_diff,
            fig_dir / "train_objective_absdiff_timeseries.png",
            date_col="rebalance_date",
            val_col="abs_diff",
            title="目的関数差（train objective）|dual-kkt|",
            ylab="|Δ objective|",
        )

        # period splits (no extra runs)
        if not bool(getattr(args, "no_period_split", False)):
            rows_obj = []
            for pname, start, end in _period_windows():
                sub = _filter_period(obj_diff, "rebalance_date", start, end)
                if sub.empty:
                    continue
                s = _summary_stats(sub["abs_diff"].to_numpy(dtype=float))
                rows_obj.append({"period": pname, "start": start, "end": end, **{f"absdiff_{k}": v for k, v in s.items()}})
            if rows_obj:
                _export_csv(pd.DataFrame(rows_obj), analysis_csv / "formulation_choice_train_objective_diff_by_period.csv")

    # Objective difference: eval-time MVO cost (per step date)
    mvo_diff = _merge_mvo_cost_diff(step_dual, step_kkt)
    if not mvo_diff.empty:
        _export_csv(mvo_diff, analysis_csv / "formulation_choice_mvo_cost_diff.csv")
        s_abs = _summary_stats(mvo_diff["abs_diff"].to_numpy(dtype=float))
        _export_csv(pd.DataFrame([{"metric": "abs_diff", **s_abs}]), analysis_csv / "formulation_choice_mvo_cost_diff_summary.csv")
        _plot_diff_timeseries(
            mvo_diff,
            fig_dir / "mvo_cost_absdiff_timeseries.png",
            date_col="date",
            val_col="abs_diff",
            title="目的関数差（MVO cost）|dual-kkt|",
            ylab="|Δ mvo_cost|",
        )

        if not bool(getattr(args, "no_period_split", False)):
            rows_obj = []
            for pname, start, end in _period_windows():
                sub = _filter_period(mvo_diff, "date", start, end)
                if sub.empty:
                    continue
                s = _summary_stats(sub["abs_diff"].to_numpy(dtype=float))
                rows_obj.append({"period": pname, "start": start, "end": end, **{f"absdiff_{k}": v for k, v in s.items()}})
            if rows_obj:
                _export_csv(pd.DataFrame(rows_obj), analysis_csv / "formulation_choice_mvo_cost_diff_by_period.csv")

    # Combined abs-diff boxplots (train objective vs mvo_cost)
    _plot_two_absdiff_boxplots(obj_diff, mvo_diff, fig_dir / "objective_absdiff_boxplots.png")

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
