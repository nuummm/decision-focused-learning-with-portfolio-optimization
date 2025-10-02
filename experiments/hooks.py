# ~/VScode/GraduationResearch/DFL_Portfolio_Optimization2/experiments/hooks.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Sequence, Optional

import numpy as np
import pandas as pd

# プロット用（インストールされていなければ: pip install matplotlib）
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class ExperimentHooks:
    """
    - 進捗ログ（一定間隔）
    - per-run CSV 保存
    - summary CSV 保存
    - 累積ログへの追記
    - 簡易プロット（per-run CSV から）
    """
    def __init__(self, outdir: Path, tag: str, log_every: int = 0, tee: bool = False):
        self.outdir = Path(outdir)
        self.tag = str(tag)
        self.log_every = int(log_every) if log_every is not None else 0
        self.tee = bool(tee)

    # ---------------------------
    # 基本ユーティリティ
    # ---------------------------
    def _ensure_outdir(self) -> None:
        self.outdir.mkdir(parents=True, exist_ok=True)

    def _basepath(self) -> Path:
        return self.outdir / self.tag

    # ---------------------------
    # 進捗ログ
    # ---------------------------
    def log_progress(self, run_idx: int, total_runs: int, seed: int, metrics: Dict[str, Any]) -> None:
        """run.pyのループから呼ぶ。log_every==0 なら抑制。"""
        if self.log_every <= 0:
            return
        if run_idx % self.log_every != 0 and run_idx != total_runs:
            return
        msg = (f"[{run_idx}/{total_runs}] seed={seed} | " +
               " | ".join(f"{k}={self._fmt(v)}" for k, v in metrics.items()))
        print(msg)

    @staticmethod
    def _fmt(v: Any) -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "nan"
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)
        except Exception:
            return str(v)

    # ---------------------------
    # 保存系
    # ---------------------------
    def save_per_run(
        self,
        seeds: Sequence[int],
        cost_list: Sequence[float],
        test_vhat_list: Sequence[float],
        train_true_list: Sequence[float],
        train_vhat_list: Sequence[float],
        r2_test_list: Sequence[float], 
        Te_list: Sequence[int],
        fail_list: Sequence[float],
        run_times: Sequence[float],
        r2_train_list: Optional[Sequence[float]] = None,
    ) -> str:
        self._ensure_outdir()
        base = self._basepath()
        run_csv = f"{base}_runs.csv"

        data = {
            "seed": list(map(int, seeds)),
            "mean_cost_test_vtrue": cost_list,
            "mean_cost_test_vhat":  test_vhat_list,
            "train_cost_vtrue":     train_true_list,
            "train_cost_vhat":      train_vhat_list,
            "mean_r2_test":         r2_test_list,       # ★ここを新名に
            "eval_rows":            Te_list,
            "fail_rate":            fail_list,
            "runtime_sec":          run_times,
        }
        if r2_train_list is not None:
            data["mean_r2_train"] = r2_train_list      # ★あれば追加

        pd.DataFrame(data).to_csv(run_csv, index=False)
        return run_csv

    def save_summary(self, summary_row: Dict[str, Any]) -> str:
        self._ensure_outdir()
        base = self._basepath()
        summary_csv = f"{base}_summary.csv"
        pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)
        return summary_csv

    def append_cumulative_log(self, summary_row: Dict[str, Any]) -> str:
        self._ensure_outdir()
        # モデル・ソルバーごとに累積ログ。なければ作成、あれば追記。
        model = str(summary_row.get("model", "unknown"))
        solver = str(summary_row.get("solver", "unknown"))
        log_path = self.outdir / f"{model}_{solver}_summary_log.csv"
        header_needed = not log_path.exists()
        pd.DataFrame([summary_row]).to_csv(log_path, mode="a", header=header_needed, index=False)
        return str(log_path)

    # ---------------------------
    # 可視化
    # ---------------------------
    def plot_from_run_csv(self, run_csv: str) -> Optional[str]:
        """per-run CSV から簡易プロットを1枚保存してパスを返す。matplotlibが無ければ何もしない。"""
        if plt is None:
            print("[hooks] matplotlib が見つからないためプロットをスキップしました。")
            return None

        df = pd.read_csv(run_csv)

        base = self._basepath()
        fig_path = f"{base}_quickplot.png"

        # 1枚に3系列（例）
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        x = np.arange(1, len(df) + 1)

        if "mean_cost_test_vtrue" in df:
            ax.plot(x, df["mean_cost_test_vtrue"], label="test cost (Vtrue)")
        if "mean_cost_test_vhat" in df:
            ax.plot(x, df["mean_cost_test_vhat"], label="test cost (Vhat)")
        if "mean_r2_test" in df:
            ax.plot(x, df["mean_r2_test"], label="R^2 test")
        if "mean_r2_train" in df:
            ax.plot(x, df["mean_r2_train"], label="R^2 train", linestyle="--")

        ax.set_xlabel("run index")
        ax.set_title(self.tag)
        ax.grid(True, alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        print(f"[hooks] quick plot saved: {fig_path}")
        return fig_path