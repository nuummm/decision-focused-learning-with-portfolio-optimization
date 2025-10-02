# ~/VScode/GraduationResearch/DFL_Portfolio_Optimization2/experiments/registry.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Tuple, List

# ルートからの相対 import を確実にする（直接実行にも耐える）
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# モデル本体（重複インポートを整理）
from models.dfl_p1_kkt import fit_dfl_p1_pyomo_ipopt as _fit_kkt
from models.dfl_p1_dual import fit_dfl_p1_dual_pyomo as _fit_dual
from models.ols_baseline import fit_ols_baseline

# 型エイリアス（分かりやすさ用）
NDArray = Any

@dataclass
class SolverSpec:
    """利用するソルバーと、そのオプション"""
    name: str = "knitro"                           # "gurobi" | "ipopt" | "knitro"
    options: Dict[str, Any] = field(default_factory=dict)
    tee: bool = True                               # ソルバーのログを出すか

# 統一インターフェース
# 返り値: (theta_hat, Z, MU, LAM, used_idx)
TrainerFn = Callable[
    [NDArray, NDArray, List[NDArray], List[int],
     Optional[int], Optional[int], float, Optional[NDArray], bool,
     str, Optional[Dict[str, Any]]],
    Tuple[NDArray, NDArray, NDArray, NDArray, List[int]]]

# --- デフォルトのソルバー仕様（必要に応じて run.py から上書き） ---
KNITRO_DEFAULTS = {
    "outlev": 4,          # 4 にすると詳細ログ
    "algorithm": 1,       # 1=Barrier
    "maxtime_real": 300,
    "maxit": 3000,
}
GUROBI_DEFAULTS = {
    # 例: "TimeLimit": 300, "Threads": 4,
}
IPOPT_DEFAULTS = {
    # 例: "tol": 1e-6, "max_iter": 5000,
}

def _merge_defaults(spec: SolverSpec) -> SolverSpec:
    """spec.options が空ならソルバーごとのデフォルトを自動で付与"""
    if spec.options:
        return spec
    name = (spec.name or "").lower()
    if name == "knitro":
        return SolverSpec(name=spec.name, options=dict(KNITRO_DEFAULTS), tee=spec.tee)
    if name == "gurobi":
        return SolverSpec(name=spec.name, options=dict(GUROBI_DEFAULTS), tee=spec.tee)
    if name == "ipopt":
        return SolverSpec(name=spec.name, options=dict(IPOPT_DEFAULTS), tee=spec.tee)
    return spec

def _wrap_kkt(spec: SolverSpec) -> TrainerFn:
    spec = _merge_defaults(spec)
    def _runner(
        X: NDArray, Y: NDArray, Vhats: List[NDArray], idx: List[int],
        start_index: Optional[int] = None, end_index: Optional[int] = None,
        delta: float = 1.0, theta_init: Optional[NDArray] = None,
        tee: bool = False,                       # 実行時の表示（モデル側）
        solver: str = "",                        # 呼び出し側で上書き可
        solver_options: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        use_solver = (solver or spec.name)
        use_opts = dict(spec.options)
        if solver_options:
            use_opts.update(solver_options)
        local_tee = bool(tee or spec.tee)
        reg_theta_l2 = float(kw.get("reg_theta_l2", 0.0))

        return _fit_kkt(
            X, Y, Vhats, idx,
            start_index=start_index, end_index=end_index,
            delta=delta, theta_init=theta_init, tee=local_tee,
            solver=use_solver, solver_options=use_opts,
            reg_theta_l2=reg_theta_l2,  
        )
    return _runner

def _wrap_dual(spec: SolverSpec) -> TrainerFn:
    spec = _merge_defaults(spec)
    def _runner(
        X: NDArray, Y: NDArray, Vhats: List[NDArray], idx: List[int],
        start_index: Optional[int] = None, end_index: Optional[int] = None,
        delta: float = 1.0, theta_init: Optional[NDArray] = None,
        tee: bool = False,
        solver: str = "",
        solver_options: Optional[Dict[str, Any]] = None,
        **kw, 
    ):
        use_solver = (solver or spec.name)
        use_opts = dict(spec.options)
        if solver_options:
            use_opts.update(solver_options)
        local_tee = bool(tee or spec.tee)
        reg_theta_l2 = float(kw.get("reg_theta_l2", 0.0))

        return _fit_dual(
            X, Y, Vhats, idx,
            start_index=start_index, end_index=end_index,
            delta=delta, theta_init=theta_init, tee=local_tee,
            solver=use_solver, solver_options=use_opts,
            reg_theta_l2=reg_theta_l2, 
        )
    return _runner

# レジストリ本体
def get_trainer(model_key: str, solver_spec: SolverSpec) -> TrainerFn:
    key = model_key.lower()
    if key in ("kkt", "dfl_p1_kkt", "dfl-kkt", "kkt-qcqp"):
        return _wrap_kkt(solver_spec)
    if key in ("dual", "dfl_p1_dual", "dfl-dual", "dual-strong"):
        return _wrap_dual(solver_spec)
    if key in ("ols", "vanilla", "baseline"):
        # OLS は Pyomo を使わないのでラッパー不要
        def _runner(
            X, Y, Vhats, idx,
            start_index=None, end_index=None,
            delta: float = 1.0, theta_init=None,
            tee: bool = False, solver: str = "", solver_options: Optional[Dict[str, Any]] = None,
            **kw,
        ):
            return fit_ols_baseline(
                X, Y, Vhats, idx,
                start_index=start_index, end_index=end_index,
                delta=delta, theta_init=theta_init, tee=tee
            )
        return _runner
    raise KeyError(f"Unknown model_key: {model_key}. Use one of: kkt, dual, ols")

def available_models() -> Dict[str, Callable]:
    # ヘルプ/表示用：実体は使われない
    return {
        "kkt": _fit_kkt,
        "dual": _fit_dual,
        "ols": fit_ols_baseline,
    }