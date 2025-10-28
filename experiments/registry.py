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
from models.dfl_p1_flex import fit_dfl_p1_flex as _fit_flex, SolverMeta as _FlexMeta
from models.ols_baseline import fit_ols_baseline
from models.ipo_closed_form import fit_ipo_closed_form
from models.ipo_nn_h import fit_ipo_nn
from models.ipo_nn_qp import fit_ipo_nn_qp

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
    "outlev": 3,
    "nlp_algorithm": 1,
    "hessopt": 2,
    "linsolver": 3,  # 3 = MA57
    "scale": 1,
    "presolve": 1,
    "feastol": 1e-8,
    "opttol": 1e-8,
    "xtol": 1e-10,
    "honorbnds": 1,
    "maxit": 20000,
    "maxtime_real": 1200,
    "numthreads": 1,
    "ms_enable": 0,
    "bar_initmu": 1e-1,
    "bar_murule": 4,
    "outmode": 1,
}
GUROBI_DEFAULTS = {
    "NonConvex": 2,
    "Threads": 1,
    "Seed": 0,
    "TimeLimit": 1200,
    "OutputFlag": 1,
    "MIPGap": 0,
    "MIPGapAbs": 0,
    "Heuristics": 0.0,
    "Cuts": 2,
    "Presolve": 2,
    "NumericFocus": 1,
    "FeasibilityTol": 1e-9,
    "OptimalityTol": 1e-9,
    "BarConvTol": 1e-12,
    "Method": 2,
}
IPOPT_DEFAULTS = {
    "tol": 1e-8,
    "acceptable_tol": 1e-6,
    "dual_inf_tol": 1e-8,
    "constr_viol_tol": 1e-8,
    "compl_inf_tol": 1e-8,
    "max_iter": 200000,
    "max_cpu_time": 1200,
    "mu_strategy": "adaptive",
    "nlp_scaling_method": "gradient-based",
    "hessian_approximation": "exact",
    "linear_solver": "ma57",
    "bound_push": 1e-12,
    "bound_frac": 1e-12,
    "warm_start_init_point": "yes",
    "print_level": 5,
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

        ret = _fit_kkt(
            X, Y, Vhats, idx,
            start_index=start_index, end_index=end_index,
            delta=delta, theta_init=theta_init, tee=local_tee,
            solver=use_solver, solver_options=use_opts,
            reg_theta_l2=reg_theta_l2,
        )
        # --- 戻り値を (.., info) 付きに正規化 ---
        if isinstance(ret, (list, tuple)):
            if len(ret) >= 6:
                return ret  # 既に info あり
            elif len(ret) == 5:
                theta_hat, Z, MU, LAM, used_idx = ret
                info = {}
                return theta_hat, Z, MU, LAM, used_idx, info
        # 想定外フォーマットの場合は例外
        raise RuntimeError("fit_kkt returned unexpected format")
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

        ret = _fit_dual(
            X, Y, Vhats, idx,
            start_index=start_index, end_index=end_index,
            delta=delta, theta_init=theta_init, tee=local_tee,
            solver=use_solver, solver_options=use_opts,
            reg_theta_l2=reg_theta_l2,
        )
        # --- 戻り値を (.., info) 付きに正規化 ---
        if isinstance(ret, (list, tuple)):
            if len(ret) >= 6:
                return ret  # 既に info あり
            elif len(ret) == 5:
                theta_hat, Z, MU, LAM, used_idx = ret
                info = {}
                return theta_hat, Z, MU, LAM, used_idx, info
        raise RuntimeError("fit_dual returned unexpected format")
    return _runner


def _wrap_flex(spec: SolverSpec) -> TrainerFn:
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

        ret = _fit_flex(
            X, Y, Vhats, idx,
            start_index=start_index, end_index=end_index,
            delta=delta, theta_init=theta_init,
            solver=use_solver, solver_options=use_opts,
            tee=local_tee,
            **kw,
        )

        if isinstance(ret, (list, tuple)) and len(ret) >= 6:
            ret_list = list(ret)
            meta = ret_list[5]
            if isinstance(meta, _FlexMeta):
                ret_list[5] = {
                    "solver": meta.solver,
                    "termination_condition": meta.termination_condition,
                    "termination_condition_str": meta.termination_condition_str,
                    "status": meta.status,
                    "status_str": meta.status_str,
                    "solver_time": meta.solver_time,
                    "message": meta.message,
                }
            return tuple(ret_list)
        raise RuntimeError("fit_flex returned unexpected format")

    return _runner

def _wrap_ipo(spec: SolverSpec) -> TrainerFn:
    # Analytic solution: ignore solver/spec options, but keep interface
    def _runner(
        X: NDArray, Y: NDArray, Vhats: List[NDArray], idx: List[int],
        start_index: Optional[int] = None, end_index: Optional[int] = None,
        delta: float = 1.0, theta_init: Optional[NDArray] = None,
        tee: bool = False,
        solver: str = "analytic",
        solver_options: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        mode = kw.pop("ipo_mode", "budget")
        psd_eps = kw.pop("ipo_psd_eps", 1e-12)
        ridge_theta = kw.pop("ipo_ridge_theta", 1e-10)
        ret = fit_ipo_closed_form(
            X, Y, Vhats, idx,
            start_index=start_index, end_index=end_index,
            delta=delta, mode=mode, psd_eps=psd_eps,
            ridge_theta=ridge_theta, tee=tee,
        )
        return ret

    return _runner


def _wrap_ipo_nn(spec: SolverSpec) -> TrainerFn:
    def _runner(
        X: NDArray, Y: NDArray, Vhats: List[NDArray], idx: List[int],
        start_index: Optional[int] = None, end_index: Optional[int] = None,
        delta: float = 1.0, theta_init: Optional[NDArray] = None,
        tee: bool = False,
        solver: str = "ipo_nn",
        solver_options: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        local_kw = dict(kw)
        local_kw.pop("reg_theta_l2", None)
        return fit_ipo_nn(
            X, Y, Vhats, idx,
            start_index=start_index, end_index=end_index,
            delta=delta,
            tee=tee,
            **local_kw,
        )

    return _runner


def _wrap_ipo_nn_qp(spec: SolverSpec) -> TrainerFn:
    def _runner(
        X: NDArray, Y: NDArray, Vhats: List[NDArray], idx: List[int],
        start_index: Optional[int] = None, end_index: Optional[int] = None,
        delta: float = 1.0, theta_init: Optional[NDArray] = None,
        tee: bool = False,
        solver: str = "ipo_nn_qp",
        solver_options: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        local_kw = dict(kw)
        local_kw.pop("reg_theta_l2", None)
        return fit_ipo_nn_qp(
            X, Y, Vhats, idx,
            start_index=start_index, end_index=end_index,
            delta=delta,
            tee=tee,
            **local_kw,
        )

    return _runner

# レジストリ本体
def get_trainer(model_key: str, solver_spec: SolverSpec) -> TrainerFn:
    key = model_key.lower()
    if key in ("kkt", "dfl_p1_kkt", "dfl-kkt", "kkt-qcqp"):
        return _wrap_kkt(solver_spec)
    if key in ("dual", "dfl_p1_dual", "dfl-dual", "dual-strong"):
        return _wrap_dual(solver_spec)
    if key in ("flex", "dfl_flex", "dfl_p1_flex"):
        return _wrap_flex(solver_spec)
    if key in ("ols", "vanilla", "baseline"):
        # OLS は Pyomo を使わないのでラッパー不要
        def _runner(
            X, Y, Vhats, idx,
            start_index=None, end_index=None,
            delta: float = 1.0, theta_init=None,
            tee: bool = False, solver: str = "", solver_options: Optional[Dict[str, Any]] = None,
            **kw,
        ):
            ret = fit_ols_baseline(
                X, Y, Vhats, idx,
                start_index=start_index, end_index=end_index,
                delta=delta, theta_init=theta_init, tee=tee
            )
            if isinstance(ret, (list, tuple)) and len(ret) == 5:
                theta_hat, Z, MU, LAM, used_idx = ret
                info = {}
                return theta_hat, Z, MU, LAM, used_idx, info
            return ret
        return _runner
    if key in ("ipo", "ipo_closed_form", "ipo-analytic"):
        return _wrap_ipo(solver_spec)
    if key in ("ipo_nn", "ipo-nn", "ipo_nn_h", "ipo_nn_linear"):
        return _wrap_ipo_nn(solver_spec)
    if key in ("ipo_nn_qp", "ipo-nn-qp"):
        return _wrap_ipo_nn_qp(solver_spec)
    raise KeyError(f"Unknown model_key: {model_key}. Use one of: kkt, dual, ols")

def available_models() -> Dict[str, Callable]:
    # ヘルプ/表示用：実体は使われない
    return {
        "kkt": _fit_kkt,
        "dual": _fit_dual,
        "flex": _fit_flex,
        "ols": fit_ols_baseline,
        "ipo": fit_ipo_closed_form,
        "ipo_nn": fit_ipo_nn,
        "ipo_nn_qp": fit_ipo_nn_qp,
        "ensemble_avg": fit_ols_baseline,
        "ensemble_weighted": fit_ols_baseline,
        "ensemble_normalized": fit_ols_baseline,
    }
