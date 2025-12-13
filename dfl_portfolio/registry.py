from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Tuple, List

from dfl_portfolio.models.dfl_p1_flex import fit_dfl_p1_flex as _fit_flex, SolverMeta as _FlexMeta
from dfl_portfolio.models.ols_baseline import fit_ols_baseline
from dfl_portfolio.models.ipo_closed_form import fit_ipo_closed_form
from dfl_portfolio.models.ipo_grad import fit_ipo_grad
from dfl_portfolio.models.spo_plus import fit_spo_plus

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
    # outmode=0 suppresses knitro.log/summary files from being created
    "outmode": 0,
    "nlp_algorithm": 3,     # Interior/Direct
    "hessopt": 4,           # BFGS
    "linsolver": 3,         # MA57 (sparse symmetric)
    "scale": 1,
    "presolve": 1,
    "feastol": 1e-6,
    "opttol": 1e-6,
    "xtol": 1e-10,
    "honorbnds": 1,
    "maxit": 20000,
    "maxtime_real": 180,
    "numthreads": 1,
    "ms_enable": 0,
    "bar_initmu": 0.1,
    "bar_murule": 4,
}

GUROBI_DEFAULTS = {
    "NonConvex": 2,
    "Threads": 1,
    "Seed": 0,
    "TimeLimit": 300,
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


def _merge_defaults(spec: SolverSpec) -> SolverSpec:
    """spec.options が空ならソルバーごとのデフォルトを自動で付与"""
    if spec.options:
        return spec
    name = (spec.name or "").lower()
    if name == "knitro":
        return SolverSpec(name=spec.name, options=dict(KNITRO_DEFAULTS), tee=spec.tee)
    if name == "gurobi":
        return SolverSpec(name=spec.name, options=dict(GUROBI_DEFAULTS), tee=spec.tee)
    return spec

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

        fit_kwargs: Dict[str, Any] = dict(
            start_index=start_index,
            end_index=end_index,
            delta=delta,
            solver=use_solver,
            solver_options=use_opts,
            tee=local_tee,
        )
        # theta_init が None の場合は渡さない（内部で shape を参照してクラッシュするため）
        if theta_init is not None:
            fit_kwargs["theta_init"] = theta_init
        fit_kwargs.update(kw)

        ret = _fit_flex(
            X, Y, Vhats, idx,
            **fit_kwargs,
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
        # δ=0 のときは IPO の二次問題が線形問題に退化し、
        # 解析解ベースの正規方程式は成り立たない。
        # この場合は「単純にリターンをよく当てるモデル」として
        # OLS ベースラインの学習器を使う。
        if delta == 0.0:
            (
                theta_hat,
                Z,
                MU,
                LAM,
                used_idx,
                meta,
            ) = fit_ols_baseline(
                X, Y, Vhats, idx,
                start_index=start_index, end_index=end_index,
                delta=delta, theta_init=theta_init, tee=tee,
            )
            meta = dict(meta)
            meta["solver"] = "ols_baseline"
            meta["message"] = (
                "delta=0: used OLS baseline instead of IPO closed-form."
            )
            return theta_hat, Z, MU, LAM, used_idx, meta
        else:
            ret = fit_ipo_closed_form(
                X, Y, Vhats, idx,
                start_index=start_index, end_index=end_index,
                delta=delta, mode=mode, psd_eps=psd_eps,
                ridge_theta=ridge_theta, tee=tee,
            )
            return ret

    return _runner


def _wrap_ipo_grad(spec: SolverSpec) -> TrainerFn:
    # IPO-GRAD: differentiable QP layer; solver_spec is unused but interface kept
    def _runner(
        X: NDArray, Y: NDArray, Vhats: List[NDArray], idx: List[int],
        start_index: Optional[int] = None, end_index: Optional[int] = None,
        delta: float = 1.0, theta_init: Optional[NDArray] = None,
        tee: bool = False,
        solver: str = "ipo_grad",
        solver_options: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        epochs = int(kw.pop("ipo_grad_epochs", 500))
        lr = float(kw.pop("ipo_grad_lr", 1e-3))
        batch_size = int(kw.pop("ipo_grad_batch_size", 0))
        qp_max_iter = int(kw.pop("ipo_grad_qp_max_iter", 5000))
        qp_tol = float(kw.pop("ipo_grad_qp_tol", 1e-6))
        debug_kkt = bool(kw.pop("ipo_grad_debug_kkt", False))
        seed = kw.pop("ipo_grad_seed", None)
        lambda_anchor = float(kw.pop("ipo_grad_lambda_anchor", 0.0))
        theta_anchor = kw.pop("ipo_grad_theta_anchor", None)
        ret = fit_ipo_grad(
            X,
            Y,
            Vhats,
            idx,
            start_index=start_index,
            end_index=end_index,
            delta=delta,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            qp_max_iter=qp_max_iter,
            qp_tol=qp_tol,
            seed=seed,
            theta_init=theta_init,
            lambda_anchor=lambda_anchor,
            theta_anchor=theta_anchor,
            tee=tee,
            debug_kkt=debug_kkt,
        )
        return ret

    return _runner


def _wrap_spo_plus(spec: SolverSpec) -> TrainerFn:
    # SPO+: linear objective + (optional) risk-constrained simplex oracle (gurobi)
    def _runner(
        X: NDArray, Y: NDArray, Vhats: List[NDArray], idx: List[int],
        start_index: Optional[int] = None, end_index: Optional[int] = None,
        delta: float = 1.0, theta_init: Optional[NDArray] = None,
        tee: bool = False,
        solver: str = "spo_plus",
        solver_options: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        epochs = int(kw.pop("spo_plus_epochs", 500))
        lr = float(kw.pop("spo_plus_lr", 1e-3))
        batch_size = int(kw.pop("spo_plus_batch_size", 0))
        lambda_reg = float(kw.pop("spo_plus_lambda_reg", 0.0))
        risk_constraint = bool(kw.pop("spo_plus_risk_constraint", True))
        risk_mult = float(kw.pop("spo_plus_risk_mult", 2.0))
        psd_eps = float(kw.pop("spo_plus_psd_eps", 1e-9))
        kw.pop("spo_plus_init_mode", None)  # consumed upstream; ignore if present
        ret = fit_spo_plus(
            X,
            Y,
            Vhats,
            idx,
            start_index=start_index,
            end_index=end_index,
            delta=delta,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            lambda_reg=lambda_reg,
            risk_constraint=risk_constraint,
            risk_mult=risk_mult,
            psd_eps=psd_eps,
            tee=tee,
            theta_init=theta_init,
        )
        return ret

    return _runner


# レジストリ本体
def get_trainer(model_key: str, solver_spec: SolverSpec) -> TrainerFn:
    key = model_key.lower()
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
    if key in ("ipo_grad", "ipo-nn", "ipo_nn"):
        return _wrap_ipo_grad(solver_spec)
    if key in ("spo_plus", "spo+", "spoplus", "spo_plus_risk"):
        return _wrap_spo_plus(solver_spec)
    raise KeyError(f"Unknown model_key: {model_key}. Use one of: flex, ols, ipo, ipo_grad, spo_plus")

def available_models() -> Dict[str, Callable]:
    # ヘルプ/表示用：実体は使われない
    return {
        "flex": _fit_flex,
        "ols": fit_ols_baseline,
        "ipo": fit_ipo_closed_form,
        "ipo_grad": fit_ipo_grad,
        "spo_plus": fit_spo_plus,
    }
