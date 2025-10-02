# ~/VScode/GraduationResearch/DFL_Portfolio_Optimization2/optimization/solvers.py
from __future__ import annotations
import os
import pyomo.environ as pyo

def _make_gurobi(model, tee: bool, options: dict | None):
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(model)
    # 代表的な既定値
    opt.set_gurobi_param('OutputFlag', 1 if tee else 0)
    if options:
        for k, v in options.items():
            opt.set_gurobi_param(k, v)
    # 非凸QCQPを許可
    opt.set_gurobi_param('NonConvex', 2)
    opt.set_gurobi_param('TimeLimit', 300)
    opt.set_gurobi_param('Threads', 8)             # 実コア数に合わせる
    # 数値安定寄り（必要に応じて）
    opt.set_gurobi_param('BarQCPConvTol', 1e-9)    # QCP収束許容
    opt.set_gurobi_param('FeasibilityTol', 1e-8)
    opt.set_gurobi_param('OptimalityTol', 1e-8)
    opt.set_gurobi_param('Presolve', 2)
    opt.set_gurobi_param('NumericFocus', 1)        # 1〜3で調整。まずは1から
    # 連続問題なら barrier を優先したい場合（任意）
    # opt.set_gurobi_param('Method', 2)            # 2=Barrier
    # opt.set_gurobi_param('Crossover', 0)         # 連続で内点解のまま出す場合
    return opt

def _make_ipopt(model, tee: bool, options: dict | None):
    opt = pyo.SolverFactory('ipopt')
    # 代表的な既定値（必要なら上書き）
    base = {
        "tol": 1e-6,
        "acceptable_tol": 1e-5,
        "max_iter": 5000,
        "linear_solver": "mumps",
        "mu_strategy": "adaptive",
        "line_search_method": "filter",
        "nlp_scaling_method": "gradient-based",
        "bound_push": 1e-6,
        "bound_relax_factor": 1e-8,
        "compl_inf_tol": 1e-6,
        "hessian_approximation": "limited-memory",
        "warm_start_init_point": "yes",
    }
    if options:
        base.update(options)
    opt.options.update(base)
    return opt

def _make_knitro(model, tee: bool, options: dict | None):
    exe = os.environ.get("KNITROAMPL", "/Users/kensei/knitro/knitroampl/knitroampl")
    opt = pyo.SolverFactory('asl')
    opt.set_executable(exe)
    opt.options['solver'] = 'knitro'

    base = {
        "outlev": 4 if tee else 2,
        # "algorithm": 1,  # ← 非推奨。nlp_algorithm を使うのでコメントアウト推奨
        "maxtime_real": 300,
        "maxit": 3000,
    }

    KNOWN = {
        # 出力・制御
        "outlev","outmode","outdir","outname",
        # 時間・反復
        "maxit","maxtime_real","maxtime",
        # アルゴリズム選択
        "nlp_algorithm",  # 0..6
        # 並列・スレッド
        "numthreads","ms_numthreads",
        # マルチスタート
        "ms_enable","ms_maxsolves","ms_terminate","ms_seed","ms_deterministic",
        "ms_initpt_cluster","ms_sub_maxtime","ms_maxtime_real",
        # 非凸QCQP初期化
        "ncvx_qcqp_init",
        # ヘッセ関連
        "hessopt","lmsize","hessian_no_f",
        # スケーリング等
        "scale","scale_vars",
        # 線形ソルバ（必要なら）
        "linsolver","linsolver_numthreads",
        # 初期点
        "bar_initpt","initpt_strategy",
        # 収束許容
        "feastol","feastol_abs","opttol","opttol_abs",
    }

    if options:
        unknown = set(options) - KNOWN
        if unknown:
            raise ValueError(f"Unknown Knitro options: {unknown}")
        base.update(options)

    opt.options.update(base)
    return opt

def make_pyomo_solver(model, solver: str, tee: bool = False, options: dict | None = None):
    """
    Parameters
    ----------
    model : ConcreteModel
    solver : {"gurobi","ipopt","knitro"}
    tee : bool
    options : dict | None
        - gurobi: Gurobi パラメータ名→値（NonConvex=2 は自動付与）
        - ipopt : Ipopt オプション名→値
        - knitro: knitroampl オプション名→値（'solver'は自動設定）

    Returns
    -------
    opt : Pyomo solver object
    """
    s = solver.lower()
    if s in ("gurobi", "gurobi_persistent"):
        return _make_gurobi(model, tee, options)
    elif s in ("ipopt",):
        return _make_ipopt(model, tee, options)
    elif s in ("knitro", "knitroampl", "asl"):
        return _make_knitro(model, tee, options)
    else:
        raise ValueError(f"Unknown solver: {solver}")