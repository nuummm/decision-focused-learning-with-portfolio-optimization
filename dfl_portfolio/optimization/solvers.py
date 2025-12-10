# ~/VScode/GraduationResearch/DFL_Portfolio_Optimization2/optimization/solvers.py
from __future__ import annotations

import os
import pyomo.environ as pyo

def _make_gurobi(model, tee: bool, options: dict | None):
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(model)
    # 代表的な既定値（後から options で上書き可）
    defaults = {
        'NonConvex': 2,
        'Threads': 1,
        'Seed': 0,
        'TimeLimit': 1200,
        'OutputFlag': 1,
        'MIPGap': 0,
        'MIPGapAbs': 0,
        'Heuristics': 0.0,
        'Cuts': 2,
        'Presolve': 2,
        'NumericFocus': 1,
        'FeasibilityTol': 1e-9,
        'OptimalityTol': 1e-9,
        'BarConvTol': 1e-12,
        'Method': 2,
    }
    for key, value in defaults.items():
        opt.set_gurobi_param(key, value)

    if options:
        for k, v in options.items():
            opt.set_gurobi_param(k, v)
    return opt

def _make_knitro(model, tee: bool, options: dict | None):
    exe = os.environ.get("KNITROAMPL", "/Users/kensei/knitro/knitroampl/knitroampl")
    opt = pyo.SolverFactory('asl')
    opt.set_executable(exe)
    opt.options['solver'] = 'knitro'

    base = {
        "outlev": 3,
        "nlp_algorithm": 1,
        "hessopt": 2,
        "linsolver": 3,  # 3 = MA57 in Knitro option coding
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
        # outmode=0 にしてファイル出力を完全に抑止する
        "outmode": 0,
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
        "scale","scale_vars","presolve",
        # 線形ソルバ（必要なら）
        "linsolver","linsolver_numthreads",
        # 初期点
        "bar_initpt","initpt_strategy",
        # 収束許容
        "feastol","feastol_abs","opttol","opttol_abs","xtol",
        # バリア関連
        "bar_initmu","bar_murule",
        # その他
        "honorbnds",
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
    elif s in ("knitro", "knitroampl", "asl"):
        return _make_knitro(model, tee, options)
    else:
        raise ValueError(f"Unknown solver: {solver}")
