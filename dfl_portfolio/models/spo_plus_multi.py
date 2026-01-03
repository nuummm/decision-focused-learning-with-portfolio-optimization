from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


def _sym_psd(V: np.ndarray, psd_eps: float) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    V = 0.5 * (V + V.T)
    if psd_eps > 0:
        V = V + float(psd_eps) * np.eye(V.shape[0], dtype=float)
    return V


def _oracle_simplex(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(-1)
    d = int(u.shape[0])
    if d == 0:
        return np.zeros(0, dtype=float)
    j = int(np.nanargmin(u))
    z = np.zeros(d, dtype=float)
    z[j] = 1.0
    return z


def _oracle_risk_qcp_gurobi(
    u: np.ndarray,
    V: np.ndarray,
    kappa: float,
    *,
    output: bool = False,
    time_limit: Optional[float] = None,
    numeric_focus: int = 1,
    psd_eps: float = 1e-9,
) -> Optional[np.ndarray]:
    if gp is None or GRB is None:
        raise ImportError("gurobipy is required for SPO+ risk-constrained oracle.")
    u = np.asarray(u, dtype=float).reshape(-1)
    d = int(u.shape[0])
    if d == 0:
        return np.zeros(0, dtype=float)
    V = _sym_psd(V, psd_eps=psd_eps)
    if V.shape != (d, d):
        raise ValueError(f"SPO+ oracle: V shape mismatch (got {V.shape}, expected {(d, d)})")
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa <= 0:
        return None

    m = gp.Model()
    m.Params.OutputFlag = 1 if output else 0
    m.Params.NumericFocus = int(numeric_focus)
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    z_vars = m.addVars(d, lb=0.0, name="z")
    m.addConstr(gp.quicksum(z_vars[i] for i in range(d)) == 1.0, name="budget")
    risk_expr = gp.QuadExpr()
    for i in range(d):
        risk_expr += float(V[i, i]) * z_vars[i] * z_vars[i]
        for j in range(i + 1, d):
            vij = float(V[i, j])
            if vij != 0.0:
                risk_expr += 2.0 * vij * z_vars[i] * z_vars[j]
    m.addQConstr(risk_expr, GRB.LESS_EQUAL, kappa, name="risk")
    obj = gp.LinExpr()
    for i in range(d):
        ui = float(u[i])
        if ui != 0.0:
            obj += ui * z_vars[i]
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        return np.asarray([z_vars[i].X for i in range(d)], dtype=float).reshape(-1)
    return None


def _solve_min_var_simplex(
    V: np.ndarray,
    *,
    output: bool = False,
    time_limit: Optional[float] = None,
    psd_eps: float = 1e-9,
) -> Optional[np.ndarray]:
    if gp is None or GRB is None:
        raise ImportError("gurobipy is required for SPO+ risk budget calibration.")
    V = np.asarray(V, dtype=float)
    d = int(V.shape[0])
    if d == 0:
        return np.zeros(0, dtype=float)
    V = _sym_psd(V, psd_eps=psd_eps)
    if V.shape != (d, d):
        raise ValueError(f"SPO+ min-var: V shape mismatch (got {V.shape}, expected {(d, d)})")

    m = gp.Model()
    m.Params.OutputFlag = 1 if output else 0
    m.Params.NumericFocus = 1
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    z_vars = m.addVars(d, lb=0.0, name="z")
    m.addConstr(gp.quicksum(z_vars[i] for i in range(d)) == 1.0, name="budget")
    obj = gp.QuadExpr()
    for i in range(d):
        obj += float(V[i, i]) * z_vars[i] * z_vars[i]
        for j in range(i + 1, d):
            vij = float(V[i, j])
            if vij != 0.0:
                obj += 2.0 * vij * z_vars[i] * z_vars[j]
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        return np.asarray([z_vars[i].X for i in range(d)], dtype=float).reshape(-1)
    return None


def _coerce_theta_matrix(theta: Optional[np.ndarray], d: int, k: int) -> np.ndarray:
    if theta is None:
        return np.zeros((d, k), dtype=float)
    arr = np.asarray(theta, dtype=float)
    if arr.ndim == 2 and arr.shape == (d, k):
        return arr
    flat = arr.reshape(-1)
    if flat.shape[0] != d * k:
        raise ValueError(f"theta shape mismatch: got {arr.shape}, expected {(d, k)} or flat length {d*k}")
    return flat.reshape(d, k)


def fit_spo_plus_multi(
    X_feat: Any,
    Y: Any,
    Vhats: Sequence[Any],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    delta: float = 1.0,  # kept for interface compatibility (not used)
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 0,
    lambda_reg: float = 0.0,
    lambda_anchor: float = 0.0,
    theta_anchor: Optional[np.ndarray] = None,
    risk_constraint: bool = True,
    risk_mult: float = 2.0,
    psd_eps: float = 1e-9,
    tee: bool = False,
    theta_init: Optional[np.ndarray] = None,
) -> Tuple[Any, Any, Any, Any, List[int], Dict[str, Any]]:
    """SPO+ trainer (multi-feature version)."""
    X_feat = np.asarray(X_feat, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X_feat.ndim != 3:
        raise ValueError(f"fit_spo_plus_multi: X_feat must be (N,d,K), got {X_feat.shape}")
    if Y.ndim != 2:
        raise ValueError(f"fit_spo_plus_multi: Y must be (N,d), got {Y.shape}")
    N, d, k = X_feat.shape
    if Y.shape != (N, d):
        raise ValueError("fit_spo_plus_multi: X_feat,Y shape mismatch")

    s = 0 if start_index is None else max(0, int(start_index))
    e = N - 1 if end_index is None else min(N - 1, int(end_index))
    if s > e:
        raise ValueError("fit_spo_plus_multi: invalid train range")

    pairs: List[Tuple[int, np.ndarray]] = [
        (int(i), np.asarray(V, float)) for i, V in zip(idx, Vhats) if s <= int(i) <= e
    ]
    if not pairs:
        theta_hat = np.zeros((d, k), dtype=float)
        Z = np.empty((0, d), dtype=float)
        MU = np.empty((0,), dtype=float)
        LAM = np.empty((0, d), dtype=float)
        meta: Dict[str, Any] = {"solver": "spo_plus_multi", "status": "skipped", "message": "no pairs in range"}
        return theta_hat, Z, MU, LAM, [], meta

    used_idx = [i for (i, _) in pairs]
    X_train = X_feat[used_idx]
    Y_train = Y[used_idx]

    rebalance_idx = e + 1
    V_lookup = {int(i): np.asarray(V, float) for i, V in zip(idx, Vhats)}
    if rebalance_idx not in V_lookup:
        rebalance_idx = int(used_idx[-1])
    V_ref = _sym_psd(V_lookup[int(rebalance_idx)], psd_eps=float(psd_eps))

    kappa = float("nan")
    risk_constraint_local = bool(risk_constraint)
    if risk_constraint_local:
        z_minvar = _solve_min_var_simplex(V_ref, output=False, psd_eps=float(psd_eps))
        if z_minvar is None:
            if tee:
                print("[SPO+-MULTI] min-variance calibration failed; disabling risk constraint for this window.")
            risk_constraint_local = False
        else:
            R_min = float(z_minvar @ V_ref @ z_minvar)
            kappa = float(risk_mult) * R_min
            if not np.isfinite(kappa) or kappa <= 0:
                if tee:
                    print("[SPO+-MULTI] invalid kappa; disabling risk constraint for this window.")
                risk_constraint_local = False

    device = torch.device("cpu")
    x_t = torch.from_numpy(X_train).to(device=device, dtype=torch.float32)  # (B,d,K)
    theta0 = torch.from_numpy(_coerce_theta_matrix(theta_init, d, k)).to(device=device, dtype=torch.float32)
    theta = torch.nn.Parameter(theta0.clone())

    theta_anchor_vec = _coerce_theta_matrix(theta_anchor, d, k)
    theta_anchor_t = torch.from_numpy(theta_anchor_vec).to(device=device, dtype=torch.float32)
    opt = torch.optim.Adam([theta], lr=float(lr))

    n_train = int(x_t.shape[0])
    if batch_size <= 0 or batch_size > n_train:
        batch_size_eff = n_train
    else:
        batch_size_eff = int(batch_size)

    c_true_np = (-Y_train).astype(float)
    w_true_np = np.zeros_like(c_true_np)
    oracle_fail_true = 0
    for row, u in enumerate(c_true_np):
        if risk_constraint_local:
            sol = _oracle_risk_qcp_gurobi(u, V_ref, kappa, output=False, psd_eps=float(psd_eps))
        else:
            sol = _oracle_simplex(u)
        if sol is None:
            oracle_fail_true += 1
            sol = _oracle_simplex(u)
        w_true_np[row] = sol
    c_true_t = torch.from_numpy(c_true_np).to(device=device, dtype=torch.float32)
    w_true_t = torch.from_numpy(w_true_np).to(device=device, dtype=torch.float32)

    patience = 50
    min_delta = 1e-5
    best_loss = float("inf")
    best_theta = theta.detach().clone()
    no_improve = 0

    oracle_fail_tilde = 0
    oracle_fallback_tilde = 0

    for epoch in range(int(epochs)):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size_eff):
            idx_b = perm[start : start + batch_size_eff]
            xb = x_t[idx_b]  # (B,d,K)
            cb = c_true_t[idx_b]
            wtb = w_true_t[idx_b]

            opt.zero_grad()
            yhat = (xb * theta.unsqueeze(0)).sum(dim=2)  # (B,d)
            c_hat = -yhat
            u_tilde = 2.0 * c_hat - cb

            u_tilde_np = u_tilde.detach().cpu().numpy().astype(float)
            w_tilde_np = np.zeros_like(u_tilde_np)
            for r, u_vec in enumerate(u_tilde_np):
                if risk_constraint_local:
                    sol = _oracle_risk_qcp_gurobi(u_vec, V_ref, kappa, output=False, psd_eps=float(psd_eps))
                else:
                    sol = _oracle_simplex(u_vec)
                if sol is None:
                    oracle_fail_tilde += 1
                    sol = _oracle_simplex(u_vec)
                if not np.all(np.isfinite(sol)) or sol.shape[0] != d:
                    oracle_fallback_tilde += 1
                    sol = _oracle_simplex(u_vec)
                w_tilde_np[r] = sol
            w_tilde = torch.from_numpy(w_tilde_np).to(device=device, dtype=torch.float32)

            term1 = ((cb - 2.0 * c_hat) * w_tilde).sum(dim=1)
            term2 = (2.0 * c_hat * wtb).sum(dim=1)
            term3 = (cb * wtb).sum(dim=1)
            loss = (term1 + term2 - term3).mean()
            if float(lambda_reg) > 0:
                loss = loss + float(lambda_reg) * (theta**2).sum()
            if float(lambda_anchor) > 0:
                loss = loss + float(lambda_anchor) * torch.sum((theta - theta_anchor_t) ** 2)

            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1

        if n_batches > 0:
            avg_loss = total_loss / float(n_batches)
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                best_theta = theta.detach().clone()
                no_improve = 0
            else:
                no_improve += 1
            if tee:
                rc = "on" if risk_constraint_local else "off"
                print(f"[SPO+-MULTI] epoch={epoch+1}/{epochs} loss={avg_loss:.6f} risk={rc}")
            if no_improve >= patience:
                if tee:
                    print(f"[SPO+-MULTI] early stopping at epoch {epoch+1} (best_loss={best_loss:.6f})")
                break

    theta.data.copy_(best_theta)
    theta_hat = theta.detach().cpu().numpy().astype(float)
    Z = np.empty((0, d), dtype=float)
    MU = np.empty((0,), dtype=float)
    LAM = np.empty((0, d), dtype=float)

    meta = {
        "solver": "spo_plus_multi",
        "status": "completed",
        "epochs": int(epochs),
        "loss_best": float(best_loss),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "lambda_reg": float(lambda_reg),
        "lambda_anchor": float(lambda_anchor),
        "risk_constraint": bool(risk_constraint_local),
        "risk_mult": float(risk_mult),
        "kappa": float(kappa) if np.isfinite(kappa) else None,
        "rebalance_idx": int(rebalance_idx),
        "oracle_fail_true": int(oracle_fail_true),
        "oracle_fail_tilde": int(oracle_fail_tilde),
        "oracle_fallback_tilde": int(oracle_fallback_tilde),
        "d": int(d),
        "k": int(k),
    }
    return theta_hat, Z, MU, LAM, used_idx, meta


__all__ = ["fit_spo_plus_multi"]

