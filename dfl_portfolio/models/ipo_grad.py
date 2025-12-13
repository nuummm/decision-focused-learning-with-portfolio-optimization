from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from qpth.qp import QPFunction
import time

NDArray = Any


def fit_ipo_grad(
    X: NDArray,
    Y: NDArray,
    Vhats: Sequence[NDArray],
    idx: Sequence[int],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    *,
    delta: float = 1.0,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 0,
    qp_max_iter: int = 5000,
    qp_tol: float = 1e-6,
    seed: Optional[int] = None,
    theta_init: Optional[NDArray] = None,
    tee: bool = False,
    debug_kkt: bool = False,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, List[int], Dict[str, Any]]:
    """IPO-GRAD (IPO-NN) trainer using a differentiable QP layer.

    This trainer optimizes theta by minimizing the realized MVO cost
    under the objective

        c(z, y; delta) = -(1-delta) z^T y + (delta/2) z^T V z

    with constraints 1^T z = 1 and z >= 0.
    """

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, d = X.shape
    if Y.shape != (N, d):
        raise ValueError("fit_ipo_grad: X,Y shape mismatch")

    # Determine training range
    s = 0 if start_index is None else max(0, int(start_index))
    e = N - 1 if end_index is None else min(N - 1, int(end_index))
    if s > e:
        raise ValueError("fit_ipo_grad: invalid train range")

    pairs: List[Tuple[int, np.ndarray]] = [
        (int(i), np.asarray(V, float)) for i, V in zip(idx, Vhats) if s <= int(i) <= e
    ]
    if not pairs:
        theta_hat = np.zeros(d, dtype=float)
        Z = np.empty((0, d), dtype=float)
        MU = np.empty((0,), dtype=float)
        LAM = np.empty((0, d), dtype=float)
        meta: Dict[str, Any] = {
            "solver": "ipo_grad",
            "status": "skipped",
            "message": "no pairs in range",
        }
        return theta_hat, Z, MU, LAM, [], meta

    used_idx = [i for (i, _) in pairs]
    X_train = X[used_idx]
    Y_train = Y[used_idx]
    V_train = np.stack([v for (_, v) in pairs], axis=0)

    device = torch.device("cpu")
    if seed is not None:
        seed_int = int(seed)
        torch.manual_seed(seed_int)
        np.random.seed(seed_int)
    x_t = torch.from_numpy(X_train).to(device=device, dtype=torch.float32)
    y_t = torch.from_numpy(Y_train).to(device=device, dtype=torch.float32)
    V_t = torch.from_numpy(V_train).to(device=device, dtype=torch.float32)

    # Initialize theta. If a warm-start (e.g., IPO closed-form) is provided,
    # use it; otherwise fall back to zeros.
    if theta_init is not None:
        theta_arr = np.asarray(theta_init, float).reshape(-1)
        if theta_arr.shape[0] != d:
            raise ValueError(f"fit_ipo_grad: theta_init dim mismatch (got {theta_arr.shape[0]}, expected {d})")
        theta_t0 = torch.from_numpy(theta_arr).to(device=device, dtype=torch.float32)
    else:
        theta_t0 = torch.zeros(d, device=device, dtype=torch.float32)
    theta = torch.nn.Parameter(theta_t0)
    opt = torch.optim.Adam([theta], lr=lr)

    n_train = x_t.shape[0]
    if batch_size <= 0 or batch_size > n_train:
        batch_size_eff = n_train
    else:
        batch_size_eff = int(batch_size)
    alpha = 1.0 - float(delta)
    if not (0.0 < delta < 1.0):
        raise ValueError(f"fit_ipo_grad: delta must be in (0,1); got {delta}")

    eye_d = torch.eye(d, device=device, dtype=torch.float32).unsqueeze(0)
    one = torch.ones(1, d, device=device, dtype=torch.float32)

    # Early stopping settings
    patience = 50
    min_delta = 1e-5
    best_loss = float("inf")
    best_theta = theta.detach().clone()
    no_improve = 0
    global_eq_viol = 0.0
    global_ineq_viol = 0.0

    for epoch in range(int(epochs)):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0
        max_eq_viol = 0.0
        max_ineq_viol = 0.0

        for start in range(0, n_train, batch_size_eff):
            idx_b = perm[start : start + batch_size_eff]
            xb = x_t[idx_b]  # (B,d)
            yb = y_t[idx_b]  # (B,d)
            Vb = V_t[idx_b]  # (B,d,d)

            opt.zero_grad()

            yhat = xb * theta  # (B,d)
            Q = delta * Vb
            # 数値安定性のため少し大きめのリッジを付与する
            eps_q = 1e-5
            Q = Q + eps_q * eye_d.expand_as(Q)

            p = -alpha * yhat  # (B,d)
            G = -eye_d.expand(Q.size(0), -1, -1)  # (B,d,d)
            h = torch.zeros(Q.size(0), d, device=device, dtype=torch.float32)
            A = one.expand(Q.size(0), -1, -1)  # (B,1,d)
            b_eq = torch.ones(Q.size(0), 1, device=device, dtype=torch.float32)

            z_star = QPFunction(
                maxIter=int(qp_max_iter),
                eps=float(qp_tol),
                # verbose=-1 で qpth 内部の警告メッセージのみ抑制し、
                # Q が非 SPD などの致命的エラーはそのまま例外として受け取る。
                verbose=-1,
            )(Q, p, G, h, A, b_eq)  # (B,d)

            ret = (z_star * yb).sum(dim=1)
            risk = torch.einsum("bi,bij,bj->b", z_star, Vb, z_star)
            loss_i = -alpha * ret + 0.5 * delta * risk
            loss = loss_i.mean()

            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1
            if debug_kkt:
                eq_viol = (z_star.sum(dim=1) - 1.0).abs().max().item()
                ineq_viol = torch.clamp(-z_star, min=0.0).max().item()
                if eq_viol > max_eq_viol:
                    max_eq_viol = float(eq_viol)
                if ineq_viol > max_ineq_viol:
                    max_ineq_viol = float(ineq_viol)

        if n_batches > 0:
            avg_loss = total_loss / float(n_batches)
            # update global constraint violation stats
            if max_eq_viol > global_eq_viol:
                global_eq_viol = max_eq_viol
            if max_ineq_viol > global_ineq_viol:
                global_ineq_viol = max_ineq_viol
            # update early-stopping statistics
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                best_theta = theta.detach().clone()
                no_improve = 0
            else:
                no_improve += 1
            if debug_kkt:
                print(
                    f"[IPO-GRAD] epoch={epoch+1}/{epochs} "
                    f"loss={avg_loss:.6f} "
                    f"eq_viol={max_eq_viol:.2e} "
                    f"ineq_viol={max_ineq_viol:.2e}"
                )
            elif tee:
                print(f"[IPO-GRAD] epoch={epoch+1}/{epochs} loss={avg_loss:.6f}")
            if no_improve >= patience:
                if debug_kkt or tee:
                    print(
                        f"[IPO-GRAD] early stopping at epoch {epoch+1} "
                        f"(best_loss={best_loss:.6f})"
                    )
                break

    # Restore best parameters before returning
    theta.data.copy_(best_theta)
    theta_hat = theta.detach().cpu().numpy().astype(float)
    Z = np.empty((0, d), dtype=float)
    MU = np.empty((0,), dtype=float)
    LAM = np.empty((0, d), dtype=float)

    meta = {
        "solver": "ipo_grad",
        "status": "completed",
        "epochs": int(epochs),
        "loss_best": float(best_loss),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "qp_max_iter": int(qp_max_iter),
        "qp_tol": float(qp_tol),
        "seed": int(seed) if seed is not None else None,
        "eq_viol_max": float(global_eq_viol),
        "ineq_viol_max": float(global_ineq_viol),
        "debug_kkt": bool(debug_kkt),
    }
    return theta_hat, Z, MU, LAM, used_idx, meta
