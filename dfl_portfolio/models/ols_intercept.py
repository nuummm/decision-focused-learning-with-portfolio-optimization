from __future__ import annotations

import numpy as np


def train_ols_intercept(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per-asset OLS with intercept.

    Model (for each asset j):
        y_{t,j} = alpha_j + beta_j * x_{t,j} + noise

    Parameters
    ----------
    X_train : (N_train, d) ndarray
        Feature matrix (e.g., momentum) per time and asset.
    Y_train : (N_train, d) ndarray
        Target returns per time and asset.
    eps : float
        Numerical stabilizer added to the variance of x.

    Returns
    -------
    theta_hat : (d, 2) ndarray
        Stacked parameters per asset: [alpha_j, beta_j].
    """
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D (N,d), got shape {X_train.shape}")
    if Y_train.ndim != 2:
        raise ValueError(f"Y_train must be 2D (N,d), got shape {Y_train.shape}")
    if X_train.shape != Y_train.shape:
        raise ValueError(f"X_train,Y_train shape mismatch: {X_train.shape} vs {Y_train.shape}")

    x_mean = X_train.mean(axis=0)
    y_mean = Y_train.mean(axis=0)
    x_centered = X_train - x_mean
    y_centered = Y_train - y_mean

    var_x = (x_centered * x_centered).sum(axis=0) + float(eps)
    cov_xy = (x_centered * y_centered).sum(axis=0)
    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean

    return np.stack([alpha, beta], axis=-1)


def predict_yhat_intercept(X: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    """
    Predict returns using per-asset intercept OLS parameters.

    ŷ_{t,j} = alpha_j + beta_j * x_{t,j}

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    theta_hat : (d, 2) ndarray
        Parameters per asset: [alpha, beta].

    Returns
    -------
    Y_hat : (N, d) ndarray
        Predicted returns.
    """
    X = np.asarray(X, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,d), got shape {X.shape}")
    if theta_hat.ndim != 2 or theta_hat.shape[1] != 2:
        raise ValueError(f"theta_hat must be 2D (d,2), got shape {theta_hat.shape}")
    if X.shape[1] != theta_hat.shape[0]:
        raise ValueError(
            f"X column count and theta_hat asset count mismatch: {X.shape[1]} vs {theta_hat.shape[0]}"
        )

    alpha = theta_hat[:, 0]
    beta = theta_hat[:, 1]
    return alpha[None, :] + X * beta[None, :]


__all__ = ["train_ols_intercept", "predict_yhat_intercept"]

