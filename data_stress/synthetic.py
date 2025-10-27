"""Synthetic data with heavy tails, regime shifts, and shock events.

The helper below intentionally keeps the same function signature as the
baseline ``data.synthetic.generate_simulation1_dataset`` so that existing
experiments can be switched over by simply adjusting the import.  Internally
it adds three forms of stress commonly discussed in portfolio-optimisation
studies:

* Heavy tails – simulated via a multivariate Student-t construction.
* Regime shifts – the signal coefficients drift after a split point and the
  volatility level steps up.
* Shock events – rare observations with amplified volatility that mimic crash
  scenarios.

The goal is to offer a richer environment where decision-focused learning can
demonstrate robustness beyond mean-performance improvements.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DOF = 3.0  # degrees of freedom for heavy-tailed noise


def _toeplitz_covariance(d: int, sigma: float, rho: float, jitter: float) -> np.ndarray:
    rho = float(np.clip(rho, -0.999, 0.999))
    idx = np.arange(d)
    lags = np.abs(idx[:, None] - idx[None, :])
    cov = (sigma**2) * (rho ** lags)
    if jitter and jitter > 0:
        cov = cov + float(jitter) * np.eye(d)
    return cov


def _sample_student_t(
    rng: np.random.Generator,
    N: int,
    L: np.ndarray,
    dof: float,
) -> np.ndarray:
    """Draw ``N`` samples from a centred multivariate Student-t.

    The resulting covariance (when it exists) matches the scale matrix ``LLᵀ``.
    """

    if dof <= 2:
        raise ValueError("degrees of freedom must exceed 2 for finite variance")

    z = rng.standard_normal((N, L.shape[0]))
    scaled = z @ L.T

    chi2 = rng.chisquare(dof, size=N)
    scale = np.sqrt(dof / chi2)
    return scaled * scale[:, None]


def generate_simulation1_dataset(
    n_samples: int = 2000,
    n_assets: int = 10,
    snr: float = 0.5,
    rho: float = 0.5,
    sigma: float = 0.0125,
    seed: int | None = 42,
    rng: np.random.Generator | None = None,
    jitter: float = 1e-12,
    *,
    dof: float = DEFAULT_DOF,
    regime_split_frac: float = 0.6,
    regime_vol_scale: float = 3.0,
    theta_shift_scale: float = 0.75,
    shock_prob: float = 0.04,
    shock_scale: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate stressed synthetic data.

    Parameters are identical to :func:`data.synthetic.generate_simulation1_dataset`
    with additional keyword-only controls (defaults chosen to mimic crisis-like
    behaviour).  The returned tuple intentionally matches the baseline helper.
    """

    if rng is None:
        rng = np.random.default_rng(seed)

    N = int(n_samples)
    d = int(n_assets)

    if not (0.0 < regime_split_frac < 1.0):
        raise ValueError("regime_split_frac must lie in (0, 1)")

    regime_split = max(1, min(N - 1, int(round(regime_split_frac * N))))

    # --- signal component ---
    theta_base = rng.standard_normal(d)
    theta_shift = theta_shift_scale * rng.standard_normal(d)

    theta_series = np.tile(theta_base, (N, 1))
    theta_series[regime_split:] = theta_base + theta_shift

    X = rng.standard_normal((N, d))

    # --- heavy-tailed noise with covariance scaling ---
    V_true = _toeplitz_covariance(d, sigma, rho, jitter)

    try:
        L = np.linalg.cholesky(V_true)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Failed to factorise covariance matrix") from exc

    eps = _sample_student_t(rng, N, L, dof=dof)

    # regime-dependent volatility amplification
    if regime_vol_scale != 1.0:
        eps[regime_split:] *= regime_vol_scale

    # rare crash events – inflate volatility drastically and bias downward
    if shock_prob > 0.0 and shock_scale > 1.0:
        mask = rng.random(N) < shock_prob
        if np.any(mask):
            eps[mask] *= shock_scale
            # introduce directionality towards losses
            drop = np.abs(rng.standard_normal((mask.sum(), d)))
            eps[mask] -= shock_scale * sigma * drop

    # --- combine signal and noise ---
    tau = np.sqrt(np.mean(theta_base**2) / (snr * sigma**2))
    Y = X * theta_series + tau * eps

    return X, Y, V_true, theta_base, tau


__all__ = ["generate_simulation1_dataset"]

