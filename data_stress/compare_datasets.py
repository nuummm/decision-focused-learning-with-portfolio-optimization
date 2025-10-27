"""Utility to visualise baseline vs stressed synthetic datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
for path in (PACKAGE_ROOT, REPO_ROOT):
    p = str(path)
    if p not in sys.path:
        sys.path.append(p)

from data.synthetic import generate_simulation1_dataset as generate_baseline
from data_stress.synthetic import generate_simulation1_dataset as generate_stress


def rolling_rms(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.abs(series)
    kernel = np.ones(window) / float(window)
    padded = np.pad(series**2, (window - 1, 0), mode="constant", constant_values=np.nan)
    conv = np.convolve(np.nan_to_num(padded, nan=0.0), kernel, mode="valid")
    out = np.sqrt(conv)
    return out


def make_figure(N: int, d: int, snr: float, rho: float, sigma: float, seed: int, output: Path) -> None:
    X_base, Y_base, _, _, _ = generate_baseline(
        n_samples=N, n_assets=d, snr=snr, rho=rho, sigma=sigma, seed=seed
    )
    X_stress, Y_stress, _, _, _ = generate_stress(
        n_samples=N, n_assets=d, snr=snr, rho=rho, sigma=sigma, seed=seed
    )

    asset = 0
    time = np.arange(N)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_ts, ax_hist, ax_theta, ax_heat = axes.flatten()

    ax_ts.plot(time, Y_base[:, asset], label="baseline", alpha=0.7)
    ax_ts.plot(time, Y_stress[:, asset], label="stress", alpha=0.7)
    ax_ts.set_title(f"Asset {asset} returns")
    ax_ts.set_xlabel("t")
    ax_ts.legend()

    bins = 60
    ax_hist.hist(Y_base[:, asset], bins=bins, alpha=0.6, label="baseline")
    ax_hist.hist(Y_stress[:, asset], bins=bins, alpha=0.6, label="stress")
    ax_hist.set_title("Return distribution (asset 0)")
    ax_hist.legend()

    # RMS of residual between regimes to visualise volatility escalation
    window = max(10, N // 20)
    rms_base = rolling_rms(Y_base[:, asset], window)
    rms_stress = rolling_rms(Y_stress[:, asset], window)
    ax_theta.plot(rms_base, label="baseline RMS")
    ax_theta.plot(rms_stress, label="stress RMS")
    ax_theta.set_title(f"Rolling RMS (window={window})")
    ax_theta.legend()

    # Compare covariance heatmaps (empirical)
    cov_base = np.cov(Y_base.T)
    cov_stress = np.cov(Y_stress.T)
    vmax = np.max(np.abs([cov_base, cov_stress]))
    im0 = ax_heat.imshow(cov_stress - cov_base, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax_heat.set_title("Covariance difference (stress - baseline)")
    fig.colorbar(im0, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.suptitle("Baseline vs stress synthetic datasets")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and stress synthetic datasets")
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--snr", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.0125)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results") / "stress_vs_baseline.png")
    args = parser.parse_args()

    make_figure(args.N, args.d, args.snr, args.rho, args.sigma, args.seed, args.output)
    print(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
