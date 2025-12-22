from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def parse_tickers(value: str) -> List[str]:
    return [t.strip().upper() for t in value.split(",") if t.strip()]


def parse_commalist(value: str) -> List[str]:
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def parse_model_train_window_spec(value: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    text = (value or "").strip()
    if not text:
        return mapping
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid model-train-window spec '{token}'. Use model:window format.")
        name, window_str = token.split(":", 1)
        model_name = name.strip().lower()
        if not model_name:
            raise ValueError(f"Missing model name in spec '{token}'.")
        try:
            window_val = int(window_str.strip())
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid train window '{window_str}' for model '{model_name}'.") from exc
        if window_val <= 0:
            raise ValueError(f"Train window for model '{model_name}' must be positive.")
        mapping[model_name] = window_val
    return mapping


def parse_trading_cost_map(value: str | None) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    text = (value or "").strip()
    if not text:
        return mapping
    for token in text.split(","):
        entry = token.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"Invalid trading-cost spec '{entry}'. Use 'TICKER:bps'.")
        ticker, val = entry.split(":", 1)
        ticker_key = ticker.strip().upper()
        if not ticker_key:
            raise ValueError(f"Missing ticker in trading-cost spec '{entry}'.")
        try:
            rate_bps = float(val.strip())
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid bps '{val}' for ticker '{ticker_key}'.") from exc
        if rate_bps < 0.0:
            raise ValueError(f"Trading cost for '{ticker_key}' must be non-negative.")
        mapping[ticker_key] = rate_bps
    return mapping


def make_output_dir(results_root: Path, base: Path | None) -> Path:
    """Create a timestamped output directory under the given results root.

    When ``base`` is provided, it is used directly instead of timestamp subdir.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = base or (results_root / timestamp)
    root.mkdir(parents=True, exist_ok=True)
    return root


def parse_delta_0_1(value: str) -> float:
    """Parse risk-aversion delta constrained to [0, 1]."""
    try:
        v = float(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"delta must be a float, got {value!r}") from exc
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(f"delta must be in [0, 1], got {v}")
    return v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-data rolling experiment runner")
    parser.add_argument("--tickers", type=str, default="SPY,GLD,EEM,TLT")
    parser.add_argument("--start", type=str, default="2006-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--price-field", type=str, default="Close")
    parser.add_argument("--return-kind", type=str, default="log", choices=["simple", "log"])
    parser.add_argument(
        "--frequency",
        type=str,
        default="weekly",
        choices=["daily", "weekly", "monthly"],
    )
    parser.add_argument("--resample-rule", type=str, default="W-FRI")
    parser.add_argument("--momentum-window", type=int, default=26)
    parser.add_argument("--return-horizon", type=int, default=1)
    parser.add_argument("--cov-window", type=int, default=13)
    parser.add_argument(
        "--cov-method",
        type=str,
        default="oas",
        choices=["diag", "oas", "robust_lw", "mini_factor"],
    )
    parser.add_argument(
        "--cov-ewma-alpha",
        type=float,
        default=0.97,
        help=(
            "EWMA の時間減衰率 α (0<α<1)。cov-method=oas のときにのみ使用され、"
            "時間重み付き共分散 S_t^{(α)} に対して OAS shrinkage を適用する。"
        ),
    )
    parser.add_argument("--cov-shrinkage", type=float, default=0.94)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    parser.add_argument("--cov-robust-huber-k", type=float, default=1.5)
    parser.add_argument("--cov-factor-rank", type=int, default=1)
    parser.add_argument("--cov-factor-shrinkage", type=float, default=0.5)
    parser.add_argument("--no-auto-adjust", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")

    parser.add_argument("--train-window", type=int, default=26)
    parser.add_argument("--rebal-interval", type=int, default=4)
    parser.add_argument(
        "--model-train-window",
        type=str,
        default="",
        help="Optional overrides e.g. 'ols:60,flex:25' to use per-model train windows.",
    )

    parser.add_argument("--delta", type=parse_delta_0_1, default=0.5)
    parser.add_argument(
        "--delta-up",
        type=str,
        default=None,
        help=(
            "Risk aversion used in the objective (default falls back to --delta). "
            "Accepts a single value in [0,1]."
        ),
    )
    parser.add_argument(
        "--delta-down",
        type=str,
        default=None,
        help=(
            "Risk aversion dedicated to DFL constraint terms. Accepts either a single "
            "value or a comma-separated grid (each in [0,1]). Defaults to delta-up."
        ),
    )
    parser.add_argument("--models", type=str, default="ols,ipo,ipo_grad,spo_plus,flex")

    # Shared theta controls (apply to both flex and IPO-GRAD when provided).
    # These options override the model-specific flags below.
    parser.add_argument(
        "--theta-init-mode",
        type=str,
        default=None,
        choices=["none", "ipo"],
        help=(
            "Shared theta initialization mode applied to both flex and IPO-GRAD when set. "
            "Overrides --flex-theta-init-mode and --ipo-grad-init-mode."
        ),
    )
    parser.add_argument(
        "--lambda-theta-anchor",
        type=float,
        default=None,
        help=(
            "Shared L2 anchor strength on theta applied to both flex and IPO-GRAD when set. "
            "Overrides --flex-lambda-theta-anchor and --ipo-grad-lambda-anchor."
        ),
    )
    parser.add_argument(
        "--theta-anchor-mode",
        type=str,
        default=None,
        choices=["ipo", "zero", "none"],
        help=(
            "Shared anchor reference for theta. 'ipo' uses IPO closed-form, 'zero' anchors to the zero vector. "
            "Applied to both flex and IPO-GRAD when set. Overrides --flex-theta-anchor-mode and "
            "--ipo-grad-theta-anchor-mode."
        ),
    )

    parser.add_argument(
        "--w-warmstart",
        "--w-warm-start",
        dest="w_warmstart",
        action="store_true",
        default=False,
        help=(
            "Enable warm-start for decision variables w when available (default: disabled). "
            "Currently affects flex warm-start initialization."
        ),
    )
    parser.add_argument(
        "--no-w-warmstart",
        "--no-w-warm-start",
        dest="w_warmstart",
        action="store_false",
        help=(
            "Disable warm-start for decision variables w, even when theta is warm-started "
            "(e.g., IPO init). This is the default."
        ),
    )

    parser.add_argument("--flex-solver", type=str, default="knitro")
    parser.add_argument(
        "--flex-formulation",
        type=str,
        default="kkt",
        help=(
            "Comma-separated flex base models to run "
            "(e.g., 'dual', 'dual,kkt', or 'dual,kkt,dual&kkt' for dual/kkt+ensemble)."
        ),
    )
    parser.add_argument("--flex-lambda-theta-anchor", type=float, default=0.0)
    parser.add_argument("--flex-lambda-theta-iso", type=float, default=0.0)
    parser.add_argument("--flex-theta-anchor-mode", type=str, default="ipo")
    parser.add_argument("--flex-theta-init-mode", type=str, default="ipo")
    parser.add_argument(
        "--flex-ensemble-weight-dual",
        type=parse_delta_0_1,
        default=0.5,
        help=(
            "Weight on flex-dual in the dual/kkt ensemble (w_dual); "
            "w_kkt = 1 - w_dual. Only used when ensemble is requested."
        ),
    )
    parser.add_argument(
        "--flex-lambda-phi-anchor",
        type=float,
        default=0.0,
        help=(
            "Additional L2 regularization strength on the covariance shrinkage "
            "coefficient phi in the V-learning flex model. "
            "Penalizes deviation of phi from 0.0 (the OAS+EWMA baseline)."
        ),
    )
    parser.add_argument(
        "--trading-cost-bps",
        type=float,
        default=1.0,
        help=(
            "Set to a positive value to enable the built-in per-asset trading cost table "
            "(values expressed in basis points). "
            "Specify --trading-cost-per-asset for custom overrides."
        ),
    )
    parser.add_argument(
        "--trading-cost-per-asset",
        type=parse_trading_cost_map,
        default={"SPY": 5.0, "GLD": 10.0, "EEM": 10.0, "TLT": 5.0},
        help="Optional overrides like 'SPY:5,GLD:8' (basis points) applied per ticker.",
    )
    parser.add_argument(
        "--ipo-grad-epochs",
        type=int,
        default=250,
        help="Number of training epochs for IPO-GRAD (IPO neural network).",
    )
    parser.add_argument(
        "--ipo-grad-lr",
        type=float,
        default=1e-3,
        help="Learning rate for IPO-GRAD (Adam).",
    )
    parser.add_argument(
        "--ipo-grad-batch-size",
        type=int,
        default=0,
        help="Batch size for IPO-GRAD (0 means full batch).",
    )
    parser.add_argument(
        "--ipo-grad-qp-max-iter",
        type=int,
        default=1500,
        help="Maximum number of iterations in the QP solver inside IPO-GRAD.",
    )
    parser.add_argument(
        "--ipo-grad-qp-tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for the QP solver inside IPO-GRAD.",
    )
    parser.add_argument(
        "--ipo-grad-init-mode",
        type=str,
        default="ipo",
        choices=["none", "ipo"],
        help="Initial theta for IPO-GRAD (none=zero init, ipo=closed-form IPO warm start).",
    )
    parser.add_argument(
        "--ipo-grad-lambda-anchor",
        type=float,
        default=0.0,
        help="L2 anchor strength on theta for IPO-GRAD (0 to disable).",
    )
    parser.add_argument(
        "--ipo-grad-theta-anchor-mode",
        type=str,
        default="ipo",
        choices=["ipo", "zero", "none"],
        help="Theta anchor for IPO-GRAD: ipo=use IPO closed-form, zero=use zero vector.",
    )
    parser.add_argument(
        "--ipo-grad-debug-kkt",
        action="store_true",
        help=(
            "Enable additional IPO-GRAD diagnostics (prints approximate "
            "constraint violations per epoch)."
        ),
    )
    parser.add_argument(
        "--ipo-grad-seed",
        type=int,
        default=0,
        help=(
            "Override the base seed used to derive IPO-GRAD's per-cycle RNG seed. "
            "When omitted, IPO-GRAD uses --base-seed like other models."
        ),
    )
    parser.add_argument(
        "--spo-plus-epochs",
        type=int,
        default=250,
        help="Number of training epochs for SPO+.",
    )
    parser.add_argument(
        "--spo-plus-lr",
        type=float,
        default=1e-3,
        help="Learning rate for SPO+ (Adam).",
    )
    parser.add_argument(
        "--spo-plus-batch-size",
        type=int,
        default=0,
        help="Batch size for SPO+ (0 means full batch).",
    )
    parser.add_argument(
        "--spo-plus-lambda-reg",
        type=float,
        default=0.0,
        help="L2 regularization strength on theta for SPO+.",
    )
    parser.add_argument(
        "--spo-plus-lambda-anchor",
        type=float,
        default=0.0,
        help="L2 anchor strength on theta for SPO+ (0 to disable).",
    )
    parser.add_argument(
        "--spo-plus-theta-anchor-mode",
        type=str,
        default="ipo",
        choices=["ipo", "zero", "none"],
        help="Theta anchor reference for SPO+: ipo=IPO closed-form, zero/none=zero vector.",
    )
    parser.add_argument(
        "--spo-plus-risk-mult",
        type=float,
        default=2.0,
        help=(
            "Risk budget multiplier for SPO+ oracle when using risk constraint: "
            "kappa = mult * min_z z^T V_S z (on the simplex)."
        ),
    )
    parser.add_argument(
        "--spo-plus-risk-constraint",
        dest="spo_plus_risk_constraint",
        action="store_true",
        default=True,
        help="Enable SPO+ risk constraint z^T V_S z <= kappa (enabled by default).",
    )
    parser.add_argument(
        "--spo-plus-no-risk-constraint",
        dest="spo_plus_risk_constraint",
        action="store_false",
        help="Disable SPO+ risk constraint (simplex-only oracle).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base seed for reproducibility (per-cycle seed is derived from this).",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=1,
        help="Seed for theta-init randomness (used for flex auxiliary init).",
    )
    parser.add_argument(
        "--spo-plus-init-mode",
        type=str,
        default="ipo",
        choices=["zero", "ipo", "none"],
        help="Initial theta for SPO+ (zero or IPO closed-form).",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="spy,1/n",
        help=(
            "Comma-separated benchmark list (spy,equal_weight,1/n,tsmom_spy). "
            "Empty string falls back to legacy flags."
        ),
    )
    parser.add_argument("--tee", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help=(
            "Number of parallel jobs for model groups (0 = auto based on selected models). "
            "Groups are: (ols+ipo), (ipo_grad), (spo_plus), (flex: dual/kkt)."
        ),
    )
    parser.add_argument(
        "--debug-roll",
        dest="debug_roll",
        action="store_true",
        default=True,
        help="Enable rolling-debug progress output (enabled by default).",
    )
    parser.add_argument(
        "--no-debug-roll",
        dest="debug_roll",
        action="store_false",
        help="Disable rolling-debug progress output.",
    )
    parser.add_argument(
        "--benchmark-ticker",
        type=str,
        default="SPY",
        help="Ticker used for buy-and-hold benchmark (leave empty to disable).",
    )
    parser.add_argument(
        "--benchmark-equal-weight",
        dest="benchmark_equal_weight",
        action="store_true",
        default=True,
        help="Include a 1/N equal-weighted portfolio benchmark built from all tickers (enabled by default).",
    )
    parser.add_argument(
        "--no-benchmark-equal-weight",
        dest="benchmark_equal_weight",
        action="store_false",
        help="Disable the equal-weighted benchmark.",
    )

    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--no-debug", action="store_true")
    return parser
