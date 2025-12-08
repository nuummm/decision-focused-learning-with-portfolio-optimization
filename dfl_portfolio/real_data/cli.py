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
    parser.add_argument("--end", type=str, default="2025-12-01")
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
    parser.add_argument("--momentum-window", type=int, default=30)
    parser.add_argument("--return-horizon", type=int, default=1)
    parser.add_argument("--cov-window", type=int, default=10)
    parser.add_argument(
        "--cov-method",
        type=str,
        default="diag",
        choices=["diag", "oas", "robust_lw", "mini_factor"],
    )
    parser.add_argument(
        "--cov-ewma-alpha",
        type=float,
        default=0.94,
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

    parser.add_argument("--train-window", type=int, default=25)
    parser.add_argument("--rebal-interval", type=int, default=4)
    parser.add_argument(
        "--model-train-window",
        type=str,
        default="",
        help="Optional overrides e.g. 'ols:60,flex:25' to use per-model train windows.",
    )

    parser.add_argument("--delta", type=parse_delta_0_1, default=0.5)
    parser.add_argument("--models", type=str, default="ols,ipo,flex")
    parser.add_argument("--flex-solver", type=str, default="knitro")
    parser.add_argument(
        "--flex-formulation",
        type=str,
        default="dual,kkt,dual&kkt",
        help=(
            "Comma-separated flex base models to run "
            "(e.g., 'dual', 'dual,kkt', or 'dual,kkt,dual&kkt' for dual/kkt+ensemble)."
        ),
    )
    parser.add_argument("--flex-lambda-theta-anchor", type=float, default=0.0)
    parser.add_argument("--flex-lambda-theta-iso", type=float, default=0.0)
    parser.add_argument("--flex-theta-anchor-mode", type=str, default="ols")
    parser.add_argument("--flex-theta-init-mode", type=str, default="none")
    parser.add_argument(
        "--flex-ensemble-weight-dual",
        type=parse_delta_0_1,
        default=0.5,
        help=(
            "Weight on flex-dual in the dual/kkt ensemble (w_dual); "
            "w_kkt = 1 - w_dual. Only used when ensemble is requested."
        ),
    )
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--debug-roll", action="store_true")
    parser.add_argument(
        "--benchmark-ticker",
        type=str,
        default="SPY",
        help="Ticker used for buy-and-hold benchmark (leave empty to disable).",
    )
    parser.add_argument(
        "--benchmark-equal-weight",
        action="store_true",
        help="Include a 1/N equal-weighted portfolio benchmark built from all tickers.",
    )

    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--no-debug", action="store_true")
    return parser
