from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .fetch_yahoo import FetchResult, fetch_yahoo_prices
from .covariance import CovarianceStats, estimate_shrinkage_covariances

ReturnKind = Literal["simple", "log"]
FrequencyKind = Literal["daily", "weekly", "monthly"]


@dataclass
class MarketLoaderConfig:
    tickers: List[str]
    start: str
    end: str
    interval: str = "1d"
    price_field: str = "Close"
    return_kind: ReturnKind = "log"
    frequency: FrequencyKind = "weekly"
    resample_rule: str = "W-FRI"
    momentum_window: int = 52
    return_horizon: int = 1
    cov_window: int = 30
    cov_method: Literal[
        "diag",
        "oas",
        "robust_lw",
        "mini_factor",
    ] = "diag"
    cov_shrinkage: float = 0.94
    cov_eps: float = 1e-6
    cov_robust_huber_k: float = 1.5
    cov_factor_rank: int = 1
    cov_factor_shrinkage: float = 0.5
    auto_adjust: bool = True
    cache_dir: Optional[Path] = None
    force_refresh: bool = False
    debug: bool = True

    @classmethod
    def for_cli(
        cls,
        *,
        tickers: Iterable[str],
        start: str,
        end: str,
        **kwargs,
    ) -> "MarketLoaderConfig":
        tickers_list = [t.strip().upper() for t in tickers if t.strip()]
        return cls(tickers=tickers_list, start=start, end=end, **kwargs)


@dataclass
class MarketDataset:
    X: np.ndarray
    Y: np.ndarray
    timestamps: List[pd.Timestamp]
    prices: pd.DataFrame
    returns: pd.DataFrame
    momentum: pd.DataFrame
    covariances: List[np.ndarray]
    cov_times: List[pd.Timestamp]
    cov_stats: List[CovarianceStats]
    fetch_result: FetchResult
    config: MarketLoaderConfig

    def summary(self) -> Dict[str, object]:
        return {
            "n_samples": int(self.X.shape[0]),
            "n_assets": int(self.X.shape[1]),
            "n_targets": int(self.Y.shape[1]),
            "cov_window": self.config.cov_window,
            "frequency": self.config.frequency,
            "cov_method": self.config.cov_method,
            "cov_shrinkage": self.config.cov_shrinkage,
            "cov_robust_huber_k": self.config.cov_robust_huber_k,
            "cov_factor_rank": self.config.cov_factor_rank,
            "timeline": [self.timestamps[0].isoformat(), self.timestamps[-1].isoformat()],
            "tickers": self.config.tickers,
        }


def _determine_fetch_start(config: MarketLoaderConfig) -> str:
    start_ts = pd.Timestamp(config.start)
    buffer = config.momentum_window + config.return_horizon + config.cov_window
    if config.frequency == "weekly":
        offset = pd.DateOffset(weeks=buffer)
    elif config.frequency == "monthly":
        offset = pd.DateOffset(months=buffer)
    else:
        offset = pd.DateOffset(days=buffer * 5)
    effective = start_ts - offset
    return effective.strftime("%Y-%m-%d")


def _resample_prices(prices: pd.DataFrame, config: MarketLoaderConfig) -> pd.DataFrame:
    if config.frequency == "daily":
        return prices
    if config.resample_rule:
        rule = config.resample_rule
    elif config.frequency == "weekly":
        rule = "W-FRI"
    else:  # monthly
        rule = "M"
    return prices.resample(rule).last().dropna(how="all")


def _compute_returns(prices: pd.DataFrame, mode: ReturnKind) -> pd.DataFrame:
    if mode == "simple":
        rets = prices.pct_change()
    elif mode == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"unknown return mode: {mode}")
    return rets


def _compute_momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return np.log(prices / prices.shift(window))


def _build_xy(
    momentum: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int,
    start_cutoff: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    if horizon < 1:
        raise ValueError("return_horizon は 1 以上を指定してください。")

    X_df = momentum.copy()
    Y_df = returns.shift(-horizon)

    mask = (~X_df.isna().any(axis=1)) & (~Y_df.isna().any(axis=1))
    mask &= (X_df.index >= start_cutoff)

    X_clean = X_df.loc[mask]
    Y_clean = Y_df.loc[mask]
    return (
        X_clean.to_numpy(dtype=float),
        Y_clean.to_numpy(dtype=float),
        list(X_clean.index),
    )


def load_market_dataset(config: MarketLoaderConfig) -> MarketDataset:
    effective_start = _determine_fetch_start(config)
    df, fetch_info = fetch_yahoo_prices(
        config.tickers,
        effective_start,
        config.end,
        interval=config.interval,
        auto_adjust=config.auto_adjust,
        cache_dir=config.cache_dir,
        force_refresh=config.force_refresh,
        debug=config.debug,
    )

    price_cols = []
    for ticker in config.tickers:
        col = f"{ticker}_{config.price_field}"
        if col not in df.columns:
            raise KeyError(f"{col} が取得されたデータに存在しません。利用可能列: {list(df.columns)}")
        price_cols.append(col)

    prices = df[price_cols].copy()
    prices.columns = config.tickers
    prices = _resample_prices(prices, config)

    returns = _compute_returns(prices, config.return_kind)
    momentum = _compute_momentum(prices, config.momentum_window)
    start_cutoff = pd.Timestamp(config.start)
    X, Y, idx = _build_xy(momentum, returns, config.return_horizon, start_cutoff)
    returns_clean = returns.dropna()
    covs, cov_times, cov_stats = estimate_shrinkage_covariances(
        returns_clean,
        config.cov_window,
        method=config.cov_method,
        shrinkage=config.cov_shrinkage,
        eps=config.cov_eps,
        robust_huber_k=config.cov_robust_huber_k,
        factor_rank=config.cov_factor_rank,
        factor_shrinkage=config.cov_factor_shrinkage,
    )

    if config.debug:
        print("[real-data] loader config:")
        print(asdict(config))
        print("[real-data] prices sample:")
        print(prices.head())
        print(prices.tail())
        print("[real-data] returns sample:")
        print(returns.head())
        print(returns.describe())
        print("[real-data] momentum sample:")
        print(momentum.head())
        print(momentum.describe())
        print("[real-data] dataset shapes:")
        print({
            "X": X.shape,
            "Y": Y.shape,
            "returns": returns.shape,
            "momentum": momentum.shape,
            "covariances": len(covs),
            "cov_eig_min": float(min(stat.eigen_min for stat in cov_stats)),
        })

    return MarketDataset(
        X=X,
        Y=Y,
        timestamps=idx,
        prices=prices,
        returns=returns,
        momentum=momentum,
        covariances=covs,
        cov_times=cov_times,
        cov_stats=cov_stats,
        fetch_result=fetch_info,
        config=config,
    )


__all__ = [
    "MarketLoaderConfig",
    "MarketDataset",
    "load_market_dataset",
]
