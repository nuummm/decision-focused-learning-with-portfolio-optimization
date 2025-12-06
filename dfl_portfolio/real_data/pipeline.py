from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dfl_portfolio.real_data.loader import MarketDataset, MarketLoaderConfig, load_market_dataset
from dfl_portfolio.real_data.covariance import CovarianceStats


@dataclass
class RealDataBundle:
    dataset: MarketDataset
    covariances: List[np.ndarray]
    cov_stats: List[CovarianceStats]
    cov_indices: np.ndarray

    def summary(self) -> Dict[str, object]:
        payload = self.dataset.summary()
        payload["cov_coverage"] = {
            "available": len(self.cov_indices),
            "total_samples": len(self.dataset.timestamps),
            "missing": len(self.dataset.timestamps) - len(self.cov_indices),
        }
        payload["cov_params"] = {
            "window": self.dataset.config.cov_window,
            "method": self.dataset.config.cov_method,
            "shrinkage": self.dataset.config.cov_shrinkage,
        }
        if self.cov_stats:
            eig_mins = np.array([stat.eigen_min for stat in self.cov_stats], dtype=float)
            payload["cov_eigen_min"] = {
                "min": float(np.min(eig_mins)),
                "median": float(np.median(eig_mins)),
            }
        payload["timeline"] = [
            self.dataset.timestamps[int(self.cov_indices[0])].isoformat(),
            self.dataset.timestamps[int(self.cov_indices[-1])].isoformat(),
        ]
        payload["covariance_samples"] = len(self.cov_indices)
        return payload


def _align_covariances(dataset: MarketDataset) -> Tuple[np.ndarray, List[np.ndarray], List[CovarianceStats]]:
    ts_index = pd.Index(dataset.timestamps)
    cov_index = pd.Index(dataset.cov_times)
    ts_to_cov = {ts: (cov, stat) for ts, cov, stat in zip(cov_index, dataset.covariances, dataset.cov_stats)}

    usable_indices: List[int] = []
    covariances: List[np.ndarray] = []
    stats: List[CovarianceStats] = []
    for idx_pos, ts in enumerate(dataset.timestamps):
        entry = ts_to_cov.get(ts)
        if entry is None:
            continue
        cov, stat = entry
        usable_indices.append(idx_pos)
        covariances.append(cov)
        stats.append(stat)

    if not usable_indices:
        raise ValueError("共分散が計算できたサンプルがありません。cov_window を小さくしてください。")

    return np.asarray(usable_indices, dtype=int), covariances, stats


@dataclass
class PipelineConfig:
    loader: MarketLoaderConfig
    debug: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "loader": asdict(self.loader),
            "debug": self.debug,
        }


def build_data_bundle(config: PipelineConfig) -> RealDataBundle:
    dataset = load_market_dataset(config.loader)
    usable_indices, covariances, cov_stats = _align_covariances(dataset)
    bundle = RealDataBundle(
        dataset=dataset,
        covariances=covariances,
        cov_stats=cov_stats,
        cov_indices=usable_indices,
    )
    if config.debug:
        print("[real-data] pipeline summary:")
        print(bundle.summary())
    return bundle
