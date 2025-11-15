"""Utilities for working with real-world market datasets.

This subpackage currently focuses on Yahoo Finance sourced price/return
series used by the real-data experiment workflow.
"""

from .loader import load_market_dataset  # noqa: F401
from .fetch_yahoo import fetch_yahoo_prices  # noqa: F401

__all__ = [
    "load_market_dataset",
    "fetch_yahoo_prices",
]
