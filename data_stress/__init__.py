"""Stress-test synthetic data generators.

This package mirrors the public API of :mod:`data.synthetic` but produces
datasets with heavy tails, regime shifts, and rare shock events so that
DFL-model behaviour can be evaluated under stressed market conditions.
"""

from .synthetic import generate_simulation1_dataset  # noqa: F401

