# Real-Data Workflow (Yahoo Finance)

This folder contains helpers for ingesting Yahoo Finance data and inspecting it
before running the main decision-focused training loops.

## Quick start

```bash
cd /Users/kensei/VScode/GraduationResearch/DFL_Portfolio_Optimization2
python -m experiments.real_data.debug_loader \
  --tickers "SPY,TLT,DBC,BIL" \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --interval 1d
```

Outputs will be written under `experiments/real_data/debug_outputs/<timestamp>/`:

- `summary.json`: configuration + dataset stats
- `prices_full.csv`, `returns_full.csv`: cleaned tables
- `X_matrix_head.csv`, `Y_matrix_head.csv`: feature/label previews
- `visuals/prices.png`, `visuals/returns.png`: quick plots

All arguments are optional so you can change tickers/dates without touching the
code. Use `--help` for the full list.

## Common data pipeline

Downstream experiments should consume the shared pipeline helper:

```python
from experiments.real_data.data_pipeline import (
    PipelineConfig,
    SplitConfig,
    build_data_bundle,
)
from data.real_data.loader import MarketLoaderConfig

loader_cfg = MarketLoaderConfig.for_cli(
    tickers=["SPY", "TLT", "DBC", "BIL"],
    start="2010-01-01",
    end="2024-12-31",
    interval="1d",
)

pipeline_cfg = PipelineConfig(
    loader=loader_cfg,
    split=SplitConfig(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2),
)

bundle = build_data_bundle(pipeline_cfg)
X_train, Y_train, ts_train = bundle.slice_split("train")
```

`bundle.summary()` returns a dictionary of dataset/timeline/split sizes so that
each experiment can log the exact data slice it used.
