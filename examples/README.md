# Examples

Self-contained scripts using **synthetic data only**. Run from the project root:

```bash
python examples/01_session_aware_ohlc.py
```

| # | Script | Description |
|---|--------|-------------|
| 01 | `01_session_aware_ohlc.py` | Generate synthetic 1-min bars and compute session-aware daily OHLC. |
| 02 | `02_pivot_levels.py` | Compute and compare all 6 pivot types on synthetic OHLC data. |
| 03 | `03_event_backtest.py` | Event-driven backtest with SL/TP exits on synthetic bars. |
| 04 | `04_vectorized_backtest.py` | Vectorized SL/TP grid search over pivot levels on synthetic data. |
| 05 | `05_walk_forward.py` | Generate walk-forward folds for 2020–2023 and print fold summary. |
