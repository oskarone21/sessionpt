# User Guide

## Session-Aware OHLC

Futures markets run nearly 24 hours. CME Globex closes at 17:00 ET, which
means a "trading day" spans two calendar dates in UTC. Naïvely grouping by
calendar date or UTC midnight produces wrong daily bars and, worse, wrong
pivot levels.

`sessionpt` solves this with three functions:

| Function | Purpose |
|---|---|
| `get_session_labels()` | Assign each bar a date label in exchange-local time, shifted past the close hour |
| `get_session_ids()` | Integer version of the above — fast for grouping |
| `build_session_mask()` | Boolean mask for named sessions (Asian, London, etc.) |

**Daily OHLC** is always computed with `calculate_daily_ohlc(df, timezone, session_close_hour)` which groups by session label, not UTC date. This is what `prepare_data_with_pivots` uses internally.

```python
from sessionpt import calculate_daily_ohlc

daily = calculate_daily_ohlc(df, timezone="America/New_York", session_close_hour=17)
```

---

## Pivot Types

Six calculation methods are supported. All take `(high, low, close)` (and
optionally `open_price` for DeMark) and return a dict of named levels.

| Type | Key difference |
|---|---|
| **Traditional** | Standard `(H+L+C)/3` centre; `R1 = 2P − L`, `S1 = 2P − H` |
| **Woodie** | Weights close double: `(H+L+2C)/4` |
| **Camarilla** | Uses fixed fractions of the range (`1.1/12`, `1.1/6`, …) from the close |
| **Fibonacci** | Applies 0.382 and 0.618 retracement ratios to the range |
| **Classic** | Same centre as Traditional but `R2/R3` and `S2/S3` are full multiples of the range |
| **DeMark (DM)** | Conditionally weights H/L/C based on open vs close relationship |

Use `calculate_pivot_levels(daily_df, pivot_type=PivotType.WOODIE)` to compute
a full DataFrame of levels, or `prepare_data_with_pivots(df, PivotType.WOODIE)`
to merge them into intraday data in one call.

---

## Event Backtester

The **event backtester** processes each bar sequentially, supporting:

- **Fixed SL / TP** in ticks
- **End-of-day (EOD) close** at the session boundary
- **Trailing stop** — activate after `trigger_ticks`, lock at `entry ± lock_ticks`
- **Breakeven stop** — move stop to entry at `trigger_pct_to_tp` of the TP distance
- **Max hold** — force exit after N elapsed calendar days (when `close_at_eod=False`)
- **Gap-aware stops** — gap-through stops fill at the next bar's open

```python
from sessionpt import (
    InstrumentSpec, ExecutionPolicy, TrailingStopPolicy,
    BreakevenPolicy, EntryEvent, run_event_backtest, Direction,
)

instrument = InstrumentSpec(
    symbol="GC", tick_size=0.10, tick_value=10.0,
    commission_round_trip=2.40, slippage_ticks=1,
    timezone="America/New_York", session_close_hour=17,
)

events = [...]  # list of EntryEvent objects
result = run_event_backtest(
    df_pivots, events, instrument,
    execution_policy=ExecutionPolicy(close_at_eod=True, max_trades_per_session=3),
    trailing_policy=TrailingStopPolicy(enabled=True, trigger_ticks=75, lock_ticks=50),
    breakeven_policy=BreakevenPolicy(enabled=True, trigger_pct_to_tp=0.5),
)
```

Exit reasons in the result: `TP`, `SL`, `TRAILING_SL`, `BREAKEVEN`, `EOD`,
`MAX_HOLD`, `DATA_END`.

---

## Vectorized Backtester

The **vectorized backtester** trades speed for simplicity — it handles fixed
SL/TP and EOD exits but does *not* support trailing or breakeven stops. It
compiles with Numba when available for 10–50× speed-up, making it ideal for
parameter grid searches.

```python
from sessionpt import VectorizedBacktester, Direction, prepare_data_with_pivots

bt = VectorizedBacktester(
    tick_size=0.10, tick_value=10.0, commission=1.20, slippage_cost=10.0,
)
result = bt.run(
    df_pivots, level_col="S1", direction=Direction.LONG,
    sl_points=100, tp_points=160,
)
print(f"Trades: {result.trades}  PF: {result.profit_factor:.2f}")
```

The vectorized path returns aggregate results. Use `run_event_backtest()` when
per-trade records or configurable EOD behavior are required.

---

## Walk-Forward Validation

Walk-forward testing prevents overfitting by optimising on a training window
and validating on a disjoint out-of-sample window.

### Fold Generation

```python
from datetime import datetime
from sessionpt import generate_walk_forward_folds

folds = generate_walk_forward_folds(
    data_start=datetime(2020, 1, 1),
    data_end=datetime(2024, 12, 31),
    train_months=12,
    test_months=6,
    step_months=6,
)
```

Each `WalkForwardFold` contains `train_start`, `train_end`, `test_start`,
`test_end` timestamps, and convenience properties `train_duration_days` and
`test_duration_days`. Endpoints are inclusive and generated train/test windows
are disjoint. `step_months` must be at least `test_months` so OOS windows do not
overlap.

### OOS Testing

1. For each fold, run the vectorized backtester on the **train** window to find
   the best (SL, TP) combination.
2. Apply that combination to the **test** window — no re-optimisation.
3. Collect `FoldResult` objects and check `pnl_profitable_oos` to see if the
   strategy holds out-of-sample.

The `OptResult` and `FoldResult` dataclasses record all key metrics (Sharpe,
Sortino, profit factor, max drawdown) for both train and test periods.
