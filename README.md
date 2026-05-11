# sessionpt

[![PyPI](https://img.shields.io/pypi/v/session-pt.svg)](https://pypi.org/project/session-pt/)
[![Python](https://img.shields.io/pypi/pyversions/session-pt.svg)](https://pypi.org/project/session-pt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/oskarone21/sessionpt/actions/workflows/ci.yml/badge.svg)](https://github.com/oskarone21/sessionpt/actions/workflows/ci.yml)

Standalone, open-source session-aware backtesting engine with pivot indicators
for futures markets.

## Why sessionpt?

Futures markets trade overnight — a CME Gold "trading day" runs from Sunday 18:00 to Monday 17:00 ET. Standard calendar-day resampling (`df.resample('D')`) splits these sessions across two days, producing **wrong** daily OHLC bars that corrupt all downstream pivot calculations.

**sessionpt** uses `groupby(session_labels)` instead, correctly grouping bars by
their trading session. This is the critical differentiator that makes pivot
point backtests produce reliable results.

The package is self-contained. It does not depend on any research repository
or local editable sibling checkout.

## Installation

```bash
pip install session-pt
```

Optional Numba acceleration:

```bash
pip install session-pt[numba]
```

## Quick Start

```python
import sessionpt as spt

# 1. Compute session-aware daily OHLC (not calendar-day resampled)
daily = spt.calculate_daily_ohlc(df, timezone="America/New_York", session_close_hour=17)

# 2. Calculate pivot levels (6 types supported)
pivots = spt.calculate_pivot_levels(daily, spt.PivotType.TRADITIONAL)

# 3. Merge pivots into intraday data
data = spt.prepare_data_with_pivots(df, spt.PivotType.TRADITIONAL)

# 4. Define instrument and execution policy
gc = spt.InstrumentSpec(
    symbol="GC", tick_size=0.10, tick_value=10.0,
    commission_round_trip=2.40, slippage_ticks=1.0,
)
exec_policy = spt.ExecutionPolicy(close_at_eod=True, max_trades_per_session=5)

# 5. Run event-driven backtest
result = spt.run_event_backtest(df, entry_events, gc, exec_policy)
print(f"Trades: {result.trades}, PnL: ${result.total_pnl_net:,.0f}, Sharpe: {result.sharpe_ratio:.2f}")
```

## Features

- **Session-aware OHLC** — Groups bars by trading session, not calendar days
- **6 pivot types** — Traditional, Woodie, Camarilla, Fibonacci, Classic, DeMark
- **Event-driven backtester** — SL/TP, trailing stop, breakeven, EOD close, MFE/MAE
- **Numba-accelerated vectorized backtester** — Fast SL/TP grid sweeps (falls back to pure Python)
- **Walk-forward validation** — Sliding-window folds, parameter grids, candidate selection, and multiple-testing helpers
- **Trend filtering** — EMA/SMA crossover bias with look-ahead prevention
- **Feature utilities** — ATR, VWAP, wick/volume filters, initial-balance levels, and pivot confluence
- **Robust analytics** — Sortino, Calmar, Kelly, drawdown, and strategy robustness checks
- **CME presets** — Built-in session configs for ES, NQ, GC, CL, ZB

## Architecture

```
sessionpt/
├── analytics/      # Robust trade and equity-curve metrics
├── features/       # ATR, VWAP, wick, volume, and entry-mask helpers
├── levels/         # Initial-balance session levels
├── enums/          # Direction, PivotType, PivotLevel, ExitReason
├── sessions/       # Session ID computation, CME presets
├── pivots/         # 6 pivot calculators, confluence, shifts, OHLC
├── filters/        # EMA/SMA trend bias filter
├── backtesting/    # Event engine + Numba vectorized fast path
└── validation/     # Walk-forward, search, selection, and leakage checks
```

## Examples

See `examples/` for self-contained scripts:

| Script | Description |
|--------|-------------|
| `01_session_aware_ohlc.py` | Compute daily OHLC using trading sessions |
| `02_pivot_levels.py` | Compare all 6 pivot types |
| `03_event_backtest.py` | Run an event-driven backtest |
| `04_vectorized_backtest.py` | SL/TP grid search with Numba |
| `05_walk_forward.py` | Generate walk-forward validation folds |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
mypy src/
```

## License

MIT
