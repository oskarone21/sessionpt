# sessionpt

**Session-aware backtesting with pivot indicators for futures.**

`sessionpt` provides intraday futures backtesting that respects exchange session
boundaries — not UTC midnight — so your EOD exits, daily OHLC aggregation, and
walk-forward folds all line up with how CME markets actually trade.

## Installation

```bash
pip install session-pt
```

## Quick Start

```python
import sessionpt as spt

# 1. Load your 1-minute OHLCV DataFrame (UTC-indexed)
#    df must have columns: Open, High, Low, Close, Volume

# 2. Compute session-aligned daily OHLC and merge pivot levels
df_pivots = spt.prepare_data_with_pivots(
    df, pivot_type=spt.PivotType.WOODIE, timezone="America/New_York"
)

# 3. Run a vectorised backtest over S1 support, LONG direction
bt = spt.VectorizedBacktester(
    tick_size=0.10, tick_value=10.0, commission=0.0, slippage_cost=0.0
)
result = bt.run(df_pivots, level_col="S1", direction=spt.Direction.LONG,
                sl_points=100, tp_points=160)
print(f"Trades: {result.trades}  Win rate: {result.win_rate:.1f}%  "
      f"Net P&L: ${result.total_pnl_net:,.0f}")
```

## Next Steps

- [API Reference](api.md) — complete list of public classes and functions
- [User Guide](guide.md) — session-aware OHLC, pivot types, backtesters, walk-forward