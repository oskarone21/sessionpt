"""Backtesting engine for session-aware futures strategies.

This module provides two execution paths:

1. **Event-driven engine** (``run_event_backtest``): Full-featured backtester
   supporting trailing stops, breakeven, EOD close, max-hold, and per-session
   trade limits. Accepts a list of EntryEvent objects and produces detailed
   trade-by-trade results.

2. **Vectorized fast path** (``VectorizedBacktester``): Numba-JIT compiled
   inner loop for rapid parameter sweeps over SL/TP grids. Falls back to
   pure Python when Numba is not installed.

Both paths produce the same ``BacktestRunResult`` type for consistency.
"""

from sessionpt.backtesting.engine import (
    EntryEvent,
    precompute_backtest_arrays,
    run_event_backtest,
    summarize_trades,
)
from sessionpt.backtesting.policies import normalize_breakeven, normalize_trailing
from sessionpt.backtesting.results import BacktestRunResult, EngineTradeRecord
from sessionpt.backtesting.specs import (
    BreakevenPolicy,
    ExecutionPolicy,
    ExitPolicy,
    InstrumentSpec,
    SessionSpec,
    TrailingStopPolicy,
)
from sessionpt.backtesting.vectorized import VectorizedBacktester

__all__ = [
    "BacktestRunResult",
    "BreakevenPolicy",
    "EngineTradeRecord",
    "EntryEvent",
    "ExecutionPolicy",
    "ExitPolicy",
    "InstrumentSpec",
    "SessionSpec",
    "TrailingStopPolicy",
    "VectorizedBacktester",
    "normalize_breakeven",
    "normalize_trailing",
    "precompute_backtest_arrays",
    "run_event_backtest",
    "summarize_trades",
]
