"""Run the public event-driven backtester on deterministic synthetic bars."""

from __future__ import annotations

import pandas as pd

from sessionpt import (
    Direction,
    EntryEvent,
    ExecutionPolicy,
    InstrumentSpec,
    run_event_backtest,
)


def build_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-08 09:30", periods=5, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [100.0, 100.0, 101.0, 102.0, 103.0],
            "High": [100.5, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.5, 99.0, 100.0, 101.0, 102.0],
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
        },
        index=index,
    )


def main() -> None:
    bars = build_bars()
    instrument = InstrumentSpec("DEMO", 1.0, 1.0, 0.25, 0.25)
    event = EntryEvent(0, Direction.LONG, 100.0, 98.0, 104.0, level_tag="P")
    result = run_event_backtest(bars, [event], instrument, ExecutionPolicy())
    print("Event-Driven Backtest Summary")
    print(f"Trades: {result.trades}")
    print(f"Net PnL: {result.total_pnl_net:.2f}")
    print(f"Exit: {result.trade_records[0].exit_reason}")


if __name__ == "__main__":
    main()
