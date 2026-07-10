"""Run a public VectorizedBacktester SL/TP grid on synthetic bars."""

from __future__ import annotations

from itertools import product

import pandas as pd

from sessionpt import Direction, VectorizedBacktester


def build_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-08 09:30", periods=8, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [100.0, 100.0, 101.0, 100.0, 99.0, 100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0, 101.0, 101.0, 102.0, 103.0, 104.0],
            "Low": [99.0, 99.0, 100.0, 98.0, 98.0, 99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 102.0, 99.0, 100.0, 101.0, 102.0, 103.0],
            "P": [100.0] * 8,
        },
        index=index,
    )


def main() -> None:
    backtester = VectorizedBacktester(1.0, 1.0, commission=0.25, slippage_cost=0.5)
    results = []
    for stop, target in product((1.0, 2.0), repeat=2):
        result = backtester.run(build_bars(), "P", Direction.LONG, stop, target)
        results.append((result.total_pnl_net, stop, target, result.trades))
    pnl, stop, target, trades = max(results)
    print("Vectorized SL/TP Grid Search")
    print(f"Best stop/target: {stop:.0f}/{target:.0f} ticks")
    print(f"Trades: {trades}, Net PnL: {pnl:.2f}")


if __name__ == "__main__":
    main()
