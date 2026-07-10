"""Compute session-aware daily OHLC with the public sessionpt API."""

from __future__ import annotations

import pandas as pd

from sessionpt import calculate_daily_ohlc


def build_bars() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-07 18:00", tz="America/New_York"),
            pd.Timestamp("2024-01-08 10:00", tz="America/New_York"),
            pd.Timestamp("2024-01-08 16:00", tz="America/New_York"),
            pd.Timestamp("2024-01-08 18:00", tz="America/New_York"),
            pd.Timestamp("2024-01-09 10:00", tz="America/New_York"),
            pd.Timestamp("2024-01-09 16:00", tz="America/New_York"),
        ]
    )
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "High": [101.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "Close": [100.5, 102.0, 103.0, 104.0, 105.0, 106.0],
            "Volume": [10, 20, 30, 10, 20, 30],
        },
        index=index,
    )


def main() -> None:
    daily = calculate_daily_ohlc(build_bars())
    print("Session-Aware Daily OHLC")
    print(daily.to_string(float_format="{:.2f}".format))


if __name__ == "__main__":
    main()
