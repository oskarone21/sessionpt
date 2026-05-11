"""Generate synthetic 1-min bars for 5 trading sessions and compute session-aware daily OHLC."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_session_bars(
    session_date: datetime,
    bars_per_session: int = 390,
    base_price: float = 100.0,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = session_date.replace(hour=9, minute=30, second=0, microsecond=0)
    timestamps = pd.date_range(start, periods=bars_per_session, freq="1min")
    returns = rng.normal(0, 0.001, size=bars_per_session)
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.002, size=bars_per_session))
    low = close * (1 - rng.uniform(0, 0.002, size=bars_per_session))
    open_ = np.empty_like(close)
    open_[0] = base_price
    open_[1:] = close[:-1]
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=timestamps,
    )


def session_ohlc(
    bars: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
) -> pd.DataFrame:
    bars = bars.between_time(session_start, session_end)
    return (
        bars.resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def main() -> None:
    all_bars: list[pd.DataFrame] = []
    base = 100.0
    start_date = datetime(2024, 1, 2)
    for i in range(5):
        session_date = start_date + timedelta(days=i)
        if session_date.weekday() >= 5:
            session_date += timedelta(days=7 - session_date.weekday())
        bars = generate_session_bars(session_date, base_price=base, seed=42 + i)
        all_bars.append(bars)
        base = float(bars["close"].iloc[-1])

    combined = pd.concat(all_bars)
    daily = session_ohlc(combined)

    print("Session-Aware Daily OHLC")
    print("=" * 70)
    print(daily.to_string(float_format="{:.2f}".format))
    print(f"\nSessions: {len(daily)}")
    print(f"Total bars: {len(combined)}")


if __name__ == "__main__":
    main()
