"""Generate synthetic OHLC and compute all 6 pivot types for comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ohlc(days: int = 5, base_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=days)
    high = base_price + rng.uniform(0.5, 2.0, size=days)
    low = base_price - rng.uniform(0.5, 2.0, size=days)
    close = base_price + rng.normal(0, 0.5, size=days)
    open_price = close + rng.normal(0, 0.2, size=days)
    return pd.DataFrame({"open": open_price, "high": high, "low": low, "close": close}, index=dates)


def classic_pivots(high: float, low: float, close: float) -> dict[str, float]:
    pivot = (high + low + close) / 3
    return {
        "P": pivot,
        "R1": 2 * pivot - low,
        "R2": pivot + (high - low),
        "R3": high + 2 * (pivot - low),
        "S1": 2 * pivot - high,
        "S2": pivot - (high - low),
        "S3": low - 2 * (high - pivot),
    }


def woodie_pivots(high: float, low: float, close: float, open_price: float) -> dict[str, float]:
    pivot = (high + low + 2 * open_price) / 4
    return {
        "P": pivot,
        "R1": 2 * pivot - low,
        "R2": pivot + (high - low),
        "R3": high + 2 * (pivot - low),
        "S1": 2 * pivot - high,
        "S2": pivot - (high - low),
        "S3": low - 2 * (high - pivot),
    }


def camarilla_pivots(high: float, low: float, close: float) -> dict[str, float]:
    price_range = high - low
    return {
        "P": (high + low + close) / 3,
        "R1": close + price_range * 1.0833,
        "R2": close + price_range * 1.1666,
        "R3": close + price_range * 1.25,
        "S1": close - price_range * 1.0833,
        "S2": close - price_range * 1.1666,
        "S3": close - price_range * 1.25,
    }


def demark_pivots(high: float, low: float, close: float, open_price: float) -> dict[str, float]:
    if close < open_price:
        x = high + low + 2 * close
    elif close > open_price:
        x = high + 2 * low + close
    else:
        x = 2 * high + low + close
    return {"P": x / 4, "R1": x / 2 - low, "S1": x / 2 - high}


def fibonacci_pivots(high: float, low: float, close: float) -> dict[str, float]:
    pivot = (high + low + close) / 3
    price_range = high - low
    return {
        "P": pivot,
        "R1": pivot + price_range * 0.382,
        "R2": pivot + price_range * 0.618,
        "R3": pivot + price_range,
        "S1": pivot - price_range * 0.382,
        "S2": pivot - price_range * 0.618,
        "S3": pivot - price_range,
    }


def mark_pivots(high: float, low: float, close: float) -> dict[str, float]:
    pivot = (high + low + close) / 3
    return {
        "P": pivot,
        "R1": 2 * pivot - low,
        "R2": pivot + (high - low),
        "S1": 2 * pivot - high,
        "S2": pivot - (high - low),
    }


PIVOT_FUNCS = {
    "classic": lambda r: classic_pivots(r["high"], r["low"], r["close"]),
    "woodie": lambda r: woodie_pivots(r["high"], r["low"], r["close"], r["open"]),
    "camarilla": lambda r: camarilla_pivots(r["high"], r["low"], r["close"]),
    "demark": lambda r: demark_pivots(r["high"], r["low"], r["close"], r["open"]),
    "fibonacci": lambda r: fibonacci_pivots(r["high"], r["low"], r["close"]),
    "mark": lambda r: mark_pivots(r["high"], r["low"], r["close"]),
}


def main() -> None:
    ohlc = generate_ohlc()
    last = ohlc.iloc[-1]
    high, low, close, open_price = last["high"], last["low"], last["close"], last["open"]

    rows: list[dict[str, str | float]] = []
    for name, func in PIVOT_FUNCS.items():
        levels = func(last)
        rows.append({"type": name, **{k: round(v, 4) for k, v in levels.items()}})

    df = pd.DataFrame(rows).set_index("type")
    print(f"Pivot Comparison Table  (H={high:.2f} L={low:.2f} C={close:.2f} O={open_price:.2f})")
    print("=" * 90)
    print(df.to_string())
    print(f"\nPivot types computed: {len(PIVOT_FUNCS)}")


if __name__ == "__main__":
    main()
