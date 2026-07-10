"""Compare every public pivot type with the sessionpt API."""

from __future__ import annotations

import pandas as pd

from sessionpt import PivotType, calculate_pivot_levels


def build_daily_ohlc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [110.0, 112.0, 114.0],
            "Low": [90.0, 92.0, 94.0],
            "Close": [105.0, 106.0, 108.0],
        },
        index=pd.date_range("2024-01-02", periods=3, freq="D"),
    )


def main() -> None:
    daily = build_daily_ohlc()
    rows = []
    for pivot_type in PivotType:
        levels = calculate_pivot_levels(daily, pivot_type).iloc[-1]
        rows.append({"type": pivot_type.value, **levels.to_dict()})
    comparison = pd.DataFrame(rows).set_index("type")
    print("Pivot Comparison Table")
    print(comparison.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
