"""Pivot point calculation functions.

Each function takes the previous session's High, Low, and Close
(and optionally Open) and returns a dictionary of pivot levels:
P, R1-R3, S1-S3.

Some pivot types (Traditional, Fibonacci, Classic, Camarilla) use
(H + L + C) / 3 as the pivot point. Woodie uses (H + L + 2*C) / 4.
DeMark (DM) uses a conditional formula based on the relationship
between Close and Open.
"""

from collections.abc import Callable

import pandas as pd

from sessionpt.constants import (
    CLOSE_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    OPEN_COLUMN,
    PIVOT_CENTRAL_COLUMN,
    PIVOT_R1_COLUMN,
    PIVOT_R2_COLUMN,
    PIVOT_R3_COLUMN,
    PIVOT_S1_COLUMN,
    PIVOT_S2_COLUMN,
    PIVOT_S3_COLUMN,
)
from sessionpt.enums.pivot_type import PivotType

StandardPivotCalculator = Callable[[float, float, float], dict[str, float]]


def calculate_traditional_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Calculate Traditional pivot levels.

    P = (H + L + C) / 3
    R1 = 2*P - L,  S1 = 2*P - H
    R2 = P + (H - L),  S2 = P - (H - L)
    R3 = H + 2*(P - L),  S3 = L - 2*(H - P)
    """
    p = (high + low + close) / 3
    r = high - low
    return {
        PIVOT_CENTRAL_COLUMN: p,
        PIVOT_R1_COLUMN: 2 * p - low,
        PIVOT_R2_COLUMN: p + r,
        PIVOT_R3_COLUMN: high + 2 * (p - low),
        PIVOT_S1_COLUMN: 2 * p - high,
        PIVOT_S2_COLUMN: p - r,
        PIVOT_S3_COLUMN: low - 2 * (high - p),
    }


def calculate_fibonacci_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Calculate Fibonacci pivot levels.

    Uses Fibonacci ratios (0.382, 0.618) of the range to compute R1/R2 and S1/S2.
    """
    p = (high + low + close) / 3
    r = high - low
    return {
        PIVOT_CENTRAL_COLUMN: p,
        PIVOT_R1_COLUMN: p + 0.382 * r,
        PIVOT_R2_COLUMN: p + 0.618 * r,
        PIVOT_R3_COLUMN: p + r,
        PIVOT_S1_COLUMN: p - 0.382 * r,
        PIVOT_S2_COLUMN: p - 0.618 * r,
        PIVOT_S3_COLUMN: p - r,
    }


def calculate_woodie_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Calculate Woodie pivot levels.

    Woodie pivots give more weight to the Close: P = (H + L + 2*C) / 4.
    """
    p = (high + low + 2 * close) / 4
    r = high - low
    return {
        PIVOT_CENTRAL_COLUMN: p,
        PIVOT_R1_COLUMN: 2 * p - low,
        PIVOT_R2_COLUMN: p + r,
        PIVOT_R3_COLUMN: high + 2 * (p - low),
        PIVOT_S1_COLUMN: 2 * p - high,
        PIVOT_S2_COLUMN: p - r,
        PIVOT_S3_COLUMN: low - 2 * (high - p),
    }


def calculate_classic_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Calculate Classic pivot levels.

    Same pivot point as Traditional (H+L+C)/3, but R2/R3 and S2/S3
    use range-based formulas instead of pivot offsets.
    """
    p = (high + low + close) / 3
    r = high - low
    return {
        PIVOT_CENTRAL_COLUMN: p,
        PIVOT_R1_COLUMN: 2 * p - low,
        PIVOT_R2_COLUMN: p + r,
        PIVOT_R3_COLUMN: p + 2 * r,
        PIVOT_S1_COLUMN: 2 * p - high,
        PIVOT_S2_COLUMN: p - r,
        PIVOT_S3_COLUMN: p - 2 * r,
    }


def calculate_dm_pivots(
    high: float, low: float, close: float, open_price: float | None = None
) -> dict[str, float]:
    """Calculate DeMark (DM) pivot levels.

    DM pivots use a conditional formula based on the Close vs Open relationship:
    - If Close < Open: X = H + 2*L + C
    - If Close > Open: X = 2*H + L + C
    - If Close == Open: X = H + L + 2*C

    Note: R2 and S2 are both equal to P in the standard DM formula.
    """
    if open_price is None:
        open_price = close

    if close < open_price:
        x = high + 2 * low + close
    elif close > open_price:
        x = 2 * high + low + close
    else:
        x = high + low + 2 * close

    p = x / 4
    return {
        PIVOT_CENTRAL_COLUMN: p,
        PIVOT_R1_COLUMN: x / 2 - low,
        PIVOT_S1_COLUMN: x / 2 - high,
        PIVOT_R2_COLUMN: p,
        PIVOT_R3_COLUMN: p,
        PIVOT_S2_COLUMN: p,
        PIVOT_S3_COLUMN: p,
    }


def calculate_camarilla_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Calculate Camarilla pivot levels.

    Camarilla pivots use the range (H-L) scaled by constants (1.1/12, 1.1/6, 1.1/4)
    to compute support and resistance levels. The pivot point is the same as Traditional.
    """
    p = (high + low + close) / 3
    r = high - low
    return {
        PIVOT_CENTRAL_COLUMN: p,
        PIVOT_R1_COLUMN: close + r * 1.1 / 12,
        PIVOT_R2_COLUMN: close + r * 1.1 / 6,
        PIVOT_R3_COLUMN: close + r * 1.1 / 4,
        PIVOT_S1_COLUMN: close - r * 1.1 / 12,
        PIVOT_S2_COLUMN: close - r * 1.1 / 6,
        PIVOT_S3_COLUMN: close - r * 1.1 / 4,
    }


STANDARD_PIVOT_CALCULATORS: dict[PivotType, StandardPivotCalculator] = {
    PivotType.TRADITIONAL: calculate_traditional_pivots,
    PivotType.FIBONACCI: calculate_fibonacci_pivots,
    PivotType.WOODIE: calculate_woodie_pivots,
    PivotType.CLASSIC: calculate_classic_pivots,
    PivotType.CAMARILLA: calculate_camarilla_pivots,
}


def calculate_pivot_levels(
    daily_df: pd.DataFrame,
    pivot_type: PivotType = PivotType.TRADITIONAL,
    intraday_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate pivot levels from daily OHLC data.

    Each row in daily_df provides the previous session's High, Low, Close
    (and optionally Open for DM pivots). The computed pivot levels are
    assigned to the next session date/name.

    Parameters
    ----------
    daily_df : pd.DataFrame
        DataFrame with columns 'Open', 'High', 'Low', 'Close'.
        Index should be session labels (from get_session_labels).
    pivot_type : PivotType
        The type of pivot calculation to use.
    intraday_df : pd.DataFrame, optional
        Not currently used; kept for API compatibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns P, R1, R2, R3, S1, S2, S3,
        indexed by session date (shifted forward by one row).
    """
    pivots_list: list[dict[str, object]] = []

    for i in range(len(daily_df) - 1):
        row = daily_df.iloc[i]
        next_date = daily_df.index[i + 1]
        high, low_price, close, open_price = (
            float(row[HIGH_COLUMN]),
            float(row[LOW_COLUMN]),
            float(row[CLOSE_COLUMN]),
            float(row[OPEN_COLUMN]),
        )

        if pivot_type == PivotType.DM:
            levels = calculate_dm_pivots(high, low_price, close, open_price)
        else:
            calc_func = STANDARD_PIVOT_CALCULATORS.get(pivot_type, calculate_traditional_pivots)
            levels = calc_func(high, low_price, close)

        pivots_list.append({**levels, "date": next_date})

    pivots_df = pd.DataFrame(pivots_list)
    pivots_df.set_index("date", inplace=True)

    return pivots_df
