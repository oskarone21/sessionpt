"""Generic pivot-level shift helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from sessionpt.constants import (
    DEFAULT_SESSION_CLOSE_HOUR,
    DEFAULT_TIMEZONE,
    PIVOT_CENTRAL_COLUMN,
    PIVOT_R1_COLUMN,
    PIVOT_R2_COLUMN,
    PIVOT_R3_COLUMN,
    PIVOT_S1_COLUMN,
    PIVOT_S2_COLUMN,
    PIVOT_S3_COLUMN,
)
from sessionpt.pivots.ohlc import prepare_data_with_pivots

RESISTANCE_PIVOT_COLUMNS = (
    PIVOT_R1_COLUMN,
    PIVOT_R2_COLUMN,
    PIVOT_R3_COLUMN,
)
SUPPORT_AND_CENTRAL_PIVOT_COLUMNS = (
    PIVOT_CENTRAL_COLUMN,
    PIVOT_S1_COLUMN,
    PIVOT_S2_COLUMN,
    PIVOT_S3_COLUMN,
)


def apply_directional_pivot_shift(df: pd.DataFrame, shift_price: float) -> pd.DataFrame:
    """Shift resistance levels up and central/support levels down."""

    shifted = df.copy()
    for column in RESISTANCE_PIVOT_COLUMNS:
        if column in shifted.columns:
            shifted[column] = shifted[column] + shift_price
    for column in SUPPORT_AND_CENTRAL_PIVOT_COLUMNS:
        if column in shifted.columns:
            shifted[column] = shifted[column] - shift_price
    return shifted


def prepare_shifted_pivot_data(
    raw_data: pd.DataFrame,
    pivot_type: Any,
    shift_price: float,
    timezone: str = DEFAULT_TIMEZONE,
    session_close_hour: int = DEFAULT_SESSION_CLOSE_HOUR,
) -> pd.DataFrame:
    """Prepare pivot data, then apply a caller-supplied directional shift."""

    pivot_data = prepare_data_with_pivots(
        raw_data,
        pivot_type=pivot_type,
        timezone=timezone,
        session_close_hour=session_close_hour,
    )
    return apply_directional_pivot_shift(pivot_data, shift_price)
