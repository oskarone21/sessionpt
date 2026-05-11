"""Session-aware OHLC aggregation and pivot level assignment.

This module provides the key functions for computing daily OHLC bars
using exchange session boundaries (not calendar dates) and merging
pivot levels into intraday data.

The session-aware OHLC computation is the critical differentiator of this
library. Standard calendar-day resampling (pandas resample('D')) incorrectly
splits futures trading sessions across two calendar days, producing wrong
Open and Close values and corrupting all downstream pivot calculations.
"""

import pandas as pd

from sessionpt.constants import (
    CLOSE_COLUMN,
    DEFAULT_SESSION_CLOSE_HOUR,
    DEFAULT_TIMEZONE,
    HIGH_COLUMN,
    LOW_COLUMN,
    OPEN_COLUMN,
    PIVOT_CENTRAL_COLUMN,
    PIVOT_LEVEL_COLUMNS,
    VOLUME_COLUMN,
)
from sessionpt.enums.pivot_type import PivotType
from sessionpt.pivots.calculator import calculate_pivot_levels
from sessionpt.sessions.core import get_session_labels

SESSION_LABEL_COLUMN = "_session"


def calculate_daily_ohlc(
    df: pd.DataFrame,
    timezone: str = DEFAULT_TIMEZONE,
    session_close_hour: int = DEFAULT_SESSION_CLOSE_HOUR,
) -> pd.DataFrame:
    """Compute daily OHLC bars using exchange session boundaries.

    Unlike pandas resample('D') which groups by calendar midnight, this
    function groups bars by trading session. For CME Gold, a "trading day"
    runs from Sunday 18:00 ET to Monday 17:00 ET (with a 1-hour break).
    Grouping by session correctly captures the Sunday evening open and
    produces accurate OHLC values for pivot calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Intraday DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume'.
        Must have a timezone-aware DatetimeIndex.
    timezone : str
        Exchange timezone string (e.g., 'America/New_York').
    session_close_hour : int
        Hour (in exchange-local time) when the session closes.
        Bars at this hour stay in the current session (close bar).
        Bars after this hour start the next session.

    Returns
    -------
    pd.DataFrame
        Daily OHLCV DataFrame indexed by session date.
    """
    session_labels = get_session_labels(
        index=df.index,
        timezone=timezone,
        session_close_hour=session_close_hour,
    )
    daily = (
        df.groupby(session_labels)
        .agg(
            {
                OPEN_COLUMN: "first",
                HIGH_COLUMN: "max",
                LOW_COLUMN: "min",
                CLOSE_COLUMN: "last",
                VOLUME_COLUMN: "sum",
            }
        )
        .dropna()
    )

    return daily


def add_pivot_levels_to_intraday(
    intraday_df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    timezone: str = DEFAULT_TIMEZONE,
    session_close_hour: int = DEFAULT_SESSION_CLOSE_HOUR,
) -> pd.DataFrame:
    """Merge pivot levels into intraday data using session-aware alignment.

    Parameters
    ----------
    intraday_df : pd.DataFrame
        Intraday price data with timezone-aware DatetimeIndex.
    pivot_df : pd.DataFrame
        Pivot levels DataFrame from calculate_pivot_levels().
    timezone : str
        Exchange timezone string.
    session_close_hour : int
        Hour when the session closes (in exchange-local time).

    Returns
    -------
    pd.DataFrame
        Intraday DataFrame with pivot level columns added.
    """
    df = intraday_df.copy()

    df[SESSION_LABEL_COLUMN] = get_session_labels(
        index=df.index,
        timezone=timezone,
        session_close_hour=session_close_hour,
    )

    for col in PIVOT_LEVEL_COLUMNS:
        df[col] = df[SESSION_LABEL_COLUMN].map(pivot_df[col])

    df.drop(SESSION_LABEL_COLUMN, axis=1, inplace=True)

    return df


def prepare_data_with_pivots(
    df: pd.DataFrame,
    pivot_type: PivotType = PivotType.TRADITIONAL,
    timezone: str = DEFAULT_TIMEZONE,
    session_close_hour: int = DEFAULT_SESSION_CLOSE_HOUR,
) -> pd.DataFrame:
    """Compute session-aware OHLC, calculate pivots, and merge into intraday data.

    This is the main entry point for pivot-based strategy preparation.
    It computes daily OHLC using session boundaries, calculates pivot
    levels for the chosen type, and merges them into the intraday data.

    Parameters
    ----------
    df : pd.DataFrame
        Intraday price data with columns 'Open', 'High', 'Low', 'Close', 'Volume'
        and timezone-aware DatetimeIndex.
    pivot_type : PivotType
        The type of pivot calculation to use.
    timezone : str
        Exchange timezone string.
    session_close_hour : int
        Hour when the session closes (in exchange-local time).

    Returns
    -------
    pd.DataFrame
        Intraday DataFrame with pivot level columns, with the first
        session's rows dropped (no prior session for pivot calc).
    """
    daily = calculate_daily_ohlc(df, timezone=timezone, session_close_hour=session_close_hour)
    pivots = calculate_pivot_levels(daily, pivot_type, intraday_df=df)
    result = add_pivot_levels_to_intraday(
        df,
        pivots,
        timezone=timezone,
        session_close_hour=session_close_hour,
    )
    result = result.dropna(subset=[PIVOT_CENTRAL_COLUMN])
    return result
