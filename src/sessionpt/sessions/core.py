"""Core session ID and mask computation for exchange-aware trading.

Handles overnight sessions (e.g., Asian 18:00-03:00 ET) and DST transitions
by converting timestamps to exchange-local time before computing boundaries.

Session assignment rule:
    A bar belongs to the trading session whose close boundary it is within.
    Bars at hour > session_close_hour start a new session.
    Bars at hour == session_close_hour stay in the current session (they are
    the close bar of the session that is ending).

For example, with session_close_hour=17 (CME Gold):
    - 16:59 ET -> current session
    - 17:00 ET -> current session (close bar)
    - 18:00 ET -> next session (new session starts)

This correctly groups bars such that a Sunday 18:00 bar and a Monday 16:59 bar
belong to the same Monday trading session, producing correct daily OHLC values
for pivot calculation.
"""

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd

from sessionpt.constants import (
    ALL_HOURS_SESSION,
    DEFAULT_SESSION_CLOSE_HOUR,
    DEFAULT_TIMEZONE,
    UTC_TIMEZONE,
)


def ensure_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    """Convert any DatetimeIndex to UTC-aware.

    Parameters
    ----------
    index : pd.Index
        DatetimeIndex, possibly timezone-naive or in a different timezone.

    Returns
    -------
    pd.DatetimeIndex
        UTC-aware DatetimeIndex.
    """
    ts = pd.DatetimeIndex(index)
    ts = ts.tz_localize(UTC_TIMEZONE) if ts.tz is None else ts.tz_convert(UTC_TIMEZONE)
    return ts


def validate_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """Return a unique, monotonically increasing DatetimeIndex.

    Causal market-data calculations depend on row order.  Failing fast avoids
    silently incorporating later observations into earlier aggregates.
    """
    ts = pd.DatetimeIndex(index)
    if not ts.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be monotonically increasing")
    if not ts.is_unique:
        raise ValueError("DatetimeIndex must not contain duplicate timestamps")
    return ts


def get_session_labels(
    index: pd.Index,
    timezone: str = DEFAULT_TIMEZONE,
    session_close_hour: int = DEFAULT_SESSION_CLOSE_HOUR,
) -> pd.DatetimeIndex:
    """Compute session labels for each bar based on exchange timezone and close hour.

    Each bar is assigned to the trading session whose date label it belongs to.
    Bars after the session close hour (e.g., 18:00+ ET for CME) get pushed
    to the next calendar day, correctly grouping overnight sessions.

    Parameters
    ----------
    index : pd.Index
        DatetimeIndex of bar timestamps.
    timezone : str
        Exchange timezone string (e.g., 'America/New_York').
    session_close_hour : int
        Hour (in exchange-local time) when the session closes.
        Bars at this hour stay in the *current* session (close bar).
        Bars after this hour start the next session.

    Returns
    -------
    pd.DatetimeIndex
        Date labels representing each bar's trading session.
    """
    ts_utc = ensure_utc_index(validate_datetime_index(index))
    ts_local = ts_utc.tz_convert(timezone)
    is_after_close = (ts_local.hour > session_close_hour).astype(np.int8)
    local_midnights = ts_local.tz_localize(None).normalize()
    labels = local_midnights + pd.to_timedelta(is_after_close, unit="D")
    return pd.DatetimeIndex(labels).tz_localize(timezone)


def get_session_ids(
    index: pd.Index,
    timezone: str = DEFAULT_TIMEZONE,
    session_close_hour: int = DEFAULT_SESSION_CLOSE_HOUR,
) -> np.ndarray:
    """Compute integer session IDs for fast comparison.

    Same logic as get_session_labels but returns int64 ordinals
    suitable for Numba-compiled loops and NumPy groupby operations.

    Parameters
    ----------
    index : pd.Index
        DatetimeIndex of bar timestamps.
    timezone : str
        Exchange timezone string.
    session_close_hour : int
        Hour when the session closes (in exchange-local time).

    Returns
    -------
    np.ndarray
        Integer session IDs (days since epoch).
    """
    labels = get_session_labels(
        index=index,
        timezone=timezone,
        session_close_hour=session_close_hour,
    )
    values = labels.tz_localize(None).values.astype("datetime64[D]").astype(np.int64)
    return np.asarray(values, dtype=np.int64)


def get_local_hours(index: pd.Index, timezone: str = DEFAULT_TIMEZONE) -> np.ndarray:
    """Extract exchange-local hour values for each bar timestamp.

    Parameters
    ----------
    index : pd.Index
        DatetimeIndex of bar timestamps.
    timezone : str
        Exchange timezone string.

    Returns
    -------
    np.ndarray
        Array of hour values (0-23) in exchange-local time.
    """
    ts_local = ensure_utc_index(index).tz_convert(timezone)
    return np.asarray(ts_local.hour, dtype=np.int16)


def build_session_mask(
    index: pd.Index,
    session_name: str,
    sessions: Mapping[str, Mapping[str, object]],
    timezone: str = DEFAULT_TIMEZONE,
) -> np.ndarray:
    """Build a boolean mask for bars within a named trading session.

    Parameters
    ----------
    index : pd.Index
        DatetimeIndex of bar timestamps.
    session_name : str
        Key in the sessions dictionary (e.g., 'asian', 'london').
    sessions : Mapping
        Dictionary of session definitions, each with 'start', 'end',
        and 'overnight' keys.
    timezone : str
        Exchange timezone string.

    Returns
    -------
    np.ndarray
        Boolean mask with True for bars within the specified session.
    """
    if session_name == ALL_HOURS_SESSION:
        return np.ones(len(index), dtype=bool)

    session = sessions[session_name]
    hours = get_local_hours(index=index, timezone=timezone)
    start_hour = int(cast(int, session["start"]))
    end_hour = int(cast(int, session["end"]))

    if bool(session["overnight"]):
        mask = (hours >= start_hour) | (hours < end_hour)
    else:
        mask = (hours >= start_hour) & (hours < end_hour)
    return np.asarray(mask, dtype=bool)
