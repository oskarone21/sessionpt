"""Session-aware VWAP feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sessionpt.constants import (
    CLOSE_COLUMN,
    DEFAULT_TIMEZONE,
    HIGH_COLUMN,
    LOW_COLUMN,
    VOLUME_COLUMN,
)
from sessionpt.sessions.core import ensure_utc_index, validate_datetime_index

RTH_ACTIVE_COLUMN = "rth_active"
RTH_SESSION_LABEL_COLUMN = "rth_session_label"
DEFAULT_RTH_START_LOCAL = "09:30"
DEFAULT_VWAP_COLUMN = "rth_vwap"


def _time_parts(hhmm: str) -> tuple[int, int]:
    hour, minute = hhmm.split(":")
    return int(hour), int(minute)


def _local_time_mask(
    ts_local: pd.DatetimeIndex,
    start_hhmm: str,
    end_hhmm: str | None,
) -> np.ndarray:
    start_hour, start_minute = _time_parts(start_hhmm)
    after_start = (ts_local.hour > start_hour) | (
        (ts_local.hour == start_hour) & (ts_local.minute >= start_minute)
    )
    if end_hhmm is None:
        return np.asarray(after_start, dtype=bool)

    end_hour, end_minute = _time_parts(end_hhmm)
    before_end = (ts_local.hour < end_hour) | (
        (ts_local.hour == end_hour) & (ts_local.minute < end_minute)
    )
    mask = after_start & before_end
    return np.asarray(mask, dtype=bool)


def add_rth_anchored_vwap(
    df: pd.DataFrame,
    timezone: str = DEFAULT_TIMEZONE,
    rth_start_local: str = DEFAULT_RTH_START_LOCAL,
    rth_end_local: str | None = None,
    vwap_col: str = DEFAULT_VWAP_COLUMN,
) -> pd.DataFrame:
    """Add VWAP reset at each local RTH session start."""

    if VOLUME_COLUMN not in df.columns:
        raise KeyError(f"{VOLUME_COLUMN} column is required for VWAP computation")

    out = df.copy()
    ts_utc = ensure_utc_index(validate_datetime_index(out.index))
    ts_local = ts_utc.tz_convert(timezone)
    active_mask = _local_time_mask(ts_local, rth_start_local, rth_end_local)

    session_label = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    local_dates = ts_local.normalize().tz_localize(None)
    session_label.loc[active_mask] = local_dates[active_mask]

    typical_price = (
        out[HIGH_COLUMN].astype(float)
        + out[LOW_COLUMN].astype(float)
        + out[CLOSE_COLUMN].astype(float)
    ) / 3.0
    volume = out[VOLUME_COLUMN].astype(float)
    price_volume = typical_price * volume

    out[vwap_col] = np.nan
    if active_mask.any():
        active_index = out.index[active_mask]
        groups = session_label.loc[active_index]
        cumulative_price_volume = price_volume.loc[active_index].groupby(groups).cumsum()
        cumulative_volume = volume.loc[active_index].groupby(groups).cumsum()
        vwap = cumulative_price_volume / cumulative_volume.replace(0, np.nan)
        out.loc[active_index, vwap_col] = vwap.values

    out[RTH_ACTIVE_COLUMN] = active_mask
    out[RTH_SESSION_LABEL_COLUMN] = session_label.values
    return out
