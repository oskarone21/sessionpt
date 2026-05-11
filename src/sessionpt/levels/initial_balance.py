"""Initial-balance level features for intraday futures data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sessionpt.constants import DEFAULT_TIMEZONE, HIGH_COLUMN, LOW_COLUMN
from sessionpt.features import add_rth_anchored_vwap
from sessionpt.sessions import ensure_utc_index

IB_SESSION_LABEL_COLUMN = "ib_session_label"
IB_SESSION_ID_COLUMN = "ib_session_id"
IB_IN_WINDOW_COLUMN = "ib_in_window"
IB_COMPLETE_COLUMN = "ib_complete"
IB_WINDOW_BAR_COUNT_COLUMN = "ib_window_bar_count"
IB_RANGE_PCT_COLUMN = "ib_range_pct_lookback20"
IB_RANGE_COUNT_COLUMN = "ib_range_lookback_count"
RTH_ACTIVE_COLUMN = "rth_active"
RTH_VWAP_COLUMN = "rth_vwap"
DEFAULT_IB_START_LOCAL = "09:30"
DEFAULT_IB_WINDOW_MINUTES = 60
DEFAULT_LOOKBACK_SESSIONS = 20

IBH_COLUMN = "IBH"
IBL_COLUMN = "IBL"
IB_RANGE_COLUMN = "IBRange"
IB_25_COLUMN = "IB_25"
IB_33_COLUMN = "IB_33"
IB_50_COLUMN = "IB_50"
IB_EXT_UP_25_COLUMN = "IB_EXT_UP_25"
IB_EXT_DN_25_COLUMN = "IB_EXT_DN_25"
TEMP_LOCAL_TS_COLUMN = "_ts_local"
IB_LEVEL_COLUMNS = (
    IBH_COLUMN,
    IBL_COLUMN,
    IB_RANGE_COLUMN,
    IB_25_COLUMN,
    IB_33_COLUMN,
    IB_50_COLUMN,
    IB_EXT_UP_25_COLUMN,
    IB_EXT_DN_25_COLUMN,
)


def _time_parts(hhmm: str) -> tuple[int, int]:
    hour, minute = hhmm.split(":")
    return int(hour), int(minute)


def _build_local_time_mask(
    ts_local: pd.DatetimeIndex,
    start_hhmm: str,
    end_hhmm: str,
) -> np.ndarray:
    start_hour, start_minute = _time_parts(start_hhmm)
    end_hour, end_minute = _time_parts(end_hhmm)
    after_start = (ts_local.hour > start_hour) | (
        (ts_local.hour == start_hour) & (ts_local.minute >= start_minute)
    )
    before_end = (ts_local.hour < end_hour) | (
        (ts_local.hour == end_hour) & (ts_local.minute < end_minute)
    )
    mask = after_start & before_end
    return np.asarray(mask, dtype=bool)


def _add_minutes(hhmm: str, minutes: int) -> str:
    hour, minute = _time_parts(hhmm)
    total = (hour * 60 + minute + int(minutes)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def add_initial_balance_levels(
    df: pd.DataFrame,
    timezone: str = DEFAULT_TIMEZONE,
    ib_start_local: str = DEFAULT_IB_START_LOCAL,
    ib_window_minutes: int = DEFAULT_IB_WINDOW_MINUTES,
    rth_end_local: str | None = None,
    include_rth_vwap: bool = True,
    rth_start_local: str | None = None,
) -> pd.DataFrame:
    """Add initial-balance levels and diagnostics without look-ahead."""

    out = df.copy()
    if out.empty:
        return out

    ts_utc = ensure_utc_index(out.index)
    ts_local = ts_utc.tz_convert(timezone)
    out[TEMP_LOCAL_TS_COLUMN] = ts_local
    local_dates = ts_local.normalize().tz_localize(None)
    out[IB_SESSION_LABEL_COLUMN] = local_dates.values

    rth_start = ib_start_local if rth_start_local is None else rth_start_local
    if rth_end_local is not None:
        out[RTH_ACTIVE_COLUMN] = _build_local_time_mask(ts_local, rth_start, rth_end_local)
    else:
        out[RTH_ACTIVE_COLUMN] = True

    ib_end_local = _add_minutes(ib_start_local, ib_window_minutes)
    out[IB_IN_WINDOW_COLUMN] = _build_local_time_mask(ts_local, ib_start_local, ib_end_local)

    for column in IB_LEVEL_COLUMNS:
        out[column] = np.nan
    out[IB_COMPLETE_COLUMN] = False
    out[IB_WINDOW_BAR_COUNT_COLUMN] = 0
    out[IB_RANGE_PCT_COLUMN] = np.nan
    out[IB_RANGE_COUNT_COLUMN] = 0
    out[IB_SESSION_ID_COLUMN] = (
        pd.Series(local_dates).astype("datetime64[ns]").astype("int64").to_numpy()
    )

    for _, group in out.groupby(IB_SESSION_LABEL_COLUMN, sort=True):
        ib_group = group[group[IB_IN_WINDOW_COLUMN]]
        if ib_group.empty:
            continue

        bar_count = len(ib_group)
        ib_high = float(ib_group[HIGH_COLUMN].max())
        ib_low = float(ib_group[LOW_COLUMN].min())
        ib_range = ib_high - ib_low
        out.loc[group.index, IB_WINDOW_BAR_COUNT_COLUMN] = int(bar_count)

        if bar_count < ib_window_minutes or not np.isfinite(ib_range) or ib_range <= 0:
            continue

        ib_end_time = ib_group[TEMP_LOCAL_TS_COLUMN].max()
        eligible_index = group.index[group[TEMP_LOCAL_TS_COLUMN] > ib_end_time]
        if len(eligible_index) == 0:
            continue

        out.loc[eligible_index, IBH_COLUMN] = ib_high
        out.loc[eligible_index, IBL_COLUMN] = ib_low
        out.loc[eligible_index, IB_RANGE_COLUMN] = ib_range
        out.loc[eligible_index, IB_25_COLUMN] = ib_low + 0.25 * ib_range
        out.loc[eligible_index, IB_33_COLUMN] = ib_low + 0.33 * ib_range
        out.loc[eligible_index, IB_50_COLUMN] = ib_low + 0.50 * ib_range
        out.loc[eligible_index, IB_EXT_UP_25_COLUMN] = ib_high + 0.25 * ib_range
        out.loc[eligible_index, IB_EXT_DN_25_COLUMN] = ib_low - 0.25 * ib_range
        out.loc[eligible_index, IB_COMPLETE_COLUMN] = True

    session_ranges = out.groupby(IB_SESSION_LABEL_COLUMN, sort=True)[IB_RANGE_COLUMN].max()
    session_complete = out.groupby(IB_SESSION_LABEL_COLUMN, sort=True)[IB_COMPLETE_COLUMN].max()
    session_df = pd.DataFrame({"ib_range": session_ranges, "ib_complete": session_complete})
    session_df = session_df[session_df["ib_complete"] & session_df["ib_range"].notna()].copy()

    if not session_df.empty:
        percentile_by_label: dict[pd.Timestamp, float] = {}
        count_by_label: dict[pd.Timestamp, int] = {}
        ranges = session_df["ib_range"].to_numpy(dtype=float)
        labels = session_df.index.to_list()
        for index, label in enumerate(labels):
            start = max(0, index - DEFAULT_LOOKBACK_SESSIONS)
            reference = ranges[start:index]
            if reference.size == 0:
                percentile_by_label[label] = np.nan
                count_by_label[label] = 0
            else:
                percentile_by_label[label] = float((reference <= ranges[index]).mean() * 100.0)
                count_by_label[label] = int(reference.size)

        for label, percentile in percentile_by_label.items():
            session_mask = (out[IB_SESSION_LABEL_COLUMN] == label) & out[IB_COMPLETE_COLUMN]
            if session_mask.any():
                out.loc[session_mask, IB_RANGE_PCT_COLUMN] = percentile
                out.loc[session_mask, IB_RANGE_COUNT_COLUMN] = count_by_label[label]

    if include_rth_vwap:
        out = add_rth_anchored_vwap(
            out,
            timezone=timezone,
            rth_start_local=rth_start,
            rth_end_local=rth_end_local,
            vwap_col=RTH_VWAP_COLUMN,
        )

    out.drop(columns=[TEMP_LOCAL_TS_COLUMN], inplace=True, errors="ignore")
    return out
