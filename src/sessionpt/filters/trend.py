"""EMA/SMA trend bias filter for trade signal refinement.

Computes EMA/SMA crossover on a higher timeframe and maps the resulting
bias back to the original (typically 1-minute) bars. A one-bar lag is
applied to eliminate look-ahead bias: the bias determined at HTF candle
close is only usable from the next HTF bar onward.

Two public functions are provided:

- ``calculate_trend_bias`` - compute the raw bias series (1 = bullish, -1 = bearish).
- ``filter_by_trend`` - apply the bias to mask rows that align with a trade direction.

A ``TrendFilterConfig`` dataclass is included for convenience when storing
and passing filter configurations through pipelines, but the core functions
accept plain parameters and do not depend on it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sessionpt.enums.direction import Direction
from sessionpt.sessions.core import validate_datetime_index


@dataclass(frozen=True)
class TrendFilterConfig:
    """Configuration for a trend bias filter.

    Parameters
    ----------
    name : str
        Human-readable label (e.g., 'ema5_sma21_1h').
    ema_period : int or None
        EMA lookback period. None disables the filter (baseline).
    sma_period : int or None
        SMA lookback period. None disables the filter (baseline).
    htf_timeframe : str or None
        Pandas resample frequency string for the higher timeframe
        (e.g., '1h', '4h'). None disables the filter.
    """

    name: str = "baseline"
    ema_period: int | None = None
    sma_period: int | None = None
    htf_timeframe: str | None = None

    @property
    def is_baseline(self) -> bool:
        """Return True if this config represents the no-filter baseline."""
        return self.ema_period is None or self.sma_period is None or self.htf_timeframe is None


def calculate_trend_bias(
    df: pd.DataFrame,
    ema_period: int,
    sma_period: int,
    timeframe: str | None = None,
    htf_timeframe: str | None = None,
) -> pd.Series:
    """Calculate trend bias from EMA/SMA crossover on a higher timeframe.

    The bias is +1 (bullish) when EMA > SMA, -1 (bearish) when EMA < SMA,
    and 0 when the averages are equal.
    A one-bar lag is applied on the higher timeframe to prevent look-ahead
    bias: the bias from HTF candle close at time *T* is only valid from *T+1*
    onward.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'Open', 'High', 'Low', 'Close' columns and a
        timezone-aware DatetimeIndex.
    ema_period : int
        Lookback period for the exponential moving average.
    sma_period : int
        Lookback period for the simple moving average.
    timeframe : str
        Pandas resample frequency for the higher timeframe (e.g., '1h', '4h').
    htf_timeframe : str
        Backward-compatible alias for ``timeframe``.

    Returns
    -------
    pd.Series
        Series of +1 / 0 / -1 values aligned to ``df.index``, with NaN during
        the warm-up period where the SMA has insufficient data.
    """
    selected_timeframe = timeframe or htf_timeframe
    if selected_timeframe is None:
        raise ValueError("calculate_trend_bias requires timeframe or htf_timeframe")
    if ema_period <= 0 or sma_period <= 0:
        raise ValueError("ema_period and sma_period must be positive")
    validate_datetime_index(df.index)

    htf_df = (
        df[["Open", "High", "Low", "Close"]]
        .resample(selected_timeframe)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )

    htf_df["ema"] = htf_df["Close"].ewm(span=ema_period, adjust=False).mean()
    htf_df["sma"] = htf_df["Close"].rolling(window=sma_period).mean()

    valid = htf_df[["ema", "sma"]].notna().all(axis=1)
    htf_df["trend_bias"] = np.nan
    htf_df.loc[valid & (htf_df["ema"] > htf_df["sma"]), "trend_bias"] = 1.0
    htf_df.loc[valid & (htf_df["ema"] < htf_df["sma"]), "trend_bias"] = -1.0
    htf_df.loc[valid & (htf_df["ema"] == htf_df["sma"]), "trend_bias"] = 0.0

    htf_df["trend_bias_lagged"] = htf_df["trend_bias"].shift(1)

    trend_series = htf_df["trend_bias_lagged"].reindex(df.index, method="ffill")
    return trend_series


def filter_by_trend(
    df: pd.DataFrame,
    trend_bias: pd.Series,
    direction: Direction,
) -> pd.DataFrame:
    """Filter a DataFrame to rows where the trend bias aligns with a trade direction.

    Parameters
    ----------
    df : pd.DataFrame
        Price data (same index used for the trend bias).
    trend_bias : pd.Series
        Bias series from ``calculate_trend_bias`` (values: +1, -1, or NaN).
    direction : Direction
        Trade direction to filter for. LONG keeps rows where bias == +1,
        SHORT keeps rows where bias == -1.

    Returns
    -------
    pd.DataFrame
        Subset of *df* where the trend bias aligns with *direction*.
        Rows where the bias is NaN (warm-up period) are excluded.
        If all bias values are NaN or zero, returns an empty frame.
    """
    aligned_bias = trend_bias.reindex(df.index, method="ffill")

    valid_mask = aligned_bias.notna()

    if direction == Direction.LONG:
        return df[valid_mask & (aligned_bias == 1)]
    if direction == Direction.SHORT:
        return df[valid_mask & (aligned_bias == -1)]
    raise ValueError(f"Unsupported direction: {direction}")
