"""Reusable NumPy-based entry confirmation indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator


EPSILON = 1e-10
DEFAULT_ATR_PERIOD = 14
DEFAULT_VOLUME_LOOKBACK = 20


@njit(cache=False)
def _atr_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    n = len(closes)
    atr = np.full(n, np.nan, dtype=np.float64)
    true_range = np.empty(n, dtype=np.float64)
    true_range[0] = highs[0] - lows[0]

    for index in range(1, n):
        high_low = highs[index] - lows[index]
        high_previous_close = abs(highs[index] - closes[index - 1])
        low_previous_close = abs(lows[index] - closes[index - 1])
        true_range[index] = max(high_low, high_previous_close, low_previous_close)

    if n < period:
        return atr

    seed = 0.0
    for index in range(period):
        seed += true_range[index]
    atr[period - 1] = seed / period

    alpha = 1.0 / period
    one_minus_alpha = 1.0 - alpha
    for index in range(period, n):
        atr[index] = atr[index - 1] * one_minus_alpha + true_range[index] * alpha

    return atr


def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = DEFAULT_ATR_PERIOD,
) -> np.ndarray:
    """Compute Wilder ATR aligned to the input bars."""

    return np.asarray(
        _atr_numba(
            highs.astype(np.float64),
            lows.astype(np.float64),
            closes.astype(np.float64),
            int(period),
        ),
        dtype=np.float64,
    )


def build_atr_mask(atr: np.ndarray, min_percentile: float) -> np.ndarray:
    """Return true where ATR is at or above the requested percentile."""

    valid = ~np.isnan(atr)
    if not np.any(valid):
        return np.zeros(len(atr), dtype=bool)
    threshold = np.percentile(atr[valid], min_percentile)
    return np.asarray(valid & (atr >= threshold), dtype=bool)


@njit(cache=False)
def _vwap_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    session_ids: np.ndarray,
) -> np.ndarray:
    n = len(closes)
    vwap = np.empty(n, dtype=np.float64)
    cumulative_price_volume = 0.0
    cumulative_volume = 0.0
    current_session_id = session_ids[0]

    for index in range(n):
        session_id = session_ids[index]
        if session_id != current_session_id:
            cumulative_price_volume = 0.0
            cumulative_volume = 0.0
            current_session_id = session_id

        typical_price = (highs[index] + lows[index] + closes[index]) / 3.0
        volume = max(volumes[index], 0.0)
        cumulative_price_volume += typical_price * volume
        cumulative_volume += volume
        vwap[index] = (
            cumulative_price_volume / cumulative_volume
            if cumulative_volume > 0.0
            else typical_price
        )

    return vwap


def compute_vwap(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    session_ids: np.ndarray,
) -> np.ndarray:
    """Compute session-reset VWAP aligned to the input bars."""

    return np.asarray(
        _vwap_numba(
            highs.astype(np.float64),
            lows.astype(np.float64),
            closes.astype(np.float64),
            volumes.astype(np.float64),
            session_ids.astype(np.int64),
        ),
        dtype=np.float64,
    )


def build_vwap_long_mask(closes: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    return closes > vwap


def build_vwap_short_mask(closes: np.ndarray, vwap: np.ndarray) -> np.ndarray:
    return closes < vwap


def compute_volume_ratio(
    volumes: np.ndarray,
    lookback: int = DEFAULT_VOLUME_LOOKBACK,
) -> np.ndarray:
    """Compute current volume divided by rolling mean volume."""

    volume = volumes.astype(np.float64)
    rolling_mean = pd.Series(volume).rolling(lookback, min_periods=1).mean().to_numpy()
    safe_mean = np.where(rolling_mean > EPSILON, rolling_mean, 1.0)
    return np.asarray(volume / safe_mean, dtype=np.float64)


def build_volume_mask(volume_ratio: np.ndarray, min_multiplier: float) -> np.ndarray:
    return volume_ratio >= min_multiplier


def compute_wick_ratios(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower and upper wick fractions of total bar range."""

    opens_float = opens.astype(np.float64)
    highs_float = highs.astype(np.float64)
    lows_float = lows.astype(np.float64)
    closes_float = closes.astype(np.float64)

    bar_range = highs_float - lows_float
    safe_range = np.where(bar_range < EPSILON, EPSILON, bar_range)
    body_high = np.maximum(opens_float, closes_float)
    body_low = np.minimum(opens_float, closes_float)

    upper_wick = (highs_float - body_high) / safe_range
    lower_wick = (body_low - lows_float) / safe_range
    return (
        np.clip(lower_wick, 0.0, 1.0).astype(np.float64),
        np.clip(upper_wick, 0.0, 1.0).astype(np.float64),
    )


def build_wick_long_mask(lower_wick_pct: np.ndarray, min_wick_ratio: float) -> np.ndarray:
    return lower_wick_pct >= min_wick_ratio


def build_wick_short_mask(upper_wick_pct: np.ndarray, min_wick_ratio: float) -> np.ndarray:
    return upper_wick_pct >= min_wick_ratio


def compute_nmin_entry_signal(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    levels: np.ndarray,
    n_bars: int,
    is_long: bool,
) -> np.ndarray:
    """Compute rolling N-bar touch and close-confirmation entry signal."""

    if n_bars <= 1:
        touch = (lows <= levels) & (highs >= levels)
        close_mask = closes > levels if is_long else closes < levels
        return np.asarray(touch & close_mask, dtype=bool)

    rolling_min_low = pd.Series(lows).rolling(n_bars, min_periods=n_bars).min().to_numpy()
    rolling_max_high = pd.Series(highs).rolling(n_bars, min_periods=n_bars).max().to_numpy()
    touch_mask = (rolling_min_low <= levels) & (rolling_max_high >= levels)
    close_mask = closes > levels if is_long else closes < levels
    return np.asarray(~np.isnan(rolling_min_low) & touch_mask & close_mask, dtype=bool)


def combine_entry_masks(
    masks: list[np.ndarray | None],
    base_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """AND together non-null boolean masks."""

    active = [mask for mask in masks if mask is not None]
    if base_mask is not None:
        active.insert(0, base_mask)
    if not active:
        return None

    result = active[0].copy()
    for mask in active[1:]:
        result &= mask
    return result
