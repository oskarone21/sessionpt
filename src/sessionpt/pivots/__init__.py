"""Pivot point calculation and session-aware OHLC aggregation.

This module provides functions for computing daily OHLC bars using
exchange session boundaries (not calendar dates), and calculating
six types of pivot point levels from those bars.

Session-aware OHLC is critical for futures markets where a "trading day"
spans midnight (e.g., CME Gold: Sunday 18:00 ET -> Monday 17:00 ET).
Using calendar-day resampling incorrectly splits each session across two
days, producing wrong Open/Close values and corrupted pivot levels.
"""

from sessionpt.constants import PIVOT_LEVEL_COLUMNS
from sessionpt.pivots.calculator import (
    calculate_camarilla_pivots,
    calculate_classic_pivots,
    calculate_dm_pivots,
    calculate_fibonacci_pivots,
    calculate_pivot_levels,
    calculate_traditional_pivots,
    calculate_woodie_pivots,
)
from sessionpt.pivots.confluence import (
    ConfluenceZone,
    calculate_all_pivot_types,
    calculate_pivot_confluence,
    find_confluence_levels,
    get_confluence_at_level,
)
from sessionpt.pivots.ohlc import (
    add_pivot_levels_to_intraday,
    calculate_daily_ohlc,
    prepare_data_with_pivots,
)
from sessionpt.pivots.shifts import (
    RESISTANCE_PIVOT_COLUMNS,
    SUPPORT_AND_CENTRAL_PIVOT_COLUMNS,
    apply_directional_pivot_shift,
    prepare_shifted_pivot_data,
)

__all__ = [
    "PIVOT_LEVEL_COLUMNS",
    "RESISTANCE_PIVOT_COLUMNS",
    "SUPPORT_AND_CENTRAL_PIVOT_COLUMNS",
    "ConfluenceZone",
    "add_pivot_levels_to_intraday",
    "apply_directional_pivot_shift",
    "calculate_all_pivot_types",
    "calculate_camarilla_pivots",
    "calculate_classic_pivots",
    "calculate_daily_ohlc",
    "calculate_dm_pivots",
    "calculate_fibonacci_pivots",
    "calculate_pivot_confluence",
    "calculate_pivot_levels",
    "calculate_traditional_pivots",
    "calculate_woodie_pivots",
    "find_confluence_levels",
    "get_confluence_at_level",
    "prepare_data_with_pivots",
    "prepare_shifted_pivot_data",
]
