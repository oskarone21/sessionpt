"""Pivot confluence utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sessionpt.enums import PivotType
from sessionpt.pivots.calculator import (
    calculate_camarilla_pivots,
    calculate_classic_pivots,
    calculate_fibonacci_pivots,
    calculate_traditional_pivots,
    calculate_woodie_pivots,
)
from sessionpt.pivots.ohlc import calculate_daily_ohlc

DEFAULT_ZONE_TOLERANCE_PCT = 0.15
DEFAULT_MIN_CONFLUENCE = 2
CONFLUENCE_STRENGTH_DENOMINATOR = 10.0
DATE_COLUMN = "date"
CONFLUENCE_ZONES_COLUMN = "confluence_zones"
MAX_CONFLUENCE_STRENGTH_COLUMN = "max_confluence_strength"
SUPPORT_CONFLUENCE_COLUMN = "support_confluence"
RESISTANCE_CONFLUENCE_COLUMN = "resistance_confluence"

PIVOT_CALCULATORS = {
    PivotType.TRADITIONAL: calculate_traditional_pivots,
    PivotType.FIBONACCI: calculate_fibonacci_pivots,
    PivotType.CAMARILLA: calculate_camarilla_pivots,
    PivotType.WOODIE: calculate_woodie_pivots,
    PivotType.CLASSIC: calculate_classic_pivots,
}


@dataclass(frozen=True)
class ConfluenceZone:
    center_price: float
    level_count: int
    levels: tuple[tuple[str, str, float], ...]
    strength: float


def _normalise_pivot_type(pivot_type: PivotType | str) -> PivotType:
    if isinstance(pivot_type, PivotType):
        return pivot_type
    return PivotType(str(pivot_type).lower())


def calculate_all_pivot_types(
    high: float,
    low: float,
    close: float,
    pivot_types: Iterable[PivotType | str] | None = None,
) -> dict[str, dict[str, float]]:
    """Calculate pivot levels for each requested pivot type."""

    selected = (
        list(PIVOT_CALCULATORS)
        if pivot_types is None
        else [_normalise_pivot_type(pivot_type) for pivot_type in pivot_types]
    )
    return {
        pivot_type.value: PIVOT_CALCULATORS[pivot_type](high, low, close)
        for pivot_type in selected
        if pivot_type in PIVOT_CALCULATORS
    }


def find_confluence_levels(
    all_pivots: dict[str, dict[str, float]],
    tolerance_pct: float = DEFAULT_ZONE_TOLERANCE_PCT,
) -> list[ConfluenceZone]:
    """Find price zones where multiple pivot levels align."""

    all_levels: list[tuple[str, str, float]] = []
    for pivot_type, levels in all_pivots.items():
        for level_name, price in levels.items():
            if price is not None and not np.isnan(price):
                all_levels.append((pivot_type, level_name, float(price)))

    if not all_levels:
        return []

    all_levels.sort(key=lambda item: item[2])
    zones: list[ConfluenceZone] = []
    used: set[int] = set()

    for index, (pivot_type, level_name, price) in enumerate(all_levels):
        if index in used:
            continue

        cluster = [(pivot_type, level_name, price)]
        used.add(index)
        tolerance = price * (tolerance_pct / 100)

        for candidate_index, candidate in enumerate(all_levels):
            if candidate_index in used:
                continue
            if abs(candidate[2] - price) <= tolerance:
                cluster.append(candidate)
                used.add(candidate_index)

        if len(cluster) >= DEFAULT_MIN_CONFLUENCE:
            unique_types = len({candidate[0] for candidate in cluster})
            strength = min(1.0, (len(cluster) * unique_types) / CONFLUENCE_STRENGTH_DENOMINATOR)
            zones.append(
                ConfluenceZone(
                    center_price=float(np.mean([candidate[2] for candidate in cluster])),
                    level_count=len(cluster),
                    levels=tuple(cluster),
                    strength=float(strength),
                )
            )

    zones.sort(key=lambda zone: zone.strength, reverse=True)
    return zones


def calculate_pivot_confluence(
    df: pd.DataFrame,
    pivot_types: Iterable[PivotType | str] | None = None,
    tolerance_pct: float = DEFAULT_ZONE_TOLERANCE_PCT,
    min_confluence: int = DEFAULT_MIN_CONFLUENCE,
    timezone: str = "America/New_York",
    session_close_hour: int = 17,
) -> pd.DataFrame:
    """Calculate daily support/resistance confluence diagnostics."""

    selected = (
        list(PIVOT_CALCULATORS)
        if pivot_types is None
        else [_normalise_pivot_type(pivot_type) for pivot_type in pivot_types]
    )
    daily = calculate_daily_ohlc(
        df,
        timezone=timezone,
        session_close_hour=session_close_hour,
    )
    rows: list[dict[str, object]] = []

    for index in range(len(daily) - 1):
        row = daily.iloc[index]
        next_date = daily.index[index + 1]
        close = float(row["Close"])
        all_pivots = calculate_all_pivot_types(
            high=float(row["High"]),
            low=float(row["Low"]),
            close=close,
            pivot_types=selected,
        )
        zones = [
            zone
            for zone in find_confluence_levels(all_pivots, tolerance_pct)
            if zone.level_count >= min_confluence
        ]
        support_zones = [zone for zone in zones if zone.center_price < close]
        resistance_zones = [zone for zone in zones if zone.center_price >= close]
        rows.append(
            {
                DATE_COLUMN: next_date,
                CONFLUENCE_ZONES_COLUMN: len(zones),
                MAX_CONFLUENCE_STRENGTH_COLUMN: max(
                    [zone.strength for zone in zones],
                    default=0.0,
                ),
                SUPPORT_CONFLUENCE_COLUMN: max(
                    [zone.strength for zone in support_zones],
                    default=0.0,
                ),
                RESISTANCE_CONFLUENCE_COLUMN: max(
                    [zone.strength for zone in resistance_zones],
                    default=0.0,
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                CONFLUENCE_ZONES_COLUMN,
                MAX_CONFLUENCE_STRENGTH_COLUMN,
                SUPPORT_CONFLUENCE_COLUMN,
                RESISTANCE_CONFLUENCE_COLUMN,
            ]
        )
    return pd.DataFrame(rows).set_index(DATE_COLUMN)


def get_confluence_at_level(
    high: float,
    low: float,
    close: float,
    target_price: float,
    pivot_types: Iterable[PivotType | str] | None = None,
    tolerance_pct: float = DEFAULT_ZONE_TOLERANCE_PCT,
) -> tuple[int, list[str]]:
    """Return count and labels of pivot levels aligned with target price."""

    tolerance = target_price * (tolerance_pct / 100)
    aligned: list[str] = []
    all_pivots = calculate_all_pivot_types(high, low, close, pivot_types)
    for pivot_type, levels in all_pivots.items():
        for level_name, price in levels.items():
            if abs(float(price) - target_price) <= tolerance:
                aligned.append(f"{pivot_type}_{level_name}")
    return len(aligned), aligned
