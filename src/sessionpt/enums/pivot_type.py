"""Pivot type enumeration."""

from enum import Enum


class PivotType(Enum):
    """Supported pivot point calculation methods.

    Each type uses a different formula to compute the daily pivot point
    and support/resistance levels from the previous session's OHLC data.
    """

    TRADITIONAL = "traditional"
    FIBONACCI = "fibonacci"
    WOODIE = "woodie"
    CLASSIC = "classic"
    DM = "dm"
    CAMARILLA = "camarilla"
