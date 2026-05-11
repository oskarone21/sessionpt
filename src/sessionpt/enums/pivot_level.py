"""Pivot level enumeration."""

from enum import Enum

_LEVEL_RANK = {
    "S5": 0,
    "S4": 1,
    "S3": 2,
    "S2": 3,
    "S1": 4,
    "P": 5,
    "R1": 6,
    "R2": 7,
    "R3": 8,
    "R4": 9,
    "R5": 10,
}


class PivotLevel(Enum):
    """Pivot support and resistance levels.

    Levels are ordered from lowest (S5) to highest (R5).
    The pivot point P sits between S1 and R1.
    """

    P = "P"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"
    R5 = "R5"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"

    @classmethod
    def supports(cls) -> list["PivotLevel"]:
        """Return support levels (S1-S5) plus P."""
        return [cls.P, cls.S1, cls.S2, cls.S3, cls.S4, cls.S5]

    @classmethod
    def resistances(cls) -> list["PivotLevel"]:
        """Return resistance levels (R1-R5) plus P."""
        return [cls.P, cls.R1, cls.R2, cls.R3, cls.R4, cls.R5]

    def __lt__(self, other: object) -> bool:
        if isinstance(other, PivotLevel):
            return _LEVEL_RANK[self.value] < _LEVEL_RANK[other.value]
        return NotImplemented
