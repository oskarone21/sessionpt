"""Trade direction enumeration."""

from enum import Enum


class Direction(Enum):
    """Trade direction for entry signals.

    Values are human-readable strings rather than numeric signs,
    making serialization and debugging clearer.
    """

    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        """Return +1 for LONG, -1 for SHORT.

        Useful for P&L calculation where direction determines
        whether price movement is favorable or adverse.
        """
        return 1 if self == Direction.LONG else -1
