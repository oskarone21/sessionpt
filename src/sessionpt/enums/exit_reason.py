"""Exit reason enumeration."""

from enum import Enum


class ExitReason(Enum):
    """Reason a trade was closed.

    Attributes:
        SL: Stop loss was hit.
        TP: Take profit target was reached.
        TRAILING_SL: Trailing stop was triggered after activation.
        BREAKEVEN: Breakeven stop was hit after breakeven activation.
        EOD: Position was closed at end of day.
        MAX_HOLD: Maximum holding period was reached.
    """

    SL = "SL"
    TP = "TP"
    TRAILING_SL = "TRAILING_SL"
    BREAKEVEN = "BREAKEVEN"
    EOD = "EOD"
    MAX_HOLD = "MAX_HOLD"
