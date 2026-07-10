"""Instrument, execution, and policy specifications for backtesting.

All specs are frozen dataclasses for immutability and hashability.
These are the primary configuration objects passed to backtesting functions.
"""

from dataclasses import dataclass
from math import isfinite

from sessionpt.constants import (
    DEFAULT_ETH_END_LOCAL,
    DEFAULT_ETH_START_LOCAL,
    DEFAULT_SESSION_CLOSE_HOUR,
    DEFAULT_TIMEZONE,
)

TP_MODE_FIXED_TICKS = "fixed_ticks"


@dataclass(frozen=True)
class InstrumentSpec:
    """Specification for a futures instrument.

    Encodes the tick mechanics and transaction costs that determine
    P&L calculation in the backtester.

    Parameters
    ----------
    symbol : str
        Instrument symbol (e.g., 'GC', 'ES').
    tick_size : float
        Minimum price increment (e.g., 0.10 for GC gold).
    tick_value : float
        Dollar value per tick (e.g., $10 for GC).
    commission_round_trip : float
        Total round-trip commission per trade in dollars.
    slippage_ticks : float
        Estimated slippage in ticks per side (total round-trip = 2x).
    timezone : str
        Exchange timezone string (e.g., 'America/New_York').
    session_close_hour : int
        Hour (exchange-local) when the trading session closes.
        Used for session grouping and EOD position management.
    """

    symbol: str
    tick_size: float
    tick_value: float
    commission_round_trip: float
    slippage_ticks: float
    timezone: str = DEFAULT_TIMEZONE
    session_close_hour: int = DEFAULT_SESSION_CLOSE_HOUR

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        numeric_values = (
            self.tick_size,
            self.tick_value,
            self.commission_round_trip,
            self.slippage_ticks,
        )
        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("instrument prices and costs must be finite")
        if self.tick_size <= 0 or self.tick_value <= 0:
            raise ValueError("tick_size and tick_value must be positive")
        if self.commission_round_trip < 0 or self.slippage_ticks < 0:
            raise ValueError("transaction costs must be non-negative")
        if not isinstance(self.session_close_hour, int) or isinstance(
            self.session_close_hour, bool
        ):
            raise ValueError("session_close_hour must be an integer")
        if not 0 <= self.session_close_hour <= 23:
            raise ValueError("session_close_hour must be between 0 and 23")

    @property
    def slippage_cost(self) -> float:
        """Total round-trip slippage cost in dollars."""
        return 2.0 * float(self.slippage_ticks) * float(self.tick_value)

    @property
    def total_cost_per_trade(self) -> float:
        """Total round-trip cost per trade (commission + slippage) in dollars."""
        return float(self.commission_round_trip) + self.slippage_cost


@dataclass(frozen=True)
class SessionSpec:
    """Trading session time specification.

    Parameters
    ----------
    name : str
        Session name (e.g., 'asian', 'new_york').
    timezone_basis : str
        Timezone for the start/end times (e.g., 'America/New_York').
    start_local_time : str
        Session start in local time (HH:MM format).
    end_local_time : str
        Session end in local time (HH:MM format).
    is_overnight : bool
        Whether the session spans midnight (e.g., Asian 18:00-03:00).
    """

    name: str
    timezone_basis: str = DEFAULT_TIMEZONE
    start_local_time: str = DEFAULT_ETH_START_LOCAL
    end_local_time: str = DEFAULT_ETH_END_LOCAL
    is_overnight: bool = False


@dataclass(frozen=True)
class BreakevenPolicy:
    """Breakeven stop-loss policy.

    When price moves a percentage of the way from entry to take-profit,
    the stop-loss is moved to the entry price (breakeven).

    Parameters
    ----------
    enabled : bool
        Whether breakeven is active.
    trigger_pct_to_tp : float
        Percentage (0-1) of the distance to TP at which breakeven activates.
        0.5 means move stop to entry when price reaches 50% of TP distance.
    """

    enabled: bool = False
    trigger_pct_to_tp: float = 0.5

    def __post_init__(self) -> None:
        if not isfinite(self.trigger_pct_to_tp) or not 0.0 <= self.trigger_pct_to_tp <= 1.0:
            raise ValueError("trigger_pct_to_tp must be between 0 and 1")


@dataclass(frozen=True)
class TrailingStopPolicy:
    """Trailing stop policy.

    After activation, the stop-loss trails the price by a fixed number
    of ticks, locking in profit as the trade moves favorably.

    Parameters
    ----------
    enabled : bool
        Whether trailing stop is active.
    trigger_ticks : float or None
        Number of favorable ticks before the trailing stop activates.
    lock_ticks : float or None
        Number of ticks to trail behind the favorable price after activation.
        For a LONG position, the new stop is set at entry_price + lock_ticks.
    """

    enabled: bool = False
    trigger_ticks: float | None = None
    lock_ticks: float | None = None

    def __post_init__(self) -> None:
        if self.enabled and (self.trigger_ticks is None or self.lock_ticks is None):
            raise ValueError("enabled trailing stops require trigger_ticks and lock_ticks")
        if self.trigger_ticks is not None and (
            not isfinite(self.trigger_ticks) or self.trigger_ticks < 0
        ):
            raise ValueError("trigger_ticks must be non-negative")
        if self.lock_ticks is not None and (not isfinite(self.lock_ticks) or self.lock_ticks < 0):
            raise ValueError("lock_ticks must be non-negative")


@dataclass(frozen=True)
class ExecutionPolicy:
    """Execution policy controlling trade management rules.

    Parameters
    ----------
    close_at_eod : bool
        If True, force-close any open position at end of trading session.
        If False, let positions run across sessions (up to max_days_to_hold).
    max_trades_per_session : int
        Maximum number of trades per trading session.
    one_trade_per_level : bool
        If True, only allow one trade per pivot level per session.
    allow_concurrent_positions : bool
        If True, multiple positions can be open simultaneously.
        If False (default), a new trade can only be entered after the
        previous one has exited.
    max_days_to_hold : int
        Maximum number of days to hold a position when close_at_eod is False.
    """

    close_at_eod: bool = True
    max_trades_per_session: int = 5
    one_trade_per_level: bool = True
    allow_concurrent_positions: bool = False
    max_days_to_hold: int = 5

    def __post_init__(self) -> None:
        integer_values = (self.max_trades_per_session, self.max_days_to_hold)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values):
            raise ValueError("trade and holding limits must be integers")
        if self.max_trades_per_session <= 0:
            raise ValueError("max_trades_per_session must be positive")
        if self.max_days_to_hold <= 0:
            raise ValueError("max_days_to_hold must be positive")


@dataclass(frozen=True)
class ExitPolicy:
    """Exit policy combining stop-loss, take-profit, and trailing parameters.

    Parameters
    ----------
    sl_ticks : float or None
        Stop-loss distance in ticks from entry price.
    tp_ticks : float or None
        Take-profit distance in ticks from entry price.
    tp_mode : str
        Take-profit mode: 'fixed_ticks' (default).
    trailing : TrailingStopPolicy or None
        Trailing stop configuration.
    breakeven : BreakevenPolicy or None
        Breakeven stop configuration.
    """

    sl_ticks: float | None = None
    tp_ticks: float | None = None
    tp_mode: str = TP_MODE_FIXED_TICKS
    trailing: TrailingStopPolicy | None = None
    breakeven: BreakevenPolicy | None = None
