"""Pre-defined session configurations for CME futures markets.

Provides ready-to-use session definitions for common futures instruments,
with correct exchange timezone and session boundaries (including the 1-hour
break between ETH close and RTH open).
"""

from dataclasses import dataclass

from sessionpt.constants import (
    ALL_HOURS_SESSION,
    DEFAULT_ETH_END_LOCAL,
    DEFAULT_ETH_START_LOCAL,
    DEFAULT_SESSION_CLOSE_HOUR,
    DEFAULT_TIMEZONE,
)

DEFAULT_GLOBEX_RTH_START_LOCAL = "08:20"
DEFAULT_GLOBEX_RTH_END_LOCAL = "13:30"
DEFAULT_INDEX_RTH_START_LOCAL = "09:30"
DEFAULT_INDEX_RTH_END_LOCAL = "16:00"
DEFAULT_CRUDE_RTH_START_LOCAL = "09:00"
DEFAULT_CRUDE_RTH_END_LOCAL = "14:30"
DEFAULT_BOND_RTH_END_LOCAL = "15:00"

GC_SYMBOL = "GC"
ES_SYMBOL = "ES"
NQ_SYMBOL = "NQ"
CL_SYMBOL = "CL"
ZB_SYMBOL = "ZB"


@dataclass(frozen=True)
class SessionPreset:
    """Pre-defined trading session configuration.

    Attributes
    ----------
    symbol : str
        Instrument symbol (e.g., 'GC', 'ES').
    name : str
        Human-readable session name.
    timezone : str
        Exchange timezone (e.g., 'America/New_York').
    eth_start_local : str
        Electronic trading session start time (HH:MM in local time).
    eth_end_local : str
        Electronic trading session end time (HH:MM in local time).
    rth_start_local : str
        Regular trading hours start time (HH:MM in local time).
    rth_end_local : str
        Regular trading hours end time (HH:MM in local time).
    session_close_hour : int
        Hour (local time) at which the session boundary falls.
        Used for daily OHLC grouping and session ID computation.
    """

    symbol: str
    name: str
    timezone: str = DEFAULT_TIMEZONE
    eth_start_local: str = DEFAULT_ETH_START_LOCAL
    eth_end_local: str = DEFAULT_ETH_END_LOCAL
    rth_start_local: str = DEFAULT_GLOBEX_RTH_START_LOCAL
    rth_end_local: str = DEFAULT_GLOBEX_RTH_END_LOCAL
    session_close_hour: int = DEFAULT_SESSION_CLOSE_HOUR


CME_SESSION_PRESETS: dict[str, SessionPreset] = {
    GC_SYMBOL: SessionPreset(
        symbol=GC_SYMBOL,
        name="CME Gold",
        timezone=DEFAULT_TIMEZONE,
        eth_start_local=DEFAULT_ETH_START_LOCAL,
        eth_end_local=DEFAULT_ETH_END_LOCAL,
        rth_start_local=DEFAULT_GLOBEX_RTH_START_LOCAL,
        rth_end_local=DEFAULT_GLOBEX_RTH_END_LOCAL,
        session_close_hour=DEFAULT_SESSION_CLOSE_HOUR,
    ),
    ES_SYMBOL: SessionPreset(
        symbol=ES_SYMBOL,
        name="CME E-mini S&P 500",
        timezone=DEFAULT_TIMEZONE,
        eth_start_local=DEFAULT_ETH_START_LOCAL,
        eth_end_local=DEFAULT_ETH_END_LOCAL,
        rth_start_local=DEFAULT_INDEX_RTH_START_LOCAL,
        rth_end_local=DEFAULT_INDEX_RTH_END_LOCAL,
        session_close_hour=DEFAULT_SESSION_CLOSE_HOUR,
    ),
    NQ_SYMBOL: SessionPreset(
        symbol=NQ_SYMBOL,
        name="CME E-mini Nasdaq-100",
        timezone=DEFAULT_TIMEZONE,
        eth_start_local=DEFAULT_ETH_START_LOCAL,
        eth_end_local=DEFAULT_ETH_END_LOCAL,
        rth_start_local=DEFAULT_INDEX_RTH_START_LOCAL,
        rth_end_local=DEFAULT_INDEX_RTH_END_LOCAL,
        session_close_hour=DEFAULT_SESSION_CLOSE_HOUR,
    ),
    CL_SYMBOL: SessionPreset(
        symbol=CL_SYMBOL,
        name="CME Crude Oil",
        timezone=DEFAULT_TIMEZONE,
        eth_start_local=DEFAULT_ETH_START_LOCAL,
        eth_end_local=DEFAULT_ETH_END_LOCAL,
        rth_start_local=DEFAULT_CRUDE_RTH_START_LOCAL,
        rth_end_local=DEFAULT_CRUDE_RTH_END_LOCAL,
        session_close_hour=DEFAULT_SESSION_CLOSE_HOUR,
    ),
    ZB_SYMBOL: SessionPreset(
        symbol=ZB_SYMBOL,
        name="CBOT 30-Year Treasury Bond",
        timezone=DEFAULT_TIMEZONE,
        eth_start_local=DEFAULT_ETH_START_LOCAL,
        eth_end_local=DEFAULT_ETH_END_LOCAL,
        rth_start_local=DEFAULT_GLOBEX_RTH_START_LOCAL,
        rth_end_local=DEFAULT_BOND_RTH_END_LOCAL,
        session_close_hour=DEFAULT_SESSION_CLOSE_HOUR,
    ),
}

# Common named trading sessions (can be used with build_session_mask)
NAMED_SESSIONS = {
    ALL_HOURS_SESSION: {"start": 0, "end": 24, "overnight": False, "description": "All Hours"},
    "asian": {"start": 18, "end": 3, "overnight": True, "description": "Asian (18:00-03:00 ET)"},
    "london": {"start": 3, "end": 8, "overnight": False, "description": "London (03:00-08:00 ET)"},
    "new_york": {
        "start": 8,
        "end": 17,
        "overnight": False,
        "description": "New York (08:00-17:00 ET)",
    },
    "london_ny_overlap": {
        "start": 8,
        "end": 12,
        "overnight": False,
        "description": "London-NY Overlap",
    },
}


def get_session_preset(symbol: str) -> SessionPreset:
    """Look up a CME session preset by instrument symbol.

    Parameters
    ----------
    symbol : str
        Instrument symbol (case-insensitive, e.g., 'GC', 'es').

    Returns
    -------
    SessionPreset
        The session preset for the given symbol.

    Raises
    ------
    KeyError
        If no preset exists for the given symbol.
    """
    s = symbol.upper()
    if s not in CME_SESSION_PRESETS:
        available = ", ".join(sorted(CME_SESSION_PRESETS.keys()))
        raise KeyError(f"No CME session preset for '{symbol}'. Available: {available}")
    return CME_SESSION_PRESETS[s]
