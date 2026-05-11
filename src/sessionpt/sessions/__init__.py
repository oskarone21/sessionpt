"""Session-aware trading utilities for futures and 24-hour markets.

This module provides utilities for computing session boundaries, labels, and masks
correctly handling overnight sessions (e.g., CME Gold 18:00-17:00 ET) and
DST transitions.

Key functions:
    - get_session_labels: Assign a date label to each bar based on its trading session.
    - get_session_ids: Numeric version of session labels for fast comparison.
    - build_session_mask: Filter bars to those within a named trading session.
    - get_local_hours: Extract local-hour array for timezone-aware filtering.
"""

from sessionpt.sessions.core import (
    build_session_mask,
    ensure_utc_index,
    get_local_hours,
    get_session_ids,
    get_session_labels,
)
from sessionpt.sessions.presets import CME_SESSION_PRESETS, NAMED_SESSIONS, get_session_preset

__all__ = [
    "CME_SESSION_PRESETS",
    "NAMED_SESSIONS",
    "build_session_mask",
    "ensure_utc_index",
    "get_local_hours",
    "get_session_ids",
    "get_session_labels",
    "get_session_preset",
]
