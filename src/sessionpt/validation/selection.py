"""Candidate filtering and ranking helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite

DEFAULT_MIN_WIN_RATE = 40.0
DEFAULT_MIN_TRADES = 0
DEFAULT_SHARPE_KEY = "sharpe_ratio"
DEFAULT_PNL_KEY = "pnl_net"
DEFAULT_WIN_RATE_KEY = "win_rate"
DEFAULT_DRAWDOWN_KEY = "max_drawdown"
ALT_WIN_RATE_KEY = "OOS Win Rate"
ALT_TRADES_KEY = "Trades"
TRADES_KEY = "trades"


def apply_hard_filters(
    rows: Iterable[Mapping],
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> list[Mapping]:
    out = []
    for row in rows:
        win_rate = float(row.get(DEFAULT_WIN_RATE_KEY, row.get(ALT_WIN_RATE_KEY, float("nan"))))
        trades = float(row.get(TRADES_KEY, row.get(ALT_TRADES_KEY, float("nan"))))
        if not isfinite(win_rate) or not isfinite(trades):
            continue
        if win_rate < min_win_rate:
            continue
        if trades < min_trades:
            continue
        out.append(row)
    return out


def rank_candidates(
    rows: Sequence[Mapping],
    sharpe_key: str = DEFAULT_SHARPE_KEY,
    pnl_key: str = DEFAULT_PNL_KEY,
    win_rate_key: str = DEFAULT_WIN_RATE_KEY,
    dd_key: str = DEFAULT_DRAWDOWN_KEY,
) -> list[Mapping]:
    required_keys = (sharpe_key, pnl_key, win_rate_key, dd_key)
    for row in rows:
        missing = [key for key in required_keys if key not in row]
        if missing:
            raise ValueError(f"Candidate is missing ranking metrics: {missing}")
        if not all(isfinite(float(row[key])) for key in required_keys):
            raise ValueError("Candidate ranking metrics must be finite")
    return sorted(
        rows,
        key=lambda row: (
            float(row.get(sharpe_key, 0.0)),
            float(row.get(pnl_key, 0.0)),
            float(row.get(win_rate_key, 0.0)),
            -float(row.get(dd_key, 0.0)),
        ),
        reverse=True,
    )
