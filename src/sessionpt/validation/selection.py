"""Candidate filtering and ranking helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

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
        if float(row.get(DEFAULT_WIN_RATE_KEY, row.get(ALT_WIN_RATE_KEY, 0.0))) < min_win_rate:
            continue
        if int(row.get(TRADES_KEY, row.get(ALT_TRADES_KEY, 0))) < min_trades:
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
