"""Event-driven backtesting engine for session-aware futures strategies.

This is the canonical backtesting engine in sessionpt. It processes a
pre-built list of EntryEvent objects bar-by-bar, supporting:

- Stop-loss and take-profit exits
- Trailing stop activation and lock
- Breakeven stop moves
- End-of-day (EOD) position closing
- Multi-day position holding (max_days_to_hold)
- Per-session trade limits
- Per-level trade exclusivity
- Maximum favorable/adverse excursion (MFE/MAE) tracking
- Session-aware EOD behavior using exchange session IDs

The vectorized fast path (VectorizedBacktester) is an alternative for
parameter sweeps where trailing/breakeven/EOD features are not needed.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sessionpt.backtesting.policies import normalize_breakeven, normalize_trailing
from sessionpt.backtesting.results import BacktestRunResult, EngineTradeRecord
from sessionpt.backtesting.specs import (
    BreakevenPolicy,
    ExecutionPolicy,
    InstrumentSpec,
    TrailingStopPolicy,
)
from sessionpt.constants import CLOSE_COLUMN, HIGH_COLUMN, LOW_COLUMN
from sessionpt.enums.direction import Direction
from sessionpt.enums.exit_reason import ExitReason
from sessionpt.sessions.core import get_session_ids


@dataclass
class EntryEvent:
    """A potential trade entry signal at a specific bar.

    Attributes
    ----------
    entry_idx : int
        Bar index where the entry condition was met.
    direction : Direction
        Trade direction (LONG or SHORT).
    entry_price : float
        Price at which the trade would be entered (typically the close).
    stop_price : float
        Initial stop-loss price.
    take_profit_price : float
        Take-profit target price.
    level_tag : str or None
        Pivot level that triggered the entry (e.g., 'P', 'S1').
    metadata : dict
        Additional entry metadata.
    """

    entry_idx: int
    direction: Direction
    entry_price: float
    stop_price: float
    take_profit_price: float
    level_tag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _direction_name(direction: Direction) -> str:
    """Convert Direction enum to uppercase string."""
    return "LONG" if direction == Direction.LONG else "SHORT"


def summarize_trades(
    trades: Sequence[EngineTradeRecord], total_cost_per_trade: float = 0.0
) -> BacktestRunResult:
    """Summarize a list of trade records into a BacktestRunResult.

    Parameters
    ----------
    trades : sequence of EngineTradeRecord
        List of trade records to summarize.
    total_cost_per_trade : float
        Transaction cost per trade (commission + slippage).

    Returns
    -------
    BacktestRunResult
        Aggregated backtest statistics.
    """
    if not trades:
        return BacktestRunResult(
            trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            gross_pnl=0.0,
            total_pnl_net=0.0,
            avg_pnl_per_trade=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            exit_reasons={reason.value: 0 for reason in ExitReason},
            avg_mfe=0.0,
            avg_mae=0.0,
            trade_records=[],
        )

    pnls = np.array([t.pnl_dollars for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    equity = np.cumsum(pnls)
    dd = np.maximum.accumulate(equity) - equity
    gross_wins = float(wins.sum()) if len(wins) else 0.0
    gross_losses = float(abs(losses.sum())) if len(losses) else 0.0

    if len(pnls) > 1:
        std = pnls.std(ddof=1)
        sharpe = float((pnls.mean() / std) * np.sqrt(len(pnls))) if std > 0 else 0.0
    else:
        sharpe = 0.0

    exit_reasons = {reason.value: 0 for reason in ExitReason}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return BacktestRunResult(
        trades=len(trades),
        wins=int((pnls > 0).sum()),
        losses=int((pnls <= 0).sum()),
        win_rate=float((pnls > 0).mean() * 100),
        gross_pnl=float(pnls.sum() + len(trades) * total_cost_per_trade),
        total_pnl_net=float(pnls.sum()),
        avg_pnl_per_trade=float(pnls.mean()),
        profit_factor=float(gross_wins / gross_losses) if gross_losses > 0 else float("inf"),
        sharpe_ratio=sharpe,
        max_drawdown=float(dd.max()) if len(dd) else 0.0,
        exit_reasons=exit_reasons,
        avg_mfe=float(np.mean([t.mfe_ticks for t in trades])),
        avg_mae=float(np.mean([t.mae_ticks for t in trades])),
        trade_records=list(trades),
    )


def precompute_backtest_arrays(df: pd.DataFrame, instrument: InstrumentSpec) -> dict[str, Any]:
    """Pre-extract arrays needed by run_event_backtest.

    Call once per data slice, then pass the result as ``precomputed``
    to avoid redundant extraction across multiple backtest runs.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'High', 'Low', 'Close' columns.
    instrument : InstrumentSpec
        Instrument specification for timezone and session info.

    Returns
    -------
    dict
        Pre-extracted arrays: highs, lows, closes, timestamps, session_ids, n_bars.
    """
    return {
        "highs": df[HIGH_COLUMN].values.astype(float),
        "lows": df[LOW_COLUMN].values.astype(float),
        "closes": df[CLOSE_COLUMN].values.astype(float),
        "timestamps": pd.DatetimeIndex(df.index).to_pydatetime(),
        "session_ids": get_session_ids(
            df.index,
            timezone=instrument.timezone,
            session_close_hour=instrument.session_close_hour,
        ),
        "n_bars": len(df),
    }


def run_event_backtest(
    df: pd.DataFrame,
    entry_events: Iterable[EntryEvent],
    instrument: InstrumentSpec,
    execution_policy: ExecutionPolicy,
    trailing_policy: TrailingStopPolicy | None = None,
    breakeven_policy: BreakevenPolicy | None = None,
    precomputed: dict[str, Any] | None = None,
) -> BacktestRunResult:
    """Run an event-driven backtest on pre-built entry signals.

    This is the primary backtesting function. It processes a sorted list of
    EntryEvent objects bar-by-bar, managing position lifecycle including
    stop-loss, take-profit, trailing stops, breakeven, EOD close, and
    multi-day holding.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'High', 'Low', 'Close' columns and DatetimeIndex.
    entry_events : iterable of EntryEvent
        Pre-built entry signals (typically from pivot level crossovers).
    instrument : InstrumentSpec
        Instrument specification (tick size, value, costs, timezone).
    execution_policy : ExecutionPolicy
        Trade management rules (EOD close, max trades, etc.).
    trailing_policy : TrailingStopPolicy or None
        Trailing stop configuration. None disables trailing.
    breakeven_policy : BreakevenPolicy or None
        Breakeven stop configuration. None disables breakeven.
    precomputed : dict or None
        Pre-extracted arrays from precompute_backtest_arrays for performance.

    Returns
    -------
    BacktestRunResult
        Aggregated backtest statistics and trade records.
    """
    if precomputed is not None:
        highs = precomputed["highs"]
        lows = precomputed["lows"]
        closes = precomputed["closes"]
        timestamps = precomputed["timestamps"]
        session_ids = precomputed["session_ids"]
        n_bars = precomputed["n_bars"]
    else:
        highs = df[HIGH_COLUMN].values.astype(float)
        lows = df[LOW_COLUMN].values.astype(float)
        closes = df[CLOSE_COLUMN].values.astype(float)
        timestamps = pd.DatetimeIndex(df.index).to_pydatetime()
        n_bars = len(df)
        session_ids = get_session_ids(
            df.index,
            timezone=instrument.timezone,
            session_close_hour=instrument.session_close_hour,
        )

    tick_size = float(instrument.tick_size)
    tick_value = float(instrument.tick_value)
    total_cost = (
        float(instrument.commission_round_trip) + float(instrument.slippage_ticks) * tick_value
    )

    trailing = normalize_trailing(trailing_policy)
    breakeven = normalize_breakeven(breakeven_policy)

    events = sorted(list(entry_events), key=lambda e: e.entry_idx)
    trades: list[EngineTradeRecord] = []

    last_exit_idx = -1
    session_trade_counts: dict[int, int] = {}
    used_levels_by_session: dict[int, set] = {}
    max_scan_bars = (
        1440 if execution_policy.close_at_eod else 1440 * execution_policy.max_days_to_hold
    )

    for e in events:
        idx = int(e.entry_idx)
        if idx < 0 or idx >= n_bars:
            continue
        if not execution_policy.allow_concurrent_positions and idx <= last_exit_idx:
            continue

        session_id = int(session_ids[idx])
        session_trade_counts.setdefault(session_id, 0)
        used_levels_by_session.setdefault(session_id, set())

        if session_trade_counts[session_id] >= execution_policy.max_trades_per_session:
            continue
        if (
            execution_policy.one_trade_per_level
            and e.level_tag is not None
            and e.level_tag in used_levels_by_session[session_id]
        ):
            continue

        direction = e.direction
        is_long = direction == Direction.LONG
        entry_price = float(e.entry_price)
        current_sl = float(e.stop_price)
        initial_sl = float(e.stop_price)
        tp_price = float(e.take_profit_price)

        trailing_activated = False
        breakeven_activated = False
        max_favorable = 0.0
        max_adverse = 0.0
        exit_idx = idx
        exit_price = entry_price
        exit_reason = (
            ExitReason.MAX_HOLD.value if not execution_policy.close_at_eod else ExitReason.EOD.value
        )

        for j in range(idx + 1, min(idx + max_scan_bars, n_bars)):
            if execution_policy.close_at_eod and session_ids[j] != session_id:
                exit_idx = j - 1
                exit_price = float(closes[exit_idx])
                exit_reason = ExitReason.EOD.value
                break

            bar_high = float(highs[j])
            bar_low = float(lows[j])

            if is_long:
                favorable = (bar_high - entry_price) / tick_size
                adverse = (entry_price - bar_low) / tick_size
            else:
                favorable = (entry_price - bar_low) / tick_size
                adverse = (bar_high - entry_price) / tick_size

            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            if (
                trailing.enabled
                and not trailing_activated
                and trailing.trigger_ticks is not None
                and trailing.lock_ticks is not None
                and max_favorable >= float(trailing.trigger_ticks)
            ):
                trailing_activated = True
                if is_long:
                    current_sl = max(
                        current_sl,
                        entry_price + float(trailing.lock_ticks) * tick_size,
                    )
                else:
                    current_sl = min(
                        current_sl,
                        entry_price - float(trailing.lock_ticks) * tick_size,
                    )

            if breakeven.enabled and not breakeven_activated:
                distance_to_tp = abs(tp_price - entry_price)
                if distance_to_tp > 0:
                    if is_long:
                        be_trigger_price = entry_price + (
                            distance_to_tp * float(breakeven.trigger_pct_to_tp)
                        )
                        if bar_high >= be_trigger_price:
                            current_sl = max(current_sl, entry_price)
                            breakeven_activated = True
                    else:
                        be_trigger_price = entry_price - (
                            distance_to_tp * float(breakeven.trigger_pct_to_tp)
                        )
                        if bar_low <= be_trigger_price:
                            current_sl = min(current_sl, entry_price)
                            breakeven_activated = True

            if is_long:
                tp_hit = bar_high >= tp_price
                sl_hit = bar_low <= current_sl
            else:
                tp_hit = bar_low <= tp_price
                sl_hit = bar_high >= current_sl

            if tp_hit and sl_hit:
                exit_idx = j
                exit_price = float(current_sl)
                if trailing_activated:
                    exit_reason = ExitReason.TRAILING_SL.value
                elif breakeven_activated and abs(current_sl - entry_price) < 1e-12:
                    exit_reason = ExitReason.BREAKEVEN.value
                else:
                    exit_reason = ExitReason.SL.value
                break
            if sl_hit:
                exit_idx = j
                exit_price = float(current_sl)
                if trailing_activated:
                    exit_reason = ExitReason.TRAILING_SL.value
                elif breakeven_activated and abs(current_sl - entry_price) < 1e-12:
                    exit_reason = ExitReason.BREAKEVEN.value
                else:
                    exit_reason = ExitReason.SL.value
                break
            if tp_hit:
                exit_idx = j
                exit_price = float(tp_price)
                exit_reason = ExitReason.TP.value
                break
        else:
            exit_idx = min(idx + max_scan_bars - 1, n_bars - 1)
            exit_price = float(closes[exit_idx])
            exit_reason = (
                ExitReason.EOD.value if execution_policy.close_at_eod else ExitReason.MAX_HOLD.value
            )

        if not execution_policy.allow_concurrent_positions:
            last_exit_idx = exit_idx
        session_trade_counts[session_id] += 1
        if execution_policy.one_trade_per_level and e.level_tag is not None:
            used_levels_by_session[session_id].add(e.level_tag)

        if is_long:
            pnl_ticks = (exit_price - entry_price) / tick_size
        else:
            pnl_ticks = (entry_price - exit_price) / tick_size
        pnl_dollars = (pnl_ticks * tick_value) - total_cost

        trades.append(
            EngineTradeRecord(
                entry_idx=idx,
                exit_idx=exit_idx,
                entry_time=timestamps[idx],
                exit_time=timestamps[exit_idx],
                direction=_direction_name(direction),
                entry_price=entry_price,
                exit_price=exit_price,
                stop_price_initial=initial_sl,
                take_profit_price=tp_price,
                pnl_ticks=float(pnl_ticks),
                pnl_dollars=float(pnl_dollars),
                exit_reason=exit_reason,
                mfe_ticks=float(max_favorable),
                mae_ticks=float(max_adverse),
                level_tag=e.level_tag,
                metadata=dict(e.metadata),
            )
        )

    return summarize_trades(trades, total_cost_per_trade=total_cost)
