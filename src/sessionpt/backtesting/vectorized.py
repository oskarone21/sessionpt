"""Numba-accelerated vectorized backtester for rapid parameter sweeps.

This module provides a fast backtesting path using NumPy vectorization and
optional Numba JIT compilation. It is designed for SL/TP grid optimization
where trailing stops, breakeven, and detailed exit tracking are not needed.

For full-featured backtesting (trailing stops, breakeven, configurable EOD, etc.),
use sessionpt.backtesting.engine.run_event_backtest instead.

When Numba is not installed, the backtester falls back to pure-Python mode.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sessionpt.constants import (
    CLOSE_COLUMN,
    DEFAULT_SESSION_CLOSE_HOUR,
    DEFAULT_TIMEZONE,
    HIGH_COLUMN,
    LOW_COLUMN,
    OPEN_COLUMN,
)
from sessionpt.enums.direction import Direction
from sessionpt.sessions.core import get_session_ids, validate_datetime_index

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(func=None, **kwargs):
        def decorator(fn):
            return fn

        if func is not None:
            return func
        return decorator


@dataclass(frozen=True)
class VectorizedBacktestResult:
    """Result from a vectorized backtest run.

    Attributes
    ----------
    trades : int
        Total number of trades.
    wins : int
        Number of winning trades (P&L > 0).
    losses : int
        Number of losing trades (P&L <= 0).
    win_rate : float
        Win rate as a percentage (0-100).
    total_pnl_net : float
        Net P&L after transaction costs.
    avg_pnl_per_trade : float
        Average P&L per trade.
    profit_factor : float
        Ratio of gross wins to gross losses.
    sharpe_ratio : float
        Per-trade Sharpe ratio of trade P&Ls (not annualized).
    max_drawdown : float
        Maximum peak-to-trough drawdown in dollars.
    """

    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_net: float
    avg_pnl_per_trade: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float


@dataclass(frozen=True)
class TradeDetail:
    """Individual trade detail from a vectorized backtest.

    Attributes
    ----------
    entry_time : pd.Timestamp
        Entry bar timestamp.
    exit_time : pd.Timestamp
        Exit bar timestamp.
    entry_price : float
        Price at entry.
    exit_price : float
        Price at exit.
    pnl_net : float
        Net P&L for this trade.
    exit_reason : str
        Reason for exit ('SL', 'TP', 'EOD', 'MAX_HOLD').
    """

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_net: float
    exit_reason: str


if NUMBA_AVAILABLE:

    @njit(cache=False)
    def _process_trades_numba(
        entry_indices: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        levels: np.ndarray,
        is_long: bool,
        sl_ticks: float,
        tp_ticks: float,
        tick_size: float,
        tick_value: float,
        total_cost: float,
        session_ids: np.ndarray,
        max_trades_per_day: int,
        once_per_day_level: bool,
    ) -> np.ndarray:
        """Numba-JIT compiled trade processing loop.

        Returns an array of net P&L values for each trade.
        This replaces the Python for-loop with compiled machine code.
        """
        n_bars = len(closes)
        max_trades = len(entry_indices)
        trades = np.empty(max_trades, dtype=np.float64)
        trade_idx = 0
        last_exit_idx = -1
        session_trade_counts: dict = {}
        used_level_sessions: dict = {}

        for i in range(max_trades):
            idx = entry_indices[i]
            if idx <= last_exit_idx:
                continue
            level = levels[idx]
            if np.isnan(level):
                continue

            sid = session_ids[idx]
            if sid not in session_trade_counts:
                session_trade_counts[sid] = 0
            if session_trade_counts[sid] >= max_trades_per_day:
                continue
            if once_per_day_level and sid in used_level_sessions:
                continue

            entry_price = closes[idx]
            sid_entry = sid

            if is_long:
                tp_price = entry_price + tp_ticks * tick_size
                sl_price = entry_price - sl_ticks * tick_size
            else:
                tp_price = entry_price - tp_ticks * tick_size
                sl_price = entry_price + sl_ticks * tick_size

            exit_idx = idx
            exit_price = entry_price
            pnl_ticks = 0.0
            closed = False

            for j in range(idx + 1, n_bars):
                if session_ids[j] != sid_entry:
                    exit_idx = j - 1
                    exit_price = closes[exit_idx]
                    if is_long:
                        pnl_ticks = (exit_price - entry_price) / tick_size
                    else:
                        pnl_ticks = (entry_price - exit_price) / tick_size
                    closed = True
                    break

                open_value = opens[j]
                h_val = highs[j]
                l_val = lows[j]

                if is_long:
                    if open_value <= sl_price:
                        exit_idx = j
                        exit_price = open_value
                        pnl_ticks = (exit_price - entry_price) / tick_size
                        closed = True
                        break
                    if open_value >= tp_price:
                        exit_idx = j
                        exit_price = tp_price
                        pnl_ticks = tp_ticks
                        closed = True
                        break
                    tp_hit = h_val >= tp_price
                    sl_hit = l_val <= sl_price
                else:
                    if open_value >= sl_price:
                        exit_idx = j
                        exit_price = open_value
                        pnl_ticks = (entry_price - exit_price) / tick_size
                        closed = True
                        break
                    if open_value <= tp_price:
                        exit_idx = j
                        exit_price = tp_price
                        pnl_ticks = tp_ticks
                        closed = True
                        break
                    tp_hit = l_val <= tp_price
                    sl_hit = h_val >= sl_price

                if tp_hit and sl_hit:
                    exit_idx = j
                    exit_price = sl_price
                    if is_long:
                        pnl_ticks = (sl_price - entry_price) / tick_size
                    else:
                        pnl_ticks = (entry_price - sl_price) / tick_size
                    closed = True
                    break
                elif tp_hit:
                    exit_idx = j
                    exit_price = tp_price
                    pnl_ticks = tp_ticks
                    closed = True
                    break
                elif sl_hit:
                    exit_idx = j
                    exit_price = sl_price
                    if is_long:
                        pnl_ticks = (sl_price - entry_price) / tick_size
                    else:
                        pnl_ticks = (entry_price - sl_price) / tick_size
                    closed = True
                    break

            if not closed:
                exit_idx = n_bars - 1
                exit_price = closes[exit_idx]
                if is_long:
                    pnl_ticks = (exit_price - entry_price) / tick_size
                else:
                    pnl_ticks = (entry_price - exit_price) / tick_size

            gross_pnl = pnl_ticks * tick_value
            net_pnl = gross_pnl - total_cost
            trades[trade_idx] = net_pnl
            trade_idx += 1

            last_exit_idx = exit_idx
            session_trade_counts[sid] += 1
            if once_per_day_level:
                used_level_sessions[sid] = True

        return trades[:trade_idx]


class VectorizedBacktester:
    """Fast vectorized backtester for SL/TP parameter sweeps.

    Designed for rapid grid search over stop-loss and take-profit parameters.
    Supports mandatory session-boundary exits and session-aware trade limits,
    but does not support trailing or breakeven stops. For configurable EOD and
    dynamic-stop behavior, use run_event_backtest instead.

    Parameters
    ----------
    tick_size : float
        Minimum price increment (e.g., 0.10 for GC).
    tick_value : float
        Dollar value per tick (e.g., $10 for GC).
    commission : float
        Round-trip commission per trade in dollars.
    slippage_cost : float
        Total slippage cost per trade in dollars.
    max_trades_per_day : int
        Maximum number of trades per trading session.
    once_per_day_level : bool
        If True, accept at most one trade per level per trading session.
    timezone : str
        Exchange timezone string.
    session_close_hour : int
        Hour (exchange-local) when session closes.
    """

    def __init__(
        self,
        tick_size: float,
        tick_value: float,
        commission: float,
        slippage_cost: float = 0.0,
        max_trades_per_day: int = 10,
        once_per_day_level: bool = True,
        timezone: str = DEFAULT_TIMEZONE,
        session_close_hour: int = DEFAULT_SESSION_CLOSE_HOUR,
    ):
        numeric_values = (tick_size, tick_value, commission, slippage_cost)
        if not np.isfinite(numeric_values).all():
            raise ValueError("prices and transaction costs must be finite")
        if tick_size <= 0 or tick_value <= 0:
            raise ValueError("tick_size and tick_value must be positive")
        if commission < 0 or slippage_cost < 0:
            raise ValueError("transaction costs must be non-negative")
        if not isinstance(max_trades_per_day, int) or isinstance(max_trades_per_day, bool):
            raise ValueError("max_trades_per_day must be an integer")
        if max_trades_per_day <= 0:
            raise ValueError("max_trades_per_day must be positive")
        if not isinstance(once_per_day_level, bool):
            raise ValueError("once_per_day_level must be boolean")
        if not isinstance(session_close_hour, int) or isinstance(session_close_hour, bool):
            raise ValueError("session_close_hour must be an integer")
        if not 0 <= session_close_hour <= 23:
            raise ValueError("session_close_hour must be between 0 and 23")
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.commission = commission
        self.slippage_cost = slippage_cost
        self.max_trades_per_day = max_trades_per_day
        self.once_per_day_level = once_per_day_level
        self.timezone = timezone
        self.session_close_hour = session_close_hour
        self._session_cache: dict[bytes | str, np.ndarray] = {}

    def _compute_session_ids(self, timestamps: np.ndarray) -> np.ndarray:
        ts_index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        return get_session_ids(
            index=ts_index,
            timezone=self.timezone,
            session_close_hour=self.session_close_hour,
        )

    def _get_or_compute_session_ids(self, timestamps: np.ndarray) -> np.ndarray:
        key: bytes | str = (
            timestamps.tobytes() if hasattr(timestamps, "tobytes") else str(timestamps)
        )
        if key not in self._session_cache:
            self._session_cache[key] = self._compute_session_ids(timestamps)
        return self._session_cache[key]

    def run(
        self,
        df: pd.DataFrame,
        level_col: str,
        direction: Direction,
        sl_points: float,
        tp_points: float,
        start_date: str | None = None,
        entry_filter_mask: np.ndarray | None = None,
    ) -> VectorizedBacktestResult:
        """Run a vectorized backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Price data with OHLCV columns and pivot level columns.
            Must have a timezone-aware DatetimeIndex.
        level_col : str
            Column name for the pivot level (e.g., 'P', 'S1', 'R1').
        direction : Direction
            Trade direction (LONG or SHORT).
        sl_points : float
            Stop-loss distance in ticks.
        tp_points : float
            Take-profit distance in ticks.
        start_date : str or None
            Optional start date filter (inclusive).
        entry_filter_mask : np.ndarray or None
            Optional boolean mask for additional entry filtering
            (e.g., session or trend filters). Length must match df.

        Returns
        -------
        VectorizedBacktestResult
            Backtest summary statistics.
        """
        data = df[df.index >= start_date] if start_date else df
        validate_datetime_index(data.index)
        if direction not in (Direction.LONG, Direction.SHORT):
            raise ValueError(f"Unsupported direction: {direction}")
        if sl_points <= 0 or tp_points <= 0:
            raise ValueError("sl_points and tp_points must be positive")

        opens = data[OPEN_COLUMN].to_numpy(dtype=float)
        closes = data[CLOSE_COLUMN].to_numpy(dtype=float)
        highs = data[HIGH_COLUMN].to_numpy(dtype=float)
        lows = data[LOW_COLUMN].to_numpy(dtype=float)
        levels = data[level_col].to_numpy(dtype=float)
        if not np.isfinite([opens, closes, highs, lows]).all():
            raise ValueError("OHLC inputs must be finite")
        is_long = direction == Direction.LONG

        if is_long:
            entry_mask = (lows <= levels) & (highs >= levels) & (closes > levels)
        else:
            entry_mask = (lows <= levels) & (highs >= levels) & (closes < levels)

        if entry_filter_mask is not None:
            if len(entry_filter_mask) != len(data):
                raise ValueError("entry_filter_mask length must match filtered dataframe length")
            entry_mask = entry_mask & np.asarray(entry_filter_mask, dtype=bool)

        entry_indices = np.where(entry_mask)[0]

        if len(entry_indices) == 0:
            return VectorizedBacktestResult(
                trades=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                total_pnl_net=0.0,
                avg_pnl_per_trade=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
            )

        session_ids = self._get_or_compute_session_ids(data.index.values)

        total_cost = self.commission + self.slippage_cost

        if NUMBA_AVAILABLE:
            trade_pnls = _process_trades_numba(
                entry_indices,
                opens,
                closes,
                highs,
                lows,
                levels,
                is_long,
                sl_points,
                tp_points,
                self.tick_size,
                self.tick_value,
                total_cost,
                session_ids,
                max_trades_per_day=self.max_trades_per_day,
                once_per_day_level=self.once_per_day_level,
            )
        else:
            trade_pnls = self._process_trades_python(
                entry_indices,
                opens,
                closes,
                highs,
                lows,
                levels,
                is_long,
                sl_points,
                tp_points,
                self.tick_size,
                self.tick_value,
                total_cost,
                session_ids,
            )

        if len(trade_pnls) == 0:
            return VectorizedBacktestResult(
                trades=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                total_pnl_net=0.0,
                avg_pnl_per_trade=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
            )

        wins = int(np.sum(trade_pnls > 0))
        losses = int(np.sum(trade_pnls <= 0))
        total_pnl = float(np.sum(trade_pnls))
        avg_pnl = float(np.mean(trade_pnls))
        gross_wins = float(np.sum(np.maximum(trade_pnls, 0)))
        gross_losses = float(abs(np.sum(np.minimum(trade_pnls, 0))))
        profit_factor = (
            gross_wins / gross_losses
            if gross_losses > 0
            else (float("inf") if gross_wins > 0 else 0.0)
        )

        cumulative = np.concatenate(([0.0], np.cumsum(trade_pnls)))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = float(np.max(drawdowns))

        if len(trade_pnls) > 1:
            std = np.std(trade_pnls, ddof=1)
            sharpe = float(np.mean(trade_pnls) / std) if std > 0 else 0.0
        else:
            sharpe = 0.0

        return VectorizedBacktestResult(
            trades=len(trade_pnls),
            wins=wins,
            losses=losses,
            win_rate=wins / len(trade_pnls) * 100,
            total_pnl_net=total_pnl,
            avg_pnl_per_trade=avg_pnl,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
        )

    def _process_trades_python(
        self,
        entry_indices: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        levels: np.ndarray,
        is_long: bool,
        sl_points: float,
        tp_points: float,
        tick_size: float,
        tick_value: float,
        total_cost: float,
        session_ids: np.ndarray,
    ) -> np.ndarray:
        """Pure-Python fallback for trade processing when Numba is not available."""
        n_bars = len(closes)
        trade_pnls = []
        last_exit_idx = -1
        session_trade_counts: dict[int, int] = {}
        used_level_sessions: set[int] = set()

        for idx in entry_indices:
            if idx <= last_exit_idx:
                continue
            level = levels[idx]
            if np.isnan(level):
                continue

            sid = int(session_ids[idx])
            session_trade_counts.setdefault(sid, 0)
            if session_trade_counts[sid] >= self.max_trades_per_day:
                continue
            if self.once_per_day_level and sid in used_level_sessions:
                continue

            entry_price = closes[idx]
            sid_entry = sid

            if is_long:
                tp_price = entry_price + tp_points * tick_size
                sl_price = entry_price - sl_points * tick_size
            else:
                tp_price = entry_price - tp_points * tick_size
                sl_price = entry_price + sl_points * tick_size

            exit_idx = idx
            exit_price = entry_price
            closed = False

            for j in range(idx + 1, n_bars):
                if session_ids[j] != sid_entry:
                    exit_idx = j - 1
                    exit_price = closes[exit_idx]
                    closed = True
                    break
                open_value = opens[j]
                h_val = highs[j]
                l_val = lows[j]

                if is_long:
                    if open_value <= sl_price:
                        exit_idx = j
                        exit_price = open_value
                        closed = True
                        break
                    if open_value >= tp_price:
                        exit_idx = j
                        exit_price = tp_price
                        closed = True
                        break
                    if h_val >= tp_price and l_val <= sl_price:
                        exit_idx = j
                        exit_price = sl_price
                        closed = True
                        break
                    if h_val >= tp_price:
                        exit_idx = j
                        exit_price = tp_price
                        closed = True
                        break
                    if l_val <= sl_price:
                        exit_idx = j
                        exit_price = sl_price
                        closed = True
                        break
                else:
                    if open_value >= sl_price:
                        exit_idx = j
                        exit_price = open_value
                        closed = True
                        break
                    if open_value <= tp_price:
                        exit_idx = j
                        exit_price = tp_price
                        closed = True
                        break
                    if l_val <= tp_price and h_val >= sl_price:
                        exit_idx = j
                        exit_price = sl_price
                        closed = True
                        break
                    if l_val <= tp_price:
                        exit_idx = j
                        exit_price = tp_price
                        closed = True
                        break
                    if h_val >= sl_price:
                        exit_idx = j
                        exit_price = sl_price
                        closed = True
                        break

            if not closed:
                exit_idx = n_bars - 1
                exit_price = closes[exit_idx]

            if is_long:
                pnl_ticks = (exit_price - entry_price) / tick_size
            else:
                pnl_ticks = (entry_price - exit_price) / tick_size
            net_pnl = pnl_ticks * tick_value - total_cost
            trade_pnls.append(net_pnl)
            last_exit_idx = exit_idx
            session_trade_counts[sid] += 1
            if self.once_per_day_level:
                used_level_sessions.add(sid)

        return np.array(trade_pnls)
