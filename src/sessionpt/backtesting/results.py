"""Backtest result and trade record dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EngineTradeRecord:
    """Complete record of a single trade from the execution engine.

    Attributes
    ----------
    entry_idx : int
        Bar index where the trade was entered.
    exit_idx : int
        Bar index where the trade was exited.
    entry_time : datetime
        Timestamp of entry bar.
    exit_time : datetime
        Timestamp of exit bar.
    direction : str
        Trade direction ('LONG' or 'SHORT').
    entry_price : float
        Price at which the position was opened.
    exit_price : float
        Price at which the position was closed.
    stop_price_initial : float
        Initial stop-loss price.
    take_profit_price : float
        Take-profit target price.
    pnl_ticks : float
        Profit/loss in ticks (can be negative).
    pnl_dollars : float
        Profit/loss in dollars (after transaction costs).
    exit_reason : str
        Reason for exit ('SL', 'TP', 'TRAILING_SL', 'BREAKEVEN', 'EOD', 'MAX_HOLD').
    mfe_ticks : float
        Maximum favorable excursion in ticks.
    mae_ticks : float
        Maximum adverse excursion in ticks.
    level_tag : str or None
        Pivot level that triggered entry (e.g., 'P', 'S1').
    metadata : dict
        Additional trade metadata.
    """

    entry_idx: int
    exit_idx: int
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_price_initial: float
    take_profit_price: float
    pnl_ticks: float
    pnl_dollars: float
    exit_reason: str
    mfe_ticks: float
    mae_ticks: float
    level_tag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestProvenance:
    """Metadata about where a backtest result came from.

    Attributes
    ----------
    strategy_family : str
        Name of the strategy (e.g., 'pivot_point').
    config_hash : str
        Hash of the configuration for reproducibility.
    data_window : str
        Description of the data window used.
    timezone_basis : str
        Timezone used for session computation.
    run_timestamp : str or None
        When the backtest was run.
    """

    strategy_family: str = ""
    config_hash: str = ""
    data_window: str = ""
    timezone_basis: str = ""
    run_timestamp: str | None = None


@dataclass
class BacktestRunResult:
    """Summary result from a backtest run.

    Attributes
    ----------
    trades : int
        Total number of trades executed.
    wins : int
        Number of profitable trades.
    losses : int
        Number of unprofitable trades (P&L <= 0).
    win_rate : float
        Percentage of winning trades (0-100).
    gross_pnl : float
        Gross P&L before transaction costs.
    total_pnl_net : float
        Net P&L after transaction costs.
    avg_pnl_per_trade : float
        Average net P&L per trade.
    profit_factor : float
        Ratio of gross wins to gross losses (inf if no losses).
    sharpe_ratio : float
        Annualized Sharpe ratio of trade P&Ls.
    max_drawdown : float
        Maximum peak-to-trough drawdown in dollars.
    exit_reasons : dict
        Count of trades by exit reason.
    avg_mfe : float
        Average maximum favorable excursion in ticks.
    avg_mae : float
        Average maximum adverse excursion in ticks.
    trade_records : list
        Detailed trade-by-trade records.
    provenance : BacktestProvenance or None
        Metadata about the backtest run.
    """

    trades: int
    wins: int
    losses: int
    win_rate: float
    gross_pnl: float
    total_pnl_net: float
    avg_pnl_per_trade: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    exit_reasons: dict[str, int]
    avg_mfe: float
    avg_mae: float
    trade_records: list[EngineTradeRecord] = field(default_factory=list)
    provenance: BacktestProvenance | None = None
