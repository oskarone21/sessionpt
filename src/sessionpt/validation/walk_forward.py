"""Walk-forward validation for robust strategy parameter selection.

Provides sliding-window fold generation and result dataclasses for
walk-forward analysis (WFA). This is a framework-agnostic module —
it does not depend on any specific strategy or backtesting engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dateutil.relativedelta import relativedelta


@dataclass(frozen=True)
class WalkForwardFold:
    """A single train/test split for walk-forward analysis.

    Attributes
    ----------
    fold_id : int
        1-based fold index.
    train_start : datetime
        Start of the training window (inclusive).
    train_end : datetime
        End of the training window (inclusive-ish; test_start is typically
        the next period).
    test_start : datetime
        Start of the out-of-sample window (inclusive).
    test_end : datetime
        End of the out-of-sample window (inclusive).
    """

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    @property
    def train_duration_days(self) -> int:
        """Training window duration in calendar days."""
        return (self.train_end - self.train_start).days

    @property
    def test_duration_days(self) -> int:
        """Test window duration in calendar days."""
        return (self.test_end - self.test_start).days


@dataclass
class OptResult:
    """Optimization result for a single parameter set within a fold.

    Attributes
    ----------
    pivot_type : str
        Pivot type name (e.g., 'traditional', 'woodie').
    level : str
        Pivot level (e.g., 'P', 'S1', 'R1').
    direction : str
        Trade direction ('LONG' or 'SHORT').
    sl : int
        Stop-loss distance in ticks.
    tp : int
        Take-profit distance in ticks.
    rr_ratio : float
        Reward-to-risk ratio (TP / SL).
    trades : int
        Number of trades.
    win_rate : float
        Percentage of winning trades (0-100).
    pnl_net : float
        Net P&L after transaction costs.
    profit_factor : float
        Gross wins / gross losses (inf if no losses).
    sharpe_ratio : float
        Annualized Sharpe ratio of trade P&Ls.
    sortino_ratio : float
        Annualized Sortino ratio (downside deviation only).
    max_drawdown : float
        Maximum peak-to-trough drawdown in dollars.
    """

    pivot_type: str = ""
    level: str = ""
    direction: str = ""
    sl: int = 0
    tp: int = 0
    rr_ratio: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    pnl_net: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class FoldResult:
    """Result from a single walk-forward fold.

    Captures both in-sample (train) and out-of-sample (test) performance
    for the best parameter set found during training.

    Attributes
    ----------
    filter_name : str
        Name of the filter configuration used.
    fold_id : int
        1-based fold index.
    best_pivot_type : str
        Best pivot type found during training.
    best_level : str
        Best pivot level found during training.
    best_direction : str
        Best trade direction found during training.
    best_sl : int
        Best stop-loss in ticks.
    best_tp : int
        Best take-profit in ticks.
    train_pnl : float
        In-sample net P&L.
    train_sharpe : float
        In-sample Sharpe ratio.
    train_sortino : float
        In-sample Sortino ratio.
    train_trades : int
        Number of in-sample trades.
    train_win_rate : float
        In-sample win rate (0-100).
    train_max_dd : float
        In-sample max drawdown in dollars.
    train_profit_factor : float
        In-sample profit factor.
    test_pnl : float
        Out-of-sample net P&L.
    test_sharpe : float
        Out-of-sample Sharpe ratio.
    test_sortino : float
        Out-of-sample Sortino ratio.
    test_trades : int
        Number of out-of-sample trades.
    test_win_rate : float
        Out-of-sample win rate (0-100).
    test_max_dd : float
        Out-of-sample max drawdown in dollars.
    test_profit_factor : float
        Out-of-sample profit factor.
    pnl_profitable_oos : bool
        Whether the out-of-sample P&L is positive.
    profit_factor_threshold_met : bool
        Whether the out-of-sample profit factor meets a threshold.
    sortino_threshold_met : bool
        Whether the out-of-sample Sortino ratio meets a threshold.
    trade_details : list
        Detailed trade records from the test period.
    """

    filter_name: str = ""
    fold_id: int = 0
    best_pivot_type: str = ""
    best_level: str = ""
    best_direction: str = ""
    best_sl: int = 0
    best_tp: int = 0
    train_pnl: float = 0.0
    train_sharpe: float = 0.0
    train_sortino: float = 0.0
    train_trades: int = 0
    train_win_rate: float = 0.0
    train_max_dd: float = 0.0
    train_profit_factor: float = 0.0
    test_pnl: float = 0.0
    test_sharpe: float = 0.0
    test_sortino: float = 0.0
    test_trades: int = 0
    test_win_rate: float = 0.0
    test_max_dd: float = 0.0
    test_profit_factor: float = 0.0
    pnl_profitable_oos: bool = False
    profit_factor_threshold_met: bool = False
    sortino_threshold_met: bool = False
    trade_details: list[Any] = field(default_factory=list)


def generate_walk_forward_folds(
    data_start: datetime,
    data_end: datetime,
    train_months: int = 6,
    test_months: int = 2,
    step_months: int = 2,
) -> list[WalkForwardFold]:
    """Generate sliding-window train/test folds for walk-forward analysis.

    Each fold advances the window by *step_months*, creating overlapping
    training periods with non-overlapping test periods. The last fold whose
    test window extends beyond *data_end* is excluded.

    Parameters
    ----------
    data_start : datetime
        Start of the available data (inclusive).
    data_end : datetime
        End of the available data (inclusive-ish; the test period must end
        on or before this date).
    train_months : int
        Length of the training window in months.
    test_months : int
        Length of the out-of-sample test window in months.
    step_months : int
        Step size in months to advance the sliding window.

    Returns
    -------
    list of WalkForwardFold
        Ordered list of folds with 1-based IDs.

    Examples
    --------
    >>> from datetime import datetime
    >>> folds = generate_walk_forward_folds(
    ...     datetime(2020, 1, 1),
    ...     datetime(2023, 1, 1),
    ...     train_months=6,
    ...     test_months=2,
    ...     step_months=2,
    ... )
    >>> len(folds)
    7
    >>> folds[0].fold_id
    1
    """
    folds: list[WalkForwardFold] = []
    fold_id = 1
    train_start = data_start

    while True:
        train_end = train_start + relativedelta(months=train_months)
        test_start = train_end
        test_end = test_start + relativedelta(months=test_months)

        if test_end > data_end:
            break

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        train_start = train_start + relativedelta(months=step_months)

    return folds
