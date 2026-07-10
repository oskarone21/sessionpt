# API Reference

All public classes and functions in `sessionpt`.

---

## Enums

### `Direction`

```python
class Direction(Enum):
    LONG = 'long'
    SHORT = 'short'
```

Trade direction enum. Used by both backtesters and trend filters. The `sign`
property returns `+1` for long and `-1` for short.

---

### `ExitReason`

```python
class ExitReason(Enum):
    SL = 'SL'
    TP = 'TP'
    TRAILING_SL = 'TRAILING_SL'
    BREAKEVEN = 'BREAKEVEN'
    EOD = 'EOD'
    MAX_HOLD = 'MAX_HOLD'
    DATA_END = 'DATA_END'
```

Reason a trade was closed.

---

### `PivotLevel`

```python
class PivotLevel(Enum):
    P = 'P'
    R1 = 'R1'
    R2 = 'R2'
    R3 = 'R3'
    R4 = 'R4'
    R5 = 'R5'
    S1 = 'S1'
    S2 = 'S2'
    S3 = 'S3'
    S4 = 'S4'
    S5 = 'S5'
```

Named pivot levels. Class methods `supports()` and `resistances()` return
relevant subsets.

---

### `PivotType`

```python
class PivotType(Enum):
    TRADITIONAL = 'traditional'
    FIBONACCI = 'fibonacci'
    WOODIE = 'woodie'
    CLASSIC = 'classic'
    DM = 'dm'
    CAMARILLA = 'camarilla'
```

Selector for the six supported pivot calculation methods.

---

## Session Utilities

### `build_session_mask`

```python
def build_session_mask(
    index: pd.Index,
    session_name: str,
    sessions: Mapping[str, Mapping[str, object]],
    timezone: str = "America/New_York",
) -> np.ndarray
```

Return a boolean mask selecting bars that fall inside the named session
(e.g. `"asian"`, `"london"`, `"all_hours"`). Overnight sessions that cross
midnight are handled automatically.

---

### `get_session_ids`

```python
def get_session_ids(
    index: pd.Index,
    timezone: str = "America/New_York",
    session_close_hour: int = 17,
) -> np.ndarray
```

Return an integer array where each unique value identifies one trading session,
derived from exchange-local time and a configurable close hour.

---

### `get_session_labels`

```python
def get_session_labels(
    index: pd.Index,
    timezone: str = "America/New_York",
    session_close_hour: int = 17,
) -> pd.DatetimeIndex
```

Like `get_session_ids` but returns normalised `DatetimeIndex` labels
(one date per session).

---

## Pivot Indicators

### `calculate_traditional_pivots`

```python
def calculate_traditional_pivots(high: float, low: float, close: float) -> Dict[str, float]
```

Standard pivot formula: `P = (H+L+C)/3`, with `R1…R3`, `S1…S3`.

---

### `calculate_woodie_pivots`

```python
def calculate_woodie_pivots(high: float, low: float, close: float) -> Dict[str, float]
```

Woodie pivots weight the close more heavily: `P = (H+L+2C)/4`.

---

### `calculate_camarilla_pivots`

```python
def calculate_camarilla_pivots(high: float, low: float, close: float) -> Dict[str, float]
```

Camarilla pivots use a fraction of the range centred on the close.

---

### `calculate_fibonacci_pivots`

```python
def calculate_fibonacci_pivots(high: float, low: float, close: float) -> Dict[str, float]
```

Fibonacci pivots apply 0.382 and 0.618 retracement ratios to the range.

---

### `calculate_classic_pivots`

```python
def calculate_classic_pivots(high: float, low: float, close: float) -> Dict[str, float]
```

Classic pivots: same centre as traditional but `R2/R3` and `S2/S3` are
multiples of the full range.

---

### `calculate_dm_pivots`

```python
def calculate_dm_pivots(
    high: float, low: float, close: float, open_price: Optional[float] = None
) -> Dict[str, float]
```

DeMark pivots conditionally weight H, L, C based on the open-to-close
relationship.

---

### `calculate_pivot_levels`

```python
def calculate_pivot_levels(
    daily_df: pd.DataFrame,
    pivot_type: PivotType = PivotType.TRADITIONAL,
    intraday_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame
```

Compute a full DataFrame of pivot levels from daily OHLC, one row per
session day. Index is the *next* session date so levels can be forward-filled.

---

### `calculate_daily_ohlc`

```python
def calculate_daily_ohlc(
    df: pd.DataFrame,
    timezone: str = 'America/New_York',
    session_close_hour: int = 17,
) -> pd.DataFrame
```

Aggregate intraday OHLCV into session-aligned daily bars, respecting the
exchange close hour rather than UTC midnight.

---

### `prepare_data_with_pivots`

```python
def prepare_data_with_pivots(
    df: pd.DataFrame,
    pivot_type: PivotType = PivotType.TRADITIONAL,
    timezone: str = 'America/New_York',
    session_close_hour: int = 17,
) -> pd.DataFrame
```

One-call convenience: compute daily OHLC, calculate pivots, and merge them
into the original intraday DataFrame, dropping the first session (no prior
day data). The result is ready for backtesting.

---

## Trend Filtering

### `TrendFilterConfig`

```python
TrendFilterConfig = Dict[str, Dict]  # preset name → {ema, sma, timeframe, description}
```

Built-in named presets (`'baseline'`, `'ema5_sma21_1h'`,
`'ema5_sma21_1h_vwap'`) that configure the EMA/SMA trend filter.

---

### `calculate_trend_bias`

```python
def calculate_trend_bias(
    df: pd.DataFrame, ema_period: int, sma_period: int, timeframe: str
) -> pd.Series
```

Compute a `+1 / -1` trend bias series on a higher timeframe, shifted forward
one bar to prevent look-ahead bias.

---

### `filter_by_trend`

```python
def filter_by_trend(df: pd.DataFrame, trend_bias: pd.Series, direction: Direction) -> pd.DataFrame
```

Filter a DataFrame to only rows where the trend bias agrees with the
trade direction (bullish for `LONG`, bearish for `SHORT`).

---

## Specification Dataclasses

### `InstrumentSpec`

```python
@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    tick_size: float          # minimum price increment (e.g. 0.10 for GC)
    tick_value: float         # dollar value per tick (e.g. 10 for GC)
    commission_round_trip: float
    slippage_ticks: float       # estimated ticks per side
    timezone: str             # exchange timezone
    session_close_hour: int   # local hour of session close
```

Describes the futures contract being traded.

---

### `ExecutionPolicy`

```python
@dataclass(frozen=True)
class ExecutionPolicy:
    close_at_eod: bool = True
    max_trades_per_session: int = 5
    one_trade_per_level: bool = True
    allow_concurrent_positions: bool = False
    max_days_to_hold: int = 5
```

Controls trade management inside the event backtester.

---

### `TrailingStopPolicy`

```python
@dataclass(frozen=True)
class TrailingStopPolicy:
    enabled: bool = False
    trigger_ticks: Optional[float] = None
    lock_ticks: Optional[float] = None
```

When enabled, the stop loss locks in profit once price moves `trigger_ticks`
in your favour, tightening to `entry ± lock_ticks`.

---

### `BreakevenPolicy`

```python
@dataclass(frozen=True)
class BreakevenPolicy:
    enabled: bool = False
    trigger_pct_to_tp: float = 0.5
```

When enabled, moves the stop to entry once price reaches `trigger_pct_to_tp`
(50 % default) of the distance to take-profit.

---

### `EntryEvent`

```python
@dataclass
class EntryEvent:
    entry_idx: int
    direction: Direction
    entry_price: float
    stop_price: float
    take_profit_price: float
    level_tag: Optional[str] = None
    metadata: Dict[str, Any] = {}
```

Pre-computed signal consumed by `run_event_backtest`.

---

## Event Backtester (Advanced)

### `run_event_backtest`

```python
def run_event_backtest(
    df: pd.DataFrame,
    entry_events: Iterable[EntryEvent],
    instrument: InstrumentSpec,
    execution_policy: ExecutionPolicy,
    trailing_policy=None,
    breakeven_policy=None,
    precomputed: Optional[Dict] = None,
) -> BacktestRunResult
```

Run a full event-driven backtest that supports trailing stops, breakeven
stops, EOD close, and max-hold exits. Returns a `BacktestRunResult` with
per-trade records including MFE/MAE.

---

### `summarize_trades`

```python
def summarize_trades(
    trades: Sequence[EngineTradeRecord],
    total_cost_per_trade: float = 0.0,
) -> BacktestRunResult
```

Aggregate a list of `EngineTradeRecord` objects into summary statistics.

---

## Vectorized Backtester (Fast Grid Search)

### `VectorizedBacktester`

```python
class VectorizedBacktester:
    def __init__(
        self,
        tick_size: float,
        tick_value: float,
        commission: float,
        slippage_cost: float,
        max_trades_per_day: int = 10,
        once_per_day_level: bool = True,
        timezone: str = 'America/New_York',
        session_close_hour: int = 17,
    ): ...

    def run(self, df, level_col, direction, sl_points, tp_points,
            start_date=None, entry_filter_mask=None) -> VectorizedBacktestResult: ...

```

High-speed Numba-accelerated backtester for SL/TP parameter sweeps. SL and
TP are specified in **ticks**. `run()` returns aggregate summary statistics;
use the event backtester when per-trade records are required.

---

## Walk-Forward Validation

### `WalkForwardFold`

```python
@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
```

One rolling train/test split. Properties `train_duration_days` and
`test_duration_days` give period lengths.

---

### `OptResult`

```python
@dataclass
class OptResult:
    pivot_type: str
    level: str
    direction: str
    sl: int
    tp: int
    rr_ratio: float
    trades: int
    win_rate: float
    pnl_net: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
```

Result of a single (pivot_type, level, direction, SL, TP) optimisation run.

---

### `FoldResult`

```python
@dataclass
class FoldResult:
    filter_name: str
    fold_id: int
    best_pivot_type: str
    best_level: str
    best_direction: str
    best_sl: int
    best_tp: int
    train_pnl: float
    train_sharpe: float
    train_sortino: float
    train_trades: int
    train_win_rate: float
    train_max_dd: float
    train_profit_factor: float
    test_pnl: float
    test_sharpe: float
    test_sortino: float
    test_trades: int
    test_win_rate: float
    test_max_dd: float
    test_profit_factor: float
    pnl_profitable_oos: bool
    profit_factor_threshold_met: bool = False
    sortino_threshold_met: bool = False
```

Train- and out-of-sample metrics for the best candidate in one fold.

---

### `generate_walk_forward_folds`

```python
def generate_walk_forward_folds(
    data_start: datetime,
    data_end: datetime,
    train_months: int,
    test_months: int,
    step_months: int,
) -> List[WalkForwardFold]
```

Create a list of `WalkForwardFold` objects using anchored, non-overlapping
test windows that step forward by `step_months`.
