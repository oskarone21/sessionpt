"""Tests for sessionpt.backtesting module."""

import numpy as np
import pandas as pd
import pytest

import sessionpt.backtesting.vectorized as vectorized_module
from sessionpt.backtesting.engine import (
    EntryEvent,
    precompute_backtest_arrays,
    run_event_backtest,
    summarize_trades,
)
from sessionpt.backtesting.policies import normalize_breakeven, normalize_trailing
from sessionpt.backtesting.specs import (
    BreakevenPolicy,
    ExecutionPolicy,
    InstrumentSpec,
    TrailingStopPolicy,
)
from sessionpt.backtesting.vectorized import VectorizedBacktester
from sessionpt.enums.direction import Direction


def _make_simple_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a simple synthetic price DataFrame for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-08 09:00", periods=n, freq="1min", tz="US/Eastern")
    close = 2000.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close - np.random.rand(n) * 0.3,
            "High": close + np.abs(np.random.randn(n)) * 0.8,
            "Low": close - np.abs(np.random.randn(n)) * 0.8,
            "Close": close,
            "Volume": np.random.randint(50, 500, size=n),
        },
        index=dates,
    )
    return df


class TestSpecs:
    def test_instrument_spec_defaults(self):
        spec = InstrumentSpec(
            symbol="GC",
            tick_size=0.10,
            tick_value=10.0,
            commission_round_trip=2.40,
            slippage_ticks=1.0,
        )
        assert spec.timezone == "America/New_York"
        assert spec.session_close_hour == 17

    def test_instrument_spec_cost(self):
        spec = InstrumentSpec(
            symbol="GC",
            tick_size=0.10,
            tick_value=10.0,
            commission_round_trip=2.40,
            slippage_ticks=1.0,
        )
        assert spec.slippage_cost == 20.0
        assert spec.total_cost_per_trade == 22.40

    def test_execution_policy_defaults(self):
        ep = ExecutionPolicy()
        assert ep.close_at_eod is True
        assert ep.max_trades_per_session == 5
        assert ep.one_trade_per_level is True

    def test_frozen_dataclass(self):
        spec = InstrumentSpec(
            symbol="GC",
            tick_size=0.10,
            tick_value=10.0,
            commission_round_trip=2.40,
            slippage_ticks=1.0,
        )
        with pytest.raises(AttributeError):
            spec.symbol = "ES"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tick_size": 0.0},
            {"tick_value": 0.0},
            {"commission_round_trip": -1.0},
            {"slippage_ticks": -1.0},
        ],
    )
    def test_invalid_instrument_values_raise(self, kwargs):
        values = {
            "symbol": "GC",
            "tick_size": 0.10,
            "tick_value": 10.0,
            "commission_round_trip": 2.40,
            "slippage_ticks": 1.0,
        }
        values.update(kwargs)
        with pytest.raises(ValueError):
            InstrumentSpec(**values)


class TestPolicies:
    def test_normalize_trailing_none(self):
        result = normalize_trailing(None)
        assert result.enabled is False

    def test_normalize_trailing_existing(self):
        policy = TrailingStopPolicy(enabled=True, trigger_ticks=50, lock_ticks=10)
        result = normalize_trailing(policy)
        assert result is policy

    def test_normalize_breakeven_none(self):
        result = normalize_breakeven(None)
        assert result.enabled is False

    def test_normalize_breakeven_existing(self):
        policy = BreakevenPolicy(enabled=True, trigger_pct_to_tp=0.5)
        result = normalize_breakeven(policy)
        assert result is policy


class TestEventBacktester:
    @pytest.fixture
    def gc_spec(self):
        return InstrumentSpec(
            symbol="GC",
            tick_size=0.10,
            tick_value=10.0,
            commission_round_trip=2.40,
            slippage_ticks=1.0,
        )

    @pytest.fixture
    def default_exec(self):
        return ExecutionPolicy(
            close_at_eod=True,
            max_trades_per_session=5,
            one_trade_per_level=True,
        )

    def test_empty_events_returns_zero(self, gc_spec, default_exec):
        df = _make_simple_df(100)
        result = run_event_backtest(df, [], gc_spec, default_exec)
        assert result.trades == 0
        assert result.total_pnl_net == 0.0

    def test_single_long_trade(self, gc_spec, default_exec):
        df = _make_simple_df(100)
        events = [
            EntryEvent(
                entry_idx=10,
                direction=Direction.LONG,
                entry_price=2000.0,
                stop_price=1995.0,
                take_profit_price=2010.0,
                level_tag="P",
            )
        ]
        result = run_event_backtest(df, events, gc_spec, default_exec)
        assert result.trades >= 1

    def test_single_short_trade(self, gc_spec, default_exec):
        df = _make_simple_df(100)
        events = [
            EntryEvent(
                entry_idx=10,
                direction=Direction.SHORT,
                entry_price=2000.0,
                stop_price=2005.0,
                take_profit_price=1990.0,
                level_tag="P",
            )
        ]
        result = run_event_backtest(df, events, gc_spec, default_exec)
        assert result.trades >= 1

    def test_precompute_backtest_arrays_is_public(self, gc_spec):
        df = _make_simple_df(100)
        arrays = precompute_backtest_arrays(df, gc_spec)
        assert set(arrays) == {
            "opens",
            "highs",
            "lows",
            "closes",
            "timestamps",
            "session_ids",
            "n_bars",
        }
        assert arrays["n_bars"] == len(df)

    def test_summarize_empty_trades(self, gc_spec):
        result = summarize_trades([], total_cost_per_trade=0.0)
        assert result.trades == 0
        assert result.total_pnl_net == 0.0
        assert result.profit_factor == 0.0

    def test_gap_through_stop_fills_at_open(self, gc_spec, default_exec):
        idx = pd.date_range("2024-01-08 09:00", periods=2, freq="1h", tz="US/Eastern")
        df = pd.DataFrame(
            {
                "Open": [100.0, 90.0],
                "High": [100.0, 92.0],
                "Low": [100.0, 88.0],
                "Close": [100.0, 91.0],
            },
            index=idx,
        )
        event = EntryEvent(0, Direction.LONG, 100.0, 95.0, 120.0)

        result = run_event_backtest(df, [event], gc_spec, default_exec)

        assert result.trade_records[0].exit_price == 90.0

    def test_dynamic_stop_changes_apply_on_next_bar(self, gc_spec, default_exec):
        idx = pd.date_range("2024-01-08 09:00", periods=2, freq="1min", tz="US/Eastern")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0],
                "High": [100.0, 106.0],
                "Low": [100.0, 94.0],
                "Close": [100.0, 100.0],
            },
            index=idx,
        )
        event = EntryEvent(0, Direction.LONG, 100.0, 95.0, 110.0)

        result = run_event_backtest(
            df,
            [event],
            gc_spec,
            default_exec,
            trailing_policy=TrailingStopPolicy(enabled=True, trigger_ticks=5.0, lock_ticks=2.0),
        )

        trade = result.trade_records[0]
        assert trade.exit_price == 95.0
        assert trade.exit_reason == "SL"

    def test_max_hold_uses_elapsed_time_not_bar_count(self, gc_spec):
        idx = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
        df = pd.DataFrame({"Open": 100.0, "High": 100.1, "Low": 99.9, "Close": 100.0}, index=idx)
        event = EntryEvent(0, Direction.LONG, 100.0, 90.0, 110.0)

        result = run_event_backtest(
            df,
            [event],
            gc_spec,
            ExecutionPolicy(close_at_eod=False, max_days_to_hold=1),
        )

        trade = result.trade_records[0]
        assert trade.exit_idx == 96
        assert trade.exit_time - trade.entry_time == pd.Timedelta(days=1)

    def test_invalid_trade_geometry_raises(self, gc_spec, default_exec):
        df = _make_simple_df(10)
        event = EntryEvent(0, Direction.LONG, 100.0, 105.0, 110.0)

        with pytest.raises(ValueError, match="stop < entry"):
            run_event_backtest(df, [event], gc_spec, default_exec)


class TestVectorizedBacktester:
    @pytest.fixture
    def gc_bt(self):
        return VectorizedBacktester(
            tick_size=0.10,
            tick_value=10.0,
            commission=2.40,
            slippage_cost=10.0,
            max_trades_per_day=5,
        )

    def test_empty_result_on_no_entries(self, gc_bt):
        df = _make_simple_df(100)
        df["P"] = 9999.0
        result = gc_bt.run(df, "P", Direction.LONG, sl_points=50, tp_points=100)
        assert result.trades == 0

    def test_run_with_pivot_column(self, gc_bt):
        df = _make_simple_df(2880)
        from sessionpt.enums.pivot_type import PivotType
        from sessionpt.pivots import prepare_data_with_pivots

        result_df = prepare_data_with_pivots(df, PivotType.TRADITIONAL)
        result = gc_bt.run(
            result_df,
            "P",
            Direction.LONG,
            sl_points=50,
            tp_points=100,
        )
        assert result.trades >= 0

    @pytest.mark.skipif(not vectorized_module.NUMBA_AVAILABLE, reason="Numba is not installed")
    def test_numba_and_python_kernels_match_at_session_boundary(self, gc_bt):
        entries = np.array([0], dtype=np.int64)
        opens = np.array([100.0, 101.0, 102.0, 103.0])
        closes = opens.copy()
        highs = opens + 0.1
        lows = opens - 0.1
        levels = np.full(4, 100.0)
        sessions = np.array([1, 1, 2, 2], dtype=np.int64)
        python_result = gc_bt._process_trades_python(
            entries,
            opens,
            closes,
            highs,
            lows,
            levels,
            True,
            50.0,
            50.0,
            1.0,
            1.0,
            0.0,
            sessions,
        )
        numba_result = vectorized_module._process_trades_numba(
            entries,
            opens,
            closes,
            highs,
            lows,
            levels,
            True,
            50.0,
            50.0,
            1.0,
            1.0,
            0.0,
            sessions,
            10,
            True,
        )

        np.testing.assert_array_equal(numba_result, python_result)
