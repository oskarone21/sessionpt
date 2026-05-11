"""Shared pytest fixtures for sessionpt tests."""

import numpy as np
import pandas as pd
import pytest

from sessionpt.backtesting.specs import (
    ExecutionPolicy,
    InstrumentSpec,
)
from sessionpt.enums.direction import Direction


@pytest.fixture
def sample_1min_df():
    """Two trading sessions of 1-minute GC (gold) bars (synthetic)."""
    dates = pd.date_range("2024-01-08 18:00", periods=1380, freq="1min", tz="US/Eastern")
    dates = dates.append(
        pd.date_range("2024-01-10 18:00", periods=1380, freq="1min", tz="US/Eastern")
    )
    np.random.seed(42)
    close = 2000.0 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close - np.random.rand(len(dates)) * 1.0,
            "High": close + np.abs(np.random.randn(len(dates))) * 1.5,
            "Low": close - np.abs(np.random.randn(len(dates))) * 1.5,
            "Close": close,
            "Volume": np.random.randint(100, 1000, size=len(dates)),
        },
        index=dates,
    )
    return df


@pytest.fixture
def gc_spec():
    """InstrumentSpec for CME Gold (GC)."""
    return InstrumentSpec(
        symbol="GC",
        tick_size=0.10,
        tick_value=10.0,
        commission_round_trip=2.40,
        slippage_ticks=1.0,
        timezone="America/New_York",
        session_close_hour=17,
    )


@pytest.fixture
def es_spec():
    """InstrumentSpec for CME E-mini S&P 500 (ES)."""
    return InstrumentSpec(
        symbol="ES",
        tick_size=0.25,
        tick_value=12.50,
        commission_round_trip=4.60,
        slippage_ticks=0.5,
        timezone="America/New_York",
        session_close_hour=17,
    )


@pytest.fixture
def default_execution():
    """Default execution policy: EOD close, 1 trade per level."""
    return ExecutionPolicy(
        close_at_eod=True,
        max_trades_per_session=5,
        one_trade_per_level=True,
        allow_concurrent_positions=False,
        max_days_to_hold=5,
    )


@pytest.fixture
def long_direction():
    return Direction.LONG


@pytest.fixture
def short_direction():
    return Direction.SHORT
