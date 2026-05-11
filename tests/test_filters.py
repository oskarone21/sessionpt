"""Tests for sessionpt.filters module."""

import numpy as np
import pandas as pd
import pytest

from sessionpt.enums.direction import Direction
from sessionpt.filters.trend import TrendFilterConfig, calculate_trend_bias, filter_by_trend


class TestTrendFilterConfig:
    def test_baseline_is_baseline(self):
        config = TrendFilterConfig()
        assert config.is_baseline is True

    def test_non_baseline_is_not_baseline(self):
        config = TrendFilterConfig(
            name="ema5_sma21_1h",
            ema_period=5,
            sma_period=21,
            htf_timeframe="1h",
        )
        assert config.is_baseline is False

    def test_partial_config_is_baseline(self):
        config = TrendFilterConfig(name="partial", ema_period=5)
        assert config.is_baseline is True


class TestCalculateTrendBias:
    @pytest.fixture
    def price_df(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        close = 2000.0 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "Open": close - np.random.rand(n) * 0.3,
                "High": close + np.abs(np.random.randn(n)) * 0.8,
                "Low": close - np.abs(np.random.randn(n)) * 0.8,
                "Close": close,
            },
            index=dates,
        )
        return df

    def test_returns_series(self, price_df):
        bias = calculate_trend_bias(price_df, ema_period=5, sma_period=21, htf_timeframe="1h")
        assert isinstance(bias, pd.Series)
        assert len(bias) == len(price_df)

    def test_bias_values(self, price_df):
        bias = calculate_trend_bias(price_df, ema_period=5, sma_period=21, htf_timeframe="1h")
        valid = bias.dropna()
        assert set(valid.unique()).issubset({1, -1})

    def test_lag_prevents_lookahead(self, price_df):
        bias = calculate_trend_bias(price_df, ema_period=5, sma_period=21, htf_timeframe="1h")
        nan_count = bias.isna().sum()
        assert nan_count > 0


class TestFilterByTrend:
    @pytest.fixture
    def price_df(self):
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": np.ones(n) * 2000,
                "High": np.ones(n) * 2001,
                "Low": np.ones(n) * 1999,
                "Close": np.ones(n) * 2000,
            },
            index=dates,
        )
        return df

    def test_long_keeps_bullish(self, price_df):
        bias = pd.Series(1, index=price_df.index)
        result = filter_by_trend(price_df, bias, Direction.LONG)
        assert len(result) == len(price_df)

    def test_short_keeps_bearish(self, price_df):
        bias = pd.Series(-1, index=price_df.index)
        result = filter_by_trend(price_df, bias, Direction.SHORT)
        assert len(result) == len(price_df)

    def test_zero_bias_returns_all(self, price_df):
        bias = pd.Series(0, index=price_df.index)
        result = filter_by_trend(price_df, bias, Direction.LONG)
        assert len(result) == len(price_df)
