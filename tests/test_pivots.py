"""Tests for sessionpt.pivots module."""

import numpy as np
import pandas as pd
import pytest

from sessionpt.enums.pivot_type import PivotType
from sessionpt.pivots.calculator import (
    calculate_camarilla_pivots,
    calculate_classic_pivots,
    calculate_dm_pivots,
    calculate_fibonacci_pivots,
    calculate_pivot_levels,
    calculate_traditional_pivots,
    calculate_woodie_pivots,
)
from sessionpt.pivots.ohlc import (
    calculate_daily_ohlc,
    prepare_data_with_pivots,
)


class TestTraditionalPivots:
    def test_basic(self):
        levels = calculate_traditional_pivots(high=2050, low=2000, close=2025)
        p = (2050 + 2000 + 2025) / 3
        assert levels["P"] == pytest.approx(p)
        assert levels["R1"] == pytest.approx(2 * p - 2000)
        assert levels["S1"] == pytest.approx(2 * p - 2050)

    def test_r1_gt_p_gt_s1(self):
        levels = calculate_traditional_pivots(high=2050, low=2000, close=2025)
        assert levels["R1"] > levels["P"] > levels["S1"]


class TestWoodiePivots:
    def test_woodie_emphasizes_close(self):
        levels = calculate_woodie_pivots(high=2050, low=2000, close=2025)
        p = (2050 + 2000 + 2 * 2025) / 4
        assert levels["P"] == pytest.approx(p)


class TestFibonacciPivots:
    def test_fibonacci_range_based(self):
        levels = calculate_fibonacci_pivots(high=2050, low=2000, close=2025)
        p = (2050 + 2000 + 2025) / 3
        r = 2050 - 2000
        assert levels["R1"] == pytest.approx(p + 0.382 * r)
        assert levels["S1"] == pytest.approx(p - 0.382 * r)


class TestClassicPivots:
    def test_classic_same_p_as_traditional(self):
        trad = calculate_traditional_pivots(2050, 2000, 2025)
        classic = calculate_classic_pivots(2050, 2000, 2025)
        assert classic["P"] == pytest.approx(trad["P"])


class TestDMPivots:
    def test_close_below_open(self):
        levels = calculate_dm_pivots(high=2050, low=2000, close=2010, open_price=2030)
        x = 2050 + 2 * 2000 + 2010
        assert levels["P"] == pytest.approx(x / 4)

    def test_close_above_open(self):
        levels = calculate_dm_pivots(high=2050, low=2000, close=2040, open_price=2010)
        x = 2 * 2050 + 2000 + 2040
        assert levels["P"] == pytest.approx(x / 4)

    def test_close_equals_open(self):
        levels = calculate_dm_pivots(high=2050, low=2000, close=2025, open_price=2025)
        x = 2050 + 2000 + 2 * 2025
        assert levels["P"] == pytest.approx(x / 4)

    def test_default_open_is_close(self):
        levels = calculate_dm_pivots(high=2050, low=2000, close=2025)
        assert levels["P"] == pytest.approx((2050 + 2000 + 2 * 2025) / 4)


class TestCamarillaPivots:
    def test_camarilla_range_based(self):
        levels = calculate_camarilla_pivots(high=2050, low=2000, close=2025)
        r = 2050 - 2000
        assert levels["R1"] == pytest.approx(2025 + r * 1.1 / 12)
        assert levels["S1"] == pytest.approx(2025 - r * 1.1 / 12)


class TestCalculatePivotLevels:
    @pytest.fixture
    def daily_df(self):
        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        return pd.DataFrame(
            {
                "Open": [2000, 2010, 2005, 2015, 2020],
                "High": [2050, 2040, 2035, 2045, 2055],
                "Low": [1980, 1995, 1990, 2000, 2005],
                "Close": [2010, 2025, 2020, 2035, 2040],
            },
            index=dates,
        )

    def test_traditional_levels(self, daily_df):
        result = calculate_pivot_levels(daily_df, PivotType.TRADITIONAL)
        assert "P" in result.columns
        assert "R1" in result.columns
        assert "S1" in result.columns
        assert len(result) == len(daily_df) - 1

    def test_woodie_levels(self, daily_df):
        result = calculate_pivot_levels(daily_df, PivotType.WOODIE)
        assert len(result) == len(daily_df) - 1

    def test_all_pivot_types(self, daily_df):
        for pt in PivotType:
            result = calculate_pivot_levels(daily_df, pt)
            assert len(result) == len(daily_df) - 1

    def test_single_session_returns_stable_empty_schema(self):
        daily = pd.DataFrame(
            {"Open": [100.0], "High": [110.0], "Low": [90.0], "Close": [105.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )

        result = calculate_pivot_levels(daily)

        assert result.empty
        assert result.columns.tolist() == ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
        assert result.index.name == "date"


class TestDailyOHLC:
    @pytest.fixture
    def intraday_df(self):
        np.random.seed(42)
        n = 2880
        dates = pd.date_range("2024-01-07 18:00", periods=n, freq="1min", tz="US/Eastern")
        close = 2000.0 + np.cumsum(np.random.randn(n) * 0.3)
        df = pd.DataFrame(
            {
                "Open": close - np.random.rand(n) * 0.5,
                "High": close + np.abs(np.random.randn(n)) * 0.8,
                "Low": close - np.abs(np.random.randn(n)) * 0.8,
                "Close": close,
                "Volume": np.random.randint(50, 500, size=n),
            },
            index=dates,
        )
        return df

    def test_session_aware_ohlc(self, intraday_df):
        daily = calculate_daily_ohlc(intraday_df)
        assert len(daily) >= 1
        assert "Open" in daily.columns
        assert "High" in daily.columns
        assert "Low" in daily.columns
        assert "Close" in daily.columns

    def test_daily_high_ge_close(self, intraday_df):
        daily = calculate_daily_ohlc(intraday_df)
        assert (daily["High"] >= daily["Close"]).all()

    def test_daily_low_le_close(self, intraday_df):
        daily = calculate_daily_ohlc(intraday_df)
        assert (daily["Low"] <= daily["Close"]).all()


class TestPrepareDataWithPivots:
    @pytest.fixture
    def intraday_df(self):
        np.random.seed(42)
        n = 2880 * 3
        dates = pd.date_range("2024-01-07 18:00", periods=n, freq="1min", tz="US/Eastern")
        close = 2000.0 + np.cumsum(np.random.randn(n) * 0.3)
        df = pd.DataFrame(
            {
                "Open": close - np.random.rand(n) * 0.5,
                "High": close + np.abs(np.random.randn(n)) * 0.8,
                "Low": close - np.abs(np.random.randn(n)) * 0.8,
                "Close": close,
                "Volume": np.random.randint(50, 500, size=n),
            },
            index=dates,
        )
        return df

    def test_prepare_adds_pivot_columns(self, intraday_df):
        result = prepare_data_with_pivots(intraday_df, PivotType.TRADITIONAL)
        assert "P" in result.columns
        assert "R1" in result.columns
        assert "S1" in result.columns

    def test_prepare_drops_first_session(self, intraday_df):
        result = prepare_data_with_pivots(intraday_df, PivotType.TRADITIONAL)
        assert not result["P"].isna().any()

    def test_prepare_preserves_ohlc(self, intraday_df):
        result = prepare_data_with_pivots(intraday_df, PivotType.TRADITIONAL)
        assert "Open" in result.columns
        assert "Close" in result.columns

    def test_spring_dst_does_not_assign_future_pivots(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-03-08 16:00", tz="America/New_York"),
                pd.Timestamp("2024-03-10 18:00", tz="America/New_York"),
                pd.Timestamp("2024-03-11 10:00", tz="America/New_York"),
                pd.Timestamp("2024-03-11 18:00", tz="America/New_York"),
                pd.Timestamp("2024-03-12 10:00", tz="America/New_York"),
            ]
        )
        df = pd.DataFrame(
            {
                "Open": [50.0, 20.0, 100.0, 30.0, 200.0],
                "High": [51.0, 21.0, 101.0, 31.0, 201.0],
                "Low": [49.0, 19.0, 99.0, 29.0, 199.0],
                "Close": [50.0, 20.0, 100.0, 30.0, 200.0],
                "Volume": [1.0] * 5,
            },
            index=idx,
        )

        result = prepare_data_with_pivots(df, PivotType.TRADITIONAL)

        assert result.loc[idx[1], "P"] == pytest.approx(50.0)
        assert result.loc[idx[2], "P"] == pytest.approx(50.0)
