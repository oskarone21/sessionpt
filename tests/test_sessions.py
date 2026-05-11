"""Tests for sessionpt.sessions module."""

import numpy as np
import pandas as pd
import pytest

from sessionpt.sessions.core import (
    build_session_mask,
    ensure_utc_index,
    get_local_hours,
    get_session_ids,
    get_session_labels,
)
from sessionpt.sessions.presets import (
    CME_SESSION_PRESETS,
    NAMED_SESSIONS,
    get_session_preset,
)


class TestEnsureUtcIndex:
    def test_naive_becomes_utc(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        result = ensure_utc_index(idx)
        assert result.tz is not None
        assert str(result.tz) == "UTC"

    def test_et_converted_to_utc(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="US/Eastern")
        result = ensure_utc_index(idx)
        assert str(result.tz) == "UTC"


class TestGetSessionLabels:
    def test_daytime_bars_same_session(self):
        idx = pd.date_range("2024-01-08 09:00", "2024-01-08 16:00", freq="h", tz="US/Eastern")
        labels = get_session_labels(idx, timezone="America/New_York", session_close_hour=17)
        normalized = labels.tz_localize(None).normalize() if labels.tz else labels.normalize()
        assert all(normalized == pd.Timestamp("2024-01-08"))

    def test_evening_bars_pushed_to_next_day(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-07 18:00", tz="US/Eastern"),
                pd.Timestamp("2024-01-08 16:00", tz="US/Eastern"),
            ]
        )
        labels = get_session_labels(idx, timezone="America/New_York", session_close_hour=17)
        assert labels[0].date() == pd.Timestamp("2024-01-08").date()
        assert labels[1].date() == pd.Timestamp("2024-01-08").date()

    def test_close_hour_bar_stays_current_session(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-08 17:00", tz="US/Eastern"),
            ]
        )
        labels = get_session_labels(idx, timezone="America/New_York", session_close_hour=17)
        assert labels[0].date() == pd.Timestamp("2024-01-08").date()


class TestGetSessionIds:
    def test_session_ids_are_ints(self):
        idx = pd.date_range("2024-01-08", periods=10, freq="h", tz="US/Eastern")
        ids = get_session_ids(idx)
        assert ids.dtype in (np.int64, np.int32)

    def test_different_sessions_get_different_ids(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-07 18:00", tz="US/Eastern"),
                pd.Timestamp("2024-01-09 18:00", tz="US/Eastern"),
            ]
        )
        ids = get_session_ids(idx, timezone="America/New_York", session_close_hour=17)
        assert ids[0] != ids[1]


class TestGetLocalHours:
    def test_hours_match(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-08 09:00", tz="US/Eastern"),
                pd.Timestamp("2024-01-08 17:00", tz="US/Eastern"),
            ]
        )
        hours = get_local_hours(idx, timezone="America/New_York")
        assert hours[0] == 9
        assert hours[1] == 17


class TestBuildSessionMask:
    def test_all_hours(self):
        idx = pd.date_range("2024-01-08", periods=24, freq="h", tz="US/Eastern")
        mask = build_session_mask(idx, "all_hours", NAMED_SESSIONS)
        assert mask.all()

    def test_asian_session(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-08 19:00", tz="US/Eastern"),
                pd.Timestamp("2024-01-08 02:00", tz="US/Eastern"),
                pd.Timestamp("2024-01-08 10:00", tz="US/Eastern"),
            ]
        )
        mask = build_session_mask(idx, "asian", NAMED_SESSIONS, timezone="America/New_York")
        assert mask[0]
        assert mask[1]
        assert not mask[2]

    def test_new_york_session(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-08 10:00", tz="US/Eastern"),
                pd.Timestamp("2024-01-08 16:00", tz="US/Eastern"),
                pd.Timestamp("2024-01-08 18:00", tz="US/Eastern"),
            ]
        )
        mask = build_session_mask(idx, "new_york", NAMED_SESSIONS, timezone="America/New_York")
        assert mask[0]
        assert mask[1]
        assert not mask[2]


class TestSessionPresets:
    def test_gc_preset(self):
        preset = get_session_preset("gc")
        assert preset.symbol == "GC"
        assert preset.session_close_hour == 17

    def test_es_preset(self):
        preset = get_session_preset("ES")
        assert preset.symbol == "ES"

    def test_unknown_symbol_raises(self):
        with pytest.raises(KeyError):
            get_session_preset("UNKNOWN")

    def test_all_presets_have_close_hour(self):
        for symbol, preset in CME_SESSION_PRESETS.items():
            assert preset.session_close_hour > 0
            assert preset.symbol == symbol

    def test_named_sessions(self):
        assert "all_hours" in NAMED_SESSIONS
        assert "asian" in NAMED_SESSIONS
        assert "new_york" in NAMED_SESSIONS
