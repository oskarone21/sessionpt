import numpy as np
import pandas as pd

from sessionpt.features import (
    add_rth_anchored_vwap,
    build_volume_mask,
    combine_entry_masks,
    compute_atr,
    compute_nmin_entry_signal,
    compute_volume_ratio,
    compute_vwap,
    compute_wick_ratios,
)


def test_compute_atr_returns_wilder_series_with_warmup():
    highs = np.array([11.0, 12.0, 13.0, 14.0])
    lows = np.array([9.0, 10.0, 11.0, 12.0])
    closes = np.array([10.0, 11.0, 12.0, 13.0])

    atr = compute_atr(highs, lows, closes, period=3)

    assert np.isnan(atr[0])
    assert np.isnan(atr[1])
    assert np.isfinite(atr[2])
    assert len(atr) == len(closes)


def test_compute_vwap_resets_by_session_id():
    highs = np.array([102.0, 104.0, 202.0])
    lows = np.array([100.0, 100.0, 200.0])
    closes = np.array([101.0, 103.0, 201.0])
    volumes = np.array([2.0, 2.0, 1.0])
    session_ids = np.array([1, 1, 2])

    vwap = compute_vwap(highs, lows, closes, volumes, session_ids)

    first_typical = (102.0 + 100.0 + 101.0) / 3
    second_typical = (104.0 + 100.0 + 103.0) / 3
    third_typical = (202.0 + 200.0 + 201.0) / 3
    assert abs(vwap[0] - first_typical) < 1e-9
    assert abs(vwap[1] - ((first_typical * 2 + second_typical * 2) / 4)) < 1e-9
    assert abs(vwap[2] - third_typical) < 1e-9


def test_rth_vwap_resets_at_session_open():
    idx = pd.DatetimeIndex(
        [
            "2024-01-02 14:29:00+00:00",
            "2024-01-02 14:30:00+00:00",
            "2024-01-02 14:31:00+00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "Open": [100, 100, 100],
            "High": [100, 102, 104],
            "Low": [100, 100, 100],
            "Close": [100, 101, 103],
            "Volume": [1, 2, 2],
        },
        index=idx,
    )

    out = add_rth_anchored_vwap(
        df,
        timezone="America/New_York",
        rth_start_local="09:30",
        rth_end_local="16:00",
    )

    tp1 = (102 + 100 + 101) / 3
    tp2 = (104 + 100 + 103) / 3
    assert np.isnan(out.loc[idx[0], "rth_vwap"])
    assert abs(out.loc[idx[1], "rth_vwap"] - tp1) < 1e-9
    assert abs(out.loc[idx[2], "rth_vwap"] - ((tp1 * 2 + tp2 * 2) / 4)) < 1e-9


def test_volume_wick_nmin_and_mask_helpers():
    volumes = np.array([10.0, 20.0, 40.0])
    volume_ratio = compute_volume_ratio(volumes, lookback=2)
    assert build_volume_mask(volume_ratio, min_multiplier=1.0).tolist() == [True, True, True]

    lower_wick, upper_wick = compute_wick_ratios(
        opens=np.array([10.0]),
        highs=np.array([12.0]),
        lows=np.array([8.0]),
        closes=np.array([11.0]),
    )
    assert abs(lower_wick[0] - 0.5) < 1e-9
    assert abs(upper_wick[0] - 0.25) < 1e-9

    signal = compute_nmin_entry_signal(
        highs=np.array([10.0, 12.0, 11.0]),
        lows=np.array([8.0, 9.0, 9.5]),
        closes=np.array([9.0, 11.0, 10.5]),
        levels=np.array([10.0, 10.0, 10.0]),
        n_bars=2,
        is_long=True,
    )
    assert signal.tolist() == [False, True, True]
    assert combine_entry_masks([signal, np.array([True, False, True])]).tolist() == [
        False,
        False,
        True,
    ]
