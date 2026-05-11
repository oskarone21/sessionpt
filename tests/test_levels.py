import numpy as np
import pandas as pd

from sessionpt.levels import add_initial_balance_levels


def _make_intraday(index: pd.DatetimeIndex) -> pd.DataFrame:
    n_rows = len(index)
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 101, n_rows),
            "High": np.linspace(101, 102, n_rows),
            "Low": np.linspace(99, 100, n_rows),
            "Close": np.linspace(100, 101, n_rows),
            "Volume": np.ones(n_rows),
        },
        index=index,
    )


def test_initial_balance_levels_compute_after_window():
    index = pd.date_range("2024-01-02 14:30:00+00:00", periods=80, freq="1min")
    df = _make_intraday(index)
    df.iloc[:60, df.columns.get_loc("High")] = 110.0
    df.iloc[:60, df.columns.get_loc("Low")] = 100.0

    out = add_initial_balance_levels(
        df,
        timezone="America/New_York",
        ib_start_local="09:30",
        ib_window_minutes=60,
        rth_start_local="09:30",
        rth_end_local="16:00",
    )

    assert out["IBH"].iloc[:60].isna().all()
    assert out["IBH"].iloc[60] == 110.0
    assert out["IBL"].iloc[60] == 100.0
    assert out["IBRange"].iloc[60] == 10.0
    assert out["IB_25"].iloc[60] == 102.5
    assert out["IB_50"].iloc[60] == 105.0
    assert out["IB_EXT_UP_25"].iloc[60] == 112.5
    assert out["IB_EXT_DN_25"].iloc[60] == 97.5
    assert bool(out["ib_complete"].iloc[60]) is True


def test_initial_balance_window_is_local_time_dst_safe():
    index_pre = pd.date_range("2024-03-08 14:30:00+00:00", periods=60, freq="1min")
    index_post = pd.date_range("2024-03-12 13:30:00+00:00", periods=60, freq="1min")
    df = pd.concat([_make_intraday(index_pre), _make_intraday(index_post)]).sort_index()

    out = add_initial_balance_levels(
        df,
        timezone="America/New_York",
        ib_start_local="09:30",
        ib_window_minutes=60,
        rth_start_local="09:30",
        rth_end_local="16:00",
    )

    counts = out.groupby("ib_session_label")["ib_in_window"].sum().astype(int).tolist()
    assert counts == [60, 60]
