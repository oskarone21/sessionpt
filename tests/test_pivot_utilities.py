import pandas as pd

from sessionpt.enums import PivotType
from sessionpt.pivots import (
    apply_directional_pivot_shift,
    calculate_all_pivot_types,
    find_confluence_levels,
    get_confluence_at_level,
)


def test_directional_pivot_shift_moves_resistance_and_support_columns():
    df = pd.DataFrame(
        {
            "P": [100.0],
            "R1": [110.0],
            "R2": [120.0],
            "S1": [90.0],
            "S2": [80.0],
        }
    )

    shifted = apply_directional_pivot_shift(df, shift_price=2.0)

    assert shifted.loc[0, "P"] == 98.0
    assert shifted.loc[0, "R1"] == 112.0
    assert shifted.loc[0, "R2"] == 122.0
    assert shifted.loc[0, "S1"] == 88.0
    assert shifted.loc[0, "S2"] == 78.0


def test_pivot_confluence_helpers_find_aligned_levels():
    all_pivots = calculate_all_pivot_types(110.0, 100.0, 105.0)
    zones = find_confluence_levels(all_pivots, tolerance_pct=0.01)
    count, labels = get_confluence_at_level(110.0, 100.0, 105.0, target_price=105.0)

    assert zones
    assert count >= 1
    assert labels


def test_confluence_supports_demark_and_negative_prices():
    all_pivots = calculate_all_pivot_types(
        high=-5.0,
        low=-15.0,
        close=-10.0,
        open_price=-12.0,
        pivot_types=[PivotType.DM],
    )
    target = all_pivots[PivotType.DM.value]["P"]

    count, labels = get_confluence_at_level(
        -5.0,
        -15.0,
        -10.0,
        target_price=target,
        pivot_types=[PivotType.DM],
        open_price=-12.0,
    )

    assert count >= 1
    assert labels == ["dm_P", "dm_R2", "dm_R3", "dm_S2", "dm_S3"]
