import pandas as pd

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
