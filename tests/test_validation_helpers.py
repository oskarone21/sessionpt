import pandas as pd
import pytest

from sessionpt.validation import (
    CorrectionMethod,
    apply_hard_filters,
    apply_multiple_testing_correction,
    build_meta_label_spec,
    iter_param_grid,
    rank_candidates,
    validate_meta_label_dataset,
)


def test_multiple_testing_corrections_validate_p_values():
    results = apply_multiple_testing_correction(
        (("a", 0.01), ("b", 0.20)),
        method=CorrectionMethod.HOLM,
    )

    assert results[0].name == "a"
    assert results[0].adjusted_p_value <= 0.02
    with pytest.raises(ValueError):
        apply_multiple_testing_correction((("bad", 1.5),), method=CorrectionMethod.HOLM)
    with pytest.raises(ValueError, match="unique"):
        apply_multiple_testing_correction((("same", 0.01), ("same", 0.20)))
    with pytest.raises(ValueError, match="alpha"):
        apply_multiple_testing_correction((("a", 0.01),), alpha=1.5)


def test_meta_label_dataset_validation_blocks_leakage_names():
    frame = pd.DataFrame(
        {
            "entry_time": pd.date_range("2024-01-01", periods=2),
            "fold_id": [1, 1],
            "pivot_distance_at_entry": [0.5, 1.0],
            "target_reaches_tp_before_sl": [1, 0],
        }
    )

    spec = build_meta_label_spec(["pivot_distance_at_entry"])
    assert validate_meta_label_dataset(frame, spec)["rows"] == 2

    leaking_spec = build_meta_label_spec(["future_return"])
    with pytest.raises(ValueError):
        validate_meta_label_dataset(frame.assign(future_return=[0.1, -0.2]), leaking_spec)

    target_feature_spec = build_meta_label_spec(["target_reaches_tp_before_sl"])
    with pytest.raises(ValueError, match="overlap"):
        validate_meta_label_dataset(frame, target_feature_spec)

    fractional = frame.assign(target_reaches_tp_before_sl=[0.5, 1.0])
    with pytest.raises(ValueError, match="Unexpected target"):
        validate_meta_label_dataset(fractional, spec)


def test_search_and_selection_helpers():
    grid = list(iter_param_grid({"a": [1, 2], "b": ["x"]}))
    rows = [
        {
            "sharpe_ratio": 1.0,
            "pnl_net": 100.0,
            "win_rate": 45.0,
            "trades": 10,
            "max_drawdown": 10.0,
        },
        {
            "sharpe_ratio": 2.0,
            "pnl_net": 50.0,
            "win_rate": 35.0,
            "trades": 10,
            "max_drawdown": 20.0,
        },
    ]

    assert grid == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
    assert len(apply_hard_filters(rows, min_win_rate=40.0, min_trades=5)) == 1
    assert rank_candidates(rows)[0]["sharpe_ratio"] == 2.0

    invalid = {**rows[0], "win_rate": float("nan"), "sharpe_ratio": float("nan")}
    assert apply_hard_filters([invalid], min_win_rate=0.0) == []
    with pytest.raises(ValueError, match="finite"):
        rank_candidates([invalid])
