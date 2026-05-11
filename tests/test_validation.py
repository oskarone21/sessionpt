"""Tests for sessionpt.validation.walk_forward module."""

from datetime import datetime

from sessionpt.validation.walk_forward import (
    FoldResult,
    OptResult,
    WalkForwardFold,
    generate_walk_forward_folds,
)


class TestWalkForwardFold:
    def test_duration_days(self):
        fold = WalkForwardFold(
            fold_id=1,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 7, 1),
            test_start=datetime(2020, 7, 1),
            test_end=datetime(2020, 9, 1),
        )
        assert fold.train_duration_days == 182
        assert fold.test_duration_days == 62


class TestGenerateWalkForwardFolds:
    def test_basic_fold_generation(self):
        folds = generate_walk_forward_folds(
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2023, 1, 1),
            train_months=6,
            test_months=2,
            step_months=2,
        )
        assert len(folds) == 15
        assert folds[0].fold_id == 1
        assert folds[-1].fold_id == 15

    def test_fold_ordering(self):
        folds = generate_walk_forward_folds(
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2023, 1, 1),
        )
        for i in range(len(folds) - 1):
            assert folds[i].test_start < folds[i + 1].test_start

    def test_no_fold_beyond_data_end(self):
        folds = generate_walk_forward_folds(
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2021, 1, 1),
            train_months=6,
            test_months=2,
            step_months=2,
        )
        for fold in folds:
            assert fold.test_end <= datetime(2021, 1, 1)

    def test_empty_when_data_too_short(self):
        folds = generate_walk_forward_folds(
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2020, 3, 1),
            train_months=6,
            test_months=2,
        )
        assert len(folds) == 0

    def test_train_test_sequential(self):
        folds = generate_walk_forward_folds(
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2022, 1, 1),
            train_months=6,
            test_months=2,
            step_months=2,
        )
        for fold in folds:
            assert fold.train_end == fold.test_start


class TestOptResult:
    def test_defaults(self):
        result = OptResult()
        assert result.pivot_type == ""
        assert result.sl == 0
        assert result.pnl_net == 0.0

    def test_custom_values(self):
        result = OptResult(pivot_type="traditional", sl=50, tp=100, pnl_net=5000.0)
        assert result.pivot_type == "traditional"
        assert result.pnl_net == 5000.0


class TestFoldResult:
    def test_defaults(self):
        result = FoldResult()
        assert result.fold_id == 0
        assert result.pnl_profitable_oos is False

    def test_pnl_profitable_flag(self):
        result = FoldResult(test_pnl=1000.0, pnl_profitable_oos=True)
        assert result.pnl_profitable_oos is True
