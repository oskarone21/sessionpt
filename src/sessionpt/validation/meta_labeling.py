"""Leakage checks for conservative meta-labeling experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

FORBIDDEN_FEATURE_TOKENS = ("future", "forward", "exit", "target", "label", "pnl")
DEFAULT_TARGET_VALUES = frozenset({0, 1})
ROWS_KEY = "rows"
FEATURES_KEY = "features"
FOLDS_KEY = "folds"
POSITIVE_LABELS_KEY = "positive_labels"


@dataclass(frozen=True)
class MetaLabelSpec:
    feature_cols: tuple[str, ...]
    target_col: str
    timestamp_col: str
    fold_col: str


def validate_meta_label_dataset(
    frame: pd.DataFrame,
    spec: MetaLabelSpec,
    allowed_target_values: frozenset[int] = DEFAULT_TARGET_VALUES,
) -> dict[str, int]:
    missing = [
        column
        for column in (*spec.feature_cols, spec.target_col, spec.timestamp_col, spec.fold_col)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing meta-label columns: {missing}")

    reserved = {spec.target_col, spec.timestamp_col, spec.fold_col}
    overlapping = sorted(set(spec.feature_cols) & reserved)
    if overlapping:
        raise ValueError(f"Feature columns overlap target/timestamp/fold columns: {overlapping}")

    leaking = [
        column
        for column in spec.feature_cols
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if leaking:
        raise ValueError(f"Feature columns look leakage-prone: {leaking}")

    target = frame[spec.target_col]
    if target.isna().any():
        raise ValueError(f"Target column contains nulls: {spec.target_col}")
    target_values = set(target.unique())
    if not target_values.issubset(allowed_target_values):
        raise ValueError(f"Unexpected target values: {sorted(target_values)}")

    if frame[spec.timestamp_col].isna().any():
        raise ValueError(f"Timestamp column contains nulls: {spec.timestamp_col}")
    timestamp_dtype = frame[spec.timestamp_col].dtype
    is_datetime = isinstance(
        timestamp_dtype, pd.DatetimeTZDtype
    ) or pd.api.types.is_datetime64_dtype(timestamp_dtype)
    if not is_datetime:
        raise ValueError(f"Timestamp column must be datetime-like: {spec.timestamp_col}")
    if frame[spec.fold_col].isna().any():
        raise ValueError(f"Fold column contains nulls: {spec.fold_col}")

    fold_ranges = frame.groupby(spec.fold_col, sort=False)[spec.timestamp_col].agg(["min", "max"])
    fold_ranges = fold_ranges.sort_values("min")
    if len(fold_ranges) > 1:
        previous_max = fold_ranges["max"].shift(1)
        if (fold_ranges["min"] <= previous_max).iloc[1:].any():
            raise ValueError("Fold timestamp ranges overlap or are temporally interleaved")

    return {
        ROWS_KEY: len(frame),
        FEATURES_KEY: len(spec.feature_cols),
        FOLDS_KEY: int(frame[spec.fold_col].nunique()),
        POSITIVE_LABELS_KEY: int((target == 1).sum()),
    }


def build_meta_label_spec(
    feature_cols: Sequence[str],
    target_col: str = "target_reaches_tp_before_sl",
    timestamp_col: str = "entry_time",
    fold_col: str = "fold_id",
) -> MetaLabelSpec:
    return MetaLabelSpec(
        feature_cols=tuple(feature_cols),
        target_col=target_col,
        timestamp_col=timestamp_col,
        fold_col=fold_col,
    )
