"""Generate walk-forward folds for 2020-2023 and print fold info."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_days: int
    test_days: int


def generate_walk_forward_folds(
    start: str = "2020-01-02",
    end: str = "2023-12-29",
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
) -> list[Fold]:
    dates = pd.bdate_range(start, end)
    folds: list[Fold] = []
    anchor = pd.Timestamp(start)
    fold_id = 0
    while True:
        train_start = anchor
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > pd.Timestamp(end):
            break
        train_days = len(dates[(dates >= train_start) & (dates <= train_end)])
        test_days = len(dates[(dates >= test_start) & (dates <= test_end)])
        if train_days > 0 and test_days > 0:
            folds.append(
                Fold(fold_id, train_start, train_end, test_start, test_end, train_days, test_days)
            )
        anchor += pd.DateOffset(months=step_months)
        fold_id += 1
    return folds


def main() -> None:
    folds = generate_walk_forward_folds()

    print("Walk-Forward Folds (2020-2023)")
    print("=" * 95)
    print(
        f"{'Fold':>4}  {'Train Start':>12}  {'Train End':>12}  {'Test Start':>12}  "
        f"{'Test End':>12}  {'Trn':>5}  {'Tst':>4}"
    )
    print("-" * 95)
    for f in folds:
        print(
            f"{f.fold_id:>4}  {f.train_start.strftime('%Y-%m-%d'):>12}  "
            f"{f.train_end.strftime('%Y-%m-%d'):>12}  {f.test_start.strftime('%Y-%m-%d'):>12}  "
            f"{f.test_end.strftime('%Y-%m-%d'):>12}  {f.train_days:>5}  {f.test_days:>4}"
        )
    print(f"\nTotal folds: {len(folds)}")
    total_train = sum(f.train_days for f in folds)
    total_test = sum(f.test_days for f in folds)
    print(f"Total train days (cumulative): {total_train}")
    print(f"Total test days (cumulative): {total_test}")


if __name__ == "__main__":
    main()
