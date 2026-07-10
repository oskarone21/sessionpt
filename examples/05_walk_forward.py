"""Generate disjoint walk-forward folds with the public sessionpt API."""

from __future__ import annotations

from datetime import datetime

from sessionpt import generate_walk_forward_folds


def main() -> None:
    folds = generate_walk_forward_folds(
        datetime(2020, 1, 2),
        datetime(2023, 12, 29, 23, 59, 59, 999999),
        train_months=12,
        test_months=3,
        step_months=3,
    )
    print("Walk-Forward Folds")
    for fold in folds:
        print(
            fold.fold_id,
            fold.train_start.date(),
            fold.train_end.date(),
            fold.test_start.date(),
            fold.test_end.date(),
        )
    print(f"Total folds: {len(folds)}")


if __name__ == "__main__":
    main()
