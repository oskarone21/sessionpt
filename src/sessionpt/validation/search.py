"""Small search-space helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from itertools import product


def iter_param_grid(grid: Mapping[str, Iterable]) -> Iterator[dict[str, object]]:
    keys = list(grid.keys())
    values = [list(grid[key]) for key in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo, strict=False))
