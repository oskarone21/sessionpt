"""Multiple-testing corrections for strategy search results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import isfinite

DEFAULT_ALPHA = 0.05


class CorrectionMethod(str, Enum):
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"


@dataclass(frozen=True)
class MultipleTestResult:
    name: str
    p_value: float
    adjusted_p_value: float
    rejected: bool
    method: CorrectionMethod
    rank: int
    n_tests: int


def _validate_tests(tests: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    validated = [(str(name), float(p_value)) for name, p_value in tests]
    for name, p_value in validated:
        if not isfinite(p_value) or p_value < 0.0 or p_value > 1.0:
            raise ValueError(f"Invalid p-value for {name}: {p_value}")
    return validated


def holm_bonferroni(
    tests: Iterable[tuple[str, float]],
    alpha: float = DEFAULT_ALPHA,
) -> tuple[MultipleTestResult, ...]:
    """Apply Holm-Bonferroni family-wise error correction."""

    ordered = sorted(_validate_tests(tests), key=lambda item: item[1])
    n_tests = len(ordered)
    adjusted_by_name: dict[str, float] = {}
    running_max = 0.0

    for rank, (name, p_value) in enumerate(ordered, start=1):
        adjusted = min(1.0, (n_tests - rank + 1) * p_value)
        running_max = max(running_max, adjusted)
        adjusted_by_name[name] = running_max

    return tuple(
        MultipleTestResult(
            name=name,
            p_value=p_value,
            adjusted_p_value=adjusted_by_name[name],
            rejected=adjusted_by_name[name] <= alpha,
            method=CorrectionMethod.HOLM,
            rank=rank,
            n_tests=n_tests,
        )
        for rank, (name, p_value) in enumerate(ordered, start=1)
    )


def benjamini_hochberg(
    tests: Iterable[tuple[str, float]],
    alpha: float = DEFAULT_ALPHA,
) -> tuple[MultipleTestResult, ...]:
    """Apply Benjamini-Hochberg false-discovery-rate correction."""

    ordered = sorted(_validate_tests(tests), key=lambda item: item[1])
    n_tests = len(ordered)
    adjusted_values: list[float] = [1.0] * n_tests
    running_min = 1.0

    for reverse_index in range(n_tests - 1, -1, -1):
        rank = reverse_index + 1
        _, p_value = ordered[reverse_index]
        adjusted = min(running_min, p_value * n_tests / rank)
        running_min = min(running_min, adjusted)
        adjusted_values[reverse_index] = min(1.0, adjusted)

    return tuple(
        MultipleTestResult(
            name=name,
            p_value=p_value,
            adjusted_p_value=adjusted_values[index],
            rejected=adjusted_values[index] <= alpha,
            method=CorrectionMethod.BENJAMINI_HOCHBERG,
            rank=index + 1,
            n_tests=n_tests,
        )
        for index, (name, p_value) in enumerate(ordered)
    )


def apply_multiple_testing_correction(
    tests: Iterable[tuple[str, float]],
    method: CorrectionMethod = CorrectionMethod.HOLM,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[MultipleTestResult, ...]:
    if method == CorrectionMethod.HOLM:
        return holm_bonferroni(tests, alpha=alpha)
    if method == CorrectionMethod.BENJAMINI_HOCHBERG:
        return benjamini_hochberg(tests, alpha=alpha)
    raise ValueError(f"Unsupported correction method: {method}")
