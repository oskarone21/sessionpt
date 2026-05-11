"""Policy normalization helpers for trailing stop and breakeven."""

from sessionpt.backtesting.specs import BreakevenPolicy, TrailingStopPolicy


def normalize_trailing(policy: TrailingStopPolicy | None) -> TrailingStopPolicy:
    """Return the policy as-is, or a disabled default if None."""
    if policy is None:
        return TrailingStopPolicy(enabled=False, trigger_ticks=None, lock_ticks=None)
    return policy


def normalize_breakeven(policy: BreakevenPolicy | None) -> BreakevenPolicy:
    """Return the policy as-is, or a disabled default if None."""
    if policy is None:
        return BreakevenPolicy(enabled=False, trigger_pct_to_tp=0.5)
    return policy
