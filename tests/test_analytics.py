import numpy as np
import pytest

from sessionpt.analytics import (
    calculate_all_robust_metrics,
    calculate_kelly_fraction,
    calculate_max_drawdown,
    calculate_max_drawdown_duration,
    calculate_max_drawdown_from_trades,
    calculate_sortino_ratio,
    check_strategy_robustness,
)


def test_drawdown_and_duration_metrics():
    equity = np.array([100.0, 120.0, 90.0, 95.0, 130.0])

    max_drawdown, peak_idx, trough_idx = calculate_max_drawdown(equity)

    assert max_drawdown == 30.0
    assert peak_idx == 1
    assert trough_idx == 2
    assert calculate_max_drawdown_duration(equity) == 2


def test_sortino_kelly_and_metric_bundle():
    trade_pnls = np.array([100.0, -50.0, 150.0, -25.0])

    assert calculate_sortino_ratio(trade_pnls) > 0
    assert calculate_kelly_fraction(win_rate=0.55, avg_win=100.0, avg_loss=-50.0) > 0
    metrics = calculate_all_robust_metrics(trade_pnls)
    assert metrics["sortino_ratio"] > 0
    assert metrics["max_drawdown"] >= 0
    assert "calmar_ratio" in metrics


def test_trade_drawdown_includes_initial_equity_baseline():
    assert calculate_max_drawdown_from_trades(np.array([-100.0, -50.0])) == 150.0
    assert calculate_all_robust_metrics(np.array([-100.0]))["max_drawdown"] == 100.0


def test_sortino_uses_fixed_observation_frequency():
    returns = np.array([2.0, -1.0, 2.0, -1.0])
    repeated = np.tile(returns, 10)

    assert calculate_sortino_ratio(returns) == pytest.approx(calculate_sortino_ratio(repeated))


def test_non_finite_metrics_fail_closed():
    result = check_strategy_robustness(float("nan"), 3.0, 1.0)
    assert result["all_passed"] is False
    with pytest.raises(ValueError, match="finite"):
        calculate_kelly_fraction(float("nan"), 100.0, -50.0)
