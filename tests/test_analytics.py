import numpy as np

from sessionpt.analytics import (
    calculate_all_robust_metrics,
    calculate_kelly_fraction,
    calculate_max_drawdown,
    calculate_max_drawdown_duration,
    calculate_sortino_ratio,
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
