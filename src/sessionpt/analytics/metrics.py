"""Reusable trade and equity-curve analytics."""

from __future__ import annotations

import numpy as np

SORTINO_RATIO_KEY = "sortino_ratio"
MAX_DRAWDOWN_KEY = "max_drawdown"
CALMAR_RATIO_KEY = "calmar_ratio"
DEFAULT_ANNUALIZATION_FACTOR = 252.0
DEFAULT_TARGET_RETURN = 0.0
DEFAULT_YEARS = 1.0
MIN_OBSERVATIONS = 2


def calculate_sortino_ratio(
    returns: np.ndarray,
    target_return: float = DEFAULT_TARGET_RETURN,
    annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    returns = np.asarray(returns, dtype=float)
    if len(returns) < MIN_OBSERVATIONS:
        return 0.0
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    if not np.isfinite(returns).all() or not np.isfinite(target_return):
        raise ValueError("returns and target_return must be finite")

    excess_returns = returns - target_return
    downside_returns = np.minimum(excess_returns, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation == 0:
        return 0.0 if np.mean(excess_returns) <= 0 else float("inf")

    return float((np.mean(excess_returns) / downside_deviation) * np.sqrt(annualization_factor))


def calculate_max_drawdown(equity_curve: np.ndarray) -> tuple[float, int, int]:
    if len(equity_curve) < MIN_OBSERVATIONS:
        return 0.0, 0, 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    max_drawdown_index = int(np.argmax(drawdowns))
    peak_index = int(np.argmax(equity_curve[: max_drawdown_index + 1]))
    return float(drawdowns[max_drawdown_index]), peak_index, max_drawdown_index


def calculate_max_drawdown_from_trades(trade_pnls: np.ndarray) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    equity_curve = np.concatenate(([0.0], np.cumsum(np.asarray(trade_pnls, dtype=float))))
    max_drawdown, _, _ = calculate_max_drawdown(equity_curve)
    return max_drawdown


def profit_factor_threshold_check(
    profit_factor: float,
    min_threshold: float = 1.5,
) -> bool:
    if np.isnan(profit_factor):
        return False
    if np.isposinf(profit_factor):
        return True
    return profit_factor >= min_threshold


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    years: float = DEFAULT_YEARS,
) -> float:
    if not np.isfinite(total_return) or not np.isfinite(max_drawdown):
        raise ValueError("total_return and max_drawdown must be finite")
    if years <= 0:
        raise ValueError("years must be positive")
    if max_drawdown <= 0:
        return 0.0 if total_return <= 0 else float("inf")
    annualized_return = total_return / years
    return float(annualized_return / max_drawdown)


def calculate_all_robust_metrics(
    trade_pnls: np.ndarray,
    target_return: float = DEFAULT_TARGET_RETURN,
    years: float = DEFAULT_YEARS,
) -> dict[str, float]:
    if len(trade_pnls) == 0:
        return {
            SORTINO_RATIO_KEY: 0.0,
            MAX_DRAWDOWN_KEY: 0.0,
            CALMAR_RATIO_KEY: 0.0,
        }

    trade_pnls = np.asarray(trade_pnls, dtype=float)
    if not np.isfinite(trade_pnls).all():
        raise ValueError("trade_pnls must be finite")
    equity_curve = np.concatenate(([0.0], np.cumsum(trade_pnls)))
    max_drawdown, _, _ = calculate_max_drawdown(equity_curve)
    return {
        SORTINO_RATIO_KEY: calculate_sortino_ratio(trade_pnls, target_return),
        MAX_DRAWDOWN_KEY: max_drawdown,
        CALMAR_RATIO_KEY: calculate_calmar_ratio(float(equity_curve[-1]), max_drawdown, years),
    }


def check_strategy_robustness(
    profit_factor: float,
    sortino_ratio: float,
    sharpe_ratio: float,
    pf_threshold: float = 1.5,
    sortino_threshold: float = 2.0,
    sharpe_threshold: float = 0.5,
) -> dict[str, bool]:
    if not np.isfinite([profit_factor, sortino_ratio, sharpe_ratio]).all() and not np.isposinf(
        profit_factor
    ):
        return {
            "profit_factor_ok": False,
            "sortino_ok": False,
            "sharpe_ok": False,
            "all_passed": False,
        }
    profit_factor_ok = profit_factor_threshold_check(profit_factor, pf_threshold)
    sortino_ok = sortino_ratio >= sortino_threshold
    sharpe_ok = sharpe_ratio >= sharpe_threshold
    return {
        "profit_factor_ok": profit_factor_ok,
        "sortino_ok": sortino_ok,
        "sharpe_ok": sharpe_ok,
        "all_passed": profit_factor_ok and sortino_ok and sharpe_ok,
    }


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    if not np.isfinite([win_rate, avg_win, avg_loss]).all():
        raise ValueError("Kelly inputs must be finite")
    if win_rate <= 0 or win_rate >= 1 or avg_win <= 0:
        return 0.0

    avg_loss_abs = abs(avg_loss) if avg_loss != 0 else 1e-10
    if avg_loss_abs <= 0:
        return 0.0

    win_loss_ratio = avg_win / avg_loss_abs
    loss_rate = 1 - win_rate
    kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
    return max(0.0, min(1.0, float(kelly)))


def calculate_max_drawdown_duration(equity_curve: np.ndarray) -> int:
    if len(equity_curve) < MIN_OBSERVATIONS:
        return 0

    equity = np.asarray(equity_curve)
    running_max = np.maximum.accumulate(equity)
    max_duration = 0
    current_duration = 0
    in_drawdown = False

    for index in range(len(equity)):
        if equity[index] < running_max[index]:
            if not in_drawdown:
                in_drawdown = True
                current_duration = 1
            else:
                current_duration += 1
        elif in_drawdown:
            max_duration = max(max_duration, current_duration)
            in_drawdown = False
            current_duration = 0

    if in_drawdown:
        max_duration = max(max_duration, current_duration)
    return int(max_duration)
