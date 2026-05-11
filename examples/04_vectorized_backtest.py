"""Generate synthetic data, compute pivots, run vectorized SL/TP grid search."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_daily_ohlc(days: int = 252, base_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=days)
    returns = rng.normal(0, 0.01, size=days)
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.005, size=days))
    low = close * (1 - rng.uniform(0, 0.005, size=days))
    open_ = np.empty_like(close)
    open_[0] = base_price
    open_[1:] = close[:-1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)


def compute_pivots(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["P"] = (df["high"] + df["low"] + df["close"]) / 3
    df["R1"] = 2 * df["P"] - df["low"]
    df["S1"] = 2 * df["P"] - df["high"]
    return df


def vectorized_backtest(
    ohlc: pd.DataFrame,
    direction: int = 1,
    atr_mult_range: tuple[float, float, float] = (0.5, 3.0, 0.5),
) -> pd.DataFrame:
    ohlc = ohlc.copy()
    ohlc["atr"] = ohlc["high"] - ohlc["low"]
    ohlc["prev_P"] = ohlc["P"].shift(1)
    ohlc["prev_atr"] = ohlc["atr"].shift(1)
    ohlc = ohlc.dropna()

    mults = np.arange(atr_mult_range[0], atr_mult_range[1] + atr_mult_range[2], atr_mult_range[2])
    results: list[dict[str, float]] = []

    for sl_m in mults:
        for tp_m in mults:
            entry = ohlc["prev_P"]
            dist = ohlc["prev_atr"]
            move = (ohlc["close"] - entry) * direction
            hit_tp = (ohlc["high"] - entry) * direction >= tp_m * dist
            hit_sl = (entry - ohlc["low"]) * direction >= sl_m * dist
            outcome = np.where(hit_tp, tp_m * dist, np.where(hit_sl, -sl_m * dist, move))
            pnl = outcome * direction
            results.append(
                {
                    "sl_mult": sl_m,
                    "tp_mult": tp_m,
                    "total_pnl": float(pnl.sum()),
                    "win_rate": float((pnl > 0).mean()) * 100,
                    "trades": len(pnl),
                }
            )

    return pd.DataFrame(results)


def main() -> None:
    ohlc = generate_daily_ohlc()
    pivots = compute_pivots(ohlc)
    grid = vectorized_backtest(pivots, direction=1)
    best = grid.loc[grid["total_pnl"].idxmax()]

    print("Vectorized SL/TP Grid Search — Best Result")
    print("=" * 55)
    print(f"SL multiplier:   {best['sl_mult']:.1f}")
    print(f"TP multiplier:   {best['tp_mult']:.1f}")
    print(f"Total PnL:       {best['total_pnl']:+.4f}")
    print(f"Win rate:        {best['win_rate']:.1f}%")
    print(f"Trades:          {int(best['trades'])}")
    print(f"\nGrid combinations tested: {len(grid)}")


if __name__ == "__main__":
    main()
