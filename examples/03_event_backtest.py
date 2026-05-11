"""Generate synthetic data, create entry events, and run an event-driven backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    pnl: float


def generate_bars(days: int = 20, base_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 390
    timestamps = pd.bdate_range("2024-01-02", periods=n, freq="1min")
    returns = rng.normal(0, 0.0005, size=n)
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.001, size=n))
    low = close * (1 - rng.uniform(0, 0.001, size=n))
    open_ = np.empty_like(close)
    open_[0] = base_price
    open_[1:] = close[:-1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=timestamps)


def generate_events(bars: pd.DataFrame, stride: int = 390) -> pd.DataFrame:
    entries = bars.iloc[::stride].copy()
    entries["direction"] = np.where(entries["close"] > entries["open"], 1, -1)
    return entries[["close", "direction"]].rename(columns={"close": "price"})


def run_backtest(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    sl_ticks: float = 0.5,
    tp_ticks: float = 1.0,
    tick_size: float = 0.01,
) -> list[Trade]:
    trades: list[Trade] = []
    for entry_time, row in events.iterrows():
        direction = int(row["direction"])
        entry_price = float(row["price"])
        sl_price = entry_price - direction * sl_ticks * tick_size * 100
        tp_price = entry_price + direction * tp_ticks * tick_size * 100
        future = bars.loc[bars.index > entry_time].head(390)
        if future.empty:
            continue
        exit_price = float(future["close"].iloc[-1])
        exit_time = future.index[-1]
        for ts, bar in future.iterrows():
            if direction == 1:
                if bar["low"] <= sl_price:
                    exit_price, exit_time = sl_price, ts
                    break
                if bar["high"] >= tp_price:
                    exit_price, exit_time = tp_price, ts
                    break
            else:
                if bar["high"] >= sl_price:
                    exit_price, exit_time = sl_price, ts
                    break
                if bar["low"] <= tp_price:
                    exit_price, exit_time = tp_price, ts
                    break
        pnl = direction * (exit_price - entry_price)
        trades.append(Trade(entry_time, exit_time, direction, entry_price, exit_price, pnl))
    return trades


def main() -> None:
    bars = generate_bars()
    events = generate_events(bars)
    trades = run_backtest(bars, events)

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    total = float(pnls.sum())
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    pf = (
        abs(float(wins.sum()) / float(losses.sum()))
        if len(losses) and losses.sum() != 0
        else float("inf")
    )

    print("Event-Driven Backtest Summary")
    print("=" * 50)
    print(f"Total trades:    {len(trades)}")
    print(f"Win rate:        {win_rate:.1f}%")
    print(f"Total PnL:       {total:+.2f}")
    print(f"Avg win:         {avg_win:+.4f}")
    print(f"Avg loss:        {avg_loss:+.4f}")
    print(f"Profit factor:   {pf:.2f}")


if __name__ == "__main__":
    main()
