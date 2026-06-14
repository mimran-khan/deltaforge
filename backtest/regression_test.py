"""Regression test: verify production MultiStrategyEngine with ADX + shock."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.multi_strategy_engine import MultiStrategyEngine

THETA_REFERENCE = 24_000
THETA_BASE = 0.30


def get_scaled_theta(nifty_price: float) -> float:
    return THETA_BASE * (nifty_price / THETA_REFERENCE)


def simulate_trade(entry_price, direction, hold_bars, close_series, entry_idx,
                   theta_per_candle, delta=0.45, base_premium=100.0, sl_pct=50.0):
    entry_prem = base_premium + settings.SLIPPAGE_POINTS
    sl_prem = entry_prem * (1 - sl_pct / 100)

    for h in range(1, hold_bars + 1):
        idx = entry_idx + h
        if idx >= len(close_series):
            break
        cur_price = close_series.iloc[idx]
        move = cur_price - entry_price
        if direction == "SHORT":
            move = -move
        cur_prem = base_premium + (move * delta) - (theta_per_candle * h)
        cur_prem = max(cur_prem, 0.05)

        if cur_prem <= sl_prem:
            exit_prem = sl_prem - settings.SLIPPAGE_POINTS
            raw_pnl = (exit_prem - entry_prem) * settings.NIFTY_LOT_SIZE
            costs = settings.BROKERAGE_PER_ORDER * 2 + settings.SLIPPAGE_POINTS * settings.NIFTY_LOT_SIZE
            return raw_pnl - costs, h, "SL"

    final_idx = min(entry_idx + hold_bars, len(close_series) - 1)
    final_price = close_series.iloc[final_idx]
    move = final_price - entry_price
    if direction == "SHORT":
        move = -move
    bars_held = final_idx - entry_idx
    final_prem = base_premium + (move * delta) - (theta_per_candle * bars_held)
    final_prem = max(final_prem, 0.05)
    exit_prem = final_prem - settings.SLIPPAGE_POINTS
    raw_pnl = (exit_prem - entry_prem) * settings.NIFTY_LOT_SIZE
    costs = settings.BROKERAGE_PER_ORDER * 2 + settings.SLIPPAGE_POINTS * settings.NIFTY_LOT_SIZE
    return raw_pnl - costs, bars_held, "TIME"


def main():
    data_path = PROJECT_ROOT / "data" / "nifty_5m_real.csv"
    df = pd.read_csv(data_path)
    dt_col = "Datetime" if "Datetime" in df.columns else "datetime"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df.set_index(dt_col, inplace=True)
    df.index.name = "datetime"

    engine = MultiStrategyEngine()
    unique_days = sorted(set(df.index.date))
    trades = []

    for day in unique_days:
        day_data = df.loc[str(day)]
        if len(day_data) < 15:
            continue

        engine.reset_day()
        indicators = engine.precompute(day_data)

        for i in range(10, len(day_data)):
            time_str = day_data.index[i].strftime("%H:%M")
            signals = engine.scan(indicators, i, time_str)
            for sig in signals:
                entry_price = day_data['close'].iloc[i]
                theta = get_scaled_theta(entry_price)
                pnl, bars, reason = simulate_trade(
                    entry_price, sig.direction, 24,
                    day_data['close'], i, theta
                )
                trades.append({
                    "date": str(day),
                    "direction": sig.direction,
                    "type": sig.signal_type,
                    "pnl": pnl,
                    "exit": reason,
                })

    tdf = pd.DataFrame(trades)
    wins = len(tdf[tdf['pnl'] > 0])
    total = len(tdf)
    wr = wins / total * 100 if total > 0 else 0

    print("=" * 60)
    print("REGRESSION TEST: Production Engine (ADX + Shock)")
    print("=" * 60)
    print(f"Total trades: {total}")
    print(f"Wins: {wins}, Losses: {total - wins}")
    print(f"Win Rate: {wr:.1f}%")
    print(f"Total PnL: Rs {tdf['pnl'].sum():.0f}")
    print(f"Average PnL/trade: Rs {tdf['pnl'].mean():.0f}")
    print()

    if wr >= 80:
        print("RESULT: PASS (WR >= 80%)")
    else:
        print("RESULT: FAIL (WR < 80%)")


if __name__ == "__main__":
    logger.remove()
    main()
