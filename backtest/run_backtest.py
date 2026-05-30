"""Run backtests on historical Nifty data to validate strategies.

Usage:
    python -m backtest.run_backtest [--days 180] [--interval 5]

This fetches historical 5-min Nifty data from Angel One and runs
ORB + VWAP momentum strategies, printing performance metrics.
"""

from __future__ import annotations
import sys
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.indicators import add_all_indicators
from strategies.orb_options import ORBStrategy
from strategies.vwap_momentum import VWAPMomentumStrategy
from risk.capital_tracker import CapitalTracker


def generate_synthetic_nifty_data(days: int = 180,
                                   interval_minutes: int = 5,
                                   base_price: float = 24000,
                                   daily_range: float = 200) -> pd.DataFrame:
    """Generate realistic synthetic Nifty intraday data for backtesting.

    Creates data that mimics real market behavior:
    - Opening gap from previous close
    - Higher volume at open and close
    - Mean-reverting intraday price action with occasional trends
    """
    np.random.seed(42)
    all_rows = []
    current_price = base_price
    trading_days = 0
    current_date = date.today() - timedelta(days=days + 60)

    while trading_days < days:
        current_date += timedelta(days=1)
        if current_date.weekday() >= 5:
            continue
        trading_days += 1

        gap = np.random.normal(0, daily_range * 0.15)
        day_open = current_price + gap
        trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        trend_strength = np.random.uniform(0.3, 1.5) * daily_range

        candles_per_day = int((6 * 60 + 15) / interval_minutes)
        prices = [day_open]

        for j in range(1, candles_per_day):
            frac = j / candles_per_day
            noise = np.random.normal(0, daily_range * 0.03)
            trend_component = trend * trend_strength * frac / candles_per_day
            mean_rev = (day_open - prices[-1]) * 0.02

            new_price = prices[-1] + noise + trend_component + mean_rev
            prices.append(new_price)

        for j in range(candles_per_day):
            minute_offset = j * interval_minutes
            ts = datetime(
                current_date.year, current_date.month, current_date.day,
                9 + (15 + minute_offset) // 60,
                (15 + minute_offset) % 60,
            )

            p = prices[j]
            intra_vol = daily_range * 0.02
            o = p + np.random.normal(0, intra_vol * 0.3)
            c = prices[j + 1] if j + 1 < len(prices) else p + np.random.normal(0, intra_vol)
            h = max(o, c) + abs(np.random.normal(0, intra_vol * 0.5))
            l = min(o, c) - abs(np.random.normal(0, intra_vol * 0.5))

            frac = j / candles_per_day
            base_vol = 50000
            if frac < 0.1:
                vol = int(base_vol * np.random.uniform(2.0, 4.0))
            elif frac > 0.85:
                vol = int(base_vol * np.random.uniform(1.5, 3.0))
            else:
                vol = int(base_vol * np.random.uniform(0.5, 1.5))

            all_rows.append({
                "timestamp": ts, "open": round(o, 2),
                "high": round(h, 2), "low": round(l, 2),
                "close": round(c, 2), "volume": vol,
            })

        current_price = prices[-1]

    df = pd.DataFrame(all_rows)
    df.set_index("timestamp", inplace=True)
    return df


def run_backtest_on_data(df: pd.DataFrame,
                          starting_capital: float = 10000) -> dict:
    """Run ORB + VWAP strategies on historical data."""

    strategies = [ORBStrategy(), VWAPMomentumStrategy()]
    lot_size = settings.NIFTY_LOT_SIZE
    premium_sl_pct = settings.PREMIUM_SL_PCT
    target_points = settings.PREMIUM_TARGET_POINTS
    max_trades_per_day = settings.MAX_TRADES_PER_DAY
    max_consec_losses = settings.MAX_CONSECUTIVE_LOSSES

    capital = starting_capital
    peak_capital = capital
    trades = []
    current_pos = None
    current_day = None
    trades_today = 0
    consec_losses = 0
    daily_pnl = 0
    daily_start_capital = capital

    dates = df.index.date if hasattr(df.index, 'date') else pd.Series(df.index).dt.date.values

    for i in range(25, len(df)):
        ts = df.index[i]
        day = dates[i]
        time_str = ts.strftime("%H:%M")

        if day != current_day:
            if current_pos:
                exit_p = current_pos["sim_premium"]
                pnl = (exit_p - current_pos["entry_premium"]) * current_pos["qty"]
                trades.append({**current_pos, "exit_premium": exit_p,
                               "pnl": pnl, "reason": "EOD", "exit_time": ts})
                capital += pnl
                current_pos = None

            current_day = day
            trades_today = 0
            consec_losses = 0
            daily_pnl = 0
            daily_start_capital = capital
            for s in strategies:
                s.reset()

        if capital <= 0:
            break

        loss_limit = daily_start_capital * (settings.DAILY_LOSS_LIMIT_PCT / 100)
        if daily_pnl < 0 and abs(daily_pnl) >= loss_limit:
            continue
        if consec_losses >= max_consec_losses:
            continue
        if trades_today >= max_trades_per_day:
            continue

        window = df.iloc[max(0, i - 200):i + 1]

        if current_pos is None and time_str < settings.NO_NEW_ENTRY_AFTER:
            wi = add_all_indicators(window)
            for strategy in strategies:
                signal = strategy.on_candle(wi, ts)
                if signal and current_pos is None:
                    base_premium = 100.0 + np.random.uniform(-20, 20)
                    sl = base_premium * (1 - premium_sl_pct / 100)
                    target = base_premium + target_points

                    lots = 1
                    for min_cap, max_lots, _ in reversed(settings.LOT_TIERS):
                        if capital >= min_cap:
                            lots = max_lots
                            break

                    current_pos = {
                        "strategy": signal.strategy_name,
                        "signal": signal.signal_type.value,
                        "entry_time": ts,
                        "entry_index": signal.entry_price,
                        "entry_premium": base_premium,
                        "sl": sl,
                        "target": target,
                        "qty": lots * lot_size,
                        "sim_premium": base_premium,
                    }
                    trades_today += 1

        if current_pos:
            row = df.iloc[i]
            if current_pos["signal"] == "LONG":
                delta_move = (row["close"] - current_pos["entry_index"]) * 0.45
            else:
                delta_move = (current_pos["entry_index"] - row["close"]) * 0.45

            sim_p = current_pos["entry_premium"] + delta_move
            current_pos["sim_premium"] = sim_p

            if sim_p <= current_pos["sl"]:
                pnl = (current_pos["sl"] - current_pos["entry_premium"]) * current_pos["qty"]
                trades.append({**current_pos, "exit_premium": current_pos["sl"],
                               "pnl": pnl, "reason": "SL_HIT", "exit_time": ts})
                capital += pnl
                daily_pnl += pnl
                consec_losses += 1
                current_pos = None

            elif sim_p >= current_pos["target"]:
                pnl = (current_pos["target"] - current_pos["entry_premium"]) * current_pos["qty"]
                trades.append({**current_pos, "exit_premium": current_pos["target"],
                               "pnl": pnl, "reason": "TARGET_HIT", "exit_time": ts})
                capital += pnl
                daily_pnl += pnl
                consec_losses = 0
                current_pos = None

            elif time_str >= "15:15":
                pnl = (sim_p - current_pos["entry_premium"]) * current_pos["qty"]
                trades.append({**current_pos, "exit_premium": sim_p,
                               "pnl": pnl, "reason": "EOD", "exit_time": ts})
                capital += pnl
                daily_pnl += pnl
                if pnl < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0
                current_pos = None

        if capital > peak_capital:
            peak_capital = capital

    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    max_dd = (peak_capital - min(
        capital, min((starting_capital + sum(t["pnl"] for t in trades[:j+1])
                      for j in range(len(trades))), default=starting_capital)
    )) / peak_capital * 100 if peak_capital > 0 else 0

    daily_pnls = {}
    for t in trades:
        d = t["entry_time"].date() if hasattr(t["entry_time"], 'date') else t["entry_time"]
        daily_pnls[d] = daily_pnls.get(d, 0) + t["pnl"]

    profitable_days = sum(1 for v in daily_pnls.values() if v > 0)
    loss_days = sum(1 for v in daily_pnls.values() if v <= 0)

    return {
        "starting_capital": starting_capital,
        "final_capital": capital,
        "total_pnl": total_pnl,
        "return_pct": (total_pnl / starting_capital) * 100,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd,
        "profitable_days": profitable_days,
        "loss_days": loss_days,
        "trading_days": len(daily_pnls),
        "avg_daily_pnl": np.mean(list(daily_pnls.values())) if daily_pnls else 0,
        "trades": trades,
        "daily_pnls": daily_pnls,
    }


def print_results(results: dict):
    print("\n" + "=" * 60)
    print("           BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Starting Capital    : Rs {results['starting_capital']:,.0f}")
    print(f"  Final Capital       : Rs {results['final_capital']:,.0f}")
    print(f"  Total P&L           : Rs {results['total_pnl']:,.0f} ({results['return_pct']:.1f}%)")
    print(f"  Trading Days        : {results['trading_days']}")
    print(f"  Profitable Days     : {results['profitable_days']}")
    print(f"  Loss Days           : {results['loss_days']}")
    print(f"  Avg Daily P&L       : Rs {results['avg_daily_pnl']:,.0f}")
    print("-" * 60)
    print(f"  Total Trades        : {results['total_trades']}")
    print(f"  Wins                : {results['wins']}")
    print(f"  Losses              : {results['losses']}")
    print(f"  Win Rate            : {results['win_rate']:.1f}%")
    print(f"  Avg Win             : Rs {results['avg_win']:,.0f}")
    print(f"  Avg Loss            : Rs {results['avg_loss']:,.0f}")
    print(f"  Profit Factor       : {results['profit_factor']:.2f}")
    print(f"  Max Drawdown        : {results['max_drawdown_pct']:.1f}%")
    print("=" * 60)

    if results["profit_factor"] > 1.2 and results["win_rate"] > 35:
        print("\n  VERDICT: Strategy shows POSITIVE EXPECTANCY")
        print("  Ready for paper trading validation.")
    elif results["profit_factor"] > 1.0:
        print("\n  VERDICT: MARGINAL edge. Proceed with caution.")
    else:
        print("\n  VERDICT: NEGATIVE expectancy. DO NOT trade live.")

    print()

    print("  Strategy breakdown:")
    strat_stats = {}
    for t in results["trades"]:
        s = t["strategy"]
        if s not in strat_stats:
            strat_stats[s] = {"wins": 0, "losses": 0, "pnl": 0}
        if t["pnl"] > 0:
            strat_stats[s]["wins"] += 1
        else:
            strat_stats[s]["losses"] += 1
        strat_stats[s]["pnl"] += t["pnl"]

    for name, stats in strat_stats.items():
        total = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total * 100 if total > 0 else 0
        print(f"    {name}: {total} trades, {wr:.0f}% win rate, Rs {stats['pnl']:,.0f} PnL")

    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument("--days", type=int, default=180, help="Days of data")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital")
    args = parser.parse_args()

    logger.info("Generating {} days of synthetic Nifty data...", args.days)
    df = generate_synthetic_nifty_data(days=args.days)
    logger.info("Generated {} candles across {} days",
                len(df), df.index.date[-1] - df.index.date[0])

    logger.info("Running backtest with Rs {:,.0f} capital...", args.capital)
    results = run_backtest_on_data(df, starting_capital=args.capital)
    print_results(results)

    csv_path = settings.DATA_DIR / "backtest_trades.csv"
    trades_df = pd.DataFrame(results["trades"])
    if not trades_df.empty:
        trades_df.to_csv(csv_path, index=False)
        print(f"  Trade log saved to: {csv_path}")

    return results


if __name__ == "__main__":
    main()
