"""Backtest with true daily compounding -- lot size scales with equity every day.

Usage:
    python -m backtest.run_backtest [--days 100] [--capital 10000]
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


def generate_nifty_data(days: int = 100, interval: int = 5,
                         base: float = 24000) -> pd.DataFrame:
    """Generate realistic Nifty 5-min data with proper ORB breakout patterns."""
    np.random.seed(42)
    rows = []
    price = base
    trading_days = 0
    cur_date = date.today() - timedelta(days=days + 80)

    while trading_days < days:
        cur_date += timedelta(days=1)
        if cur_date.weekday() >= 5:
            continue
        trading_days += 1

        day_type = np.random.choice(
            ["strong_up", "strong_down", "mild_up", "mild_down", "range"],
            p=[0.15, 0.12, 0.18, 0.15, 0.40])

        gap = np.random.normal(0, 30)
        day_open = price + gap

        candles = int((6 * 60 + 15) / interval)
        p = day_open
        day_prices = []

        for j in range(candles):
            mins = j * interval
            hour = 9 + (15 + mins) // 60
            minute = (15 + mins) % 60
            if hour > 15 or (hour == 15 and minute > 30):
                break
            ts = datetime(cur_date.year, cur_date.month, cur_date.day, hour, minute)

            frac = j / candles

            if day_type == "strong_up":
                drift = np.random.uniform(1.5, 4.0)
                if j < 3:
                    drift = np.random.uniform(-2, 2)
            elif day_type == "strong_down":
                drift = np.random.uniform(-4.0, -1.5)
                if j < 3:
                    drift = np.random.uniform(-2, 2)
            elif day_type == "mild_up":
                drift = np.random.uniform(0.3, 2.0)
            elif day_type == "mild_down":
                drift = np.random.uniform(-2.0, -0.3)
            else:
                drift = np.random.normal(0, 1.0)
                drift += (day_open - p) * 0.03

            noise = np.random.normal(0, 3.0)
            p = p + drift + noise

            spread = abs(np.random.normal(0, 5))
            o = p + np.random.normal(0, 2)
            c = p + drift + np.random.normal(0, 2)
            h = max(o, c) + spread
            l = min(o, c) - spread

            if frac < 0.05 or frac > 0.9:
                vol = int(np.random.uniform(200000, 500000))
            elif 0.15 < frac < 0.25:
                vol = int(np.random.uniform(120000, 350000))
            else:
                vol = int(np.random.uniform(50000, 150000))

            rows.append({"timestamp": ts, "open": round(o, 2), "high": round(h, 2),
                         "low": round(l, 2), "close": round(c, 2), "volume": vol})
            day_prices.append(c)

        if day_prices:
            price = day_prices[-1]

    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    return df


def run_compound_backtest(df: pd.DataFrame,
                           starting_capital: float = 10000,
                           lot_size: int = 75,
                           deploy_pct: float = 80.0) -> dict:
    """Full compound backtest with ORB + VWAP + momentum signals.

    Instead of relying on complex strategy objects (which are optimized
    for real-time), this uses direct indicator-based signal logic that
    generates trades at realistic frequency (1-3 per day on active days).
    """
    from engine.indicators import ema, rsi, atr, vwap_intraday, supertrend_fast

    capital = starting_capital
    peak = capital
    trades = []
    equity_curve = []

    unique_days = sorted(set(df.index.date))

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            equity_curve.append({"date": day, "capital": capital, "daily_pnl": 0,
                                  "trades": 0, "lots": 0})
            continue

        day_start_cap = capital
        day_pnl = 0
        day_trades = 0
        consec_loss = 0
        daily_loss_limit = day_start_cap * 0.20

        # Dynamic lot sizing: deploy_pct of capital
        avg_premium = 95.0
        cost_per_lot = avg_premium * lot_size
        deployable = capital * (deploy_pct / 100)
        day_lots = max(1, int(deployable / cost_per_lot))

        # Compute indicators for the day
        close = day_df["close"]
        high = day_df["high"]
        low = day_df["low"]
        volume = day_df["volume"]

        ema9 = ema(close, 9)
        ema20 = ema(close, 20)
        rsi14 = rsi(close, 14)
        vol_sma = volume.rolling(20, min_periods=5).mean()

        cum_tp_vol = ((high + low + close) / 3 * volume).cumsum()
        cum_vol = volume.cumsum().replace(0, np.nan)
        vwap_line = cum_tp_vol / cum_vol

        # ORB range (first 15 min = first 3 candles of 5-min)
        orb_candles = day_df.iloc[:3]
        orb_high = orb_candles["high"].max()
        orb_low = orb_candles["low"].min()
        orb_range = orb_high - orb_low

        signals = []

        for i in range(4, len(day_df)):
            ts = day_df.index[i]
            time_str = ts.strftime("%H:%M")
            if time_str < "09:30" or time_str > "14:30":
                continue

            c_now = close.iloc[i]
            c_prev = close.iloc[i - 1]
            v_now = volume.iloc[i]
            v_avg = vol_sma.iloc[i] if not np.isnan(vol_sma.iloc[i]) else 100000
            e9 = ema9.iloc[i]
            e20 = ema20.iloc[i]
            e9_prev = ema9.iloc[i - 1]
            e20_prev = ema20.iloc[i - 1]
            r = rsi14.iloc[i] if not np.isnan(rsi14.iloc[i]) else 50
            vw = vwap_line.iloc[i] if not np.isnan(vwap_line.iloc[i]) else c_now

            # ── ORB Breakout ────────────────────────────────────
            if orb_range < 250 and orb_range > 5 and time_str <= "11:30":
                if c_now > orb_high and c_prev <= orb_high and c_now > vw:
                    if v_now > v_avg * 1.2:
                        signals.append(("ORB", "LONG", ts, c_now, orb_low,
                                        c_now + orb_range * 1.5))

                elif c_now < orb_low and c_prev >= orb_low and c_now < vw:
                    if v_now > v_avg * 1.2:
                        signals.append(("ORB", "SHORT", ts, c_now, orb_high,
                                        c_now - orb_range * 1.5))

            # ── EMA Crossover + VWAP ───────────────────────────
            cross_up = e9_prev <= e20_prev and e9 > e20
            cross_down = e9_prev >= e20_prev and e9 < e20

            if cross_up and c_now > vw and r > 48:
                signals.append(("VWAP_MOM", "LONG", ts, c_now,
                                c_now - 40, c_now + 55))

            elif cross_down and c_now < vw and r < 52:
                signals.append(("VWAP_MOM", "SHORT", ts, c_now,
                                c_now + 40, c_now - 55))

            # ── Momentum burst (price moves 0.2%+ in 1 candle) ─
            candle_move = (c_now - c_prev) / c_prev * 100
            if abs(candle_move) > 0.15 and v_now > v_avg * 1.5:
                if candle_move > 0 and c_now > vw:
                    signals.append(("MOMENTUM", "LONG", ts, c_now,
                                    c_now - 35, c_now + 50))
                elif candle_move < 0 and c_now < vw:
                    signals.append(("MOMENTUM", "SHORT", ts, c_now,
                                    c_now + 35, c_now - 50))

        # ── Execute signals sequentially ────────────────────────
        for sig in signals:
            strat, direction, entry_ts, entry_idx, sl_idx, tgt_idx = sig

            if day_trades >= 5:
                break
            if consec_loss >= 2:
                break
            if day_pnl < 0 and abs(day_pnl) >= daily_loss_limit:
                break

            # Simulate premium
            base_prem = 85 + np.random.uniform(-10, 30)
            sl_prem = base_prem * 0.60     # 40% SL
            tgt_prem = base_prem + 25      # Rs 25 target
            qty = day_lots * lot_size

            # Walk forward from entry to see if SL or target hits first
            entry_i = day_df.index.get_loc(entry_ts)
            exit_prem = base_prem
            exit_reason = "EOD"
            exit_time = day_df.index[-1]

            for k in range(entry_i + 1, len(day_df)):
                future_close = close.iloc[k]
                if direction == "LONG":
                    idx_move = future_close - entry_idx
                else:
                    idx_move = entry_idx - future_close

                delta = 0.48 + np.random.uniform(-0.03, 0.03)
                sim_prem = base_prem + idx_move * delta

                if sim_prem <= sl_prem:
                    exit_prem = sl_prem
                    exit_reason = "SL_HIT"
                    exit_time = day_df.index[k]
                    break
                elif sim_prem >= tgt_prem:
                    exit_prem = tgt_prem
                    exit_reason = "TARGET_HIT"
                    exit_time = day_df.index[k]
                    break

                if day_df.index[k].strftime("%H:%M") >= "15:15":
                    exit_prem = sim_prem
                    exit_reason = "EOD"
                    exit_time = day_df.index[k]
                    break

            pnl = (exit_prem - base_prem) * qty
            capital += pnl
            day_pnl += pnl
            day_trades += 1

            if pnl < 0:
                consec_loss += 1
            else:
                consec_loss = 0

            trades.append({
                "strategy": strat, "signal": direction,
                "entry_time": entry_ts, "exit_time": exit_time,
                "entry_premium": round(base_prem, 2),
                "exit_premium": round(exit_prem, 2),
                "qty": qty, "lots": day_lots,
                "pnl": round(pnl, 0), "reason": exit_reason,
                "capital_after": round(capital, 0),
            })

            if capital <= 0:
                break

        if capital > peak:
            peak = capital

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades, "lots": day_lots,
        })

        if capital <= 0:
            break

    # ── Stats ───────────────────────────────────────────────────
    total_pnl = capital - starting_capital
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    eq_vals = [e["capital"] for e in equity_curve]
    peak_arr = np.maximum.accumulate(eq_vals) if eq_vals else [starting_capital]
    dd_arr = [(p - v) / p * 100 for p, v in zip(peak_arr, eq_vals)]
    max_dd = max(dd_arr) if dd_arr else 0

    active_days = [e for e in equity_curve if e["trades"] > 0]
    daily_rets = [(e["daily_pnl"] / max(e["capital"] - e["daily_pnl"], 1)) * 100
                  for e in active_days]
    avg_daily_ret = np.mean(daily_rets) if daily_rets else 0

    profitable_days = sum(1 for e in equity_curve if e["daily_pnl"] > 0)
    loss_days = sum(1 for e in equity_curve if e["daily_pnl"] < 0)
    flat_days = sum(1 for e in equity_curve if e["daily_pnl"] == 0)

    return {
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / starting_capital * 100, 1),
        "total_trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 0), "avg_loss": round(avg_loss, 0),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "profitable_days": profitable_days,
        "loss_days": loss_days, "flat_days": flat_days,
        "trading_days": len(equity_curve),
        "active_trading_days": len(active_days),
        "avg_daily_return_pct": round(avg_daily_ret, 1),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def print_results(results: dict):
    ec = results["equity_curve"]

    print("\n" + "=" * 65)
    print("     BACKTEST: DAILY COMPOUNDING RESULTS")
    print("=" * 65)
    print(f"  Starting Capital      : Rs {results['starting_capital']:>12,.0f}")
    print(f"  Final Capital         : Rs {results['final_capital']:>12,.0f}")
    pnl = results['total_pnl']
    s = "+" if pnl >= 0 else ""
    print(f"  Total P&L             : Rs {s}{pnl:>11,.0f} ({results['return_pct']}%)")
    print(f"  Avg Daily Return      : {results['avg_daily_return_pct']}% (on active days)")
    print("-" * 65)
    print(f"  Calendar Days         : {results['trading_days']}")
    print(f"  Active Trading Days   : {results['active_trading_days']}")
    print(f"  Profitable Days       : {results['profitable_days']}")
    print(f"  Loss Days             : {results['loss_days']}")
    print(f"  No-Trade Days         : {results['flat_days']}")
    print("-" * 65)
    print(f"  Total Trades          : {results['total_trades']}")
    print(f"  Wins                  : {results['wins']}")
    print(f"  Losses                : {results['losses']}")
    print(f"  Win Rate              : {results['win_rate']}%")
    print(f"  Avg Win               : Rs {results['avg_win']:>10,.0f}")
    print(f"  Avg Loss              : Rs {results['avg_loss']:>10,.0f}")
    print(f"  Profit Factor         : {results['profit_factor']}")
    print(f"  Max Drawdown          : {results['max_drawdown_pct']}%")
    print("=" * 65)

    # Strategy breakdown
    strat_stats = {}
    for t in results["trades"]:
        s = t["strategy"]
        if s not in strat_stats:
            strat_stats[s] = {"w": 0, "l": 0, "pnl": 0, "n": 0}
        strat_stats[s]["n"] += 1
        if t["pnl"] > 0:
            strat_stats[s]["w"] += 1
        else:
            strat_stats[s]["l"] += 1
        strat_stats[s]["pnl"] += t["pnl"]

    print("\n  Strategy Breakdown:")
    for name, st in sorted(strat_stats.items()):
        wr = st["w"] / st["n"] * 100 if st["n"] > 0 else 0
        print(f"    {name:12s}: {st['n']:>3d} trades | "
              f"{wr:>4.0f}% win | Rs {st['pnl']:>10,.0f}")

    # Capital journey
    if ec:
        print("\n  Capital Growth (Compound Journey):")
        print(f"    {'Day':>5s}  {'Date':>12s}  {'Capital':>12s}  "
              f"{'Day PnL':>10s}  {'Lots':>5s}  {'Trades':>6s}")
        print("    " + "-" * 58)

        show = set([0, len(ec) - 1])
        for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            show.add(min(int(len(ec) * pct / 100), len(ec) - 1))
        for i, e in enumerate(ec):
            if e["daily_pnl"] != 0 and len(show) < 25:
                show.add(i)

        for i in sorted(show):
            e = ec[i]
            dpnl = e["daily_pnl"]
            s = "+" if dpnl >= 0 else ""
            print(f"    {i+1:>5d}  {e['date']}  Rs {e['capital']:>10,.0f}  "
                  f"Rs {s}{dpnl:>8,.0f}  {e['lots']:>5d}  {e['trades']:>6d}")

    # Theoretical 10% daily compound comparison
    print("\n  10% Daily Compound Target (theoretical):")
    for d in [10, 20, 30, 50, 75, 100]:
        if d <= results["trading_days"]:
            theoretical = results["starting_capital"] * (1.10 ** d)
            print(f"    Day {d:>3d}: Rs {theoretical:>12,.0f}")

    print()
    if results["profit_factor"] > 1.5 and results["win_rate"] > 45:
        print("  VERDICT: STRONG edge -- ready for paper trading")
    elif results["profit_factor"] > 1.2:
        print("  VERDICT: Moderate edge -- paper trade to confirm")
    elif results["profit_factor"] > 1.0:
        print("  VERDICT: Marginal -- needs tuning")
    else:
        print("  VERDICT: Negative expectancy -- do NOT trade live")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--deploy-pct", type=float, default=80)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    print(f"\nGenerating {args.days} trading days of synthetic Nifty data...")
    df = generate_nifty_data(days=args.days)
    print(f"Generated {len(df)} candles across {len(set(df.index.date))} days")

    print(f"Running compound backtest: Rs {args.capital:,.0f}, "
          f"{args.deploy_pct:.0f}% deployed...")

    results = run_compound_backtest(
        df, starting_capital=args.capital,
        lot_size=settings.NIFTY_LOT_SIZE,
        deploy_pct=args.deploy_pct,
    )
    print_results(results)

    pd.DataFrame(results["trades"]).to_csv(
        settings.DATA_DIR / "backtest_trades.csv", index=False)
    pd.DataFrame(results["equity_curve"]).to_csv(
        settings.DATA_DIR / "equity_curve.csv", index=False)
    print(f"  Saved: data/backtest_trades.csv, data/equity_curve.csv")

    return results


if __name__ == "__main__":
    main()
