"""6-Month Realistic Backtest with Withdrawals + Trend Analysis.

Runs a 6-month (130 trading day) backtest with:
  - Realistic cost model (spread + STT + exchange + market impact)
  - Withdrawal rule: withdraw Rs 30K every time capital crosses Rs 1L milestones
  - Weekly and monthly trend analysis
  - Lot size 65 (Jan 2026 NSE standard)
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, run_compound_backtest, _calc_realistic_costs
from engine.multi_strategy_engine import MultiStrategyEngine
from engine.premium_model import create_premium_state


def run_with_withdrawals(df: pd.DataFrame, starting_capital: float = 10000,
                          lot_size: int = 65, withdraw_amount: float = 30000,
                          withdraw_every: float = 100000) -> dict:
    """Compound backtest with periodic withdrawals.

    Every time capital crosses a Rs 1L milestone, withdraw Rs 30K.
    Tracks total withdrawn separately from trading capital.
    """
    engine = MultiStrategyEngine()
    capital = starting_capital
    peak = capital
    total_withdrawn = 0.0
    next_withdraw_at = withdraw_every
    trades = []
    equity_curve = []
    withdrawals = []

    unique_days = sorted(set(df.index.date))

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            equity_curve.append({
                "date": day, "capital": capital, "daily_pnl": 0,
                "trades": 0, "lots": 0, "withdrawn": total_withdrawn,
                "net_worth": capital + total_withdrawn,
            })
            continue

        engine.reset_day()
        day_start_cap = capital
        day_pnl = 0.0
        day_trades = 0
        consec_loss = 0
        daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)

        per_lot = getattr(settings, 'CAPITAL_PER_LOT', 6000)
        day_lots = max(1, int(capital / per_lot))
        day_lots = min(day_lots, getattr(settings, 'MAX_LOTS_CAP', 20))

        indicators = engine.precompute(day_df)
        open_positions = []

        for i in range(10, len(day_df)):
            ts = day_df.index[i]
            time_str = ts.strftime("%H:%M")
            nifty_price = day_df["close"].iloc[i]

            closed_this_bar = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(
                    nifty_price, pos["candles_held"])
                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                trail_floor = pos["prem_state"].update_trail(
                    cur_prem, trigger_pct=12, trail_pct=8)

                exit_reason = None
                exit_prem = cur_prem

                if cur_prem <= pos["sl_premium"]:
                    exit_reason = "SL"
                    exit_prem = pos["sl_premium"]
                elif cur_prem >= pos["prem_state"].target_premium:
                    exit_reason = "TGT"
                    exit_prem = pos["prem_state"].target_premium
                elif trail_floor is not None and cur_prem <= trail_floor:
                    exit_reason = "TRAIL"
                    exit_prem = trail_floor
                elif pos["candles_held"] >= settings.PULLBACK_HOLD_CANDLES:
                    exit_reason = "TIME"
                elif time_str >= settings.SQUARE_OFF_TIME:
                    exit_reason = "EOD"

                if exit_reason:
                    costs = _calc_realistic_costs(
                        pos["entry_premium"], exit_prem, pos["qty"], day_lots)
                    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                    net_pnl = raw_pnl - costs

                    capital += net_pnl
                    day_pnl += net_pnl
                    day_trades += 1
                    consec_loss = consec_loss + 1 if net_pnl < 0 else 0

                    trades.append({
                        "strategy": pos["signal_type"], "signal": pos["direction"],
                        "entry_time": pos["entry_time"], "exit_time": ts,
                        "entry_premium": round(pos["entry_premium"], 2),
                        "exit_premium": round(exit_prem, 2),
                        "qty": pos["qty"], "lots": day_lots,
                        "pnl": round(net_pnl, 0), "reason": exit_reason,
                        "capital_after": round(capital, 0),
                    })
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

            max_sim = 2
            if len(open_positions) >= max_sim:
                continue
            if consec_loss >= settings.MAX_CONSECUTIVE_LOSSES:
                continue
            if day_pnl < 0 and abs(day_pnl) >= daily_loss_limit:
                continue

            open_dirs = {p["direction"] for p in open_positions}
            signals = engine.scan(indicators, i, time_str)

            for signal in signals:
                if signal.direction in open_dirs:
                    continue
                if signal.confidence < getattr(settings, 'PULLBACK_MIN_CONFIDENCE', 50):
                    continue

                theta = settings.get_scaled_theta(nifty_price)
                prem_state = create_premium_state(
                    entry_index_price=nifty_price,
                    direction=signal.direction,
                    base_premium=settings.PREMIUM_BASE,
                    delta=settings.PREMIUM_DELTA,
                    theta_per_candle=theta,
                    sl_pct=settings.PREMIUM_SL_PCT,
                    confluence_score=signal.confidence,
                    signal_type=signal.signal_type,
                )
                spread = getattr(settings, 'BID_ASK_SPREAD', 0.30)
                entry_premium = prem_state.entry_premium + spread
                from engine.premium_model import STRATEGY_SL_PCT
                eff_sl = STRATEGY_SL_PCT.get(signal.signal_type, settings.PREMIUM_SL_PCT)
                sl_premium = entry_premium * (1 - eff_sl / 100)
                qty = day_lots * lot_size

                open_positions.append({
                    "direction": signal.direction,
                    "signal_type": signal.signal_type,
                    "entry_time": ts,
                    "entry_premium": entry_premium,
                    "sl_premium": sl_premium,
                    "qty": qty,
                    "prem_state": prem_state,
                    "candles_held": 0,
                    "peak_premium": entry_premium,
                    "signal": signal,
                })

        if capital > peak:
            peak = capital

        # Withdrawal check: every Rs 1L milestone, withdraw Rs 30K
        while capital >= next_withdraw_at:
            capital -= withdraw_amount
            total_withdrawn += withdraw_amount
            withdrawals.append({
                "date": day, "amount": withdraw_amount,
                "capital_before": capital + withdraw_amount,
                "capital_after": capital,
                "total_withdrawn": total_withdrawn,
            })
            next_withdraw_at += withdraw_every

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades, "lots": day_lots,
            "withdrawn": total_withdrawn,
            "net_worth": round(capital + total_withdrawn, 0),
        })

    return {
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_withdrawn": total_withdrawn,
        "net_worth": round(capital + total_withdrawn, 0),
        "trades": trades,
        "equity_curve": equity_curve,
        "withdrawals": withdrawals,
        "trading_days": len(unique_days),
        "total_trades": len(trades),
    }


def print_trend_analysis(results: dict):
    """Print weekly and monthly trend analysis."""
    ec = pd.DataFrame(results["equity_curve"])
    ec["date"] = pd.to_datetime(ec["date"])
    trades_df = pd.DataFrame(results["trades"]) if results["trades"] else pd.DataFrame()

    print(f"\n{'=' * 75}")
    print(f"  6-MONTH BACKTEST — REALISTIC COSTS + Rs 30K WITHDRAWAL PER Rs 1L")
    print(f"{'=' * 75}")
    print(f"  Starting Capital   : Rs {results['starting_capital']:>12,.0f}")
    print(f"  Final Capital      : Rs {results['final_capital']:>12,.0f}")
    print(f"  Total Withdrawn    : Rs {results['total_withdrawn']:>12,.0f}")
    print(f"  NET WORTH (cap+wd) : Rs {results['net_worth']:>12,.0f}")
    print(f"  Total P&L          : Rs {results['net_worth'] - results['starting_capital']:>+12,.0f}")
    print(f"  Total Trades       : {results['total_trades']}")
    print(f"  Trading Days       : {results['trading_days']}")

    if results["withdrawals"]:
        print(f"\n{'─' * 75}")
        print(f"  WITHDRAWALS")
        print(f"{'─' * 75}")
        for w in results["withdrawals"]:
            print(f"  {w['date']}  Withdrew Rs {w['amount']:,.0f}  "
                  f"(capital {w['capital_before']:,.0f} → {w['capital_after']:,.0f})  "
                  f"Total out: Rs {w['total_withdrawn']:,.0f}")

    # Monthly breakdown
    active = ec[ec["trades"] > 0].copy()
    if not active.empty:
        active["month"] = active["date"].dt.to_period("M")
        print(f"\n{'─' * 75}")
        print(f"  MONTHLY TREND")
        print(f"{'─' * 75}")
        print(f"  {'Month':<10s} {'Days':>5s} {'Trades':>7s} {'P&L':>12s} "
              f"{'Avg/Day':>10s} {'Win Days':>9s} {'Capital':>12s} {'Net Worth':>12s}")
        print(f"  {'─' * 73}")

        for month, grp in active.groupby("month"):
            m_pnl = grp["daily_pnl"].sum()
            m_trades = grp["trades"].sum()
            m_days = len(grp)
            win_days = (grp["daily_pnl"] > 0).sum()
            last_cap = grp["capital"].iloc[-1]
            last_nw = grp["net_worth"].iloc[-1]
            avg_day = m_pnl / m_days if m_days > 0 else 0
            print(f"  {str(month):<10s} {m_days:>5d} {m_trades:>7d} "
                  f"Rs {m_pnl:>+10,.0f} Rs {avg_day:>+8,.0f} "
                  f"{win_days:>4d}/{m_days:<4d} Rs {last_cap:>10,.0f} Rs {last_nw:>10,.0f}")

    # Weekly breakdown
    if not active.empty:
        active["week"] = active["date"].dt.isocalendar().week.astype(int)
        active["year"] = active["date"].dt.year
        active["yw"] = active["year"].astype(str) + "-W" + active["week"].astype(str).str.zfill(2)

        print(f"\n{'─' * 75}")
        print(f"  WEEKLY TREND")
        print(f"{'─' * 75}")
        print(f"  {'Week':<10s} {'Days':>5s} {'Trades':>7s} {'P&L':>12s} "
              f"{'Win Days':>9s} {'Result':>8s}")
        print(f"  {'─' * 55}")

        for yw, grp in active.groupby("yw"):
            w_pnl = grp["daily_pnl"].sum()
            w_trades = grp["trades"].sum()
            w_days = len(grp)
            win_days = (grp["daily_pnl"] > 0).sum()
            result = "GREEN" if w_pnl > 0 else "RED" if w_pnl < 0 else "FLAT"
            print(f"  {yw:<10s} {w_days:>5d} {w_trades:>7d} "
                  f"Rs {w_pnl:>+10,.0f} {win_days:>4d}/{w_days:<4d} {result:>8s}")

        green_weeks = sum(1 for _, g in active.groupby("yw") if g["daily_pnl"].sum() > 0)
        total_weeks = active.groupby("yw").ngroups
        print(f"\n  Green weeks: {green_weeks}/{total_weeks} "
              f"({green_weeks/total_weeks*100:.0f}%)")

    # Strategy breakdown
    if not trades_df.empty:
        print(f"\n{'─' * 75}")
        print(f"  STRATEGY PERFORMANCE")
        print(f"{'─' * 75}")
        for strat, grp in trades_df.groupby("strategy"):
            wins = (grp["pnl"] > 0).sum()
            total = len(grp)
            wr = wins / total * 100 if total > 0 else 0
            total_pnl = grp["pnl"].sum()
            avg_w = grp[grp["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
            avg_l = grp[grp["pnl"] <= 0]["pnl"].mean() if (total - wins) > 0 else 0
            print(f"  {strat:<14s} {total:>3d} trades | {wr:>5.1f}% WR | "
                  f"Rs {total_pnl:>+10,.0f} | "
                  f"Avg W: Rs {avg_w:>+8,.0f} | Avg L: Rs {avg_l:>+8,.0f}")

    # Risk metrics
    if not active.empty:
        daily_returns = active["daily_pnl"] / active["capital"].shift(1).fillna(
            results["starting_capital"])
        daily_returns = daily_returns.replace([np.inf, -np.inf], 0).dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0

        max_dd_pct = 0
        running_peak = results["starting_capital"]
        for _, row in ec.iterrows():
            nw = row["net_worth"]
            if nw > running_peak:
                running_peak = nw
            dd = (running_peak - nw) / running_peak * 100 if running_peak > 0 else 0
            if dd > max_dd_pct:
                max_dd_pct = dd

        wins = trades_df[trades_df["pnl"] > 0]["pnl"] if not trades_df.empty else pd.Series()
        losses = trades_df[trades_df["pnl"] <= 0]["pnl"] if not trades_df.empty else pd.Series()

        print(f"\n{'─' * 75}")
        print(f"  RISK METRICS")
        print(f"{'─' * 75}")
        print(f"  Annualized Sharpe   : {sharpe:.2f}")
        print(f"  Max Drawdown (NW)   : {max_dd_pct:.1f}%")
        print(f"  Win Rate            : {len(wins)}/{len(trades_df)} "
              f"({len(wins)/len(trades_df)*100:.1f}%)" if not trades_df.empty else "  N/A")
        print(f"  Avg Win / Avg Loss  : Rs {wins.mean():,.0f} / Rs {losses.mean():,.0f}"
              if len(wins) > 0 and len(losses) > 0 else "  N/A")
        pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0
        print(f"  Profit Factor       : {pf:.2f}")
        print(f"  Best Trade          : Rs {trades_df['pnl'].max():+,.0f}"
              if not trades_df.empty else "")
        print(f"  Worst Trade         : Rs {trades_df['pnl'].min():+,.0f}"
              if not trades_df.empty else "")

    # Equity curve milestones
    print(f"\n{'─' * 75}")
    print(f"  NET WORTH JOURNEY (capital + withdrawals)")
    print(f"{'─' * 75}")
    milestones = [10000, 25000, 50000, 100000, 200000, 300000, 500000, 750000, 1000000]
    hit = set()
    for _, row in ec.iterrows():
        nw = row["net_worth"]
        for m in milestones:
            if nw >= m and m not in hit:
                hit.add(m)
                print(f"  Rs {m:>10,.0f}  reached on  {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}")

    print(f"\n{'=' * 75}\n")


if __name__ == "__main__":
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    days = 130
    print(f"\nLoading {days} trading days of Nifty data (~6 months)...")
    df = load_real_data(days=days)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days")
    print(f"Date range: {unique_days[0]} → {unique_days[-1]}")

    results = run_with_withdrawals(
        df, starting_capital=10000,
        lot_size=settings.NIFTY_LOT_SIZE,
        withdraw_amount=30000,
        withdraw_every=100000,
    )
    print_trend_analysis(results)
