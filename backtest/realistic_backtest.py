"""Realistic backtest -- simulates adaptive mode to match live trading behavior.

Runs the same compound backtest but applies:
  1. Adaptive mode state machine (NORMAL/DEFENSIVE/AGGRESSIVE transitions)
  2. Per-mode SL multiplier, trail params, min_confidence, lot scaling
  3. Compares realistic vs baseline performance

Usage:
    python -m backtest.realistic_backtest [--days 100] [--capital 10000]
"""

from __future__ import annotations
import sys
import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import (
    load_real_data, _calc_realistic_costs, print_results,
)


@dataclass
class AdaptiveState:
    mode: str = "NORMAL"
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    day_trades: int = 0
    day_wins: int = 0
    day_pnl: float = 0.0

    def reset_day(self):
        self.mode = "NORMAL"
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.day_trades = 0
        self.day_wins = 0
        self.day_pnl = 0.0

    def on_trade(self, pnl: float, day_start_cap: float):
        self.day_trades += 1
        self.day_pnl += pnl
        won = pnl > 0

        if won:
            self.day_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        pnl_pct = self.day_pnl / day_start_cap * 100
        wr = self.day_wins / self.day_trades * 100 if self.day_trades else 0

        prev = self.mode

        if self.consecutive_losses >= 3:
            self.mode = "HALT"
        elif pnl_pct <= -5:
            self.mode = "HALT"
        elif pnl_pct <= -3:
            self.mode = "DEFENSIVE"
        elif self.consecutive_losses >= 2:
            self.mode = "DEFENSIVE"
        elif self.day_trades >= 3 and wr < 40:
            self.mode = "DEFENSIVE"
        elif (self.mode in ("NORMAL", "AGGRESSIVE")
              and pnl_pct >= 5 and self.consecutive_wins >= 2 and wr >= 60):
            self.mode = "AGGRESSIVE"
        elif prev == "DEFENSIVE" and won:
            self.mode = "NORMAL"
        elif prev == "AGGRESSIVE" and not won:
            self.mode = "NORMAL"

    @property
    def profile(self):
        profiles = {
            "AGGRESSIVE": {
                "min_confidence": 60, "max_trades": 10, "max_sim": 2,
                "lot_mult": 1.0, "sl_mult": 1.0, "target_mult": 1.3,
                "trail_trigger": 12.0, "trail_pct": 6.0,
            },
            "NORMAL": {
                "min_confidence": 65, "max_trades": 8, "max_sim": 2,
                "lot_mult": 1.0, "sl_mult": 1.0, "target_mult": 1.0,
                "trail_trigger": 10.0, "trail_pct": 5.0,
            },
            "DEFENSIVE": {
                "min_confidence": 75, "max_trades": 4, "max_sim": 1,
                "lot_mult": 0.5, "sl_mult": 0.7, "target_mult": 0.8,
                "trail_trigger": 8.0, "trail_pct": 4.0,
            },
            "HALT": {
                "min_confidence": 100, "max_trades": 0, "max_sim": 0,
                "lot_mult": 0.0, "sl_mult": 1.0, "target_mult": 1.0,
                "trail_trigger": 10.0, "trail_pct": 5.0,
            },
        }
        return profiles[self.mode]


def run_realistic_backtest(df: pd.DataFrame,
                           starting_capital: float = 10000,
                           lot_size: int = 65,
                           engine_override=None) -> dict:
    """Backtest with adaptive mode simulation matching live engine."""
    from engine.multi_strategy_engine import MultiStrategyEngine
    from engine.premium_model import create_premium_state, STRATEGY_SL_PCT

    engine = engine_override if engine_override is not None else MultiStrategyEngine()
    adaptive = AdaptiveState()

    capital = starting_capital
    peak = capital
    trades = []
    equity_curve = []
    mode_log = []

    unique_days = sorted(set(df.index.date))

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            equity_curve.append({"date": day, "capital": capital, "daily_pnl": 0,
                                  "trades": 0, "lots": 0, "mode": "NORMAL"})
            continue

        prev_day_data = None
        if day_idx > 0:
            prev_day = unique_days[day_idx - 1]
            prev_df = df[df.index.date == prev_day]
            if len(prev_df) > 0:
                prev_day_data = {
                    "high": prev_df["high"].max(),
                    "low": prev_df["low"].min(),
                    "close": prev_df["close"].iloc[-1],
                }
        engine.reset_day(prev_day_data)
        adaptive.reset_day()

        day_start_cap = capital
        day_pnl = 0.0
        day_trades = 0
        daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)

        per_lot = getattr(settings, 'CAPITAL_PER_LOT', 10_000)
        base_lots = max(1, int(capital / per_lot))
        base_lots = min(base_lots, getattr(settings, 'MAX_LOTS_CAP', 10))

        indicators = engine.precompute(day_df)
        open_positions = []

        for i in range(10, len(day_df)):
            ts = day_df.index[i]
            time_str = ts.strftime("%H:%M")
            nifty_price = day_df["close"].iloc[i]
            prof = adaptive.profile

            closed_this_bar = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(
                    nifty_price, pos["candles_held"])

                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                trail_floor = pos["prem_state"].update_trail(
                    cur_prem,
                    trigger_pct=prof["trail_trigger"],
                    trail_pct=prof["trail_pct"])

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
                    if exit_reason == "SL":
                        engine.record_sl_exit(pos["signal_type"], i)

                    day_lots = max(1, int(base_lots * pos.get("lot_mult", 1.0)))
                    costs = _calc_realistic_costs(
                        pos["entry_premium"], exit_prem,
                        pos["qty"], day_lots)

                    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                    net_pnl = raw_pnl - costs

                    capital += net_pnl
                    day_pnl += net_pnl
                    day_trades += 1

                    adaptive.on_trade(net_pnl, day_start_cap)

                    peak_gain = (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100

                    trades.append({
                        "strategy": pos["signal_type"], "signal": pos["direction"],
                        "entry_time": pos["entry_time"], "exit_time": ts,
                        "entry_premium": round(pos["entry_premium"], 2),
                        "exit_premium": round(exit_prem, 2),
                        "peak_premium": round(pos["peak_premium"], 2),
                        "peak_gain_pct": round(peak_gain, 2),
                        "qty": pos["qty"], "lots": day_lots,
                        "pnl": round(net_pnl, 0), "reason": exit_reason,
                        "capital_after": round(capital, 0),
                        "mode": adaptive.mode,
                    })
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

            if len(open_positions) >= prof["max_sim"]:
                continue
            if adaptive.day_trades >= prof["max_trades"]:
                continue
            if day_pnl < 0 and abs(day_pnl) >= daily_loss_limit:
                continue

            open_dirs = {p["direction"] for p in open_positions}
            signals = engine.scan(
                indicators, i, time_str,
                max_total_override=prof["max_trades"],
            )

            for signal in signals:
                if signal.direction in open_dirs:
                    continue
                if signal.confidence < prof["min_confidence"]:
                    continue

                lot_mult = prof["lot_mult"]
                day_lots = max(1, int(base_lots * lot_mult))

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
                eff_sl = STRATEGY_SL_PCT.get(signal.signal_type, settings.PREMIUM_SL_PCT)
                eff_sl *= prof["sl_mult"]
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
                    "lot_mult": lot_mult,
                })
                break

            if capital <= 0:
                break

        for pos in open_positions:
            nifty_price = day_df["close"].iloc[-1]
            exit_prem = pos["prem_state"].current_premium(
                nifty_price, pos["candles_held"])
            costs = _calc_realistic_costs(
                pos["entry_premium"], exit_prem,
                pos["qty"], base_lots)
            raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
            net_pnl = raw_pnl - costs
            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1

        if capital > peak:
            peak = capital

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades, "lots": base_lots,
            "mode": adaptive.mode,
        })

        if capital <= 0:
            break

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

    mode_counts = {}
    for t in trades:
        m = t.get("mode", "NORMAL")
        mode_counts[m] = mode_counts.get(m, 0) + 1

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
        "mode_distribution": mode_counts,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--capital", type=float, default=10000)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    print(f"\nLoading {args.days} trading days of Nifty data...")
    df = load_real_data(days=args.days)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days\n")

    lot_size = settings.NIFTY_LOT_SIZE

    print("=" * 65)
    print("  REALISTIC BACKTEST (with adaptive mode simulation)")
    print("  Parameters: NORMAL trail=10%/5%, DEFENSIVE trail=8%/4%,")
    print("  NORMAL min_conf=65, DEFENSIVE min_conf=75, DEFENSIVE SL*0.7")
    print("=" * 65)

    results_realistic = run_realistic_backtest(
        df, starting_capital=args.capital, lot_size=lot_size)
    print_results(results_realistic)

    mode_dist = results_realistic.get("mode_distribution", {})
    if mode_dist:
        print("  Adaptive Mode Distribution:")
        for mode, count in sorted(mode_dist.items()):
            print(f"    {mode:12s}: {count} trades")
        print()

    from backtest.run_backtest import run_compound_backtest
    print("=" * 65)
    print("  BASELINE BACKTEST (no adaptive mode, original params)")
    print("=" * 65)

    results_baseline = run_compound_backtest(
        df, starting_capital=args.capital, lot_size=lot_size)
    print_results(results_baseline)

    print("=" * 65)
    print("  COMPARISON: Realistic vs Baseline")
    print("=" * 65)
    print(f"  {'Metric':25s} {'Baseline':>12s} {'Realistic':>12s} {'Delta':>10s}")
    print(f"  {'-'*60}")
    for key in ["win_rate", "profit_factor", "total_trades", "return_pct",
                "max_drawdown_pct", "avg_daily_return_pct"]:
        b = results_baseline[key]
        r = results_realistic[key]
        d = r - b
        sign = "+" if d >= 0 else ""
        print(f"  {key:25s} {b:>12} {r:>12} {sign}{d:>9.1f}")
    print(f"\n  Final Capital: Baseline Rs {results_baseline['final_capital']:,.0f}"
          f" vs Realistic Rs {results_realistic['final_capital']:,.0f}")
    print()

    pd.DataFrame(results_realistic["trades"]).to_csv(
        settings.DATA_DIR / "realistic_backtest_trades.csv", index=False)
    pd.DataFrame(results_realistic["equity_curve"]).to_csv(
        settings.DATA_DIR / "realistic_equity_curve.csv", index=False)
    print("  Saved: data/realistic_backtest_trades.csv, data/realistic_equity_curve.csv")

    return results_realistic, results_baseline


if __name__ == "__main__":
    main()
