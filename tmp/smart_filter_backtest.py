#!/usr/bin/env python3
"""100-day backtest: smarter loss-reduction strategies.

Tests three approaches that target losses WITHOUT blocking entries:
  1. Breakeven trail after +2% — moves SL to entry once trade gains 2%
  2. Half-lot after 2 consecutive losses in same day
  3. Block TREND_RIDE 10:00-10:59 — the only near-breakeven strategy+hour combo
  4. All three combined

These are "surgical" fixes that don't block profitable signals.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["LOGURU_LEVEL"] = "ERROR"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs, _compute_dte_for_date
from engine.premium_model import STRATEGY_SL_PCT, create_premium_state
from engine.multi_strategy_engine import MultiStrategyEngine
from risk.adaptive_mode import AdaptiveModeController

WARMUP_DAYS = 5


@dataclass
class SmartFilters:
    breakeven_trail_pct: float = 0.0  # Move SL to entry after this % gain (0 = disabled)
    half_lot_after_consec: int = 0    # Half lots after N consecutive losses (0 = disabled)
    block_trend_ride_10: bool = False  # Block TREND_RIDE between 10:00-10:59

    @property
    def label(self) -> str:
        parts = []
        if self.breakeven_trail_pct:
            parts.append(f"BE@{self.breakeven_trail_pct}%")
        if self.half_lot_after_consec:
            parts.append(f"HalfLot>{self.half_lot_after_consec}L")
        if self.block_trend_ride_10:
            parts.append("NoTR@10")
        return "+".join(parts) if parts else "BASELINE"


def run_smart_backtest(
    df: pd.DataFrame,
    filters: SmartFilters,
    starting_capital: float = 10_000,
    lot_size: int | None = None,
) -> dict:
    """Day-by-day replay with smart loss-reduction filters."""
    lot_size = lot_size or settings.NIFTY_LOT_SIZE
    engine = MultiStrategyEngine()
    adaptive = AdaptiveModeController()

    capital = starting_capital
    peak = capital
    trades: list[dict] = []
    equity_curve: list[dict] = []
    blocked_log: list[dict] = []

    unique_days = sorted(set(df.index.date))

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            equity_curve.append({"date": day, "capital": capital, "daily_pnl": 0, "trades": 0})
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
        adaptive.reset()

        day_start_cap = capital
        day_pnl = 0.0
        day_trades = 0
        day_wins = 0
        day_losses = 0
        consec_loss = 0
        daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)
        daily_profit_target = day_start_cap * (
            getattr(settings, "DAILY_PROFIT_TARGET_PCT", 35) / 100
        )

        per_lot = getattr(settings, "CAPITAL_PER_LOT", 10_000)
        deploy_pct = getattr(settings, "DEPLOY_PCT", 100)
        deployable = capital * (deploy_pct / 100)
        day_lots = max(1, int(deployable / per_lot))
        day_lots = min(day_lots, getattr(settings, "MAX_LOTS_CAP", 20))

        warmup_start = max(0, day_idx - WARMUP_DAYS)
        warmup_days = unique_days[warmup_start : day_idx + 1]
        warmup_day_set = set(warmup_days)
        warmup_df = df[df.index.map(lambda t: t.date() in warmup_day_set)]
        indicators = engine.precompute(warmup_df)
        today_indices = [i for i, ts in enumerate(warmup_df.index) if ts.date() == day]
        open_positions: list[dict] = []

        for i in today_indices:
            if i < 10:
                continue
            ts = warmup_df.index[i]
            time_str = ts.strftime("%H:%M")
            nifty_price = float(warmup_df["close"].iloc[i])

            adaptive.on_bar()
            ap = adaptive.profile
            ap_min_conf = ap.min_confidence
            ap_lot_mult = ap.lot_multiplier
            ap_sl_mult = ap.sl_multiplier
            ap_target_mult = ap.target_multiplier
            ap_max_sim = ap.max_simultaneous
            ap_max_trades = ap.max_trades_per_day
            ap_trail_trigger = ap.trail_trigger_pct
            ap_trail_pct = ap.trail_pct

            closed_this_bar = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(nifty_price, pos["candles_held"])

                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                # FILTER 1: Breakeven trail — move SL to entry after X% gain
                if filters.breakeven_trail_pct > 0 and not pos.get("be_activated"):
                    gain_pct = (cur_prem - pos["entry_premium"]) / pos["entry_premium"] * 100
                    if gain_pct >= filters.breakeven_trail_pct:
                        pos["sl_premium"] = pos["entry_premium"]
                        pos["be_activated"] = True

                trail_floor = pos["prem_state"].update_trail(
                    cur_prem, trigger_pct=ap_trail_trigger, trail_pct=ap_trail_pct
                )

                # Partial profit booking
                partial_pct = getattr(settings, "PARTIAL_PROFIT_PCT", 25)
                if (
                    not pos.get("partial_booked")
                    and pos["entry_premium"] > 0
                    and (cur_prem - pos["entry_premium"]) / pos["entry_premium"] * 100 >= partial_pct
                    and pos["qty"] > lot_size
                ):
                    half_qty = (pos["qty"] // (2 * lot_size)) * lot_size
                    if half_qty >= lot_size:
                        partial_pnl = (cur_prem - pos["entry_premium"]) * half_qty
                        partial_costs = _calc_realistic_costs(
                            pos["entry_premium"], cur_prem, half_qty, day_lots
                        )
                        net_partial = partial_pnl - partial_costs
                        capital += net_partial
                        day_pnl += net_partial
                        pos["qty"] -= half_qty
                        pos["partial_booked"] = True
                        pos["sl_premium"] = pos["entry_premium"]
                        trades.append({
                            "strategy": pos["signal_type"],
                            "signal": pos["direction"],
                            "entry_time": pos["entry_time"],
                            "exit_time": ts,
                            "entry_premium": round(pos["entry_premium"], 2),
                            "exit_premium": round(cur_prem, 2),
                            "peak_premium": round(pos["peak_premium"], 2),
                            "peak_gain_pct": round(partial_pct, 2),
                            "qty": half_qty,
                            "lots": pos["eff_lots"],
                            "pnl": round(net_partial, 0),
                            "reason": "PARTIAL",
                            "capital_after": round(capital, 0),
                            "candles_held": pos["candles_held"],
                        })
                        day_trades += 1
                        day_wins += 1
                        consec_loss = 0

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

                    costs = _calc_realistic_costs(
                        pos["entry_premium"], exit_prem, pos["qty"], pos["eff_lots"]
                    )
                    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                    net_pnl = raw_pnl - costs
                    capital += net_pnl
                    day_pnl += net_pnl
                    day_trades += 1
                    won = net_pnl > 0
                    if won:
                        day_wins += 1
                        consec_loss = 0
                    else:
                        day_losses += 1
                        consec_loss += 1

                    daily_pnl_pct = (day_pnl / day_start_cap * 100) if day_start_cap > 0 else 0
                    adaptive.update(
                        daily_pnl_pct=daily_pnl_pct,
                        wins=day_wins,
                        losses=day_losses,
                        consecutive_losses=consec_loss,
                        trades=day_trades,
                        last_trade_won=won,
                    )

                    peak_gain = (
                        (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100
                    )
                    trades.append({
                        "strategy": pos["signal_type"],
                        "signal": pos["direction"],
                        "entry_time": pos["entry_time"],
                        "exit_time": ts,
                        "entry_premium": round(pos["entry_premium"], 2),
                        "exit_premium": round(exit_prem, 2),
                        "peak_premium": round(pos["peak_premium"], 2),
                        "peak_gain_pct": round(peak_gain, 2),
                        "qty": pos["qty"],
                        "lots": pos["eff_lots"],
                        "pnl": round(net_pnl, 0),
                        "reason": exit_reason,
                        "capital_after": round(capital, 0),
                        "candles_held": pos["candles_held"],
                    })
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

            if len(open_positions) >= ap_max_sim:
                continue
            if consec_loss >= settings.MAX_CONSECUTIVE_LOSSES:
                continue
            if day_pnl < 0 and abs(day_pnl) >= daily_loss_limit:
                continue
            if day_pnl >= daily_profit_target:
                continue
            if adaptive.mode.value == "HALT":
                continue
            if day_trades >= ap_max_trades:
                continue

            open_dirs = {p["direction"] for p in open_positions}
            signals = engine.scan(indicators, i, time_str, max_total_override=ap_max_trades)

            for signal in signals:
                if signal.direction in open_dirs:
                    continue
                if signal.confidence < ap_min_conf:
                    continue

                # FILTER 3: Block TREND_RIDE at 10:00-10:59
                if filters.block_trend_ride_10:
                    hour = ts.hour
                    if signal.signal_type == "TREND_RIDE" and hour == 10:
                        blocked_log.append({
                            "date": day,
                            "time": time_str,
                            "strategy": signal.signal_type,
                            "direction": signal.direction,
                            "confidence": round(signal.confidence, 1),
                            "reason": "TREND_RIDE blocked at 10:xx",
                        })
                        continue

                theta = settings.get_scaled_theta(nifty_price)
                dte = _compute_dte_for_date(day)
                prem_state = create_premium_state(
                    entry_index_price=nifty_price,
                    direction=signal.direction,
                    base_premium=settings.PREMIUM_BASE,
                    delta=settings.PREMIUM_DELTA,
                    theta_per_candle=theta,
                    sl_pct=settings.PREMIUM_SL_PCT,
                    confluence_score=signal.confidence,
                    signal_type=signal.signal_type,
                    dte=dte,
                )

                if ap_target_mult != 1.0:
                    prem_state.target_premium = (
                        prem_state.entry_premium
                        + (prem_state.target_premium - prem_state.entry_premium) * ap_target_mult
                    )

                spread = getattr(settings, "BID_ASK_SPREAD", 0.30)
                entry_premium = prem_state.entry_premium + spread
                vol_ratio = getattr(signal, "vol_ratio", 1.0)
                eff_sl = (
                    STRATEGY_SL_PCT.get(signal.signal_type, settings.PREMIUM_SL_PCT)
                    * ap_sl_mult
                    * vol_ratio
                )
                sl_premium = entry_premium * (1 - eff_sl / 100)

                # FILTER 2: Half-lot after N consecutive losses
                eff_lots = max(1, int(day_lots * ap_lot_mult))
                if (
                    filters.half_lot_after_consec > 0
                    and consec_loss >= filters.half_lot_after_consec
                ):
                    eff_lots = max(1, eff_lots // 2)

                qty = eff_lots * lot_size

                open_positions.append({
                    "direction": signal.direction,
                    "signal_type": signal.signal_type,
                    "entry_time": ts,
                    "entry_premium": entry_premium,
                    "sl_premium": sl_premium,
                    "qty": qty,
                    "eff_lots": eff_lots,
                    "prem_state": prem_state,
                    "candles_held": 0,
                    "peak_premium": entry_premium,
                })
                break

            if capital <= 0:
                break

        # EOD square-off
        for pos in open_positions:
            last_bar_idx = today_indices[-1]
            nifty_price = float(warmup_df["close"].iloc[last_bar_idx])
            exit_prem = pos["prem_state"].current_premium(nifty_price, pos["candles_held"])
            brokerage = getattr(settings, "BROKERAGE_PER_ORDER", 20) * 2
            stt = exit_prem * pos["qty"] * getattr(settings, "STT_SELL_PCT", 0.05) / 100
            slippage = getattr(settings, "SLIPPAGE_POINTS", 0.30) * pos["qty"]
            costs = brokerage + stt + slippage
            raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
            net_pnl = raw_pnl - costs
            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1
            peak_gain = (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100
            trades.append({
                "strategy": pos["signal_type"],
                "signal": pos["direction"],
                "entry_time": pos["entry_time"],
                "exit_time": warmup_df.index[last_bar_idx],
                "entry_premium": round(pos["entry_premium"], 2),
                "exit_premium": round(exit_prem, 2),
                "peak_premium": round(pos["peak_premium"], 2),
                "peak_gain_pct": round(peak_gain, 2),
                "qty": pos["qty"],
                "lots": pos["eff_lots"],
                "pnl": round(net_pnl, 0),
                "reason": "EOD",
                "capital_after": round(capital, 0),
                "candles_held": pos["candles_held"],
            })

        if capital > peak:
            peak = capital
        equity_curve.append({
            "date": day,
            "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades,
        })
        if capital <= 0:
            break

    return _summarize(starting_capital, capital, trades, equity_curve, blocked_log, filters.label)


def _summarize(
    starting_capital: float,
    capital: float,
    trades: list[dict],
    equity_curve: list[dict],
    blocked_log: list[dict],
    label: str,
) -> dict:
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    eq_vals = [e["capital"] for e in equity_curve] or [starting_capital]
    peak_arr = np.maximum.accumulate(eq_vals)
    dd_arr = [(p - v) / p * 100 if p > 0 else 0 for p, v in zip(peak_arr, eq_vals)]
    max_dd = max(dd_arr) if dd_arr else 0

    sl_trades = [t for t in trades if t["reason"] == "SL"]
    sl_total_loss = abs(sum(t["pnl"] for t in sl_trades))

    return {
        "label": label,
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_pnl": round(capital - starting_capital, 0),
        "return_pct": round((capital - starting_capital) / starting_capital * 100, 1),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "blocked_signals": len(blocked_log),
        "profitable_days": sum(1 for e in equity_curve if e["daily_pnl"] > 0),
        "loss_days": sum(1 for e in equity_curve if e["daily_pnl"] < 0),
        "sl_count": len(sl_trades),
        "sl_total_loss": round(sl_total_loss, 0),
        "trades": trades,
        "blocked_log": blocked_log,
        "equity_curve": equity_curve,
    }


def print_comparison(results: list[dict], days: int):
    print("\n" + "=" * 120)
    print(f"  SMART FILTER BACKTEST — {days} trading days | Starting capital Rs 10,000")
    print("=" * 120)
    hdr = (
        f"{'Scenario':<24} {'Final':>12} {'P&L':>12} {'Ret%':>8} "
        f"{'Trades':>7} {'WR%':>6} {'PF':>6} {'MaxDD':>7} {'SLs':>5} {'SL Loss':>10} {'ProfDays':>9}"
    )
    print(hdr)
    print("-" * 120)
    for r in results:
        pnl = r["total_pnl"]
        sign = "+" if pnl >= 0 else ""
        print(
            f"{r['label']:<24} "
            f"Rs{r['final_capital']:>10,.0f} "
            f"{sign}Rs{pnl:>9,.0f} "
            f"{r['return_pct']:>7.1f}% "
            f"{r['total_trades']:>7} "
            f"{r['win_rate']:>5.1f}% "
            f"{r['profit_factor']:>6.2f} "
            f"{r['max_drawdown_pct']:>6.1f}% "
            f"{r['sl_count']:>5} "
            f"Rs{r['sl_total_loss']:>8,.0f} "
            f"{r['profitable_days']:>4}/{r['loss_days']:<4}"
        )

    print("-" * 120)
    base = results[0]
    for r in results[1:]:
        delta = r["final_capital"] - base["final_capital"]
        dd_delta = r["max_drawdown_pct"] - base["max_drawdown_pct"]
        sl_delta = r["sl_total_loss"] - base["sl_total_loss"]
        print(
            f"  {r['label']:<20} vs BASELINE: "
            f"Final Rs {delta:+,.0f} | "
            f"MaxDD {dd_delta:+.1f}pp | "
            f"SL Loss Rs {sl_delta:+,.0f} | "
            f"WR {r['win_rate'] - base['win_rate']:+.1f}pp"
        )

    print()


def print_sl_analysis(results: list[dict]):
    """Show how each filter affected SL outcomes."""
    print("  SL EXIT ANALYSIS (breakeven trail converts SLs to flat/small win):")
    print(f"  {'Scenario':<24} {'SL Exits':>8} {'BE Saves':>9} {'Avg SL Loss':>12}")
    print(f"  {'-'*60}")
    for r in results:
        sl_trades = [t for t in r["trades"] if t["reason"] == "SL"]
        be_flat = [t for t in r["trades"] if t["reason"] == "SL" and t["pnl"] == 0]
        avg_sl = np.mean([t["pnl"] for t in sl_trades]) if sl_trades else 0
        print(
            f"  {r['label']:<24} {len(sl_trades):>8} {len(be_flat):>9} Rs{avg_sl:>10,.0f}"
        )
    print()


def main():
    days = 100
    capital = 10_000
    print(f"Loading {days} trading days of real Nifty 5m data...")
    df = load_real_data(days=days)
    unique_days = sorted(set(df.index.date))
    print(f"  {len(df):,} candles | {len(unique_days)} days | {unique_days[0]} -> {unique_days[-1]}")

    scenarios = [
        SmartFilters(),                                                  # BASELINE
        SmartFilters(breakeven_trail_pct=2.0),                           # BE trail at 2%
        SmartFilters(half_lot_after_consec=2),                           # Half lot after 2 consec
        SmartFilters(block_trend_ride_10=True),                          # Block TR@10
        SmartFilters(breakeven_trail_pct=2.0, half_lot_after_consec=2, block_trend_ride_10=True),  # All
    ]

    results = []
    for f in scenarios:
        label = f.label
        print(f"  Running {label}...")
        results.append(run_smart_backtest(df, f, starting_capital=capital))

    print_comparison(results, len(unique_days))
    print_sl_analysis(results)

    out = ROOT / "tmp" / "smart_filter_results.csv"
    rows = []
    for r in results:
        rows.append({k: r[k] for k in r if k not in ("trades", "blocked_log", "equity_curve")})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Results saved: {out}")


if __name__ == "__main__":
    main()
