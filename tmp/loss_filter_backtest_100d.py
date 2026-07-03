#!/usr/bin/env python3
"""100-day replay: baseline vs loss-reduction filters.

Tests two filters individually and combined:
  1. HTF RSI > 70 blocks LONG entries (overbought filter)
  2. Post-SL cooldown of 4 bars (20 min) blocks all entries after a SL

Uses production MultiStrategyEngine + compound backtest loop.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs, run_compound_backtest
from engine.premium_model import STRATEGY_SL_PCT, create_premium_state
from engine.multi_strategy_engine import MultiStrategyEngine
from risk.adaptive_mode import AdaptiveModeController


@dataclass
class LossFilters:
    rsi70_block: bool = False
    sl_cooldown_bars: int = 0

    @property
    def label(self) -> str:
        parts = []
        if self.rsi70_block:
            parts.append("RSI70")
        if self.sl_cooldown_bars:
            parts.append(f"SLcd{self.sl_cooldown_bars}")
        return "+".join(parts) if parts else "BASELINE"


class DayContext:
    def __init__(self):
        self.last_sl_bar: int | None = None

    def record_sl(self, bar_idx: int):
        self.last_sl_bar = bar_idx


def _blocked(
    filters: LossFilters,
    ctx: DayContext,
    bar_i: int,
    signal,
) -> str | None:
    if filters.rsi70_block and signal.direction == "LONG" and signal.htf_rsi > 70:
        return f"HTF_RSI={signal.htf_rsi:.0f} > 70 (overbought LONG)"

    if filters.sl_cooldown_bars and ctx.last_sl_bar is not None:
        bars_since = bar_i - ctx.last_sl_bar
        if bars_since < filters.sl_cooldown_bars:
            return f"SL cooldown ({bars_since}/{filters.sl_cooldown_bars} bars)"

    return None


def run_filtered_backtest(
    df: pd.DataFrame,
    filters: LossFilters,
    starting_capital: float = 10_000,
    lot_size: int | None = None,
) -> dict:
    """Day-by-day replay with optional loss-reduction filters."""
    lot_size = lot_size or settings.NIFTY_LOT_SIZE
    engine = MultiStrategyEngine()
    adaptive = AdaptiveModeController()

    capital = starting_capital
    peak = capital
    trades: list[dict] = []
    equity_curve: list[dict] = []
    blocked_log: list[dict] = []

    unique_days = sorted(set(df.index.date))
    warmup_days_n = 5

    for day_idx, day in enumerate(unique_days):
        day_ctx = DayContext()
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
        day_lots = max(1, min(int(capital / per_lot), getattr(settings, "MAX_LOTS_CAP", 10)))

        warmup_start = max(0, day_idx - warmup_days_n)
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

                trail_floor = pos["prem_state"].update_trail(
                    cur_prem, trigger_pct=ap_trail_trigger, trail_pct=ap_trail_pct
                )

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
                        day_ctx.record_sl(i)

                    costs = _calc_realistic_costs(
                        pos["entry_premium"], exit_prem, pos["qty"], day_lots
                    )
                    net_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"] - costs
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
                    trades.append({
                        "date": day,
                        "strategy": pos["signal_type"],
                        "direction": pos["direction"],
                        "entry_time": pos["entry_time"],
                        "exit_time": ts,
                        "pnl": round(net_pnl, 0),
                        "reason": exit_reason,
                        "capital_after": round(capital, 0),
                        "hold_bars": pos["candles_held"],
                        "htf_rsi": pos.get("htf_rsi", 0),
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

                reason = _blocked(filters, day_ctx, i, signal)
                if reason:
                    blocked_log.append({
                        "date": day,
                        "time": time_str,
                        "strategy": signal.signal_type,
                        "direction": signal.direction,
                        "confidence": round(signal.confidence, 1),
                        "htf_rsi": round(signal.htf_rsi, 1),
                        "reason": reason,
                    })
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
                spread = getattr(settings, "BID_ASK_SPREAD", 0.30)
                entry_premium = prem_state.entry_premium + spread
                eff_sl = STRATEGY_SL_PCT.get(signal.signal_type, settings.PREMIUM_SL_PCT)
                sl_premium = entry_premium * (1 - eff_sl / 100)
                eff_lots = max(1, int(day_lots * ap_lot_mult))
                qty = eff_lots * lot_size

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
                    "htf_rsi": signal.htf_rsi,
                })
                break

            if capital <= 0:
                break

        for pos in open_positions:
            last_bar_idx = today_indices[-1]
            nifty_price = float(warmup_df["close"].iloc[last_bar_idx])
            exit_prem = pos["prem_state"].current_premium(nifty_price, pos["candles_held"])
            brokerage = getattr(settings, "BROKERAGE_PER_ORDER", 20) * 2
            stt = exit_prem * pos["qty"] * getattr(settings, "STT_PCT", 0.0125) / 100
            slippage = getattr(settings, "SLIPPAGE_POINTS", 0.5) * pos["qty"]
            costs = brokerage + stt + slippage
            net_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"] - costs
            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1
            trades.append({
                "date": day,
                "strategy": pos["signal_type"],
                "direction": pos["direction"],
                "entry_time": pos["entry_time"],
                "exit_time": warmup_df.index[last_bar_idx],
                "pnl": round(net_pnl, 0),
                "reason": "EOD",
                "capital_after": round(capital, 0),
                "hold_bars": pos["candles_held"],
                "htf_rsi": pos.get("htf_rsi", 0),
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
        "trades": trades,
        "blocked_log": blocked_log,
        "equity_curve": equity_curve,
    }


def print_comparison(results: list[dict], days: int):
    print("\n" + "=" * 110)
    print(f"  LOSS FILTER BACKTEST — {days} trading days | Starting capital Rs 10,000")
    print("=" * 110)
    hdr = (
        f"{'Scenario':<22} {'Final':>10} {'P&L':>10} {'Ret%':>7} "
        f"{'Trades':>7} {'WR%':>6} {'PF':>6} {'MaxDD':>7} {'Blocked':>8} {'ProfDays':>9}"
    )
    print(hdr)
    print("-" * 110)
    for r in results:
        pnl = r["total_pnl"]
        sign = "+" if pnl >= 0 else ""
        print(
            f"{r['label']:<22} "
            f"Rs{r['final_capital']:>8,.0f} "
            f"{sign}Rs{pnl:>7,.0f} "
            f"{r['return_pct']:>6.1f}% "
            f"{r['total_trades']:>7} "
            f"{r['win_rate']:>5.1f}% "
            f"{r['profit_factor']:>6.2f} "
            f"{r['max_drawdown_pct']:>6.1f}% "
            f"{r['blocked_signals']:>8} "
            f"{r['profitable_days']:>4}/{r['loss_days']:<4}"
        )

    if len(results) > 1:
        base = results[0]
        best = max(results[1:], key=lambda r: r["final_capital"])
        delta = best["final_capital"] - base["final_capital"]
        print("-" * 110)
        print(
            f"  BEST filter ({best['label']}) vs BASELINE: "
            f"Rs {delta:+,.0f} final capital | "
            f"{best['total_trades'] - base['total_trades']:+d} trades | "
            f"WR {best['win_rate'] - base['win_rate']:+.1f}pp | "
            f"MaxDD {best['max_drawdown_pct'] - base['max_drawdown_pct']:+.1f}pp"
        )

    print()


def print_blocked_detail(results: list[dict]):
    for r in results:
        if not r["blocked_log"]:
            continue
        print(f"\n  Blocked signals — {r['label']} ({len(r['blocked_log'])} total):")
        print(f"    {'Date':<12} {'Time':<6} {'Strategy':<14} {'Dir':<6} {'Conf':>5} {'RSI':>5} {'Reason'}")
        print(f"    {'-'*80}")
        for b in r["blocked_log"][:30]:
            print(
                f"    {str(b['date']):<12} {b['time']:<6} {b['strategy']:<14} "
                f"{b['direction']:<6} {b['confidence']:>5.0f} {b['htf_rsi']:>5.0f} "
                f"{b['reason']}"
            )
        if len(r["blocked_log"]) > 30:
            print(f"    ... and {len(r['blocked_log']) - 30} more")


def print_counterfactual(results: list[dict]):
    """For the combined filter, show what baseline trades were avoided."""
    if len(results) < 2:
        return

    base = results[0]
    combined = results[-1]

    def _trade_key(t):
        d = t.get("date", "")
        if not d and "entry_time" in t:
            d = str(t["entry_time"])[:10]
        return (str(d), str(t.get("entry_time", "")))

    base_trades = {_trade_key(t): t for t in base["trades"]}
    combined_trades = {_trade_key(t): t for t in combined["trades"]}

    avoided = []
    for key, t in base_trades.items():
        if key not in combined_trades:
            avoided.append(t)

    if not avoided:
        print("\n  No trades were avoided by the filters.")
        return

    avoided_wins = [t for t in avoided if t["pnl"] > 0]
    avoided_losses = [t for t in avoided if t["pnl"] <= 0]

    print(f"\n  COUNTERFACTUAL: Trades in BASELINE but NOT in {combined['label']}:")
    print(f"    Avoided {len(avoided)} trades: {len(avoided_wins)} winners, {len(avoided_losses)} losers")
    print(f"    Avoided winner PnL: Rs {sum(t['pnl'] for t in avoided_wins):,.0f}")
    print(f"    Avoided loser PnL:  Rs {sum(t['pnl'] for t in avoided_losses):,.0f}")
    print(f"    Net benefit:        Rs {-sum(t['pnl'] for t in avoided):,.0f}")
    print()
    print(f"    {'Date':<12} {'Strategy':<14} {'Dir':<6} {'PnL':>8} {'Reason':<8} {'Bars':>5} {'RSI':>5}")
    print(f"    {'-'*70}")
    for t in sorted(avoided, key=lambda x: x["pnl"])[:25]:
        d = t.get("date", str(t.get("entry_time", ""))[:10])
        hold = t.get("hold_bars", t.get("candles_held", "?"))
        rsi = t.get("htf_rsi", 0)
        print(
            f"    {str(d):<12} {t.get('strategy', '?'):<14} {t.get('direction', t.get('signal', '?')):<6} "
            f"Rs{t['pnl']:>6,.0f} {t.get('reason', '?'):<8} {str(hold):>5} "
            f"{rsi:>5.0f}"
        )


def main():
    days = 100
    capital = 10_000
    print(f"Loading {days} trading days of real Nifty 5m data...")
    df = load_real_data(days=days)
    unique_days = sorted(set(df.index.date))
    print(f"  {len(df):,} candles | {len(unique_days)} days | {unique_days[0]} -> {unique_days[-1]}")

    scenarios = [
        LossFilters(),
        LossFilters(rsi70_block=True),
        LossFilters(sl_cooldown_bars=4),
        LossFilters(rsi70_block=True, sl_cooldown_bars=4),
    ]

    results = []
    for f in scenarios:
        label = f.label
        print(f"  Running {label}...")
        if label == "BASELINE":
            raw = run_compound_backtest(df, starting_capital=capital)
            results.append({
                "label": "BASELINE",
                "starting_capital": capital,
                "final_capital": raw["final_capital"],
                "total_pnl": raw["total_pnl"],
                "return_pct": raw["return_pct"],
                "total_trades": raw["total_trades"],
                "wins": raw["wins"],
                "losses": raw["losses"],
                "win_rate": raw["win_rate"],
                "profit_factor": raw["profit_factor"],
                "max_drawdown_pct": raw["max_drawdown_pct"],
                "blocked_signals": 0,
                "profitable_days": raw["profitable_days"],
                "loss_days": raw["loss_days"],
                "trades": raw["trades"],
                "blocked_log": [],
                "equity_curve": raw.get("equity_curve", []),
            })
        else:
            results.append(run_filtered_backtest(df, f, starting_capital=capital))

    print_comparison(results, len(unique_days))
    print_blocked_detail(results)
    print_counterfactual(results)

    out = ROOT / "tmp" / "loss_filter_results.csv"
    rows = []
    for r in results:
        rows.append({k: r[k] for k in r if k not in ("trades", "blocked_log", "equity_curve")})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Results saved: {out}")


if __name__ == "__main__":
    main()
