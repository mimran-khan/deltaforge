#!/usr/bin/env python3
"""100-day replay: baseline vs intraday guardrails (Rs 10k start).

Uses production MultiStrategyEngine + compound backtest loop from run_backtest.py.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from backtest.run_backtest import (
    load_real_data,
    _calc_realistic_costs,
    run_compound_backtest,
)
from engine.premium_model import STRATEGY_SL_PCT, create_premium_state
from engine.multi_strategy_engine import MultiStrategyEngine
from risk.adaptive_mode import AdaptiveModeController


@dataclass
class Guardrails:
    below_vwap: bool = False
    htf_ltf_gap: bool = False
    global_sl_cooldown_bars: int = 0  # 6 = 30 min
    session_high_pts: float = 0.0     # block PB/STOCH LONG if this far below high
    lower_highs: bool = False

    @property
    def label(self) -> str:
        parts = []
        if self.below_vwap:
            parts.append("VWAP")
        if self.htf_ltf_gap:
            parts.append("HTF/LTF")
        if self.global_sl_cooldown_bars:
            parts.append(f"SLcd{self.global_sl_cooldown_bars}")
        if self.session_high_pts:
            parts.append(f"Hi-{int(self.session_high_pts)}")
        if self.lower_highs:
            parts.append("LH")
        return "+".join(parts) if parts else "BASELINE"


BASELINE = Guardrails()
ALL_FILTERS = Guardrails(
    below_vwap=True,
    htf_ltf_gap=True,
    global_sl_cooldown_bars=6,
    session_high_pts=40.0,
    lower_highs=True,
)


class DayContext:
    """Intraday state for guardrail checks."""

    def __init__(self):
        self.session_high = 0.0
        self.twap_sum = 0.0
        self.bar_count = 0
        self.peak_high = 0.0
        self.last_swing_high = 0.0
        self.lower_highs = False
        self.last_sl_bar: int | None = None

    def update_bar(self, high: float, close: float):
        self.bar_count += 1
        tp = (high + close) / 2  # approx when using close-only updates
        self.twap_sum += (high + close + close) / 3  # typical price
        self.session_high = max(self.session_high, high)

        if high >= self.session_high - 0.01:
            self.peak_high = high
            self.last_swing_high = high
        elif self.last_swing_high > 0 and high < self.last_swing_high:
            self.lower_highs = True

    @property
    def twap(self) -> float:
        return self.twap_sum / self.bar_count if self.bar_count else close  # noqa


def _blocked(
    guards: Guardrails,
    ctx: DayContext,
    bar_i: int,
    close: float,
    signal,
) -> str | None:
    if signal.direction == "LONG" and guards.below_vwap:
        twap = ctx.twap_sum / ctx.bar_count
        if close < twap:
            return f"below TWAP ({close:.0f}<{twap:.0f})"

    if signal.direction == "LONG" and guards.htf_ltf_gap:
        htf, ltf = signal.htf_rsi, signal.ltf_rsi
        if htf >= 58 and ltf <= htf - 18:
            return f"HTF/LTF gap ({htf:.0f}/{ltf:.0f})"

    if guards.global_sl_cooldown_bars and ctx.last_sl_bar is not None:
        if bar_i - ctx.last_sl_bar < guards.global_sl_cooldown_bars:
            return f"global SL cooldown ({bar_i - ctx.last_sl_bar} bars)"

    if (
        signal.direction == "LONG"
        and signal.signal_type in ("PULLBACK", "STOCH_CROSS")
        and guards.session_high_pts > 0
    ):
        dist = ctx.session_high - close
        if dist >= guards.session_high_pts:
            return f"{dist:.0f}pts below session high"

    if signal.direction == "LONG" and guards.lower_highs and ctx.lower_highs:
        return "lower highs after peak"

    return None


def run_guarded_backtest(
    df: pd.DataFrame,
    guards: Guardrails,
    starting_capital: float = 10_000,
    lot_size: int | None = None,
) -> dict:
    """Mirror run_compound_backtest with optional entry guardrails."""
    from engine.multi_strategy_engine import MultiStrategyEngine

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
            equity_curve.append(
                {"date": day, "capital": capital, "daily_pnl": 0, "trades": 0, "lots": 0}
            )
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
            h = float(warmup_df["high"].iloc[i])
            c = float(warmup_df["close"].iloc[i])
            nifty_price = c
            day_ctx.update_bar(h, c)

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
                        day_ctx.last_sl_bar = i

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
                    peak_gain = (
                        (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100
                    )
                    trades.append(
                        {
                            "date": day,
                            "strategy": pos["signal_type"],
                            "signal": pos["direction"],
                            "entry_time": pos["entry_time"],
                            "exit_time": ts,
                            "pnl": round(net_pnl, 0),
                            "reason": exit_reason,
                            "capital_after": round(capital, 0),
                            "peak_gain_pct": round(peak_gain, 2),
                        }
                    )
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

                reason = _blocked(guards, day_ctx, i, c, signal)
                if reason:
                    blocked_log.append(
                        {
                            "date": day,
                            "time": time_str,
                            "strategy": signal.signal_type,
                            "direction": signal.direction,
                            "reason": reason,
                        }
                    )
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

                open_positions.append(
                    {
                        "direction": signal.direction,
                        "signal_type": signal.signal_type,
                        "entry_time": ts,
                        "entry_premium": entry_premium,
                        "sl_premium": sl_premium,
                        "qty": qty,
                        "prem_state": prem_state,
                        "candles_held": 0,
                        "peak_premium": entry_premium,
                    }
                )
                break

            if capital <= 0:
                break

        # EOD square-off
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
            trades.append(
                {
                    "date": day,
                    "strategy": pos["signal_type"],
                    "signal": pos["direction"],
                    "entry_time": pos["entry_time"],
                    "exit_time": warmup_df.index[last_bar_idx],
                    "pnl": round(net_pnl, 0),
                    "reason": "EOD",
                    "capital_after": round(capital, 0),
                    "peak_gain_pct": 0,
                }
            )

        if capital > peak:
            peak = capital
        equity_curve.append(
            {
                "date": day,
                "capital": round(capital, 0),
                "daily_pnl": round(day_pnl, 0),
                "trades": day_trades,
                "lots": day_lots,
            }
        )
        if capital <= 0:
            break

    return _summarize(starting_capital, capital, trades, equity_curve, blocked_log, guards.label)


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
    dd_arr = [(p - v) / p * 100 for p, v in zip(peak_arr, eq_vals)]
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
    }


def print_comparison(results: list[dict], days: int):
    print("\n" + "=" * 100)
    print(f"  GUARDRAIL REPLAY — {days} trading days | Starting capital Rs 10,000")
    print("=" * 100)
    hdr = (
        f"{'Scenario':<22} {'Final':>10} {'P&L':>10} {'Ret%':>7} "
        f"{'Trades':>7} {'WR%':>6} {'PF':>5} {'MaxDD':>7} {'Blocked':>8}"
    )
    print(hdr)
    print("-" * 100)
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
            f"{r['profit_factor']:>5.2f} "
            f"{r['max_drawdown_pct']:>6.1f}% "
            f"{r['blocked_signals']:>8}"
        )

    base = results[0]
    all_f = next(r for r in results if r["label"] == ALL_FILTERS.label)
    delta = all_f["final_capital"] - base["final_capital"]
    print("-" * 100)
    print(
        f"  ALL filters vs baseline: Rs {delta:+,.0f} final capital | "
        f"{all_f['total_trades'] - base['total_trades']:+d} trades | "
        f"blocked {all_f['blocked_signals']} signals"
    )

    # Strategy breakdown for baseline vs all filters
    for label in (base["label"], all_f["label"]):
        r = next(x for x in results if x["label"] == label)
        print(f"\n  Strategy mix — {label}:")
        stats: dict[str, dict] = {}
        for t in r["trades"]:
            s = t["strategy"]
            stats.setdefault(s, {"n": 0, "pnl": 0, "w": 0})
            stats[s]["n"] += 1
            stats[s]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                stats[s]["w"] += 1
        for s, st in sorted(stats.items()):
            wr = st["w"] / st["n"] * 100 if st["n"] else 0
            print(f"    {s:14s}: {st['n']:>4d} trades | {wr:4.0f}% WR | Rs {st['pnl']:>9,.0f}")

    print()


def main():
    days = 100
    capital = 10_000
    print(f"Loading {days} trading days...")
    df = load_real_data(days=days)
    unique_days = sorted(set(df.index.date))
    print(f"  {len(df):,} candles | {len(unique_days)} days | {unique_days[0]} → {unique_days[-1]}")

    scenarios = [
        BASELINE,
        Guardrails(below_vwap=True),
        Guardrails(htf_ltf_gap=True),
        Guardrails(global_sl_cooldown_bars=6),
        Guardrails(session_high_pts=40.0),
        Guardrails(lower_highs=True),
        ALL_FILTERS,
    ]

    results = []
    for g in scenarios:
        print(f"  Running {g.label}...")
        if g.label == "BASELINE":
            # use canonical backtest for baseline parity
            raw = run_compound_backtest(df, starting_capital=capital)
            results.append(
                {
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
                }
            )
        else:
            results.append(run_guarded_backtest(df, g, starting_capital=capital))

    print_comparison(results, len(unique_days))

    out = ROOT / "tmp" / "guardrail_backtest_results.csv"
    rows = []
    for r in results:
        rows.append({k: r[k] for k in r if k not in ("trades", "blocked_log")})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Saved summary: {out}")


if __name__ == "__main__":
    main()
