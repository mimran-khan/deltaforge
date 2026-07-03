#!/usr/bin/env python3
"""Counterfactual paper test: Jun 16 2026 full-day vs late-start vs actual live."""

from __future__ import annotations

import copy
import sys
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.check_today import download_nifty_5m
from backtest.run_backtest import _calc_realistic_costs, run_compound_backtest
from config import settings
from engine.multi_strategy_engine import MultiStrategyEngine
from engine.premium_model import STRATEGY_SL_PCT, create_premium_state
from risk.adaptive_mode import AdaptiveModeController

TARGET = date(2026, 6, 16)
START_CAPITAL = 18_609.24  # day-start capital from live
ACTUAL_PNL = 673.44
LATE_START = time(11, 26)


def _load_df() -> pd.DataFrame:
    df = download_nifty_5m(trading_days=8)
    if (df["volume"] == 0).all():
        df["volume"] = 5000
    return df


def simulate_day(
    df: pd.DataFrame,
    *,
    label: str,
    entry_not_before: time | None = None,
    entry_not_after: str | None = None,
    disabled_override: set[str] | None = None,
    max_sim_override: int | None = None,
    allow_pyramid: bool = False,
    lots_override: int | None = None,
) -> dict:
    """Single-day replay using production scan + risk + premium model."""
    engine = MultiStrategyEngine()
    if disabled_override is not None:
        engine.DISABLED_STRATEGIES = set(disabled_override)

    adaptive = AdaptiveModeController()
    lot_size = settings.NIFTY_LOT_SIZE
    capital = START_CAPITAL
    day_start_cap = capital

    unique_days = sorted(set(df.index.date))
    if TARGET not in unique_days:
        raise RuntimeError(f"{TARGET} not in downloaded data: {unique_days}")

    day_idx = unique_days.index(TARGET)
    prev_day_data = None
    if day_idx > 0:
        prev_df = df[df.index.date == unique_days[day_idx - 1]]
        if len(prev_df):
            prev_day_data = {
                "high": prev_df["high"].max(),
                "low": prev_df["low"].min(),
                "close": prev_df["close"].iloc[-1],
            }

    warmup_start = max(0, day_idx - 5)
    warmup_days = set(unique_days[warmup_start : day_idx + 1])
    warmup_df = df[df.index.map(lambda t: t.date() in warmup_days)]
    engine.reset_day(prev_day_data)
    adaptive.reset()

    per_lot = settings.CAPITAL_PER_LOT
    day_lots = lots_override if lots_override is not None else max(
        1, min(int(capital / per_lot), settings.MAX_LOTS_CAP)
    )
    daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)
    daily_profit_target = day_start_cap * (settings.DAILY_PROFIT_TARGET_PCT / 100)
    entry_end = entry_not_after or settings.ENTRY_END

    indicators = engine.precompute(warmup_df)
    today_indices = [i for i, ts in enumerate(warmup_df.index) if ts.date() == TARGET]

    open_positions: list[dict] = []
    trades: list[dict] = []
    signals_log: list[dict] = []
    day_trades = day_wins = day_losses = 0
    consec_loss = 0
    day_pnl = 0.0

    def current_adx(bar_i: int) -> float:
        adx = indicators.get("adx")
        if adx is None or bar_i >= len(adx):
            return 0.0
        v = adx.iloc[bar_i]
        return 0.0 if pd.isna(v) else float(v)

    def should_runner(pos: dict, adx_val: float, now_str: str) -> bool:
        if not settings.TREND_RUNNER_ENABLED:
            return False
        if pos["signal_type"] not in settings.TREND_RUNNER_STRATEGIES:
            return False
        if now_str >= settings.TREND_RUNNER_CUTOFF_TIME:
            return False
        return adx_val >= settings.TREND_RUNNER_ADX_MIN

    for i in today_indices:
        if i < settings.SCAN_WARMUP_BARS:
            continue

        ts = warmup_df.index[i]
        bar_time = ts.time()
        time_str = ts.strftime("%H:%M")
        nifty_price = float(warmup_df["close"].iloc[i])
        adx_val = current_adx(i)

        adaptive.on_bar()
        ap = adaptive.profile
        ap_max_sim = max_sim_override if max_sim_override is not None else ap.max_simultaneous

        closed_this_bar: list[dict] = []
        for pos in open_positions:
            pos["candles_held"] += 1
            cur_prem = pos["prem_state"].current_premium(nifty_price, pos["candles_held"])
            if cur_prem > pos["peak_premium"]:
                pos["peak_premium"] = cur_prem

            trail_floor = pos["prem_state"].update_trail(
                cur_prem,
                trigger_pct=ap.trail_trigger_pct,
                trail_pct=ap.trail_pct,
            )

            exit_reason = None
            exit_prem = cur_prem

            if cur_prem <= pos["sl_premium"]:
                exit_reason = "SL"
                exit_prem = pos["sl_premium"]
            elif cur_prem >= pos["prem_state"].target_premium:
                if not pos["runner_mode"] and should_runner(pos, adx_val, time_str):
                    pos["runner_mode"] = True
                    pos["runner_bars"] = 0
                    pos["sl_premium"] = max(pos["sl_premium"], pos["entry_premium"])
                elif not pos["runner_mode"]:
                    exit_reason = "TGT"
                    exit_prem = pos["prem_state"].target_premium
            elif pos["runner_mode"]:
                pos["runner_bars"] += 1
                runner_floor = pos["peak_premium"] * (1 - settings.TREND_RUNNER_TRAIL_PCT / 100)
                if cur_prem <= runner_floor:
                    exit_reason = "RUNNER_TRAIL"
                    exit_prem = runner_floor
                elif adx_val < settings.TREND_RUNNER_ADX_EXIT:
                    exit_reason = "RUNNER_WEAK"
                elif pos["runner_bars"] >= settings.TREND_RUNNER_MAX_BARS:
                    exit_reason = "RUNNER_TIME"
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
                    pos["entry_premium"], exit_prem, pos["qty"], pos["lots"]
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

                adaptive.update(
                    daily_pnl_pct=day_pnl / day_start_cap * 100,
                    wins=day_wins,
                    losses=day_losses,
                    consecutive_losses=consec_loss,
                    trades=day_trades,
                    last_trade_won=won,
                )

                trades.append({
                    "strategy": pos["signal_type"],
                    "direction": pos["direction"],
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "entry_premium": round(pos["entry_premium"], 2),
                    "exit_premium": round(exit_prem, 2),
                    "pnl": round(net_pnl, 2),
                    "reason": exit_reason,
                    "lots": pos["lots"],
                    "qty": pos["qty"],
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
        if day_trades >= ap.max_trades_per_day:
            continue
        if entry_not_before and bar_time < entry_not_before:
            continue
        if time_str > entry_end:
            continue

        open_dirs = {p["direction"] for p in open_positions}
        signals = engine.scan(indicators, i, time_str, max_total_override=ap.max_trades_per_day)

        for signal in signals:
            if not allow_pyramid and signal.direction in open_dirs:
                continue
            if signal.confidence < ap.min_confidence:
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
            spread = settings.BID_ASK_SPREAD
            entry_premium = prem_state.entry_premium + spread
            eff_sl = STRATEGY_SL_PCT.get(signal.signal_type, settings.PREMIUM_SL_PCT)
            sl_premium = entry_premium * (1 - eff_sl / 100)
            eff_lots = max(1, int(day_lots * ap.lot_multiplier))
            qty = eff_lots * lot_size

            signals_log.append({
                "time": time_str,
                "strategy": signal.signal_type,
                "direction": signal.direction,
                "confidence": signal.confidence,
            })

            open_positions.append({
                "direction": signal.direction,
                "signal_type": signal.signal_type,
                "entry_time": ts,
                "entry_premium": entry_premium,
                "sl_premium": sl_premium,
                "qty": qty,
                "lots": eff_lots,
                "prem_state": prem_state,
                "candles_held": 0,
                "peak_premium": entry_premium,
                "runner_mode": False,
                "runner_bars": 0,
            })
            break

    for pos in open_positions:
        last_i = today_indices[-1]
        ts = warmup_df.index[last_i]
        nifty_price = float(warmup_df["close"].iloc[last_i])
        exit_prem = pos["prem_state"].current_premium(nifty_price, pos["candles_held"])
        costs = _calc_realistic_costs(pos["entry_premium"], exit_prem, pos["qty"], pos["lots"])
        net_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"] - costs
        capital += net_pnl
        day_pnl += net_pnl
        day_trades += 1
        trades.append({
            "strategy": pos["signal_type"],
            "direction": pos["direction"],
            "entry_time": pos["entry_time"],
            "exit_time": ts,
            "entry_premium": round(pos["entry_premium"], 2),
            "exit_premium": round(exit_prem, 2),
            "pnl": round(net_pnl, 2),
            "reason": "EOD",
            "lots": pos["lots"],
            "qty": pos["qty"],
        })

    wins = sum(1 for t in trades if t["pnl"] > 0)
    return {
        "label": label,
        "pnl": round(day_pnl, 2),
        "trades": trades,
        "trade_count": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "signals": signals_log,
        "final_capital": round(capital, 2),
    }


def print_scenario(result: dict) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {result['label']}")
    print(f"{'─' * 72}")
    if not result["trades"]:
        print("  No trades")
    else:
        for t in result["trades"]:
            tag = "WIN" if t["pnl"] > 0 else "LOSS" if t["pnl"] < 0 else "FLAT"
            et = t["entry_time"].strftime("%H:%M")
            xt = t["exit_time"].strftime("%H:%M")
            print(
                f"  {t['direction']:5s} {t['strategy']:14s} {et}->{xt}  "
                f"₹{t['entry_premium']:.1f}->₹{t['exit_premium']:.1f}  "
                f"{t['reason']:12s}  ₹{t['pnl']:+,.0f}  [{tag}]  ({t['lots']} lot)"
            )
    wr = result["wins"] / result["trade_count"] * 100 if result["trade_count"] else 0
    print(
        f"\n  Day P&L: ₹{result['pnl']:+,.0f}  |  "
        f"Trades: {result['trade_count']} ({result['wins']}W/{result['losses']}L, {wr:.0f}% WR)  |  "
        f"Signals taken: {len(result['signals'])}"
    )


def main() -> None:
    print(f"\n{'=' * 72}")
    print(f"  JUN 16 2026 COUNTERFACTUAL PAPER TEST")
    print(f"  Capital: ₹{START_CAPITAL:,.0f}  |  Lot size: {settings.NIFTY_LOT_SIZE}  |  "
          f"Lots/day: {max(1, min(int(START_CAPITAL / settings.CAPITAL_PER_LOT), settings.MAX_LOTS_CAP))}")
    print(f"{'=' * 72}")

    df = _load_df()
    today_bars = df[df.index.date == TARGET]
    print(
        f"\n  Data: {len(today_bars)} bars  "
        f"({today_bars.index[0].strftime('%H:%M')} – {today_bars.index[-1].strftime('%H:%M')})  "
        f"Nifty {today_bars['close'].iloc[0]:,.0f} → {today_bars['close'].iloc[-1]:,.0f}  "
        f"({today_bars['close'].iloc[-1] - today_bars['close'].iloc[0]:+.0f} pts)"
    )

    base_disabled = set(MultiStrategyEngine.DISABLED_STRATEGIES)
    all_enabled = base_disabled - {
        "ADX_BREAKOUT", "ORB_BREAKOUT", "SUPERTREND", "VWAP_MOMENTUM",
    }

    scenarios = [
        None,  # filled with actual live below
        simulate_day(
            df,
            label="FULL DAY @ 9:15 — 1 lot (matches live risk approval)",
            entry_not_before=time(9, 15),
            lots_override=1,
        ),
        simulate_day(
            df,
            label="LATE START @ 11:26 — 1 lot (today's late boot)",
            entry_not_before=LATE_START,
            lots_override=1,
        ),
        simulate_day(
            df,
            label="FULL DAY — re-enabled trend strategies, 1 lot",
            entry_not_before=time(9, 15),
            disabled_override=all_enabled,
            lots_override=1,
        ),
        simulate_day(
            df,
            label="FULL DAY — 2 slots + pyramiding, 1 lot each",
            entry_not_before=time(9, 15),
            max_sim_override=2,
            allow_pyramid=True,
            lots_override=1,
        ),
    ]

    # Actual live was a single manual path — inject known result
    scenarios[0] = {
        "label": "ACTUAL LIVE (what happened)",
        "pnl": ACTUAL_PNL,
        "trades": [{
            "strategy": "TREND_RIDE",
            "direction": "LONG",
            "entry_time": datetime(2026, 6, 16, 13, 45),
            "exit_time": datetime(2026, 6, 16, 14, 45),
            "entry_premium": 103.02,
            "exit_premium": 114.16,
            "pnl": ACTUAL_PNL,
            "reason": "TRAIL",
            "lots": 1,
            "qty": settings.NIFTY_LOT_SIZE,
        }],
        "trade_count": 1,
        "wins": 1,
        "losses": 0,
        "signals": [{"time": "13:40", "strategy": "TREND_RIDE", "direction": "LONG", "confidence": 74}],
        "final_capital": START_CAPITAL + ACTUAL_PNL,
    }

    for result in scenarios:
        if result is None:
            continue
        print_scenario(result)

    full = scenarios[1]
    late = scenarios[2]
    actual = scenarios[0]

    print(f"\n{'=' * 72}")
    print("  OPPORTUNITY GAP")
    print(f"{'=' * 72}")
    print(f"  Actual live P&L          : ₹{actual['pnl']:+,.0f}")
    print(f"  Full-day paper (engine)  : ₹{full['pnl']:+,.0f}  "
          f"(missed ₹{full['pnl'] - actual['pnl']:+,.0f} vs live)")
    print(f"  Late-start paper         : ₹{late['pnl']:+,.0f}  "
          f"(what 11:26 start alone explains)")
    print(f"  Morning miss (full − late): ₹{full['pnl'] - late['pnl']:+,.0f}")
    print(f"  Best scenario (aggressive): ₹{scenarios[4]['pnl']:+,.0f}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
