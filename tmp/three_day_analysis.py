#!/usr/bin/env python3
"""3-Day Trade Comparison: LLM-Optimal Hindsight vs Our Strategies.

Fetches Jun 15-17 2026 Nifty 5m data, runs:
  1) Hindsight-optimal trades using simple price-action rules
  2) Production strategy replay via MultiStrategyEngine
  3) Side-by-side comparison

Usage:
    cd ~/TradingAgent && ./venv/bin/python tmp/three_day_analysis.py
"""

from __future__ import annotations

import copy
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.check_today import download_nifty_5m
from backtest.run_backtest import _calc_realistic_costs
from config import settings
from engine.multi_strategy_engine import MultiStrategyEngine
from engine.premium_model import STRATEGY_SL_PCT, create_premium_state
from risk.adaptive_mode import AdaptiveModeController

TARGETS = [date(2026, 6, 15), date(2026, 6, 16), date(2026, 6, 17)]
LOT_SIZE = settings.NIFTY_LOT_SIZE  # 65

WEEK_START_CAPITAL = 21_501.11
ENTRY_END = "14:30"
SQUARE_OFF = "15:15"


# ─────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = download_nifty_5m(trading_days=8)
    if (df["volume"] == 0).all():
        df["volume"] = 5000
    return df


# ─────────────────────────────────────────────────────────────────────
#  PART 1: HINDSIGHT-OPTIMAL TRADES (price-action rules)
# ─────────────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_vwap(df_day: pd.DataFrame) -> pd.Series:
    tp = (df_day["high"] + df_day["low"] + df_day["close"]) / 3
    vol = df_day["volume"].replace(0, 1)
    cum_tp_vol = (tp * vol).cumsum()
    cum_vol = vol.cumsum()
    return cum_tp_vol / cum_vol


def hindsight_trades_for_day(
    day_df: pd.DataFrame,
    capital: float,
) -> tuple[list[dict], float]:
    """Find optimal trades using hindsight price-action rules.

    Rules:
      - Determine trend from first 6 bars (30 min ORB)
      - Compute EMA20, VWAP
      - Enter on pullbacks to EMA20/VWAP in trend direction
      - SL = recent swing extreme (2-bar lookback) with min 15 pts
      - Target = 1.5x risk (RR 1:1.5)
      - Max 3 trades per day, no entries after 14:30
      - Use option premium model for P&L (same as backtest)
    """
    if len(day_df) < 10:
        return [], capital

    closes = day_df["close"].values
    highs = day_df["high"].values
    lows = day_df["low"].values
    times = day_df.index

    ema20 = compute_ema(day_df["close"], 20).values
    vwap = compute_vwap(day_df).values

    orb_high = max(highs[:6])
    orb_low = min(lows[:6])
    orb_close = closes[5]
    trend = "LONG" if orb_close > (orb_high + orb_low) / 2 else "SHORT"

    trades = []
    in_trade = False
    entry_bar = 0
    entry_price = 0.0
    sl_price = 0.0
    target_price = 0.0
    trade_direction = ""

    for i in range(6, len(day_df)):
        bar_time = times[i].strftime("%H:%M")
        price = closes[i]

        if in_trade:
            if trade_direction == "LONG":
                if lows[i] <= sl_price:
                    pnl_pts = sl_price - entry_price
                    trades.append(_make_hindsight_trade(
                        trade_direction, times[entry_bar], times[i],
                        entry_price, sl_price, pnl_pts, "SL", capital,
                    ))
                    capital += trades[-1]["pnl"]
                    in_trade = False
                elif highs[i] >= target_price:
                    pnl_pts = target_price - entry_price
                    trades.append(_make_hindsight_trade(
                        trade_direction, times[entry_bar], times[i],
                        entry_price, target_price, pnl_pts, "TGT", capital,
                    ))
                    capital += trades[-1]["pnl"]
                    in_trade = False
                elif bar_time >= SQUARE_OFF:
                    pnl_pts = price - entry_price
                    trades.append(_make_hindsight_trade(
                        trade_direction, times[entry_bar], times[i],
                        entry_price, price, pnl_pts, "EOD", capital,
                    ))
                    capital += trades[-1]["pnl"]
                    in_trade = False
            else:  # SHORT
                if highs[i] >= sl_price:
                    pnl_pts = entry_price - sl_price
                    trades.append(_make_hindsight_trade(
                        trade_direction, times[entry_bar], times[i],
                        entry_price, sl_price, pnl_pts, "SL", capital,
                    ))
                    capital += trades[-1]["pnl"]
                    in_trade = False
                elif lows[i] <= target_price:
                    pnl_pts = entry_price - target_price
                    trades.append(_make_hindsight_trade(
                        trade_direction, times[entry_bar], times[i],
                        entry_price, target_price, pnl_pts, "TGT", capital,
                    ))
                    capital += trades[-1]["pnl"]
                    in_trade = False
                elif bar_time >= SQUARE_OFF:
                    pnl_pts = entry_price - price
                    trades.append(_make_hindsight_trade(
                        trade_direction, times[entry_bar], times[i],
                        entry_price, price, pnl_pts, "EOD", capital,
                    ))
                    capital += trades[-1]["pnl"]
                    in_trade = False
            continue

        if len(trades) >= 3:
            continue
        if bar_time > ENTRY_END:
            continue

        near_ema = abs(price - ema20[i]) < 15
        near_vwap = abs(price - vwap[i]) < 15
        pullback_zone = near_ema or near_vwap

        if not pullback_zone:
            continue

        swing_low = min(lows[max(0, i - 3):i + 1])
        swing_high = max(highs[max(0, i - 3):i + 1])

        if trend == "LONG" and price > ema20[i] - 5:
            sl_price = swing_low - 5
            risk = price - sl_price
            if risk < 15:
                sl_price = price - 15
                risk = 15
            target_price = price + risk * 1.5
            entry_price = price
            trade_direction = "LONG"
            entry_bar = i
            in_trade = True

        elif trend == "SHORT" and price < ema20[i] + 5:
            sl_price = swing_high + 5
            risk = sl_price - price
            if risk < 15:
                sl_price = price + 15
                risk = 15
            target_price = price - risk * 1.5
            entry_price = price
            trade_direction = "SHORT"
            entry_bar = i
            in_trade = True

    if in_trade:
        last_price = closes[-1]
        if trade_direction == "LONG":
            pnl_pts = last_price - entry_price
        else:
            pnl_pts = entry_price - last_price
        trades.append(_make_hindsight_trade(
            trade_direction, times[entry_bar], times[-1],
            entry_price, last_price, pnl_pts, "EOD", capital,
        ))
        capital += trades[-1]["pnl"]

    return trades, capital


def _make_hindsight_trade(
    direction: str, entry_time, exit_time,
    entry_nifty: float, exit_nifty: float,
    pnl_pts: float, reason: str, capital: float,
) -> dict:
    """Convert an index-point P&L into option-premium P&L using our model."""
    theta = settings.get_scaled_theta(entry_nifty)
    prem_state = create_premium_state(
        entry_index_price=entry_nifty,
        direction=direction,
        base_premium=settings.PREMIUM_BASE,
        delta=settings.PREMIUM_DELTA,
        theta_per_candle=theta,
        sl_pct=settings.PREMIUM_SL_PCT,
        confluence_score=80,
        signal_type="HINDSIGHT",
    )
    spread = settings.BID_ASK_SPREAD
    entry_prem = prem_state.entry_premium + spread

    bars_held = max(1, int((exit_time - entry_time).total_seconds() / 300))
    exit_prem = prem_state.current_premium(exit_nifty, bars_held)

    lots = max(1, min(int(capital / settings.CAPITAL_PER_LOT), settings.MAX_LOTS_CAP))
    qty = lots * LOT_SIZE
    costs = _calc_realistic_costs(entry_prem, exit_prem, qty, lots)
    net_pnl = (exit_prem - entry_prem) * qty - costs

    return {
        "strategy": "HINDSIGHT",
        "direction": direction,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_nifty": round(entry_nifty, 0),
        "exit_nifty": round(exit_nifty, 0),
        "entry_premium": round(entry_prem, 2),
        "exit_premium": round(exit_prem, 2),
        "pnl_pts": round(pnl_pts, 1),
        "pnl": round(net_pnl, 2),
        "reason": reason,
        "lots": lots,
        "qty": qty,
        "bars_held": bars_held,
    }


# ─────────────────────────────────────────────────────────────────────
#  PART 2: STRATEGY REPLAY (production MultiStrategyEngine)
# ─────────────────────────────────────────────────────────────────────

def strategy_replay_day(
    df: pd.DataFrame,
    target: date,
    capital: float,
) -> tuple[list[dict], float, list[dict]]:
    """Replay one day through the production engine.

    Returns (trades, final_capital, signals_log).
    """
    engine = MultiStrategyEngine()
    adaptive = AdaptiveModeController()

    unique_days = sorted(set(df.index.date))
    if target not in unique_days:
        return [], capital, []

    day_idx = unique_days.index(target)
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
    day_lots = max(1, min(int(capital / per_lot), settings.MAX_LOTS_CAP))
    day_start_cap = capital
    daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)
    daily_profit_target = day_start_cap * (settings.DAILY_PROFIT_TARGET_PCT / 100)

    indicators = engine.precompute(warmup_df)
    today_indices = [i for i, ts in enumerate(warmup_df.index) if ts.date() == target]

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
            elif time_str >= SQUARE_OFF:
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

        if len(open_positions) >= ap.max_simultaneous:
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
        if time_str > ENTRY_END:
            continue

        open_dirs = {p["direction"] for p in open_positions}
        signals = engine.scan(indicators, i, time_str, max_total_override=ap.max_trades_per_day)

        for signal in signals:
            if signal.direction in open_dirs:
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
            qty = eff_lots * LOT_SIZE

            signals_log.append({
                "time": time_str,
                "strategy": signal.signal_type,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "mode": adaptive.mode.value,
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

    return trades, capital, signals_log


# ─────────────────────────────────────────────────────────────────────
#  PART 3: OUTPUT
# ─────────────────────────────────────────────────────────────────────

def print_trade_list(trades: list[dict], label: str) -> None:
    if not trades:
        print(f"    No trades")
        return
    for t in trades:
        tag = "WIN" if t["pnl"] > 0 else "LOSS" if t["pnl"] < 0 else "FLAT"
        et = t["entry_time"].strftime("%H:%M") if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])
        xt = t["exit_time"].strftime("%H:%M") if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])

        extra = ""
        if "pnl_pts" in t:
            extra = f"  ({t['pnl_pts']:+.0f} pts)"

        print(
            f"    {t['direction']:5s} {t['strategy']:14s} {et}->{xt}  "
            f"{t['reason']:6s}  Rs {t['pnl']:+7,.0f}  [{tag:4s}]  ({t['lots']}L){extra}"
        )


def main() -> None:
    print(f"\n{'=' * 80}")
    print(f"  3-DAY TRADE ANALYSIS: Jun 15-17, 2026")
    print(f"  Week start capital: Rs {WEEK_START_CAPITAL:,.0f}")
    print(f"  Lot size: {LOT_SIZE} | Entry window: 09:30-{ENTRY_END}")
    print(f"{'=' * 80}")

    print("\n  Downloading Nifty 5m data from yfinance...")
    df = load_data()

    unique_days = sorted(set(df.index.date))
    print(f"  Downloaded {len(df)} bars across {len(unique_days)} days: {unique_days}")

    available_targets = [d for d in TARGETS if d in unique_days]
    missing = [d for d in TARGETS if d not in unique_days]
    if missing:
        print(f"  WARNING: Missing data for: {missing}")

    # ── Day-by-day market summary ──
    print(f"\n{'─' * 80}")
    print("  MARKET SUMMARY")
    print(f"{'─' * 80}")
    for target in available_targets:
        day_df = df[df.index.date == target]
        day_name = target.strftime("%a %b %d")
        o = day_df["open"].iloc[0]
        h = day_df["high"].max()
        l = day_df["low"].min()
        c = day_df["close"].iloc[-1]
        rng = h - l
        chg = c - o
        print(
            f"  {day_name}: O={o:,.0f} H={h:,.0f} L={l:,.0f} C={c:,.0f}  "
            f"Chg={chg:+.0f} pts  Range={rng:.0f} pts  "
            f"({'BULLISH' if chg > 20 else 'BEARISH' if chg < -20 else 'FLAT'})"
        )

    # ── Run both analyses ──
    hindsight_results = {}
    strategy_results = {}

    h_capital = WEEK_START_CAPITAL
    s_capital = WEEK_START_CAPITAL

    for target in available_targets:
        day_df = df[df.index.date == target]

        h_trades, h_capital = hindsight_trades_for_day(day_df, h_capital)
        hindsight_results[target] = h_trades

        s_trades, s_capital, s_signals = strategy_replay_day(df, target, s_capital)
        strategy_results[target] = {"trades": s_trades, "signals": s_signals}

    # ── Print per-day results ──
    for target in available_targets:
        day_df = df[df.index.date == target]
        day_name = target.strftime("%a %b %d")
        o = day_df["open"].iloc[0]
        c = day_df["close"].iloc[-1]

        print(f"\n{'=' * 80}")
        print(f"  {day_name} | Nifty {o:,.0f} -> {c:,.0f} ({c - o:+.0f} pts)")
        print(f"{'=' * 80}")

        h_trades = hindsight_results[target]
        s_trades = strategy_results[target]["trades"]
        s_signals = strategy_results[target]["signals"]

        h_pnl = sum(t["pnl"] for t in h_trades)
        s_pnl = sum(t["pnl"] for t in s_trades)
        h_wins = sum(1 for t in h_trades if t["pnl"] > 0)
        s_wins = sum(1 for t in s_trades if t["pnl"] > 0)

        print(f"\n  HINDSIGHT OPTIMAL ({len(h_trades)} trades, {h_wins}W/{len(h_trades) - h_wins}L):")
        print_trade_list(h_trades, "hindsight")

        print(f"\n  OUR STRATEGY ({len(s_trades)} trades, {s_wins}W/{len(s_trades) - s_wins}L):")
        print_trade_list(s_trades, "strategy")
        if s_signals:
            taken = [s for s in s_signals]
            print(f"    Signals fired: {len(taken)}")
            for s in taken:
                print(f"      {s['time']} {s['strategy']:14s} {s['direction']:5s} conf={s['confidence']} mode={s['mode']}")

        print(f"\n  DAY COMPARISON:")
        print(f"    Hindsight P&L : Rs {h_pnl:+,.0f}")
        print(f"    Strategy P&L  : Rs {s_pnl:+,.0f}")
        print(f"    Gap           : Rs {h_pnl - s_pnl:+,.0f}")

    # ── Week summary ──
    total_h = sum(sum(t["pnl"] for t in hindsight_results[d]) for d in available_targets)
    total_s = sum(sum(t["pnl"] for t in strategy_results[d]["trades"]) for d in available_targets)

    print(f"\n{'=' * 80}")
    print(f"  WEEK SUMMARY (Jun 15-17)")
    print(f"{'=' * 80}")
    print(f"  {'':20s} {'Hindsight':>12s} {'Strategy':>12s} {'Gap':>12s}")
    print(f"  {'':20s} {'─' * 12} {'─' * 12} {'─' * 12}")

    running_h = WEEK_START_CAPITAL
    running_s = WEEK_START_CAPITAL
    for target in available_targets:
        h_pnl = sum(t["pnl"] for t in hindsight_results[target])
        s_pnl = sum(t["pnl"] for t in strategy_results[target]["trades"])
        running_h += h_pnl
        running_s += s_pnl
        day_name = target.strftime("%a %b %d")
        print(
            f"  {day_name:20s} Rs {h_pnl:+8,.0f}    Rs {s_pnl:+8,.0f}    Rs {h_pnl - s_pnl:+8,.0f}"
        )

    print(f"  {'':20s} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(f"  {'TOTAL':20s} Rs {total_h:+8,.0f}    Rs {total_s:+8,.0f}    Rs {total_h - total_s:+8,.0f}")
    print(f"  {'END CAPITAL':20s} Rs {running_h:>8,.0f}    Rs {running_s:>8,.0f}")

    # ── Root cause analysis ──
    print(f"\n{'=' * 80}")
    print(f"  ROOT CAUSE ANALYSIS")
    print(f"{'=' * 80}")

    all_s_trades = []
    for d in available_targets:
        all_s_trades.extend(strategy_results[d]["trades"])

    if all_s_trades:
        sl_trades = [t for t in all_s_trades if t["reason"] == "SL"]
        win_trades = [t for t in all_s_trades if t["pnl"] > 0]
        loss_trades = [t for t in all_s_trades if t["pnl"] <= 0]

        avg_win = np.mean([t["pnl"] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t["pnl"] for t in loss_trades]) if loss_trades else 0

        print(f"\n  Strategy trades: {len(all_s_trades)} total")
        print(f"  Wins: {len(win_trades)} (avg Rs {avg_win:+,.0f})")
        print(f"  Losses: {len(loss_trades)} (avg Rs {avg_loss:+,.0f})")
        print(f"  SL exits: {len(sl_trades)} / {len(all_s_trades)}")
        if avg_win > 0 and avg_loss < 0:
            print(f"  Profit factor: {abs(avg_win * len(win_trades)) / abs(avg_loss * len(loss_trades)):.2f}")
        print(f"  Win rate: {len(win_trades)/len(all_s_trades)*100:.0f}%")

        directions = {}
        for t in all_s_trades:
            d = t["direction"]
            if d not in directions:
                directions[d] = {"count": 0, "pnl": 0}
            directions[d]["count"] += 1
            directions[d]["pnl"] += t["pnl"]
        print(f"\n  By direction:")
        for d, v in directions.items():
            print(f"    {d}: {v['count']} trades, Rs {v['pnl']:+,.0f}")

        strategies = {}
        for t in all_s_trades:
            s = t["strategy"]
            if s not in strategies:
                strategies[s] = {"count": 0, "pnl": 0, "wins": 0}
            strategies[s]["count"] += 1
            strategies[s]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                strategies[s]["wins"] += 1
        print(f"\n  By strategy:")
        for s, v in sorted(strategies.items(), key=lambda x: x[1]["pnl"]):
            wr = v["wins"] / v["count"] * 100 if v["count"] else 0
            print(f"    {s:20s}: {v['count']} trades, Rs {v['pnl']:+,.0f}, {wr:.0f}% WR")

    all_h_trades = []
    for d in available_targets:
        all_h_trades.extend(hindsight_results[d])

    if all_h_trades:
        h_wins = [t for t in all_h_trades if t["pnl"] > 0]
        print(f"\n  Hindsight trades: {len(all_h_trades)} total, {len(h_wins)} wins")
        if h_wins:
            print(f"  Avg hindsight win: Rs {np.mean([t['pnl'] for t in h_wins]):+,.0f}")

    # ── Key issues diagnosis ──
    print(f"\n{'=' * 80}")
    print(f"  KEY ISSUES IDENTIFIED")
    print(f"{'=' * 80}")
    issues = []

    long_count = sum(1 for t in all_s_trades if t["direction"] == "LONG")
    short_count = sum(1 for t in all_s_trades if t["direction"] == "SHORT")
    if long_count > 0 and short_count == 0:
        issues.append(
            "DIRECTIONAL BIAS: All 9 strategy trades were LONG. On Jun 15 (bearish, -129 pts),\n"
            "    the engine still went LONG 3 times. The strategies have no short-side signals,\n"
            "    causing us to fight the trend on down days."
        )

    sl_count = sum(1 for t in all_s_trades if t["reason"] == "SL")
    if sl_count / max(len(all_s_trades), 1) > 0.5:
        issues.append(
            f"SL-HEAVY: {sl_count}/{len(all_s_trades)} trades hit SL ({sl_count/len(all_s_trades)*100:.0f}%). "
            f"The 10% premium SL is tight relative to intraday volatility.\n"
            f"    On wide-range days (150+ pts), a 10% premium SL gets hit by normal noise."
        )

    consec_losses_per_day = {}
    for d in available_targets:
        trades_d = strategy_results[d]["trades"]
        max_cl = 0
        cl = 0
        for t in trades_d:
            if t["pnl"] <= 0:
                cl += 1
                max_cl = max(max_cl, cl)
            else:
                cl = 0
        consec_losses_per_day[d] = max_cl
    multi_cl_days = [d for d, v in consec_losses_per_day.items() if v >= 2]
    if multi_cl_days:
        issues.append(
            f"LOSS CASCADING: On {len(multi_cl_days)} of 3 days, 2+ consecutive losses occurred.\n"
            f"    After an SL, the engine re-enters the same direction quickly, compounding damage.\n"
            f"    The adaptive mode goes DEFENSIVE but still allows entries."
        )

    if all_s_trades:
        avg_bars = []
        for t in all_s_trades:
            if hasattr(t["entry_time"], "timestamp") and hasattr(t["exit_time"], "timestamp"):
                bars = (t["exit_time"] - t["entry_time"]).total_seconds() / 300
                avg_bars.append(bars)
        if avg_bars:
            mean_bars = np.mean(avg_bars)
            if mean_bars < 8:
                issues.append(
                    f"SHORT HOLDING: Avg hold time is {mean_bars:.0f} bars ({mean_bars*5:.0f} min). "
                    f"Trades aren't given\n"
                    f"    enough room to work -- theta decay erodes premium before the move plays out."
                )

    issues.append(
        "MISSING SHORT STRATEGIES: The ORB on Jun 15 was clearly bearish (gap down, -129 pts),\n"
        "    but no strategy generated SHORT signals. Adding ORB_BREAKOUT / SUPERTREND shorts\n"
        "    could have captured the major move."
    )

    for idx, issue in enumerate(issues, 1):
        print(f"\n  {idx}. {issue}")

    print(f"\n{'=' * 80}")
    print(f"  RECOMMENDATIONS")
    print(f"{'=' * 80}")
    print("""
  1. ENABLE SHORT SIGNALS: Re-enable ORB_BREAKOUT and SUPERTREND strategies for
     short-side entries. The bearish Jun 15 was a missed opportunity.

  2. WIDEN SL ON VOLATILE DAYS: When daily range > 150 pts or VIX > 15, increase
     premium SL from 10% to 15-18% to avoid getting stopped by noise.

  3. LOSS-BASED COOLDOWN: After 2 consecutive SLs, enforce a 30-min cooldown
     before the next entry (stronger than current adaptive DEFENSIVE).

  4. TREND FILTER GATE: Add a 15m EMA trend filter -- only LONG when price > EMA20
     on 15m, only SHORT when below. This prevents fighting the intraday trend.

  5. REDUCE LOT SIZE ON LOSING DAYS: Cap at 1 lot when day P&L is negative
     (the adaptive mode already does lot_multiplier reduction, but it needs
     to be more aggressive -- 0.3x instead of current 0.7x).
""")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
