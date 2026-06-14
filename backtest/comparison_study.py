"""Position-management comparison study — 4 scenarios side-by-side.

Runs baseline, tightened trail, reduced-risk stop, and combined configs
on identical signal entries with withdrawal logic and peak-premium tracking.

Usage:
    python -m backtest.comparison_study
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.multi_strategy_engine import MultiStrategyEngine
from engine.premium_model import STRATEGY_SL_PCT, create_premium_state
from backtest.run_backtest import (
    _calc_realistic_costs,
    generate_nifty_data,
    load_real_data,
)

DATA_DIR = PROJECT_ROOT / "data"
STUDY_DAYS = 100
STARTING_CAPITAL = 10_000
WITHDRAWAL_THRESHOLD = 100_000
WITHDRAWAL_AMOUNT = 30_000


def load_study_data(days: int = STUDY_DAYS) -> pd.DataFrame:
    """Load Nifty 5m data: yfinance → CSV → synthetic fallback."""
    # 1. Yahoo Finance (recent 60d max for 5m)
    try:
        import yfinance as yf

        raw = yf.Ticker("^NSEI").history(period="60d", interval="5m")
        if raw is not None and not raw.empty:
            df = raw.copy()
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[["open", "high", "low", "close", "volume"]].sort_index()
            unique_days = sorted(set(df.index.date))
            if len(unique_days) >= days:
                logger.info(
                    "Loaded {} candles / {} days from Yahoo Finance",
                    len(df), len(unique_days),
                )
                selected = set(unique_days[-days:])
                return df[df.index.map(lambda t: t.date() in selected)]
            logger.debug(
                "Yahoo Finance only has {} days (need {}), trying other sources",
                len(unique_days), days,
            )
    except Exception as exc:
        logger.debug("Yahoo Finance unavailable: {}", exc)

    # 2. Local CSV sources via load_real_data
    df = load_real_data(days=days)
    if not df.empty:
        unique_days = sorted(set(df.index.date))
        if len(unique_days) >= days:
            logger.info(
                "Loaded {} candles / {} days from local CSV",
                len(df), len(unique_days),
            )
            selected = set(unique_days[-days:])
            return df[df.index.map(lambda t: t.date() in selected)]
        if len(unique_days) >= 30:
            logger.info(
                "Local CSV has {} days (requested {}), using available data",
                len(unique_days), days,
            )
            return df

    # 3. Synthetic fallback (reproducible seed=42)
    logger.info("Falling back to synthetic data ({} days)", days)
    return generate_nifty_data(days=days)


def _apply_withdrawals(capital: float, total_withdrawn: float,
                        threshold: float, amount: float) -> tuple[float, float]:
    """Withdraw fixed amount each time capital crosses threshold."""
    while capital >= threshold:
        capital -= amount
        total_withdrawn += amount
    return capital, total_withdrawn


def _finalize_trade(pos: dict, exit_reason: str, exit_prem: float,
                    ts, day_lots: int, capital: float,
                    total_withdrawn: float,
                    withdrawal_threshold: float,
                    withdrawal_amount: float) -> tuple[dict, float, float]:
    """Close a position and return trade dict + updated capital/withdrawn."""
    costs = _calc_realistic_costs(
        pos["entry_premium"], exit_prem, pos["qty"], day_lots,
    )
    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
    net_pnl = raw_pnl - costs
    capital += net_pnl

    peak_gain_pct = (
        (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100
    )
    was_green_then_red = peak_gain_pct > 8 and exit_reason == "SL"
    peak_before_sl = peak_gain_pct if exit_reason == "SL" else None

    breakeven_exit = (
        exit_reason == "SL" and pos.get("reduced_risk_active", False)
    )
    reason = "BREAKEVEN" if breakeven_exit else exit_reason

    capital, total_withdrawn = _apply_withdrawals(
        capital, total_withdrawn, withdrawal_threshold, withdrawal_amount,
    )

    trade = {
        "strategy": pos["signal_type"],
        "signal": pos["direction"],
        "entry_time": pos["entry_time"],
        "exit_time": ts,
        "entry_premium": round(pos["entry_premium"], 2),
        "exit_premium": round(exit_prem, 2),
        "peak_premium": round(pos["peak_premium"], 2),
        "peak_gain_pct": round(peak_gain_pct, 2),
        "was_green_then_red": was_green_then_red,
        "peak_before_sl": round(peak_before_sl, 2) if peak_before_sl is not None else None,
        "qty": pos["qty"],
        "lots": day_lots,
        "pnl": round(net_pnl, 0),
        "reason": reason,
        "capital_after": round(capital, 0),
        "total_withdrawn": round(total_withdrawn, 0),
    }
    return trade, capital, total_withdrawn


def _update_position_exits(
    pos: dict,
    cur_prem: float,
    time_str: str,
    trail_trigger_pct: float,
    trail_pct: float,
    reduced_risk_enabled: bool,
    reduced_risk_threshold_pct: float,
    reduced_risk_sl_pct: float,
) -> tuple[str | None, float]:
    """Apply peak tracking, reduced-risk SL, trail, and return exit if any."""
    if cur_prem > pos["peak_premium"]:
        pos["peak_premium"] = cur_prem

    if reduced_risk_enabled:
        peak_gain_pct = (
            (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100
        )
        if peak_gain_pct >= reduced_risk_threshold_pct:
            new_sl = pos["entry_premium"] * (1 - reduced_risk_sl_pct / 100)
            if new_sl > pos["sl_premium"]:
                pos["sl_premium"] = new_sl
                pos["reduced_risk_active"] = True

    trail_floor = pos["prem_state"].update_trail(
        cur_prem, trigger_pct=trail_trigger_pct, trail_pct=trail_pct,
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

    return exit_reason, exit_prem


def capture_entry_schedule(
    df: pd.DataFrame,
    starting_capital: float = STARTING_CAPITAL,
    lot_size: int | None = None,
) -> list[dict]:
    """Run baseline entry logic once; return chronological entry schedule."""
    if lot_size is None:
        lot_size = settings.NIFTY_LOT_SIZE

    engine = MultiStrategyEngine()
    capital = starting_capital
    schedule: list[dict] = []
    unique_days = sorted(set(df.index.date))

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
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

        day_start_cap = capital
        day_pnl = 0
        consec_loss = 0
        daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)

        per_lot = getattr(settings, "CAPITAL_PER_LOT", 10_000)
        day_lots = max(1, int(capital / per_lot))
        day_lots = min(day_lots, getattr(settings, "MAX_LOTS_CAP", 10))

        indicators = engine.precompute(day_df)
        open_positions: list[dict] = []

        for i in range(10, len(day_df)):
            ts = day_df.index[i]
            time_str = ts.strftime("%H:%M")
            nifty_price = day_df["close"].iloc[i]

            closed_this_bar: list[dict] = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(
                    nifty_price, pos["candles_held"],
                )
                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                trail_floor = pos["prem_state"].update_trail(
                    cur_prem, trigger_pct=12, trail_pct=8,
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
                    costs = _calc_realistic_costs(
                        pos["entry_premium"], exit_prem, pos["qty"], day_lots,
                    )
                    net_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"] - costs
                    capital += net_pnl
                    day_pnl += net_pnl
                    consec_loss = consec_loss + 1 if net_pnl < 0 else 0
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

            if len(open_positions) >= 2:
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
                if signal.confidence < getattr(settings, "PULLBACK_MIN_CONFIDENCE", 50):
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
                eff_sl = STRATEGY_SL_PCT.get(
                    signal.signal_type, settings.PREMIUM_SL_PCT,
                )
                sl_premium = entry_premium * (1 - eff_sl / 100)
                qty = day_lots * lot_size

                schedule.append({
                    "day": day,
                    "bar_idx": i,
                    "entry_time": ts,
                    "direction": signal.direction,
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "nifty_price": nifty_price,
                    "entry_premium": entry_premium,
                    "sl_premium": sl_premium,
                    "qty": qty,
                    "day_lots": day_lots,
                })

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
                })
                break

            if capital <= 0:
                break

        if capital <= 0:
            break

    return schedule


def _open_position_from_entry(entry: dict, lot_size: int) -> dict:
    """Recreate a live position dict from a schedule entry."""
    theta = settings.get_scaled_theta(entry["nifty_price"])
    prem_state = create_premium_state(
        entry_index_price=entry["nifty_price"],
        direction=entry["direction"],
        base_premium=settings.PREMIUM_BASE,
        delta=settings.PREMIUM_DELTA,
        theta_per_candle=theta,
        sl_pct=settings.PREMIUM_SL_PCT,
        confluence_score=entry["confidence"],
        signal_type=entry["signal_type"],
    )
    spread = getattr(settings, "BID_ASK_SPREAD", 0.30)
    entry_premium = prem_state.entry_premium + spread
    eff_sl = STRATEGY_SL_PCT.get(entry["signal_type"], settings.PREMIUM_SL_PCT)
    sl_premium = entry_premium * (1 - eff_sl / 100)

    return {
        "direction": entry["direction"],
        "signal_type": entry["signal_type"],
        "entry_time": entry["entry_time"],
        "entry_premium": entry_premium,
        "sl_premium": sl_premium,
        "qty": entry["qty"],
        "prem_state": prem_state,
        "candles_held": 0,
        "peak_premium": entry_premium,
        "reduced_risk_active": False,
    }


def run_scenario(
    df: pd.DataFrame,
    scenario_name: str,
    starting_capital: float,
    trail_trigger_pct: float,
    trail_pct: float,
    reduced_risk_enabled: bool,
    reduced_risk_threshold_pct: float,
    reduced_risk_sl_pct: float,
    withdrawal_threshold: float,
    withdrawal_amount: float,
    entry_schedule: list[dict],
    lot_size: int | None = None,
) -> dict:
    """Replay fixed entry schedule with scenario-specific exit management."""
    if lot_size is None:
        lot_size = settings.NIFTY_LOT_SIZE

    capital = starting_capital
    total_withdrawn = 0.0
    peak_capital = capital
    trades: list[dict] = []
    equity_curve: list[dict] = []

    schedule_by_day: dict = {}
    for entry in entry_schedule:
        schedule_by_day.setdefault(entry["day"], []).append(entry)
    for day_entries in schedule_by_day.values():
        day_entries.sort(key=lambda e: e["entry_time"])

    unique_days = sorted(set(df.index.date))

    for day in unique_days:
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            effective = capital + total_withdrawn
            equity_curve.append({
                "date": day, "capital": capital, "daily_pnl": 0,
                "trades": 0, "lots": 0, "total_withdrawn": total_withdrawn,
                "effective_wealth": round(effective, 0),
            })
            continue

        day_start_cap = capital
        day_pnl = 0
        day_trades = 0
        day_entries = schedule_by_day.get(day, [])
        entry_ptr = 0
        open_positions: list[dict] = []

        per_lot = getattr(settings, "CAPITAL_PER_LOT", 10_000)
        day_lots = max(1, int(capital / per_lot))
        day_lots = min(day_lots, getattr(settings, "MAX_LOTS_CAP", 10))

        for i in range(10, len(day_df)):
            ts = day_df.index[i]
            time_str = ts.strftime("%H:%M")
            nifty_price = day_df["close"].iloc[i]

            while entry_ptr < len(day_entries) and day_entries[entry_ptr]["bar_idx"] == i:
                pos = _open_position_from_entry(day_entries[entry_ptr], lot_size)
                pos["qty"] = day_entries[entry_ptr]["qty"]
                open_positions.append(pos)
                entry_ptr += 1

            closed_this_bar: list[dict] = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(
                    nifty_price, pos["candles_held"],
                )

                exit_reason, exit_prem = _update_position_exits(
                    pos, cur_prem, time_str,
                    trail_trigger_pct, trail_pct,
                    reduced_risk_enabled,
                    reduced_risk_threshold_pct,
                    reduced_risk_sl_pct,
                )

                if exit_reason:
                    trade, capital, total_withdrawn = _finalize_trade(
                        pos, exit_reason, exit_prem, ts, day_lots, capital,
                        total_withdrawn, withdrawal_threshold, withdrawal_amount,
                    )
                    trades.append(trade)
                    day_pnl += trade["pnl"]
                    day_trades += 1
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

        for pos in open_positions:
            nifty_price = day_df["close"].iloc[-1]
            exit_prem = pos["prem_state"].current_premium(
                nifty_price, pos["candles_held"],
            )
            trade, capital, total_withdrawn = _finalize_trade(
                pos, "EOD", exit_prem, day_df.index[-1], day_lots, capital,
                total_withdrawn, withdrawal_threshold, withdrawal_amount,
            )
            trades.append(trade)
            day_pnl += trade["pnl"]
            day_trades += 1

        effective = capital + total_withdrawn
        if effective > peak_capital:
            peak_capital = effective

        equity_curve.append({
            "date": day,
            "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades,
            "lots": day_lots,
            "total_withdrawn": round(total_withdrawn, 0),
            "effective_wealth": round(effective, 0),
        })

    total_pnl = capital + total_withdrawn - starting_capital
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    eq_effective = [e["effective_wealth"] for e in equity_curve]
    peak_arr = np.maximum.accumulate(eq_effective) if eq_effective else [starting_capital]
    dd_arr = [(p - v) / p * 100 for p, v in zip(peak_arr, eq_effective)]
    max_dd = max(dd_arr) if dd_arr else 0

    sl_trades = [t for t in trades if t["reason"] == "SL"]
    tgt_trades = [t for t in trades if t["reason"] == "TGT"]
    trail_trades = [t for t in trades if t["reason"] == "TRAIL"]
    be_trades = [t for t in trades if t["reason"] == "BREAKEVEN"]

    avg_sl_loss = np.mean([t["pnl"] for t in sl_trades + be_trades]) if (sl_trades or be_trades) else 0
    total_sl_damage = sum(t["pnl"] for t in sl_trades + be_trades)
    avg_trail_gain = np.mean([t["pnl"] for t in trail_trades]) if trail_trades else 0

    return {
        "scenario_name": scenario_name,
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_withdrawn": round(total_withdrawn, 0),
        "effective_wealth": round(capital + total_withdrawn, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / starting_capital * 100, 1),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "sl_exits": len(sl_trades),
        "tgt_exits": len(tgt_trades),
        "trail_exits": len(trail_trades),
        "breakeven_exits": len(be_trades),
        "avg_sl_loss": round(avg_sl_loss, 0),
        "total_sl_damage": round(total_sl_damage, 0),
        "avg_trail_gain": round(avg_trail_gain, 0),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def _fmt_rs(val: float) -> str:
    return f"Rs {val:,.0f}"


def _fmt_pct(val: float) -> str:
    return f"{val:.1f}%"


def print_comparison_table(
    df: pd.DataFrame,
    results: dict[str, dict],
) -> None:
    """Print formatted side-by-side comparison."""
    unique_days = sorted(set(df.index.date))
    start_date = unique_days[0] if unique_days else "N/A"
    end_date = unique_days[-1] if unique_days else "N/A"

    print("\n" + "═" * 71)
    print("  DELTAFORGE POSITION MANAGEMENT COMPARISON STUDY")
    print(
        f"  Data: {len(unique_days)} trading days | {len(df)} candles | "
        f"Start: {start_date} | End: {end_date}"
    )
    print("═" * 71)

    cols = ["A: BASELINE", "B: TIGHT TRAIL", "C: RED-RISK", "D: COMBINED"]
    keys = ["A", "B", "C", "D"]
    r = {k: results[k] for k in keys}

    def row(label: str, values: list[str]) -> None:
        print(f"  {label:<25} │ {values[0]:>11} │ {values[1]:>14} │ "
              f"{values[2]:>11} │ {values[3]:>11}")

    print()
    row("Metric", cols)
    print("  " + "─" * 25 + "┼" + "─" * 13 + "┼" + "─" * 16 + "┼" + "─" * 13 + "┼" + "─" * 13)

    row("Final Capital", [_fmt_rs(r[k]["final_capital"]) for k in keys])
    row("Total Withdrawn", [_fmt_rs(r[k]["total_withdrawn"]) for k in keys])
    row("Effective Wealth", [_fmt_rs(r[k]["effective_wealth"]) for k in keys])
    row("Total Return %", [_fmt_pct(r[k]["return_pct"]) for k in keys])

    print("  " + "─" * 25 + "┼" + "─" * 13 + "┼" + "─" * 16 + "┼" + "─" * 13 + "┼" + "─" * 13)

    row("Total Trades", [str(r[k]["total_trades"]) for k in keys])
    row("Win Rate", [_fmt_pct(r[k]["win_rate"]) for k in keys])
    row("Profit Factor", [str(r[k]["profit_factor"]) for k in keys])
    row("Max Drawdown", [_fmt_pct(r[k]["max_drawdown_pct"]) for k in keys])

    print("  " + "─" * 25 + "┼" + "─" * 13 + "┼" + "─" * 16 + "┼" + "─" * 13 + "┼" + "─" * 13)

    row("SL Exits", [str(r[k]["sl_exits"]) for k in keys])
    row("TGT Exits", [str(r[k]["tgt_exits"]) for k in keys])
    row("TRAIL Exits", [str(r[k]["trail_exits"]) for k in keys])
    row("BREAKEVEN Exits", [str(r[k]["breakeven_exits"]) for k in keys])

    print("  " + "─" * 25 + "┼" + "─" * 13 + "┼" + "─" * 16 + "┼" + "─" * 13 + "┼" + "─" * 13)

    row("Avg SL Loss", [_fmt_rs(r[k]["avg_sl_loss"]) for k in keys])
    row("Total SL Damage", [_fmt_rs(r[k]["total_sl_damage"]) for k in keys])
    row("Avg TRAIL Gain", [_fmt_rs(r[k]["avg_trail_gain"]) for k in keys])
    print()


def analyze_peak_premium(baseline_trades: list[dict]) -> dict:
    """Peak premium analysis from baseline SL trades."""
    sl_trades = [t for t in baseline_trades if t["reason"] == "SL"]
    total_sl = len(sl_trades)

    thresholds = [5, 8, 10, 12, 15]
    threshold_stats = {}
    for thr in thresholds:
        count = sum(1 for t in sl_trades if (t.get("peak_gain_pct") or 0) > thr)
        pct = count / total_sl * 100 if total_sl else 0
        threshold_stats[thr] = {"count": count, "total": total_sl, "pct": pct}

    peaks = [t["peak_gain_pct"] for t in sl_trades if t.get("peak_gain_pct") is not None]
    avg_peak = float(np.mean(peaks)) if peaks else 0.0
    median_peak = float(np.median(peaks)) if peaks else 0.0

    breakeven_sims = {}
    for thr in [8, 10, 15]:
        saved = [t for t in sl_trades if (t.get("peak_gain_pct") or 0) >= thr]
        recovered = sum(abs(t["pnl"]) for t in saved)
        breakeven_sims[thr] = {"saved": len(saved), "recovered": recovered}

    return {
        "threshold_stats": threshold_stats,
        "avg_peak_before_sl": avg_peak,
        "median_peak_before_sl": median_peak,
        "breakeven_sims": breakeven_sims,
        "sl_trades": sl_trades,
    }


def print_peak_premium_analysis(analysis: dict) -> None:
    """Print peak premium section."""
    print("  PEAK PREMIUM ANALYSIS (from Scenario A baseline):")
    print("  " + "─" * 50)

    for thr in [5, 8, 10, 12, 15]:
        st = analysis["threshold_stats"][thr]
        print(
            f"  SL trades that peaked above +{thr}%:  "
            f"{st['count']}/{st['total']} ({st['pct']:.0f}%)"
        )

    print()
    print(f"  Average peak gain before SL:      {analysis['avg_peak_before_sl']:.1f}%")
    print(f"  Median peak gain before SL:       {analysis['median_peak_before_sl']:.1f}%")
    print()

    for thr in [8, 10, 15]:
        sim = analysis["breakeven_sims"][thr]
        print(
            f"  If breakeven at +{thr}%:  {sim['saved']} trades saved, "
            f"{_fmt_rs(sim['recovered'])} recovered"
        )
    print()


def save_outputs(results: dict[str, dict], peak_analysis: dict) -> None:
    """Write trade logs and peak premium CSV."""
    DATA_DIR.mkdir(exist_ok=True)

    for key, label in [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")]:
        path = DATA_DIR / f"comparison_trades_{label}.csv"
        pd.DataFrame(results[key]["trades"]).to_csv(path, index=False)

    rows = []
    for thr in [5, 8, 10, 12, 15]:
        st = peak_analysis["threshold_stats"][thr]
        rows.append({
            "threshold_pct": thr,
            "sl_trades_above": st["count"],
            "total_sl_trades": st["total"],
            "pct": round(st["pct"], 1),
        })
    for thr in [8, 10, 15]:
        sim = peak_analysis["breakeven_sims"][thr]
        rows.append({
            "threshold_pct": f"breakeven_{thr}",
            "sl_trades_above": sim["saved"],
            "total_sl_trades": "",
            "pct": round(sim["recovered"], 0),
        })
    rows.append({
        "threshold_pct": "avg_peak_before_sl",
        "sl_trades_above": round(peak_analysis["avg_peak_before_sl"], 2),
        "total_sl_trades": "",
        "pct": "",
    })
    rows.append({
        "threshold_pct": "median_peak_before_sl",
        "sl_trades_above": round(peak_analysis["median_peak_before_sl"], 2),
        "total_sl_trades": "",
        "pct": "",
    })

    pd.DataFrame(rows).to_csv(DATA_DIR / "peak_premium_analysis.csv", index=False)
    print(f"  Saved: {DATA_DIR}/comparison_trades_{{A,B,C,D}}.csv")
    print(f"  Saved: {DATA_DIR}/peak_premium_analysis.csv")


SCENARIOS = {
    "A": {
        "name": "A: BASELINE",
        "trail_trigger_pct": 12,
        "trail_pct": 8,
        "reduced_risk_enabled": False,
        "reduced_risk_threshold_pct": 15,
        "reduced_risk_sl_pct": 10,
    },
    "B": {
        "name": "B: TIGHT TRAIL",
        "trail_trigger_pct": 10,
        "trail_pct": 6,
        "reduced_risk_enabled": False,
        "reduced_risk_threshold_pct": 15,
        "reduced_risk_sl_pct": 10,
    },
    "C": {
        "name": "C: RED-RISK",
        "trail_trigger_pct": 12,
        "trail_pct": 8,
        "reduced_risk_enabled": True,
        "reduced_risk_threshold_pct": 15,
        "reduced_risk_sl_pct": 10,
    },
    "D": {
        "name": "D: COMBINED",
        "trail_trigger_pct": 10,
        "trail_pct": 6,
        "reduced_risk_enabled": True,
        "reduced_risk_threshold_pct": 15,
        "reduced_risk_sl_pct": 10,
    },
}


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    print(f"\nLoading {STUDY_DAYS} trading days of Nifty data...")
    df = load_study_data(days=STUDY_DAYS)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days")

    lot_size = settings.NIFTY_LOT_SIZE

    print("\nCapturing entry schedule (baseline signal generation)...")
    entry_schedule = capture_entry_schedule(
        df, starting_capital=STARTING_CAPITAL, lot_size=lot_size,
    )
    print(f"  {len(entry_schedule)} entries captured for replay")

    results: dict[str, dict] = {}
    for key, cfg in SCENARIOS.items():
        print(f"Running scenario {cfg['name']}...")
        results[key] = run_scenario(
            df,
            scenario_name=cfg["name"],
            starting_capital=STARTING_CAPITAL,
            trail_trigger_pct=cfg["trail_trigger_pct"],
            trail_pct=cfg["trail_pct"],
            reduced_risk_enabled=cfg["reduced_risk_enabled"],
            reduced_risk_threshold_pct=cfg["reduced_risk_threshold_pct"],
            reduced_risk_sl_pct=cfg["reduced_risk_sl_pct"],
            withdrawal_threshold=WITHDRAWAL_THRESHOLD,
            withdrawal_amount=WITHDRAWAL_AMOUNT,
            entry_schedule=entry_schedule,
            lot_size=lot_size,
        )

    print_comparison_table(df, results)

    peak_analysis = analyze_peak_premium(results["A"]["trades"])
    print_peak_premium_analysis(peak_analysis)

    save_outputs(results, peak_analysis)
