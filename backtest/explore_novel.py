"""Novel CDGR exploration — fundamentally different approaches beyond parameter sweeps.

Usage:
    python backtest/explore_novel.py
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs
from engine import premium_model
from engine.multi_strategy_engine import MultiStrategyEngine
from engine.premium_model import STRATEGY_SL_PCT, STRATEGY_TARGET_MULT, create_premium_state

DAYS = 100
STARTING_CAPITAL = 10_000
LOT_SIZE = settings.NIFTY_LOT_SIZE
DEPLOY_PCT = getattr(settings, "CAPITAL_DEPLOY_PCT", 80.0)
CDGR_TARGET = 0.08

# ── Snapshots for restore ────────────────────────────────────────────────────
_ORIG_MAX_TOTAL = MultiStrategyEngine.MAX_TOTAL_PER_DAY
_ORIG_SL_PCT = copy.deepcopy(premium_model.STRATEGY_SL_PCT)
_ORIG_TARGET_MULT = copy.deepcopy(premium_model.STRATEGY_TARGET_MULT)
_ORIG_TRAIL_TRIGGER = settings.TRAIL_TRIGGER_PCT
_ORIG_TRAIL_PCT = settings.TRAIL_PCT
_ORIG_MAX_SIM = settings.MAX_SIMULTANEOUS_POSITIONS
_ORIG_PREMIUM_DELTA = settings.PREMIUM_DELTA
_ORIG_PREMIUM_SL = settings.PREMIUM_SL_PCT
_ORIG_COOLDOWN = MultiStrategyEngine.COOLDOWN_BARS


def calc_cdgr(final: float, initial: float, active_days: int) -> float:
    if active_days <= 0 or initial <= 0 or final <= 0:
        return 0.0
    return (final / initial) ** (1 / active_days) - 1


def restore_all() -> None:
    MultiStrategyEngine.MAX_TOTAL_PER_DAY = _ORIG_MAX_TOTAL
    MultiStrategyEngine.COOLDOWN_BARS = _ORIG_COOLDOWN
    settings.TRAIL_TRIGGER_PCT = _ORIG_TRAIL_TRIGGER
    settings.TRAIL_PCT = _ORIG_TRAIL_PCT
    settings.MAX_SIMULTANEOUS_POSITIONS = _ORIG_MAX_SIM
    settings.PREMIUM_DELTA = _ORIG_PREMIUM_DELTA
    settings.PREMIUM_SL_PCT = _ORIG_PREMIUM_SL
    premium_model.STRATEGY_SL_PCT.clear()
    premium_model.STRATEGY_SL_PCT.update(_ORIG_SL_PCT)
    premium_model.STRATEGY_TARGET_MULT.clear()
    premium_model.STRATEGY_TARGET_MULT.update(_ORIG_TARGET_MULT)


def apply_baseline_patches() -> None:
    """Production-like baseline: Scenario A + MAX8 + SL8/trail10-5/sim2/delta0.45."""
    MultiStrategyEngine.MAX_TOTAL_PER_DAY = 8
    settings.TRAIL_TRIGGER_PCT = 10.0
    settings.TRAIL_PCT = 5.0
    settings.MAX_SIMULTANEOUS_POSITIONS = 2
    settings.PREMIUM_DELTA = 0.45
    settings.PREMIUM_SL_PCT = 8.0
    premium_model.STRATEGY_SL_PCT["PULLBACK"] = 6.0
    premium_model.STRATEGY_TARGET_MULT["EMA_MOMENTUM"] = {70: 1.50, 50: 1.40, 0: 1.30}


@dataclass
class NovelConfig:
    label: str = "baseline"
    atr_mult: float | None = None
    partial_book_pct: float | None = None
    premium_delta: float | None = None
    regime_adaptive: bool = False
    bypass_cooldown_after_win: bool = False


def _regime_params(adx: float) -> dict | None:
    """Return per-regime trade params, or None to skip entry."""
    if adx < 20:
        return None
    if adx < 30:
        return {"sl_pct": 6.0, "trail_trigger": 12.0, "trail_pct": 5.0, "max_sim": 1}
    if adx < 40:
        return {"sl_pct": 8.0, "trail_trigger": 10.0, "trail_pct": 5.0, "max_sim": 2}
    return {
        "sl_pct": 10.0,
        "trail_trigger": 8.0,
        "trail_pct": 4.0,
        "max_sim": 2,
        "target_boost": 1.10,
    }


def run_compound_backtest_novel(
    df: pd.DataFrame,
    cfg: NovelConfig,
    *,
    starting_capital: float = STARTING_CAPITAL,
    lot_size: int = LOT_SIZE,
    deploy_pct: float = DEPLOY_PCT,
    engine_override: MultiStrategyEngine | None = None,
) -> dict:
    """Modified run_compound_backtest supporting novel exit/entry features."""
    engine = engine_override if engine_override is not None else MultiStrategyEngine()
    delta = cfg.premium_delta if cfg.premium_delta is not None else settings.PREMIUM_DELTA

    capital = starting_capital
    peak = capital
    trades: list[dict] = []
    equity_curve: list[dict] = []

    unique_days = sorted(set(df.index.date))

    for day_idx, day in enumerate(unique_days):
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

        day_start_cap = capital
        day_pnl = 0
        day_trades = 0
        consec_loss = 0
        daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)

        per_lot = getattr(settings, "CAPITAL_PER_LOT", 10_000)
        day_lots = max(1, int(capital / per_lot))
        day_lots = min(day_lots, getattr(settings, "MAX_LOTS_CAP", 10))

        indicators = engine.precompute(day_df)
        open_positions: list[dict] = []
        cooldown_bypass_bar: int | None = None

        for i in range(10, len(day_df)):
            ts = day_df.index[i]
            time_str = ts.strftime("%H:%M")
            nifty_price = day_df["close"].iloc[i]

            if cooldown_bypass_bar == i and cfg.bypass_cooldown_after_win:
                engine._used_bars.discard(i)

            closed_this_bar: list[dict] = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(
                    nifty_price, pos["candles_held"]
                )

                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                # Approach 2: partial profit booking
                if cfg.partial_book_pct is not None and not pos.get("partial_booked"):
                    book_level = pos["entry_premium"] * (1 + cfg.partial_book_pct / 100)
                    if cur_prem >= book_level:
                        partial_qty = pos["qty"] // 2
                        if partial_qty > 0:
                            costs = _calc_realistic_costs(
                                pos["entry_premium"], cur_prem, partial_qty, day_lots
                            )
                            raw_pnl = (cur_prem - pos["entry_premium"]) * partial_qty
                            net_pnl = raw_pnl - costs
                            capital += net_pnl
                            day_pnl += net_pnl
                            day_trades += 1
                            pos["qty"] -= partial_qty
                            pos["sl_premium"] = pos["entry_premium"]
                            pos["partial_booked"] = True
                            trades.append(
                                {
                                    "strategy": pos["signal_type"],
                                    "signal": pos["direction"],
                                    "entry_time": pos["entry_time"],
                                    "exit_time": ts,
                                    "entry_premium": round(pos["entry_premium"], 2),
                                    "exit_premium": round(cur_prem, 2),
                                    "peak_premium": round(pos["peak_premium"], 2),
                                    "peak_gain_pct": round(
                                        (cur_prem - pos["entry_premium"])
                                        / pos["entry_premium"]
                                        * 100,
                                        2,
                                    ),
                                    "qty": partial_qty,
                                    "lots": day_lots,
                                    "pnl": round(net_pnl, 0),
                                    "reason": f"PARTIAL_{cfg.partial_book_pct:.0f}pct",
                                    "capital_after": round(capital, 0),
                                }
                            )

                trail_trigger = pos.get("trail_trigger_pct", settings.TRAIL_TRIGGER_PCT)
                trail_pct = pos.get("trail_pct", settings.TRAIL_PCT)
                trail_floor = pos["prem_state"].update_trail(
                    cur_prem, trigger_pct=trail_trigger, trail_pct=trail_pct
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

                    costs = _calc_realistic_costs(
                        pos["entry_premium"], exit_prem, pos["qty"], day_lots
                    )
                    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                    net_pnl = raw_pnl - costs

                    capital += net_pnl
                    day_pnl += net_pnl
                    day_trades += 1
                    consec_loss = consec_loss + 1 if net_pnl < 0 else 0

                    if (
                        cfg.bypass_cooldown_after_win
                        and net_pnl > 0
                        and exit_reason in ("TGT", "TRAIL")
                    ):
                        cooldown_bypass_bar = i + 1

                    peak_gain = (
                        (pos["peak_premium"] - pos["entry_premium"])
                        / pos["entry_premium"]
                        * 100
                    )

                    trades.append(
                        {
                            "strategy": pos["signal_type"],
                            "signal": pos["direction"],
                            "entry_time": pos["entry_time"],
                            "exit_time": ts,
                            "entry_premium": round(pos["entry_premium"], 2),
                            "exit_premium": round(exit_prem, 2),
                            "peak_premium": round(pos["peak_premium"], 2),
                            "peak_gain_pct": round(peak_gain, 2),
                            "qty": pos["qty"],
                            "lots": day_lots,
                            "pnl": round(net_pnl, 0),
                            "reason": exit_reason,
                            "capital_after": round(capital, 0),
                        }
                    )
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

            max_sim = getattr(settings, "MAX_SIMULTANEOUS_POSITIONS", 2)
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
                if signal.confidence < getattr(settings, "PULLBACK_MIN_CONFIDENCE", 50):
                    continue

                # Approach 4: regime-adaptive params
                regime = None
                if cfg.regime_adaptive:
                    adx_series = indicators.get("adx", pd.Series())
                    adx_val = (
                        float(adx_series.iloc[i])
                        if i < len(adx_series) and not pd.isna(adx_series.iloc[i])
                        else 25.0
                    )
                    regime = _regime_params(adx_val)
                    if regime is None:
                        continue
                    if len(open_positions) >= regime["max_sim"]:
                        continue

                theta = settings.get_scaled_theta(nifty_price)
                prem_state = create_premium_state(
                    entry_index_price=nifty_price,
                    direction=signal.direction,
                    base_premium=settings.PREMIUM_BASE,
                    delta=delta,
                    theta_per_candle=theta,
                    sl_pct=settings.PREMIUM_SL_PCT,
                    confluence_score=signal.confidence,
                    signal_type=signal.signal_type,
                )

                if regime and regime.get("target_boost"):
                    prem_state.target_premium = round(
                        prem_state.target_premium * regime["target_boost"], 2
                    )

                spread = getattr(settings, "BID_ASK_SPREAD", 0.30)
                entry_premium = prem_state.entry_premium + spread

                eff_sl = STRATEGY_SL_PCT.get(signal.signal_type, settings.PREMIUM_SL_PCT)
                if regime:
                    eff_sl = regime["sl_pct"]

                # Approach 1: ATR-based dynamic SL
                atr_series = indicators.get("atr", pd.Series())
                atr = (
                    float(atr_series.iloc[i])
                    if i < len(atr_series) and not pd.isna(atr_series.iloc[i])
                    else 30.0
                )
                if cfg.atr_mult is not None:
                    sl_premium = entry_premium - (atr * cfg.atr_mult * delta)
                    sl_premium = max(sl_premium, entry_premium * 0.5)
                    trail_trigger = (
                        (atr * cfg.atr_mult * 0.5 * delta) / entry_premium * 100
                    )
                    trail_pct = settings.TRAIL_PCT
                elif regime:
                    sl_premium = entry_premium * (1 - eff_sl / 100)
                    trail_trigger = regime["trail_trigger"]
                    trail_pct = regime["trail_pct"]
                else:
                    sl_premium = entry_premium * (1 - eff_sl / 100)
                    trail_trigger = settings.TRAIL_TRIGGER_PCT
                    trail_pct = settings.TRAIL_PCT

                qty = day_lots * lot_size

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
                        "partial_booked": False,
                        "trail_trigger_pct": trail_trigger,
                        "trail_pct": trail_pct,
                    }
                )
                break

            if capital <= 0:
                break

        for pos in open_positions:
            nifty_price = day_df["close"].iloc[-1]
            exit_prem = pos["prem_state"].current_premium(
                nifty_price, pos["candles_held"]
            )
            brokerage = getattr(settings, "BROKERAGE_PER_ORDER", 20) * 2
            stt = (
                exit_prem
                * pos["qty"]
                * getattr(settings, "STT_PCT", 0.0125)
                / 100
            )
            slippage = getattr(settings, "SLIPPAGE_POINTS", 0.5) * pos["qty"]
            costs = brokerage + stt + slippage
            raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
            net_pnl = raw_pnl - costs
            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1
            peak_gain = (
                (pos["peak_premium"] - pos["entry_premium"])
                / pos["entry_premium"]
                * 100
            )

            trades.append(
                {
                    "strategy": pos["signal_type"],
                    "signal": pos["direction"],
                    "entry_time": pos["entry_time"],
                    "exit_time": day_df.index[-1],
                    "entry_premium": round(pos["entry_premium"], 2),
                    "exit_premium": round(exit_prem, 2),
                    "peak_premium": round(pos["peak_premium"], 2),
                    "peak_gain_pct": round(peak_gain, 2),
                    "qty": pos["qty"],
                    "lots": day_lots,
                    "pnl": round(net_pnl, 0),
                    "reason": "EOD",
                    "capital_after": round(capital, 0),
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

    total_pnl = capital - starting_capital
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    eq_vals = [e["capital"] for e in equity_curve]
    peak_arr = np.maximum.accumulate(eq_vals) if eq_vals else [starting_capital]
    dd_arr = [(p - v) / p * 100 for p, v in zip(peak_arr, eq_vals)]
    max_dd = max(dd_arr) if dd_arr else 0

    active_days = [e for e in equity_curve if e["trades"] > 0]
    cdgr = calc_cdgr(capital, starting_capital, len(active_days))

    return {
        "label": cfg.label,
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_pnl": round(total_pnl, 0),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "active_trading_days": len(active_days),
        "cdgr": cdgr,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def run_with_baseline(cfg: NovelConfig, df: pd.DataFrame) -> dict:
    restore_all()
    apply_baseline_patches()
    try:
        return run_compound_backtest_novel(df, cfg)
    finally:
        restore_all()


def format_result(r: dict) -> str:
    cdgr_pct = r["cdgr"] * 100
    hit = "YES" if r["cdgr"] >= CDGR_TARGET else "no"
    return (
        f"{r['label']:<42} "
        f"{r['total_trades']:>5} "
        f"{r['win_rate']:>5.1f}% "
        f"{r['profit_factor']:>5.2f} "
        f"Rs {r['final_capital']:>10,} "
        f"{r['max_drawdown_pct']:>5.1f}% "
        f"{cdgr_pct:>6.2f}% "
        f"{r['active_trading_days']:>4} "
        f"{hit:>3}"
    )


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("=" * 120)
    print("NOVEL CDGR EXPLORATION — Target: 8% CDGR")
    print("=" * 120)
    print(
        f"Baseline: MAX_TOTAL=8, SL=8%/PB6%, Trail=10%/5%, MAX_SIM=2, "
        f"delta=0.45, lot={LOT_SIZE}, capital=Rs {STARTING_CAPITAL:,}"
    )
    print()

    print("Loading 100-day backtest data...")
    df = load_real_data(days=DAYS)
    cal_days = len(set(df.index.date))
    print(f"  {len(df)} bars, {cal_days} calendar days\n")

    all_results: list[dict] = []

    header = (
        f"{'Scenario':<42} {'Trds':>5} {'WR':>6} {'PF':>5} "
        f"{'Final Cap':>12} {'MaxDD':>6} {'CDGR':>7} {'Act':>4} {'8%':>3}"
    )
    print(header)
    print("-" * len(header))

    # Baseline
    baseline = run_with_baseline(NovelConfig(label="0 Baseline"), df)
    all_results.append(baseline)
    print(format_result(baseline))

    # Approach 1: ATR-based dynamic SL
    print("\n--- Approach 1: ATR-Based Dynamic SL ---")
    best_a1 = baseline
    for mult in (1.0, 1.5, 2.0, 2.5):
        cfg = NovelConfig(label=f"1 ATR mult={mult}", atr_mult=mult)
        r = run_with_baseline(cfg, df)
        all_results.append(r)
        print(format_result(r))
        if r["cdgr"] > best_a1["cdgr"]:
            best_a1 = r

    # Approach 2: Partial profit booking
    print("\n--- Approach 2: Partial Profit Booking (50% at target) ---")
    best_a2 = baseline
    for pct in (8, 10, 12, 15):
        cfg = NovelConfig(label=f"2 Partial book +{pct}%", partial_book_pct=float(pct))
        r = run_with_baseline(cfg, df)
        all_results.append(r)
        print(format_result(r))
        if r["cdgr"] > best_a2["cdgr"]:
            best_a2 = r

    # Approach 3: Higher delta
    print("\n--- Approach 3: Higher Delta (ITM options) ---")
    best_a3 = baseline
    for d in (0.45, 0.55, 0.65, 0.75):
        cfg = NovelConfig(label=f"3 Delta={d}", premium_delta=d)
        r = run_with_baseline(cfg, df)
        all_results.append(r)
        print(format_result(r))
        if r["cdgr"] > best_a3["cdgr"]:
            best_a3 = r

    # Approach 4: Regime-adaptive
    print("\n--- Approach 4: Regime-Adaptive (ADX) ---")
    cfg = NovelConfig(label="4 Regime-adaptive ADX", regime_adaptive=True)
    r4 = run_with_baseline(cfg, df)
    all_results.append(r4)
    print(format_result(r4))
    best_a4 = r4

    # Approach 5: Re-enter after win
    print("\n--- Approach 5: Bypass Cooldown After Win ---")
    cfg = NovelConfig(label="5 Win cooldown bypass", bypass_cooldown_after_win=True)
    r5 = run_with_baseline(cfg, df)
    all_results.append(r5)
    print(format_result(r5))
    best_a5 = r5

    # Approach 6: Best combination
    print("\n--- Approach 6: Best Combination ---")
    combo = NovelConfig(label="6 Best combo")
    if "ATR" in best_a1["label"]:
        combo.atr_mult = float(best_a1["label"].split("=")[-1])
    if "Partial" in best_a2["label"]:
        combo.partial_book_pct = float(best_a2["label"].split("+")[-1].replace("%", ""))
    if "Delta=" in best_a3["label"]:
        combo.premium_delta = float(best_a3["label"].split("=")[-1])
    if best_a4["cdgr"] > baseline["cdgr"]:
        combo.regime_adaptive = True
    if best_a5["cdgr"] > baseline["cdgr"]:
        combo.bypass_cooldown_after_win = True

    combo_parts = []
    if combo.atr_mult:
        combo_parts.append(f"ATR={combo.atr_mult}")
    if combo.partial_book_pct:
        combo_parts.append(f"partial+{combo.partial_book_pct:.0f}%")
    if combo.premium_delta and combo.premium_delta != 0.45:
        combo_parts.append(f"delta={combo.premium_delta}")
    if combo.regime_adaptive:
        combo_parts.append("regime")
    if combo.bypass_cooldown_after_win:
        combo_parts.append("win-bypass")
    combo.label = "6 Best combo (" + ", ".join(combo_parts) + ")" if combo_parts else "6 Best combo (baseline only)"

    r6 = run_with_baseline(combo, df)
    all_results.append(r6)
    print(format_result(r6))

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    baseline_cdgr = baseline["cdgr"] * 100
    print(f"Baseline CDGR: {baseline_cdgr:.2f}%  (Final: Rs {baseline['final_capital']:,})")

    ranked = sorted(all_results, key=lambda x: x["cdgr"], reverse=True)
    print("\nTop 5 by CDGR:")
    for i, r in enumerate(ranked[:5], 1):
        print(
            f"  {i}. {r['label']}: CDGR={r['cdgr']*100:.2f}%, "
            f"WR={r['win_rate']:.1f}%, PF={r['profit_factor']:.2f}, "
            f"Final=Rs {r['final_capital']:,}, MaxDD={r['max_drawdown_pct']:.1f}%"
        )

    hits = [r for r in all_results if r["cdgr"] >= CDGR_TARGET]
    print(f"\n8% CDGR target reached: {'YES — ' + str(len(hits)) + ' scenario(s)' if hits else 'NO'}")
    if hits:
        for r in hits:
            print(f"  ✓ {r['label']}: {r['cdgr']*100:.2f}% CDGR")

    best = ranked[0]
    delta_vs_baseline = (best["cdgr"] - baseline["cdgr"]) * 100
    print(
        f"\nBest overall: {best['label']} "
        f"({best['cdgr']*100:.2f}% CDGR, {delta_vs_baseline:+.2f}pp vs baseline)"
    )

    print("\nBest per approach:")
    print(f"  A1 ATR SL:     {best_a1['label']} -> {best_a1['cdgr']*100:.2f}%")
    print(f"  A2 Partial:    {best_a2['label']} -> {best_a2['cdgr']*100:.2f}%")
    print(f"  A3 Delta:      {best_a3['label']} -> {best_a3['cdgr']*100:.2f}%")
    print(f"  A4 Regime:     {best_a4['label']} -> {best_a4['cdgr']*100:.2f}%")
    print(f"  A5 Win bypass: {best_a5['label']} -> {best_a5['cdgr']*100:.2f}%")
    print(f"  A6 Combo:      {r6['label']} -> {r6['cdgr']*100:.2f}%")


if __name__ == "__main__":
    main()
