"""Backtest with adaptive mode controller wired in.

Runs the same compound backtest as run_backtest.py but integrates
the AdaptiveModeController to dynamically modulate parameters
(max_trades, max_sim, lot_multiplier, SL/target/trail multipliers,
min_confidence) based on intra-day performance.

Usage:
    python -m backtest.test_adaptive [--days 100] [--capital 10000]
    python -m backtest.test_adaptive --sweep   # sweep adaptive thresholds
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs, print_results


def run_adaptive_backtest(df: pd.DataFrame,
                          starting_capital: float = 10000,
                          lot_size: int = 65,
                          adaptive_enabled: bool = True,
                          adaptive_overrides: dict | None = None) -> dict:
    """Compound backtest with adaptive mode controller.

    When adaptive_enabled=False, runs the baseline (identical to
    run_compound_backtest). When True, the controller modulates
    max_trades, max_sim, lots, SL/target/trail multipliers per trade.
    """
    from engine.multi_strategy_engine import MultiStrategyEngine
    from engine.premium_model import create_premium_state, STRATEGY_SL_PCT
    from risk.adaptive_mode import AdaptiveModeController, Mode

    engine = MultiStrategyEngine()
    adaptive = AdaptiveModeController() if adaptive_enabled else None

    if adaptive_overrides and adaptive_enabled:
        for key, val in adaptive_overrides.items():
            setattr(settings, key, val)

    capital = starting_capital
    peak = capital
    trades = []
    equity_curve = []
    mode_at_trade: list[str] = []

    unique_days = sorted(set(df.index.date))

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            equity_curve.append({"date": day, "capital": capital,
                                 "daily_pnl": 0, "trades": 0, "lots": 0})
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
        if adaptive:
            adaptive.reset()

        day_start_cap = capital
        day_pnl = 0.0
        day_trades = 0
        wins_today = 0
        losses_today = 0
        consec_loss = 0
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

            if adaptive:
                adaptive.on_bar()
                ap = adaptive.profile
            else:
                ap = None

            max_sim = ap.max_simultaneous if ap else getattr(settings, 'MAX_SIMULTANEOUS_POSITIONS', 2)
            max_total = ap.max_trades_per_day if ap else getattr(settings, 'MAX_TOTAL_PER_DAY', 8)
            lot_mult = ap.lot_multiplier if ap else 1.0
            sl_mult = ap.sl_multiplier if ap else 1.0
            tgt_mult = ap.target_multiplier if ap else 1.0
            trail_trigger = ap.trail_trigger_pct if ap else settings.TRAIL_TRIGGER_PCT
            trail_pct = ap.trail_pct if ap else settings.TRAIL_PCT
            min_conf = ap.min_confidence if ap else getattr(settings, 'PULLBACK_MIN_CONFIDENCE', 50)

            # --- Update open positions ---
            closed_this_bar = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(
                    nifty_price, pos["candles_held"])

                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                trail_floor = pos["prem_state"].update_trail(
                    cur_prem,
                    trigger_pct=trail_trigger,
                    trail_pct=trail_pct)

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
                        pos["entry_premium"], exit_prem,
                        pos["qty"], pos.get("lots_used", base_lots))

                    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                    net_pnl = raw_pnl - costs

                    capital += net_pnl
                    day_pnl += net_pnl
                    day_trades += 1
                    won = net_pnl > 0
                    if won:
                        wins_today += 1
                        consec_loss = 0
                    else:
                        losses_today += 1
                        consec_loss += 1

                    if adaptive:
                        daily_pnl_pct = (day_pnl / day_start_cap * 100) if day_start_cap > 0 else 0
                        adaptive.update(
                            daily_pnl_pct=daily_pnl_pct,
                            wins=wins_today,
                            losses=losses_today,
                            consecutive_losses=consec_loss,
                            trades=wins_today + losses_today,
                            last_trade_won=won,
                        )

                    peak_gain = (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100

                    trades.append({
                        "strategy": pos["signal_type"], "signal": pos["direction"],
                        "entry_time": pos["entry_time"], "exit_time": ts,
                        "entry_premium": round(pos["entry_premium"], 2),
                        "exit_premium": round(exit_prem, 2),
                        "peak_premium": round(pos["peak_premium"], 2),
                        "peak_gain_pct": round(peak_gain, 2),
                        "qty": pos["qty"], "lots": pos.get("lots_used", base_lots),
                        "pnl": round(net_pnl, 0), "reason": exit_reason,
                        "capital_after": round(capital, 0),
                    })
                    mode_at_trade.append(adaptive.mode.value if adaptive else "NORMAL")
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

            # --- Entry logic ---
            if len(open_positions) >= max_sim:
                continue
            if consec_loss >= settings.MAX_CONSECUTIVE_LOSSES:
                continue
            if day_pnl < 0 and abs(day_pnl) >= daily_loss_limit:
                continue

            open_dirs = {p["direction"] for p in open_positions}
            signals = engine.scan(indicators, i, time_str,
                                  max_total_override=max_total)

            for signal in signals:
                if signal.direction in open_dirs:
                    continue
                if signal.confidence < min_conf:
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

                if tgt_mult != 1.0:
                    prem_state.target_premium = (
                        prem_state.entry_premium
                        + (prem_state.target_premium - prem_state.entry_premium) * tgt_mult
                    )

                spread = getattr(settings, 'BID_ASK_SPREAD', 0.30)
                entry_premium = prem_state.entry_premium + spread
                eff_sl = STRATEGY_SL_PCT.get(signal.signal_type, settings.PREMIUM_SL_PCT) * sl_mult
                sl_premium = entry_premium * (1 - eff_sl / 100)

                eff_lots = max(1, int(base_lots * lot_mult))
                qty = eff_lots * lot_size

                open_positions.append({
                    "direction": signal.direction,
                    "signal_type": signal.signal_type,
                    "entry_time": ts,
                    "entry_premium": entry_premium,
                    "sl_premium": sl_premium,
                    "qty": qty,
                    "lots_used": eff_lots,
                    "prem_state": prem_state,
                    "candles_held": 0,
                    "peak_premium": entry_premium,
                })
                mode_at_trade.append(adaptive.mode.value if adaptive else "NORMAL")
                break

            if capital <= 0:
                break

        # EOD square-off
        for pos in open_positions:
            nifty_price = day_df["close"].iloc[-1]
            exit_prem = pos["prem_state"].current_premium(
                nifty_price, pos["candles_held"])
            brokerage = getattr(settings, 'BROKERAGE_PER_ORDER', 20) * 2
            stt = exit_prem * pos["qty"] * getattr(settings, 'STT_PCT', 0.0125) / 100
            slippage = getattr(settings, 'SLIPPAGE_POINTS', 0.5) * pos["qty"]
            costs = brokerage + stt + slippage
            raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
            net_pnl = raw_pnl - costs
            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1
            peak_gain = (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100

            trades.append({
                "strategy": pos["signal_type"], "signal": pos["direction"],
                "entry_time": pos["entry_time"], "exit_time": day_df.index[-1],
                "entry_premium": round(pos["entry_premium"], 2),
                "exit_premium": round(exit_prem, 2),
                "peak_premium": round(pos["peak_premium"], 2),
                "peak_gain_pct": round(peak_gain, 2),
                "qty": pos["qty"], "lots": pos.get("lots_used", base_lots),
                "pnl": round(net_pnl, 0), "reason": "EOD",
                "capital_after": round(capital, 0),
            })
            mode_at_trade.append(adaptive.mode.value if adaptive else "NORMAL")

        if capital > peak:
            peak = capital

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades, "lots": base_lots,
        })

        if capital <= 0:
            break

    # ── Stats ──
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

    n_days = len(equity_curve)
    cdgr = ((capital / starting_capital) ** (1 / n_days) - 1) * 100 if n_days > 0 and capital > 0 else 0

    mode_counts = Counter(mode_at_trade)

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
        "cdgr": round(cdgr, 2),
        "mode_distribution": dict(mode_counts),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def print_adaptive_results(results: dict, label: str = ""):
    """Print results with mode distribution."""
    tag = f" ({label})" if label else ""
    print(f"\n{'=' * 65}")
    print(f"     ADAPTIVE BACKTEST RESULTS{tag}")
    print("=" * 65)
    print(f"  Starting Capital      : Rs {results['starting_capital']:>12,.0f}")
    print(f"  Final Capital         : Rs {results['final_capital']:>12,.0f}")
    pnl = results['total_pnl']
    s = "+" if pnl >= 0 else ""
    print(f"  Total P&L             : Rs {s}{pnl:>11,.0f} ({results['return_pct']}%)")
    print(f"  CDGR                  : {results['cdgr']}%")
    print(f"  Avg Daily Return      : {results['avg_daily_return_pct']}%")
    print("-" * 65)
    print(f"  Total Trades          : {results['total_trades']}")
    print(f"  Win Rate              : {results['win_rate']}%")
    print(f"  Profit Factor         : {results['profit_factor']}")
    print(f"  Max Drawdown          : {results['max_drawdown_pct']}%")
    print(f"  Trading Days          : {results['trading_days']}")
    print(f"  Active Days           : {results['active_trading_days']}")
    print(f"  Profitable / Loss     : {results['profitable_days']} / {results['loss_days']}")
    print("-" * 65)

    md = results.get("mode_distribution", {})
    if md:
        total = sum(md.values())
        print("  Mode Distribution (at trade time):")
        for mode in ["AGGRESSIVE", "NORMAL", "DEFENSIVE", "HALT"]:
            cnt = md.get(mode, 0)
            pct = cnt / total * 100 if total > 0 else 0
            print(f"    {mode:12s}: {cnt:>4d} ({pct:>5.1f}%)")
    print("=" * 65)


def run_sweep(df: pd.DataFrame, starting_capital: float = 10000,
              lot_size: int = 65):
    """Sweep adaptive threshold combos and report best config."""
    defensive_triggers = [-3.0, -5.0, -7.0]
    aggressive_triggers = [3.0, 5.0, 8.0]
    halt_consecutives = [2, 3, 4]

    results_grid = []

    total = len(defensive_triggers) * len(aggressive_triggers) * len(halt_consecutives)
    idx = 0

    for def_pnl in defensive_triggers:
        for agg_pnl in aggressive_triggers:
            for halt_c in halt_consecutives:
                idx += 1
                overrides = {
                    "ADAPTIVE_DEFENSIVE_PNL_PCT": def_pnl,
                    "ADAPTIVE_AGGRESSIVE_PNL_PCT": agg_pnl,
                    "ADAPTIVE_HALT_CONSECUTIVE": halt_c,
                }

                res = run_adaptive_backtest(
                    df, starting_capital=starting_capital,
                    lot_size=lot_size, adaptive_enabled=True,
                    adaptive_overrides=overrides,
                )

                row = {
                    "def_pnl": def_pnl,
                    "agg_pnl": agg_pnl,
                    "halt_consec": halt_c,
                    "cdgr": res["cdgr"],
                    "pf": res["profit_factor"],
                    "wr": res["win_rate"],
                    "max_dd": res["max_drawdown_pct"],
                    "trades": res["total_trades"],
                    "final_cap": res["final_capital"],
                }
                results_grid.append(row)

                print(f"  [{idx:>2d}/{total}] def={def_pnl:>5.1f}% agg={agg_pnl:>4.1f}% "
                      f"halt_c={halt_c} => CDGR={res['cdgr']:.2f}% "
                      f"PF={res['profit_factor']:.2f} DD={res['max_drawdown_pct']:.1f}%")

    # Filter: DD < 35%
    valid = [r for r in results_grid if r["max_dd"] < 35]
    if not valid:
        valid = results_grid

    best = max(valid, key=lambda r: r["pf"])

    print(f"\n{'=' * 65}")
    print("  SWEEP RESULTS (sorted by PF, DD < 35%)")
    print("=" * 65)
    print(f"  {'Def%':>6s} {'Agg%':>6s} {'HaltC':>5s} "
          f"{'CDGR':>6s} {'PF':>6s} {'WR':>6s} {'DD':>6s} {'Trades':>6s}")
    print("  " + "-" * 55)

    for r in sorted(valid, key=lambda x: x["pf"], reverse=True):
        marker = " <-- BEST" if r == best else ""
        print(f"  {r['def_pnl']:>6.1f} {r['agg_pnl']:>6.1f} {r['halt_consec']:>5d} "
              f"{r['cdgr']:>6.2f} {r['pf']:>6.2f} {r['wr']:>5.1f}% {r['max_dd']:>5.1f}% "
              f"{r['trades']:>6d}{marker}")

    print(f"\n  BEST CONFIG: def={best['def_pnl']}% agg={best['agg_pnl']}% "
          f"halt_c={best['halt_consec']}")
    print(f"    CDGR={best['cdgr']}% | PF={best['pf']} | WR={best['wr']}% | DD={best['max_dd']}%")
    print()

    return best, results_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--sweep", action="store_true",
                        help="Run threshold sweep instead of single backtest")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    print(f"\nLoading {args.days} trading days of Nifty data...")
    df = load_real_data(days=args.days)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days")

    lot_size = settings.NIFTY_LOT_SIZE

    if args.sweep:
        print(f"\n{'=' * 65}")
        print("  ADAPTIVE THRESHOLD SWEEP")
        print(f"{'=' * 65}")
        best, grid = run_sweep(df, starting_capital=args.capital, lot_size=lot_size)
        return

    # --- Baseline (no adaptive) ---
    print(f"\n  ── BASELINE (no adaptive) ──")
    baseline = run_adaptive_backtest(
        df, starting_capital=args.capital,
        lot_size=lot_size, adaptive_enabled=False,
    )
    print_adaptive_results(baseline, "BASELINE")

    # --- Adaptive ---
    print(f"\n  ── ADAPTIVE MODE ENABLED ──")
    adaptive_res = run_adaptive_backtest(
        df, starting_capital=args.capital,
        lot_size=lot_size, adaptive_enabled=True,
    )
    print_adaptive_results(adaptive_res, "ADAPTIVE")

    # --- Comparison ---
    print(f"\n{'=' * 65}")
    print("  COMPARISON: BASELINE vs ADAPTIVE")
    print("=" * 65)
    metrics = ["cdgr", "profit_factor", "win_rate", "max_drawdown_pct",
               "total_trades", "avg_daily_return_pct"]
    labels = ["CDGR (%)", "Profit Factor", "Win Rate (%)", "Max Drawdown (%)",
              "Total Trades", "Avg Daily Ret (%)"]

    for metric, label in zip(metrics, labels):
        b = baseline.get(metric, 0)
        a = adaptive_res.get(metric, 0)
        diff = a - b
        arrow = "+" if diff > 0 else ""
        print(f"  {label:>22s}: {b:>8.2f} -> {a:>8.2f}  ({arrow}{diff:.2f})")
    print("=" * 65)


if __name__ == "__main__":
    main()
