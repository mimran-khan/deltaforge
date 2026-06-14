"""Compare baseline vs runner-mode exit strategy.

Runner mode: when target is hit AND ADX > threshold, defer exit and switch
to trailing stop. This lets winners run during strong trend days.

Also tests trend continuation re-entry: after a TGT exit, allow re-entry
with relaxed RSI ceiling if pullback occurs near EMA9.

Usage:
    python -m backtest.runner_backtest [--days 100] [--capital 10000]
"""

from __future__ import annotations
import sys
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
import copy

import pandas as pd
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs, generate_nifty_data


# ── Continuation Settings ──
CONT_RSI_CEILING = 80           # Relaxed RSI ceiling
CONT_WINDOW_BARS = 8            # Re-enter within 8 bars of exit
CONT_MIN_ADX = 40               # Min ADX for continuation
CONT_MAX_PER_DAY = 2            # Max continuations per day


def run_enhanced_backtest(df: pd.DataFrame,
                          starting_capital: float = 10000,
                          lot_size: int = 65,
                          enable_runner: bool = True,
                          enable_continuation: bool = True,
                          label: str = "ENHANCED") -> dict:
    """Backtest with runner mode and/or continuation re-entry."""
    from engine.multi_strategy_engine import MultiStrategyEngine
    from engine.premium_model import create_premium_state, STRATEGY_SL_PCT

    engine = MultiStrategyEngine()

    capital = starting_capital
    peak = capital
    trades = []
    equity_curve = []

    unique_days = sorted(set(df.index.date))
    WARMUP_DAYS = 5  # Match check_today.py and live candle_builder

    runner_activations = 0
    runner_extra_profit = 0
    cont_entries = 0

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            equity_curve.append({"date": day, "capital": capital, "daily_pnl": 0,
                                  "trades": 0, "lots": 0})
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

        per_lot = getattr(settings, 'CAPITAL_PER_LOT', 10_000)
        day_lots = max(1, int(capital / per_lot))
        day_lots = min(day_lots, getattr(settings, 'MAX_LOTS_CAP', 10))

        # Multi-day warmup (like check_today.py and live engine)
        warmup_start = max(0, day_idx - WARMUP_DAYS)
        warmup_days = unique_days[warmup_start:day_idx + 1]
        warmup_day_set = set(warmup_days)
        warmup_df = df[df.index.map(lambda t: t.date() in warmup_day_set)]
        indicators = engine.precompute(warmup_df)
        today_indices = [i for i, ts in enumerate(warmup_df.index) if ts.date() == day]

        open_positions = []
        
        # Continuation tracking
        last_trend_win = None  # {direction, exit_bar, strategy}
        cont_count_today = 0

        for i in today_indices:
            if i < 10:
                continue
            ts = warmup_df.index[i]
            time_str = ts.strftime("%H:%M")
            nifty_price = warmup_df["close"].iloc[i]
            
            # Get current ADX for runner logic
            adx_val = 0
            adx_prev = 0
            try:
                adx_series = indicators.get('adx', pd.Series())
                if hasattr(adx_series, 'iloc') and len(adx_series) > i:
                    adx_val = float(adx_series.iloc[i]) if not np.isnan(adx_series.iloc[i]) else 0
                if hasattr(adx_series, 'iloc') and len(adx_series) > i - 1 and i > 0:
                    adx_prev = float(adx_series.iloc[i-1]) if not np.isnan(adx_series.iloc[i-1]) else 0
            except Exception:
                pass

            closed_this_bar = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(
                    nifty_price, pos["candles_held"])

                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                trail_floor = pos["prem_state"].update_trail(
                    cur_prem,
                    trigger_pct=settings.TRAIL_TRIGGER_PCT,
                    trail_pct=settings.TRAIL_PCT)

                exit_reason = None
                exit_prem = cur_prem

                # ── SL check ──
                if cur_prem <= pos["sl_premium"]:
                    exit_reason = "SL"
                    exit_prem = pos["sl_premium"]
                
                # ── TGT check with Runner Mode ──
                elif cur_prem >= pos["prem_state"].target_premium:
                    if (enable_runner 
                        and not pos.get("runner_mode", False)
                        and pos["signal_type"] in settings.TREND_RUNNER_STRATEGIES
                        and adx_val >= settings.TREND_RUNNER_ADX_MIN
                        and time_str < settings.TREND_RUNNER_CUTOFF_TIME):
                        # ACTIVATE RUNNER MODE
                        pos["runner_mode"] = True
                        pos["runner_bars"] = 0
                        pos["runner_entry_prem"] = cur_prem
                        # Move SL to breakeven
                        pos["sl_premium"] = max(pos["sl_premium"], pos["entry_premium"])
                        runner_activations += 1
                        # Don't exit -- continue holding
                    elif pos.get("runner_mode", False):
                        # Already in runner mode, target keeps getting hit -- keep running
                        pass
                    else:
                        exit_reason = "TGT"
                        exit_prem = pos["prem_state"].target_premium
                
                # ── Runner Mode exit checks ──
                elif pos.get("runner_mode", False):
                    pos["runner_bars"] = pos.get("runner_bars", 0) + 1
                    
                    # Runner trail: 8% below peak
                    runner_floor = pos["peak_premium"] * (1 - settings.TREND_RUNNER_TRAIL_PCT / 100)
                    
                    if cur_prem <= runner_floor:
                        exit_reason = "RUNNER_TRAIL"
                        exit_prem = cur_prem
                        extra = (cur_prem - pos["prem_state"].target_premium) * pos["qty"]
                        runner_extra_profit += max(0, extra)
                    elif adx_val < settings.TREND_RUNNER_ADX_EXIT:
                        exit_reason = "RUNNER_WEAK"
                        exit_prem = cur_prem
                        extra = (cur_prem - pos["prem_state"].target_premium) * pos["qty"]
                        runner_extra_profit += max(0, extra)
                    elif pos["runner_bars"] >= settings.TREND_RUNNER_MAX_BARS:
                        exit_reason = "RUNNER_TIME"
                        exit_prem = cur_prem
                        extra = (cur_prem - pos["prem_state"].target_premium) * pos["qty"]
                        runner_extra_profit += max(0, extra)
                
                # ── TRAIL check ──
                elif trail_floor is not None and cur_prem <= trail_floor:
                    exit_reason = "TRAIL"
                    exit_prem = trail_floor
                
                # ── TIME check ──
                if not exit_reason and pos["candles_held"] >= settings.PULLBACK_HOLD_CANDLES:
                    exit_reason = "TIME"
                
                # ── EOD check ──
                if not exit_reason and time_str >= settings.SQUARE_OFF_TIME:
                    exit_reason = "EOD"

                if exit_reason:
                    if exit_reason == "SL":
                        engine.record_sl_exit(pos["signal_type"], i)

                    costs = _calc_realistic_costs(
                        pos["entry_premium"], exit_prem,
                        pos["qty"], day_lots)

                    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                    net_pnl = raw_pnl - costs

                    capital += net_pnl
                    day_pnl += net_pnl
                    day_trades += 1
                    consec_loss = consec_loss + 1 if net_pnl < 0 else 0

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
                        "runner_mode": pos.get("runner_mode", False),
                        "runner_bars": pos.get("runner_bars", 0),
                        "is_continuation": pos.get("is_continuation", False),
                    })
                    closed_this_bar.append(pos)
                    
                    # Track for continuation
                    if (enable_continuation 
                        and exit_reason in ("TGT", "RUNNER_TRAIL", "RUNNER_WEAK", "RUNNER_TIME")
                        and net_pnl > 0
                        and pos["signal_type"] in settings.TREND_RUNNER_STRATEGIES):
                        last_trend_win = {
                            "direction": pos["direction"],
                            "exit_bar": i,
                            "strategy": pos["signal_type"],
                        }

            for pos in closed_this_bar:
                open_positions.remove(pos)

            # ── Entry Logic ──
            max_sim = getattr(settings, 'MAX_SIMULTANEOUS_POSITIONS', 2)
            if len(open_positions) >= max_sim:
                continue
            if consec_loss >= settings.MAX_CONSECUTIVE_LOSSES:
                continue
            if day_pnl < 0 and abs(day_pnl) >= daily_loss_limit:
                continue

            open_dirs = {p["direction"] for p in open_positions}
            
            # Try continuation first
            cont_signal = None
            if (enable_continuation 
                and last_trend_win is not None 
                and cont_count_today < CONT_MAX_PER_DAY
                and i - last_trend_win["exit_bar"] <= CONT_WINDOW_BARS
                and i - last_trend_win["exit_bar"] >= 2  # Wait at least 2 bars
                and adx_val >= CONT_MIN_ADX
                and time_str < settings.TREND_RUNNER_CUTOFF_TIME):
                
                # Check for pullback
                try:
                    ema9 = float(indicators.get('ema_9', pd.Series()).iloc[i]) if len(indicators.get('ema_9', pd.Series())) > i else None
                    rsi_5m = float(indicators.get('rsi_5m', pd.Series()).iloc[i]) if len(indicators.get('rsi_5m', pd.Series())) > i else None
                    atr = float(indicators.get('atr', pd.Series()).iloc[i]) if len(indicators.get('atr', pd.Series())) > i else 30
                except Exception:
                    ema9 = None
                    rsi_5m = None
                    atr = 30
                
                if ema9 and rsi_5m:
                    direction = last_trend_win["direction"]
                    pullback_ok = False
                    rsi_ok = False
                    
                    if direction == "LONG":
                        pullback_ok = abs(nifty_price - ema9) <= atr * 0.4
                        rsi_ok = 55 <= rsi_5m <= CONT_RSI_CEILING
                    else:
                        pullback_ok = abs(ema9 - nifty_price) <= atr * 0.4
                        rsi_ok = (100 - CONT_RSI_CEILING) <= rsi_5m <= 45
                    
                    if pullback_ok and rsi_ok and direction not in open_dirs:
                        from engine.multi_strategy_engine import TradeSignal
                        cont_signal = TradeSignal(
                            direction=direction,
                            signal_type=last_trend_win["strategy"],
                            confidence=70,
                            htf_rsi=0,
                            ltf_rsi=rsi_5m,
                            nifty_price=nifty_price,
                            reason="CONTINUATION",
                        )
                        cont_entries += 1
                        cont_count_today += 1
                        last_trend_win = None  # Consumed
            
            # Expire stale continuation window
            if (last_trend_win and i - last_trend_win["exit_bar"] > CONT_WINDOW_BARS):
                last_trend_win = None
            
            # Normal signal scan
            signals = engine.scan(indicators, i, time_str)
            
            # Use continuation signal if no normal signal, or normal signal
            entry_signal = None
            if cont_signal and cont_signal.direction not in open_dirs:
                entry_signal = cont_signal
            elif signals:
                for sig in signals:
                    if sig.direction not in open_dirs and sig.confidence >= getattr(settings, 'PULLBACK_MIN_CONFIDENCE', 50):
                        entry_signal = sig
                        break

            if entry_signal:
                theta = settings.get_scaled_theta(nifty_price)
                prem_state = create_premium_state(
                    entry_index_price=nifty_price,
                    direction=entry_signal.direction,
                    base_premium=settings.PREMIUM_BASE,
                    delta=settings.PREMIUM_DELTA,
                    theta_per_candle=theta,
                    sl_pct=settings.PREMIUM_SL_PCT,
                    confluence_score=entry_signal.confidence,
                    signal_type=entry_signal.signal_type,
                )

                spread = getattr(settings, 'BID_ASK_SPREAD', 0.30)
                entry_premium = prem_state.entry_premium + spread
                eff_sl = STRATEGY_SL_PCT.get(entry_signal.signal_type, settings.PREMIUM_SL_PCT)
                
                # Half size for continuation
                is_cont = (cont_signal is not None and entry_signal == cont_signal)
                lot_mult = 0.5 if is_cont else 1.0
                
                sl_premium = entry_premium * (1 - eff_sl / 100)
                qty = int(day_lots * lot_size * lot_mult)

                open_positions.append({
                    "direction": entry_signal.direction,
                    "signal_type": entry_signal.signal_type,
                    "entry_time": ts,
                    "entry_premium": entry_premium,
                    "sl_premium": sl_premium,
                    "qty": qty,
                    "prem_state": prem_state,
                    "candles_held": 0,
                    "peak_premium": entry_premium,
                    "runner_mode": False,
                    "runner_bars": 0,
                    "is_continuation": is_cont,
                })

            if capital <= 0:
                break

        # EOD square off remaining
        for pos in open_positions:
            last_bar_idx = today_indices[-1]
            nifty_price = warmup_df["close"].iloc[last_bar_idx]
            exit_prem = pos["prem_state"].current_premium(
                nifty_price, pos["candles_held"])
            costs = _calc_realistic_costs(pos["entry_premium"], exit_prem, pos["qty"], day_lots)
            raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
            net_pnl = raw_pnl - costs
            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1
            peak_gain = (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100

            trades.append({
                "strategy": pos["signal_type"], "signal": pos["direction"],
                "entry_time": pos["entry_time"], "exit_time": warmup_df.index[last_bar_idx],
                "entry_premium": round(pos["entry_premium"], 2),
                "exit_premium": round(exit_prem, 2),
                "peak_premium": round(pos["peak_premium"], 2),
                "peak_gain_pct": round(peak_gain, 2),
                "qty": pos["qty"], "lots": day_lots,
                "pnl": round(net_pnl, 0), "reason": "EOD",
                "capital_after": round(capital, 0),
                "runner_mode": pos.get("runner_mode", False),
                "runner_bars": pos.get("runner_bars", 0),
                "is_continuation": pos.get("is_continuation", False),
            })

        if capital > peak:
            peak = capital

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades, "lots": day_lots,
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

    # CDGR
    n_days = len(equity_curve)
    cdgr = ((capital / starting_capital) ** (1 / n_days) - 1) * 100 if n_days > 0 and capital > 0 else 0

    runner_trades = [t for t in trades if t.get("runner_mode")]
    cont_trades = [t for t in trades if t.get("is_continuation")]

    return {
        "label": label,
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
        "avg_daily_return_pct": round(avg_daily_ret, 1),
        "cdgr_pct": round(cdgr, 2),
        "trading_days": len(equity_curve),
        "active_trading_days": len(active_days),
        "runner_activations": runner_activations,
        "runner_extra_profit": round(runner_extra_profit, 0),
        "runner_trades": len(runner_trades),
        "cont_entries": cont_entries,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def print_comparison(baseline: dict, enhanced: dict):
    """Print side-by-side comparison."""
    print("\n" + "=" * 75)
    print("  RUNNER MODE + CONTINUATION BACKTEST COMPARISON")
    print("=" * 75)
    
    metrics = [
        ("Starting Capital", "starting_capital", "Rs {:>12,.0f}"),
        ("Final Capital", "final_capital", "Rs {:>12,.0f}"),
        ("Total P&L", "total_pnl", "Rs {:>12,.0f}"),
        ("Return %", "return_pct", "{:>12.1f}%"),
        ("CDGR (Compound Daily)", "cdgr_pct", "{:>12.2f}%"),
        ("Avg Daily Return", "avg_daily_return_pct", "{:>12.1f}%"),
        ("", None, ""),
        ("Total Trades", "total_trades", "{:>12d}"),
        ("Wins", "wins", "{:>12d}"),
        ("Losses", "losses", "{:>12d}"),
        ("Win Rate", "win_rate", "{:>12.1f}%"),
        ("Avg Win", "avg_win", "Rs {:>10,.0f}"),
        ("Avg Loss", "avg_loss", "Rs {:>10,.0f}"),
        ("Profit Factor", "profit_factor", "{:>12.2f}"),
        ("Max Drawdown", "max_drawdown_pct", "{:>12.1f}%"),
        ("", None, ""),
        ("Trading Days", "trading_days", "{:>12d}"),
        ("Active Days", "active_trading_days", "{:>12d}"),
    ]
    
    print(f"\n  {'Metric':<25s} {'BASELINE':>15s} {'ENHANCED':>15s} {'DELTA':>12s}")
    print("  " + "-" * 70)
    
    for label, key, fmt in metrics:
        if key is None:
            print()
            continue
        bv = baseline.get(key, 0)
        ev = enhanced.get(key, 0)
        
        bfmt = fmt.format(bv) if bv != 0 else fmt.format(bv)
        efmt = fmt.format(ev) if ev != 0 else fmt.format(ev)
        
        if isinstance(bv, (int, float)) and isinstance(ev, (int, float)):
            delta = ev - bv
            if "pct" in key or "rate" in key or "factor" in key or "cdgr" in key:
                dfmt = f"{delta:>+10.1f}"
            else:
                dfmt = f"{delta:>+10,.0f}"
        else:
            dfmt = ""
        
        print(f"  {label:<25s} {bfmt:>15s} {efmt:>15s} {dfmt:>12s}")
    
    # Runner-specific stats
    print(f"\n  {'─── RUNNER MODE STATS ───':─<70}")
    print(f"  Runner Activations     : {enhanced.get('runner_activations', 0)}")
    print(f"  Runner Trades          : {enhanced.get('runner_trades', 0)}")
    print(f"  Runner Extra Profit    : Rs {enhanced.get('runner_extra_profit', 0):,.0f}")
    print(f"  Continuation Entries   : {enhanced.get('cont_entries', 0)}")
    
    # Exit reason breakdown
    for label, results in [("BASELINE", baseline), ("ENHANCED", enhanced)]:
        print(f"\n  Exit Reasons ({label}):")
        reasons = {}
        for t in results["trades"]:
            r = t["reason"]
            if r not in reasons:
                reasons[r] = {"count": 0, "pnl": 0}
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t["pnl"]
        for r, s in sorted(reasons.items()):
            avg = s["pnl"] / s["count"] if s["count"] > 0 else 0
            print(f"    {r:15s}: {s['count']:>4d} trades | Rs {s['pnl']:>10,.0f} | avg Rs {avg:>8,.0f}")
    
    # Strategy breakdown comparison
    for label, results in [("BASELINE", baseline), ("ENHANCED", enhanced)]:
        strat_stats = {}
        for t in results["trades"]:
            s = t["strategy"]
            if s not in strat_stats:
                strat_stats[s] = {"w": 0, "l": 0, "pnl": 0, "n": 0}
            strat_stats[s]["n"] += 1
            if t["pnl"] > 0:
                strat_stats[s]["w"] += 1
            else:
                strat_stats[s]["l"] += 1
            strat_stats[s]["pnl"] += t["pnl"]

        print(f"\n  Strategy Breakdown ({label}):")
        for name, st in sorted(strat_stats.items()):
            wr = st["w"] / st["n"] * 100 if st["n"] > 0 else 0
            print(f"    {name:15s}: {st['n']:>3d} trades | {wr:>4.0f}% win | Rs {st['pnl']:>10,.0f}")

    # Verdict
    print("\n" + "=" * 75)
    pnl_diff = enhanced["total_pnl"] - baseline["total_pnl"]
    wr_diff = enhanced["win_rate"] - baseline["win_rate"]
    dd_diff = enhanced["max_drawdown_pct"] - baseline["max_drawdown_pct"]
    
    if pnl_diff > 0 and dd_diff <= 5:
        print(f"  VERDICT: ENHANCED is BETTER")
        print(f"    +Rs {pnl_diff:,.0f} more profit | WR diff: {wr_diff:+.1f}% | DD diff: {dd_diff:+.1f}%")
        if enhanced["cdgr_pct"] > baseline["cdgr_pct"]:
            print(f"    CDGR improved: {baseline['cdgr_pct']:.2f}% → {enhanced['cdgr_pct']:.2f}%")
    elif pnl_diff > 0 and dd_diff > 5:
        print(f"  VERDICT: ENHANCED has more profit but HIGHER RISK")
        print(f"    +Rs {pnl_diff:,.0f} more profit BUT drawdown is {dd_diff:+.1f}% worse")
    else:
        print(f"  VERDICT: BASELINE is better or equal")
        print(f"    P&L diff: Rs {pnl_diff:,.0f} | WR diff: {wr_diff:+.1f}%")
    print("=" * 75)


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
    print(f"Loaded {len(df)} candles across {len(unique_days)} days")

    lot_size = settings.NIFTY_LOT_SIZE

    # ── BASELINE (current system) ──
    print(f"\n{'='*75}")
    print("  Running BASELINE (current exit logic)...")
    print(f"{'='*75}")
    from backtest.run_backtest import run_compound_backtest
    baseline = run_compound_backtest(df, starting_capital=args.capital, lot_size=lot_size)
    baseline["label"] = "BASELINE"
    baseline["cdgr_pct"] = round(
        ((baseline["final_capital"] / args.capital) ** (1 / len(unique_days)) - 1) * 100, 2
    ) if len(unique_days) > 0 and baseline["final_capital"] > 0 else 0
    
    # ── ENHANCED: Runner Only ──
    print(f"\n{'='*75}")
    print("  Running RUNNER ONLY (defer TGT when ADX strong)...")
    print(f"{'='*75}")
    runner_only = run_enhanced_backtest(
        df, starting_capital=args.capital, lot_size=lot_size,
        enable_runner=True, enable_continuation=False,
        label="RUNNER_ONLY")

    # ── ENHANCED: Continuation Only ──
    print(f"\n{'='*75}")
    print("  Running CONTINUATION ONLY (re-entry after win)...")
    print(f"{'='*75}")
    cont_only = run_enhanced_backtest(
        df, starting_capital=args.capital, lot_size=lot_size,
        enable_runner=False, enable_continuation=True,
        label="CONT_ONLY")

    # ── ENHANCED: Both ──
    print(f"\n{'='*75}")
    print("  Running FULL ENHANCED (runner + continuation)...")
    print(f"{'='*75}")
    full_enhanced = run_enhanced_backtest(
        df, starting_capital=args.capital, lot_size=lot_size,
        enable_runner=True, enable_continuation=True,
        label="FULL_ENHANCED")

    # ── Print all comparisons ──
    print("\n\n" + "#" * 75)
    print("  COMPARISON: BASELINE vs RUNNER ONLY")
    print("#" * 75)
    print_comparison(baseline, runner_only)

    print("\n\n" + "#" * 75)
    print("  COMPARISON: BASELINE vs CONTINUATION ONLY")
    print("#" * 75)
    print_comparison(baseline, cont_only)

    print("\n\n" + "#" * 75)
    print("  COMPARISON: BASELINE vs FULL ENHANCED (Runner + Continuation)")
    print("#" * 75)
    print_comparison(baseline, full_enhanced)

    # ── Summary table ──
    print("\n\n" + "=" * 85)
    print("  FINAL SUMMARY")
    print("=" * 85)
    configs = [baseline, runner_only, cont_only, full_enhanced]
    print(f"\n  {'Config':<20s} {'Final Cap':>12s} {'Return':>8s} {'CDGR':>7s} {'WR':>6s} {'PF':>6s} {'MaxDD':>7s} {'Trades':>7s}")
    print("  " + "-" * 75)
    for c in configs:
        print(f"  {c['label']:<20s} Rs {c['final_capital']:>9,.0f} {c['return_pct']:>7.1f}% {c['cdgr_pct']:>6.2f}% {c['win_rate']:>5.1f}% {c['profit_factor']:>5.2f} {c['max_drawdown_pct']:>6.1f}% {c['total_trades']:>7d}")
    print("=" * 85)
    
    # Save trades
    pd.DataFrame(full_enhanced["trades"]).to_csv(
        settings.DATA_DIR / "enhanced_trades.csv", index=False)
    print(f"\n  Saved: data/enhanced_trades.csv")


if __name__ == "__main__":
    main()
