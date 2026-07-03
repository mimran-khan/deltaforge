"""Backtest using REAL option premium data from Angel One.

Instead of the synthetic premium model (delta * price_move - theta),
this reads actual 5-min CE/PE candle data fetched from the broker.

When the MultiStrategyEngine fires a LONG signal, we look up the real
ATM CE premium at that exact candle timestamp. When it fires SHORT,
we look up the ATM PE premium. All P&L is from real market prices.

Usage:
    # First fetch the data (run once):
    python -m backtest.fetch_option_data --days 30

    # Then run the real-premium backtest:
    python -m backtest.real_premium_backtest --days 22
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import settings
from backtest.run_backtest import load_real_data
from engine.multi_strategy_engine import MultiStrategyEngine
from risk.adaptive_mode import AdaptiveModeController

OPTION_DATA_DIR = BASE_DIR / "data" / "option_candles"


def load_option_data(trade_date) -> dict:
    """Load real option candle data for a specific day.

    Returns dict with 'CE' and 'PE' DataFrames indexed by timestamp.
    """
    cache_file = OPTION_DATA_DIR / f"{trade_date}.csv"
    if not cache_file.exists():
        return {}

    df = pd.read_csv(cache_file, parse_dates=["timestamp"], index_col="timestamp")

    result = {}
    for opt_type in ["CE", "PE"]:
        opt_df = df[df["option_type"] == opt_type].copy()
        if not opt_df.empty:
            opt_df = opt_df[~opt_df.index.duplicated(keep="last")]
            opt_df.sort_index(inplace=True)
            result[opt_type] = opt_df
    return result


def _find_premium_at_time(option_df: pd.DataFrame, target_time: pd.Timestamp,
                          field: str = "close") -> float:
    """Find the option premium closest to the given timestamp."""
    if option_df.empty:
        return np.nan

    t = target_time
    if t.tzinfo is not None and option_df.index.tz is None:
        t = t.tz_localize(None)
    elif t.tzinfo is None and option_df.index.tz is not None:
        t = t.tz_localize(option_df.index.tz)

    idx = option_df.index.searchsorted(t, side="right") - 1
    if idx < 0:
        idx = 0
    if idx >= len(option_df):
        idx = len(option_df) - 1

    return float(option_df[field].iloc[idx])


def run_real_premium_backtest(
    nifty_df: pd.DataFrame,
    starting_capital: float = 45000,
    lot_size: int = 65,
    deploy_pct: float = 100,
    engine_override: MultiStrategyEngine = None,
    use_adaptive: bool = True,
) -> dict:
    """Bar-by-bar backtest using real option premium data.

    Same signal generation as production, but P&L from actual
    ATM option prices fetched from Angel One.
    """
    engine = engine_override or MultiStrategyEngine()
    adaptive = AdaptiveModeController() if use_adaptive else None

    capital = starting_capital
    trades = []
    equity_curve = []

    unique_days = sorted(set(nifty_df.index.date))
    total_pnl = 0
    peak_capital = capital

    for day_idx, day in enumerate(unique_days):
        engine.reset_day()
        if adaptive:
            adaptive.reset()

        option_data = load_option_data(day)
        if not option_data:
            equity_curve.append({
                "date": str(day), "capital": round(capital, 0),
                "daily_pnl": 0, "lots": 0, "trades": 0,
            })
            continue

        day_nifty = nifty_df[nifty_df.index.date == day]
        if day_nifty.empty:
            continue

        warmup_start = max(0, day_idx - 5)
        warmup_days = unique_days[warmup_start:day_idx + 1]
        warmup_df = nifty_df[nifty_df.index.map(lambda t: t.date() in set(warmup_days))]

        indicators = engine.precompute(warmup_df)
        today_mask = warmup_df.index.map(lambda t: t.date() == day)
        today_indices = [i for i, m in enumerate(today_mask) if m]

        if not today_indices:
            continue

        day_start_cap = capital
        per_lot = getattr(settings, 'CAPITAL_PER_LOT', 3000)
        deployable = capital * (deploy_pct / 100)
        day_lots = max(1, int(deployable / per_lot))
        day_lots = min(day_lots, getattr(settings, 'MAX_LOTS_CAP', 50))

        day_pnl = 0
        day_trades = 0
        day_wins = 0
        day_losses = 0
        consec_loss = 0
        daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)

        open_positions = []

        ap_min_conf = 65
        ap_lot_mult = 1.0
        ap_sl_mult = 1.0
        ap_max_sim = getattr(settings, 'MAX_SIMULTANEOUS_POSITIONS', 2)
        ap_max_trades = 8
        if adaptive:
            ap = adaptive.profile
            ap_min_conf = ap.min_confidence
            ap_lot_mult = ap.lot_multiplier
            ap_max_sim = ap.max_simultaneous
            ap_max_trades = ap.max_trades_per_day

        for i in today_indices:
            ts = warmup_df.index[i]
            time_str = ts.strftime("%H:%M")

            if time_str < "09:20":
                continue

            nifty_price = warmup_df["close"].iloc[i]

            if adaptive:
                adaptive.on_bar()

            closed_this_bar = []
            for pos in open_positions:
                pos["candles_held"] += 1

                opt_type = "CE" if pos["direction"] == "LONG" else "PE"
                opt_df = option_data.get(opt_type)
                if opt_df is None:
                    continue

                cur_prem = _find_premium_at_time(opt_df, ts)
                if np.isnan(cur_prem):
                    continue

                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                exit_reason = None
                exit_prem = cur_prem

                pnl_pct = (cur_prem - pos["entry_premium"]) / pos["entry_premium"] * 100
                loss_pct = -pnl_pct if pnl_pct < 0 else 0

                sl_pct = pos.get("sl_pct", 15.0)
                target_pct = pos.get("target_pct", 30.0)

                if loss_pct >= sl_pct:
                    exit_reason = "SL"
                    exit_prem = pos["entry_premium"] * (1 - sl_pct / 100)
                elif pnl_pct >= target_pct:
                    exit_reason = "TGT"
                    exit_prem = pos["entry_premium"] * (1 + target_pct / 100)

                trail_trigger = 12.0
                trail_pct = 8.0
                peak_gain_pct = (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100
                if peak_gain_pct >= trail_trigger:
                    trail_floor = pos["peak_premium"] * (1 - trail_pct / 100)
                    if cur_prem <= trail_floor:
                        exit_reason = "TRAIL"
                        exit_prem = trail_floor

                if pos["candles_held"] >= settings.PULLBACK_HOLD_CANDLES:
                    exit_reason = "TIME"
                elif time_str >= settings.SQUARE_OFF_TIME:
                    exit_reason = "EOD"

                if exit_reason:
                    if exit_reason == "SL":
                        engine.record_sl_exit(pos["signal_type"], i)

                    qty = pos["qty"]
                    spread_cost = 0.30 * qty * 2
                    stt = exit_prem * qty * 0.000625
                    costs = spread_cost + stt

                    raw_pnl = (exit_prem - pos["entry_premium"]) * qty
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

                    if adaptive:
                        daily_pnl_pct = (day_pnl / day_start_cap * 100) if day_start_cap > 0 else 0
                        adaptive.update(
                            daily_pnl_pct=daily_pnl_pct,
                            wins=day_wins, losses=day_losses,
                            consecutive_losses=consec_loss,
                            trades=day_trades, last_trade_won=won,
                        )

                    trades.append({
                        "strategy": pos["signal_type"],
                        "signal": pos["direction"],
                        "entry_time": pos["entry_time"],
                        "exit_time": ts,
                        "entry_premium": round(pos["entry_premium"], 2),
                        "exit_premium": round(exit_prem, 2),
                        "peak_premium": round(pos["peak_premium"], 2),
                        "qty": qty, "lots": day_lots,
                        "pnl": round(net_pnl, 0),
                        "reason": exit_reason,
                        "capital_after": round(capital, 0),
                        "real_premium": True,
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
            if adaptive and adaptive.mode.value == "HALT":
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

                opt_type = "CE" if signal.direction == "LONG" else "PE"
                opt_df = option_data.get(opt_type)
                if opt_df is None:
                    continue

                entry_prem = _find_premium_at_time(opt_df, ts)
                if np.isnan(entry_prem) or entry_prem <= 1.0:
                    continue

                from engine.premium_model import STRATEGY_SL_PCT, STRATEGY_TARGET_MULT
                sl_pct = STRATEGY_SL_PCT.get(signal.signal_type, 15.0)

                tgt_map = STRATEGY_TARGET_MULT.get(
                    signal.signal_type, {70: 1.45, 50: 1.35, 0: 1.25})
                abs_conf = abs(signal.confidence)
                if abs_conf >= 70:
                    tgt_mult = tgt_map[70]
                elif abs_conf >= 50:
                    tgt_mult = tgt_map[50]
                else:
                    tgt_mult = tgt_map[0]
                target_pct = (tgt_mult - 1.0) * 100

                risk_per_trade = max(capital, 0) * 0.05
                sl_amount = entry_prem * (sl_pct / 100)
                if sl_amount <= 0:
                    continue
                risk_lots = int(risk_per_trade / (sl_amount * lot_size))
                eff_lots = max(1, min(risk_lots, day_lots))
                qty = eff_lots * lot_size

                open_positions.append({
                    "direction": signal.direction,
                    "signal_type": signal.signal_type,
                    "entry_time": ts,
                    "entry_premium": entry_prem,
                    "sl_pct": sl_pct,
                    "target_pct": target_pct,
                    "qty": qty,
                    "candles_held": 0,
                    "peak_premium": entry_prem,
                })
                break

            if capital <= lot_size * 5:
                break

        for pos in open_positions:
            opt_type = "CE" if pos["direction"] == "LONG" else "PE"
            opt_df = option_data.get(opt_type)
            if opt_df is not None and not opt_df.empty:
                exit_prem = float(opt_df["close"].iloc[-1])
            else:
                exit_prem = pos["entry_premium"]

            qty = pos["qty"]
            raw_pnl = (exit_prem - pos["entry_premium"]) * qty
            costs = 0.30 * qty * 2 + exit_prem * qty * 0.000625
            net_pnl = raw_pnl - costs
            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1
            if net_pnl > 0:
                day_wins += 1
            else:
                day_losses += 1

            trades.append({
                "strategy": pos["signal_type"],
                "signal": pos["direction"],
                "entry_time": pos["entry_time"],
                "exit_time": str(day) + " 15:25",
                "entry_premium": round(pos["entry_premium"], 2),
                "exit_premium": round(exit_prem, 2),
                "peak_premium": round(pos["peak_premium"], 2),
                "qty": qty, "lots": day_lots,
                "pnl": round(net_pnl, 0),
                "reason": "EOD",
                "capital_after": round(capital, 0),
                "real_premium": True,
            })

        total_pnl += day_pnl
        peak_capital = max(peak_capital, capital)

        equity_curve.append({
            "date": str(day), "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "lots": day_lots, "trades": day_trades,
        })

    wins_list = [t for t in trades if t["pnl"] > 0]
    losses_list = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins_list) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins_list]) if wins_list else 0
    avg_loss = np.mean([t["pnl"] for t in losses_list]) if losses_list else 0
    gross_win = sum(t["pnl"] for t in wins_list)
    gross_loss = abs(sum(t["pnl"] for t in losses_list))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    drawdowns = []
    running_peak = starting_capital
    for e in equity_curve:
        running_peak = max(running_peak, e["capital"])
        dd = (running_peak - e["capital"]) / running_peak * 100
        drawdowns.append(dd)
    max_dd = max(drawdowns) if drawdowns else 0

    active_days = [e for e in equity_curve if e["trades"] > 0]
    profitable_days = sum(1 for e in active_days if e["daily_pnl"] > 0)
    loss_days = sum(1 for e in active_days if e["daily_pnl"] < 0)

    daily_rets = []
    for e in equity_curve:
        if e["trades"] > 0:
            cap_before = e["capital"] - e["daily_pnl"]
            if cap_before > 0:
                daily_rets.append(e["daily_pnl"] / cap_before)
    avg_daily_ret = np.mean(daily_rets) * 100 if daily_rets else 0

    return {
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / starting_capital * 100, 1),
        "total_trades": len(trades),
        "wins": len(wins_list), "losses": len(losses_list),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 0), "avg_loss": round(avg_loss, 0),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "profitable_days": profitable_days, "loss_days": loss_days,
        "active_trading_days": len(active_days),
        "avg_daily_return_pct": round(avg_daily_ret, 1),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def print_results(results: dict):
    print("\n" + "=" * 70)
    print("  REAL-PREMIUM BACKTEST RESULTS (actual option prices from Angel One)")
    print("=" * 70)
    print(f"  Starting Capital : Rs {results['starting_capital']:>12,.0f}")
    print(f"  Final Capital    : Rs {results['final_capital']:>12,.0f}")
    pnl = results["total_pnl"]
    s = "+" if pnl >= 0 else ""
    print(f"  Total P&L        : Rs {s}{pnl:>11,.0f} ({results['return_pct']}%)")
    print(f"  Avg Daily Return : {results['avg_daily_return_pct']}%")
    print("-" * 70)
    print(f"  Active Days      : {results['active_trading_days']}")
    print(f"  Green Days       : {results['profitable_days']}")
    print(f"  Red Days         : {results['loss_days']}")
    print("-" * 70)
    print(f"  Total Trades     : {results['total_trades']}")
    print(f"  Wins / Losses    : {results['wins']} / {results['losses']}")
    print(f"  Win Rate         : {results['win_rate']}%")
    print(f"  Avg Win          : Rs {results['avg_win']:>8,.0f}")
    print(f"  Avg Loss         : Rs {results['avg_loss']:>8,.0f}")
    print(f"  Profit Factor    : {results['profit_factor']:.2f}")
    print(f"  Max Drawdown     : {results['max_drawdown_pct']:.1f}%")
    print("=" * 70)

    strat_stats = {}
    for t in results["trades"]:
        s = t["strategy"]
        if s not in strat_stats:
            strat_stats[s] = {"trades": 0, "wins": 0, "pnl": 0}
        strat_stats[s]["trades"] += 1
        if t["pnl"] > 0:
            strat_stats[s]["wins"] += 1
        strat_stats[s]["pnl"] += t["pnl"]

    print("\n  Strategy Breakdown:")
    for s, st in sorted(strat_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr = st["wins"] / st["trades"] * 100 if st["trades"] > 0 else 0
        print(f"    {s:>16}: {st['trades']:>3} trades | {wr:>5.1f}% WR | Rs{st['pnl']:>+10,.0f}")

    ec = results["equity_curve"]
    print(f"\n  Day-by-Day:")
    print(f"    {'Date':>12}  {'Capital':>12}  {'Day PnL':>10}  {'Lots':>5}  {'Trades':>6}")
    for e in ec:
        ps = "+" if e["daily_pnl"] >= 0 else ""
        print(f"    {e['date']}  Rs{e['capital']:>10,.0f}  Rs{ps}{e['daily_pnl']:>8,.0f}"
              f"  {e['lots']:>5}  {e['trades']:>6}")

    daily_rets = []
    for e in ec:
        if e["trades"] > 0:
            cap_before = e["capital"] - e["daily_pnl"]
            if cap_before > 0:
                daily_rets.append(e["daily_pnl"] / cap_before)
    if daily_rets:
        geo = ((np.prod([1 + r for r in daily_rets])) ** (1 / len(daily_rets)) - 1) * 100
        sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if np.std(daily_rets) > 0 else 0
        print(f"\n  Geometric Daily Return: {geo:.2f}%")
        print(f"  Sharpe (annualised)   : {sharpe:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=22)
    parser.add_argument("--capital", type=float, default=45000)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    nifty_df = load_real_data(days=args.days + 10)
    unique_days = sorted(set(nifty_df.index.date))[-args.days:]
    selected = set(unique_days)
    nifty_df = nifty_df[nifty_df.index.map(lambda t: t.date() in selected)]

    available = sum(1 for d in unique_days if (OPTION_DATA_DIR / f"{d}.csv").exists())
    print(f"Option data available for {available}/{len(unique_days)} days")

    if available == 0:
        print("\nNo real option data found! Run this first:")
        print("  python -m backtest.fetch_option_data --days 30")
        sys.exit(1)

    engine = MultiStrategyEngine()
    results = run_real_premium_backtest(
        nifty_df, starting_capital=args.capital,
        lot_size=65, deploy_pct=100,
        engine_override=engine, use_adaptive=True,
    )
    print_results(results)
