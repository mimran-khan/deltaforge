"""Quick check: run MultiStrategyEngine on today's Nifty 5m candles.

Downloads the latest 5 trading days via yfinance (^NSEI), uses prior days
for indicator warmup, scans today only, and prints signals + snapshot.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.multi_strategy_engine import MultiStrategyEngine
from engine.premium_model import create_premium_state


def download_nifty_5m(trading_days: int = 5) -> pd.DataFrame:
    """Download recent Nifty 50 5m candles; keep the last N trading days."""
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="10d", interval="5m")
    if df.empty:
        ticker = yf.Ticker("NIFTY_50.NS")
        df = ticker.history(period="10d", interval="5m")
    if df.empty:
        raise RuntimeError("No intraday data from yfinance (^NSEI / NIFTY_50.NS)")

    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]]
    # Yahoo ^NSEI often reports volume=0; use synthetic volume for bar-quality checks
    if (df["volume"] == 0).all():
        df["volume"] = 5000
    df = df.sort_index()

    unique_days = sorted(set(df.index.date))
    if len(unique_days) > trading_days:
        keep = set(unique_days[-trading_days:])
        df = df[df.index.map(lambda t: t.date() in keep)]

    return df


def _st_label(st_dir: float) -> str:
    if st_dir == 1:
        return "BULLISH"
    if st_dir == -1:
        return "BEARISH"
    return "NEUTRAL"


def _vwap_position(close: float, vwap: float) -> str:
    if np.isnan(vwap):
        return "N/A"
    if close > vwap:
        return f"ABOVE VWAP ({close - vwap:+.1f})"
    if close < vwap:
        return f"BELOW VWAP ({close - vwap:+.1f})"
    return "AT VWAP"


def main(target_date: date | None = None) -> None:
    today = target_date or date.today()
    print(f"\n{'=' * 65}")
    print(f"  NIFTY 50 SIGNAL CHECK — {today.strftime('%A %Y-%m-%d')}")
    print(f"{'=' * 65}\n")

    print("Downloading latest 5 trading days of 5m data (^NSEI)...")
    df = download_nifty_5m(trading_days=5)
    days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(days)} days: "
          f"{days[0]} → {days[-1]}\n")

    today_df = df[df.index.date == today]
    if today_df.empty:
        print(f"WARNING: No candles for {today} in downloaded data.")
        print(f"Available dates: {', '.join(str(d) for d in days)}")
        return

    print(f"Today's session: {len(today_df)} candles "
          f"({today_df.index[0].strftime('%H:%M')} – "
          f"{today_df.index[-1].strftime('%H:%M')})\n")

    engine = MultiStrategyEngine()
    engine.reset_day()
    indicators = engine.precompute(df)

    signals_today = []
    today_indices = [i for i, ts in enumerate(df.index) if ts.date() == today]

    for i in today_indices:
        if i < 10:
            continue
        ts = df.index[i]
        time_str = ts.strftime("%H:%M")
        found = engine.scan(indicators, i, time_str)
        for sig in found:
            signals_today.append((ts, sig))

    print(f"{'─' * 65}")
    print("  SIGNALS TODAY")
    print(f"{'─' * 65}")
    if not signals_today:
        print("  (no signals generated)\n")
    else:
        for ts, sig in signals_today:
            print(f"  {ts.strftime('%Y-%m-%d %H:%M')}  "
                  f"{sig.direction:5s}  {sig.signal_type:14s}  "
                  f"conf={sig.confidence:5.1f}  {sig.reason}")
        print()

    # ── P&L Simulation ──
    lot_size = getattr(settings, 'NIFTY_LOT_SIZE', 75)
    capital = 10_000
    per_lot = getattr(settings, 'CAPITAL_PER_LOT', 6_000)
    day_lots = max(1, int(capital * 0.8 / per_lot))

    open_positions = []
    closed_trades = []

    for i in today_indices:
        ts = df.index[i]
        time_str = ts.strftime("%H:%M")
        nifty_price = df["close"].iloc[i]

        still_open = []
        for pos in open_positions:
            pos["candles_held"] += 1
            cur_prem = pos["prem_state"].current_premium(
                nifty_price, pos["candles_held"])
            if cur_prem > pos["peak_prem"]:
                pos["peak_prem"] = cur_prem

            trail_floor = pos["prem_state"].update_trail(cur_prem, 12, 8)

            exit_reason = None
            exit_prem = cur_prem
            if cur_prem <= pos["sl_prem"]:
                exit_reason = "SL"
                exit_prem = pos["sl_prem"]
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
                qty = day_lots * lot_size
                brokerage = getattr(settings, 'BROKERAGE_PER_ORDER', 20) * 2
                slippage = getattr(settings, 'SLIPPAGE_POINTS', 0.5) * qty
                raw_pnl = (exit_prem - pos["entry_prem"]) * qty
                net_pnl = raw_pnl - brokerage - slippage
                closed_trades.append({
                    "signal": pos["sig"],
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "entry_prem": pos["entry_prem"],
                    "exit_prem": round(exit_prem, 2),
                    "exit_reason": exit_reason,
                    "pnl": round(net_pnl, 0),
                    "lots": day_lots,
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # Check signals for this bar
        for entry_ts, sig in signals_today:
            if entry_ts == ts and not open_positions:
                theta = settings.get_scaled_theta(nifty_price)
                prem = create_premium_state(
                    entry_index_price=nifty_price,
                    direction=sig.direction,
                    base_premium=settings.PREMIUM_BASE,
                    delta=settings.PREMIUM_DELTA,
                    theta_per_candle=theta,
                    sl_pct=settings.PREMIUM_SL_PCT,
                    confluence_score=sig.confidence,
                    signal_type=sig.signal_type,
                )
                entry_prem = prem.entry_premium + getattr(settings, 'SLIPPAGE_POINTS', 0.5)
                sl_prem = entry_prem * (1 - settings.PREMIUM_SL_PCT / 100)
                open_positions.append({
                    "sig": sig,
                    "prem_state": prem,
                    "entry_time": ts,
                    "entry_prem": entry_prem,
                    "sl_prem": sl_prem,
                    "peak_prem": entry_prem,
                    "candles_held": 0,
                })

    # Mark-to-market any still open
    for pos in open_positions:
        last_price = df["close"].iloc[today_indices[-1]]
        cur_prem = pos["prem_state"].current_premium(
            last_price, pos["candles_held"])
        qty = day_lots * lot_size
        raw_pnl = (cur_prem - pos["entry_prem"]) * qty
        closed_trades.append({
            "signal": pos["sig"],
            "entry_time": pos["entry_time"],
            "exit_time": df.index[today_indices[-1]],
            "entry_prem": pos["entry_prem"],
            "exit_prem": round(cur_prem, 2),
            "exit_reason": "OPEN (MTM)",
            "pnl": round(raw_pnl, 0),
            "lots": day_lots,
        })

    print(f"{'─' * 65}")
    print("  TRADE SIMULATION (Rs 10K capital, paper)")
    print(f"{'─' * 65}")
    total_pnl = 0
    if not closed_trades:
        print("  No trades executed today.\n")
    else:
        for t in closed_trades:
            s = t["signal"]
            pnl_str = f"Rs {t['pnl']:+,.0f}"
            tag = "WIN" if t["pnl"] > 0 else "LOSS" if t["pnl"] < 0 else "FLAT"
            print(f"  {s.direction:5s} {s.signal_type:14s} "
                  f"{t['entry_time'].strftime('%H:%M')}->{t['exit_time'].strftime('%H:%M')}  "
                  f"Prem {t['entry_prem']:.1f}->{t['exit_prem']:.1f}  "
                  f"{t['exit_reason']:6s}  {pnl_str:>10s}  [{tag}]")
            total_pnl += t["pnl"]
        print()
        wins = sum(1 for t in closed_trades if t["pnl"] > 0)
        print(f"  Total P&L  : Rs {total_pnl:+,.0f}")
        print(f"  Trades     : {len(closed_trades)} ({wins}W / {len(closed_trades)-wins}L)")
        print(f"  Win Rate   : {wins/len(closed_trades)*100:.0f}%")
        print()

    # ── Market Snapshot ──
    last_i = today_indices[-1]
    close = engine._sv(indicators["close"], last_i)
    rsi = engine._sv(indicators.get("rsi_5m", pd.Series()), last_i, 50)
    st_dir = engine._sv(indicators.get("supertrend_dir", pd.Series()), last_i, 0)
    vwap = engine._sv(indicators.get("vwap", pd.Series()), last_i, np.nan)
    adx = engine._sv(indicators.get("adx", pd.Series()), last_i, 0)
    last_ts = df.index[last_i]

    print(f"{'─' * 65}")
    print(f"  MARKET SNAPSHOT @ {last_ts.strftime('%H:%M')}")
    print(f"{'─' * 65}")
    print(f"  Nifty Price       : {close:,.2f}")
    print(f"  RSI (14, 5m)      : {rsi:.1f}")
    print(f"  Supertrend (10,3) : {_st_label(st_dir)}")
    print(f"  VWAP Position     : {_vwap_position(close, vwap)}")
    print(f"  ADX (14)          : {adx:.1f}")
    print()

    print(f"{'=' * 65}")
    print(f"  SUMMARY: {len(signals_today)} signal(s), {len(closed_trades)} trade(s)")
    print(f"  Day P&L: Rs {total_pnl:+,.0f}")
    if signals_today:
        by_type: dict[str, int] = {}
        for _, sig in signals_today:
            by_type[sig.signal_type] = by_type.get(sig.signal_type, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
        print(f"  Signals: {breakdown}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check today's Nifty signals")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    main(target)
