"""Parameter sweep for PULLBACK strategy on real data with correct HTF RSI.

Sweeps dead zone, ADX, oscillator thresholds, min confirmations,
and min confidence to find a profitable PULLBACK configuration.

Usage:
    python -m backtest.sweep_pullback
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.multi_strategy_engine import MultiStrategyEngine, TradeSignal
from engine.premium_model import create_premium_state

THETA_REFERENCE = 24_000
THETA_BASE = 0.30


def get_scaled_theta(nifty_price: float) -> float:
    return THETA_BASE * (nifty_price / THETA_REFERENCE)


def simulate_trade(entry_price, direction, hold_bars, close_series, entry_idx,
                   theta_per_candle, delta=None, base_premium=100.0, sl_pct=None):
    if delta is None:
        delta = settings.PREMIUM_DELTA
    if sl_pct is None:
        sl_pct = settings.PREMIUM_SL_PCT

    entry_prem = base_premium + settings.SLIPPAGE_POINTS
    sl_prem = entry_prem * (1 - sl_pct / 100)

    for h in range(1, hold_bars + 1):
        idx = entry_idx + h
        if idx >= len(close_series):
            break
        cur_price = close_series.iloc[idx]
        move = cur_price - entry_price
        if direction == "SHORT":
            move = -move
        cur_prem = base_premium + (move * delta) - (theta_per_candle * h)
        cur_prem = max(cur_prem, 0.05)

        if cur_prem <= sl_prem:
            exit_prem = sl_prem - settings.SLIPPAGE_POINTS
            raw_pnl = (exit_prem - entry_prem) * settings.NIFTY_LOT_SIZE
            costs = settings.BROKERAGE_PER_ORDER * 2 + settings.SLIPPAGE_POINTS * settings.NIFTY_LOT_SIZE
            return raw_pnl - costs, h, "SL"

    final_idx = min(entry_idx + hold_bars, len(close_series) - 1)
    final_price = close_series.iloc[final_idx]
    move = final_price - entry_price
    if direction == "SHORT":
        move = -move
    bars_held = final_idx - entry_idx
    final_prem = base_premium + (move * delta) - (theta_per_candle * bars_held)
    final_prem = max(final_prem, 0.05)
    exit_prem = final_prem - settings.SLIPPAGE_POINTS
    raw_pnl = (exit_prem - entry_prem) * settings.NIFTY_LOT_SIZE
    costs = settings.BROKERAGE_PER_ORDER * 2 + settings.SLIPPAGE_POINTS * settings.NIFTY_LOT_SIZE
    return raw_pnl - costs, bars_held, "TIME"


def load_2026() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "nifty_5m_real.csv"
    df = pd.read_csv(path)
    dt_col = "Datetime" if "Datetime" in df.columns else "datetime"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df.set_index(dt_col, inplace=True)
    df.index.name = "datetime"
    return df


def load_oos() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "nifty_5m_oos_2015_2024.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)
    return df


def run_sweep_config(df: pd.DataFrame, config: dict) -> dict:
    """Run a single sweep configuration and return results."""
    engine = MultiStrategyEngine()

    engine.HTF_DEAD_ZONE_LO = config["dz_lo"]
    engine.HTF_DEAD_ZONE_HI = config["dz_hi"]
    engine.MIN_ADX = config["min_adx"]

    min_pb_count = config["min_pb"]
    min_conf = config["min_conf"]

    unique_days = sorted(set(df.index.date))
    trades = []

    for day in unique_days:
        day_data = df.loc[str(day)]
        if len(day_data) < 15:
            continue

        engine.reset_day()
        indicators = engine.precompute(day_data)

        for i in range(10, len(day_data)):
            time_str = day_data.index[i].strftime("%H:%M")
            signals = engine.scan(indicators, i, time_str)

            for sig in signals:
                if sig.signal_type != "PULLBACK":
                    continue
                if sig.confidence < min_conf:
                    continue
                if sig.pullback_count < min_pb_count:
                    continue

                entry_price = day_data["close"].iloc[i]
                theta = get_scaled_theta(entry_price)
                pnl, bars, reason = simulate_trade(
                    entry_price, sig.direction, 24,
                    day_data["close"], i, theta,
                )
                trades.append({"pnl": pnl, "reason": reason, "direction": sig.direction})

    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "pnl": 0, "tpd": 0}

    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / n * 100
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gp / gl if gl > 0 else float("inf")
    total_pnl = sum(t["pnl"] for t in trades)
    tpd = n / len(unique_days) if unique_days else 0

    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(total_pnl, 0), "tpd": round(tpd, 2)}


def main():
    logger.remove()

    print("Loading 2026 in-sample data...")
    df_is = load_2026()
    n_days_is = len(set(df_is.index.date))
    print(f"  {len(df_is)} candles, {n_days_is} days")
    print(f"  Using delta={settings.PREMIUM_DELTA}, SL={settings.PREMIUM_SL_PCT}%")

    sweep_grid = {
        "dz_lo":    [0, 3, 5, 8],
        "dz_hi":    [8, 10, 15, 20],
        "min_adx":  [15, 18, 20, 25],
        "min_pb":   [1, 2],
        "min_conf": [50, 60, 65, 70],
    }

    keys = list(sweep_grid.keys())
    combos = list(itertools.product(*(sweep_grid[k] for k in keys)))
    print(f"\nSweeping {len(combos)} configurations on in-sample data...")

    results = []
    for i, vals in enumerate(combos):
        config = dict(zip(keys, vals))
        r = run_sweep_config(df_is, config)
        r["config"] = config
        results.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    viable = [r for r in results if r["wr"] >= 50 and r["n"] >= 3]
    viable.sort(key=lambda r: (r["wr"], r["pf"], r["tpd"]), reverse=True)

    print(f"\nViable configs (WR >= 50%, n >= 3): {len(viable)} / {len(combos)}")
    print(f"\n{'Rank':>4} | {'N':>3} | {'WR%':>5} | {'PF':>5} | {'PnL':>10} | {'t/d':>5} | Config")
    print("-" * 90)

    for rank, r in enumerate(viable[:20], 1):
        c = r["config"]
        cfg_str = f"dz=[{c['dz_lo']},{c['dz_hi']}) adx>={c['min_adx']} pb>={c['min_pb']} conf>={c['min_conf']}"
        sign = "+" if r["pnl"] >= 0 else ""
        print(f"{rank:>4} | {r['n']:>3} | {r['wr']:>5.1f} | {r['pf']:>5.2f} | Rs {sign}{r['pnl']:>7,.0f} | {r['tpd']:>5.2f} | {cfg_str}")

    # Validate top configs on OOS data
    if viable:
        print(f"\n{'='*80}")
        print("OUT-OF-SAMPLE VALIDATION (2022-2024)")
        print(f"{'='*80}")

        try:
            df_oos_full = load_oos()
            df_oos = df_oos_full[df_oos_full.index.year >= 2022].copy()
            n_days_oos = len(set(df_oos.index.date))
            print(f"  {len(df_oos)} candles, {n_days_oos} days")
        except Exception as e:
            print(f"  Failed to load OOS: {e}")
            df_oos = None

        if df_oos is not None and len(df_oos) > 0:
            for rank, r in enumerate(viable[:5], 1):
                config = r["config"]
                r_oos = run_sweep_config(df_oos, config)
                c = config
                cfg_str = f"dz=[{c['dz_lo']},{c['dz_hi']}) adx>={c['min_adx']} pb>={c['min_pb']} conf>={c['min_conf']}"
                sign = "+" if r_oos["pnl"] >= 0 else ""
                print(f"  IS#{rank} -> OOS: {r_oos['n']:>3}t WR={r_oos['wr']:>5.1f}% PF={r_oos['pf']:>5.2f} PnL=Rs {sign}{r_oos['pnl']:>7,.0f} t/d={r_oos['tpd']:.2f} | {cfg_str}")

    # STOCH_CROSS baseline for comparison
    print(f"\n{'='*80}")
    print("STOCH_CROSS BASELINE (2026 in-sample)")
    print(f"{'='*80}")
    engine = MultiStrategyEngine()
    unique_days = sorted(set(df_is.index.date))
    sc_trades = []
    for day in unique_days:
        day_data = df_is.loc[str(day)]
        if len(day_data) < 15:
            continue
        engine.reset_day()
        indicators = engine.precompute(day_data)
        for i in range(10, len(day_data)):
            time_str = day_data.index[i].strftime("%H:%M")
            signals = engine.scan(indicators, i, time_str)
            for sig in signals:
                if sig.signal_type != "STOCH_CROSS":
                    continue
                entry_price = day_data["close"].iloc[i]
                theta = get_scaled_theta(entry_price)
                pnl, bars, reason = simulate_trade(entry_price, sig.direction, 24, day_data["close"], i, theta)
                sc_trades.append({"pnl": pnl})

    if sc_trades:
        n = len(sc_trades)
        wins = sum(1 for t in sc_trades if t["pnl"] > 0)
        wr = wins / n * 100
        gp = sum(t["pnl"] for t in sc_trades if t["pnl"] > 0)
        gl = abs(sum(t["pnl"] for t in sc_trades if t["pnl"] <= 0))
        pf = gp / gl if gl > 0 else float("inf")
        total = sum(t["pnl"] for t in sc_trades)
        print(f"  STOCH_CROSS: {n} trades, WR={wr:.0f}%, PF={pf:.2f}, PnL=Rs {total:+,.0f}, t/d={n/n_days_is:.2f}")


if __name__ == "__main__":
    main()
