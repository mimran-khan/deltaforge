"""Phase A validation: test ADX filter, shock detector, and combined on OOS + 2026 data.

Runs the production MultiStrategyEngine with external gates applied in test code only.
No production code is modified. Results determine whether to proceed with implementation.

Usage:
    python -m backtest.validate_improvements
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
from engine import indicators as ind

THETA_REFERENCE = 24_000
THETA_BASE = 0.30


def load_year(year: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / f"nifty_5m_oos_{year}.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)
    return df


def load_2026() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "nifty_5m_real.csv"
    df = pd.read_csv(path)
    # Handle both 'datetime' and 'Datetime' column names
    dt_col = "Datetime" if "Datetime" in df.columns else "datetime"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df.set_index(dt_col, inplace=True)
    df.index.name = "datetime"
    return df


def get_scaled_theta(nifty_price: float) -> float:
    return THETA_BASE * (nifty_price / THETA_REFERENCE)


def simulate_trade(entry_price, direction, hold_bars, close_series, entry_idx,
                   theta_per_candle, delta=0.45, base_premium=100.0, sl_pct=50.0):
    entry_prem = base_premium + settings.SLIPPAGE_POINTS
    sl_prem = entry_prem * (1 - sl_pct / 100)
    peak_prem = entry_prem

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
        peak_prem = max(peak_prem, cur_prem)

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


def run_backtest(df: pd.DataFrame, use_adx: bool = False, use_shock: bool = False,
                 use_dynamic_theta: bool = True, min_adx: float = 20.0,
                 shock_pct: float = 1.5, shock_lookback: int = 3, shock_halt: int = 6):
    """Run MultiStrategyEngine on data with optional external ADX/shock gates."""
    engine = MultiStrategyEngine()
    unique_days = sorted(set(df.index.date))
    trades = []

    for day in unique_days:
        day_data = df.loc[str(day)]
        if len(day_data) < 15:
            continue

        engine.reset_day()
        indicators = engine.precompute(day_data)

        adx_series = None
        if use_adx:
            adx_series = ind.adx(day_data['high'], day_data['low'], day_data['close'], 14)[0]

        shock_halt_until = -1

        for i in range(10, len(day_data)):
            time_str = day_data.index[i].strftime("%H:%M")

            if use_adx and adx_series is not None:
                adx_val = adx_series.iloc[i] if i < len(adx_series) and not np.isnan(adx_series.iloc[i]) else 25.0
                if adx_val < min_adx:
                    continue

            if use_shock:
                if i <= shock_halt_until:
                    continue
                if i >= shock_lookback:
                    cur = day_data['close'].iloc[i]
                    past = day_data['close'].iloc[i - shock_lookback]
                    if abs(cur - past) / past > (shock_pct / 100):
                        shock_halt_until = i + shock_halt
                        continue

            signals = engine.scan(indicators, i, time_str)
            for sig in signals:
                entry_price = day_data['close'].iloc[i]
                theta = get_scaled_theta(entry_price) if use_dynamic_theta else settings.PREMIUM_THETA_PER_CANDLE
                pnl, bars, reason = simulate_trade(
                    entry_price, sig.direction, 24,
                    day_data['close'], i, theta
                )
                trades.append({
                    "date": str(day),
                    "time": time_str,
                    "direction": sig.direction,
                    "signal_type": sig.signal_type,
                    "confidence": sig.confidence,
                    "entry_price": entry_price,
                    "pnl": pnl,
                    "bars_held": bars,
                    "exit_reason": reason,
                })

    return trades


def summarize(trades: list, label: str):
    if not trades:
        print(f"  {label}: 0 trades")
        return {}
    df = pd.DataFrame(trades)
    wins = len(df[df['pnl'] > 0])
    losses = len(df[df['pnl'] <= 0])
    wr = wins / len(df) * 100
    total_pnl = df['pnl'].sum()
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = abs(df[df['pnl'] <= 0]['pnl'].mean()) if losses > 0 else 1
    pf = avg_win * wins / (avg_loss * losses) if losses > 0 else float('inf')

    days = df['date'].nunique()
    tpd = len(df) / days if days > 0 else 0

    print(f"  {label}: {len(df)} trades | WR={wr:.0f}% | PF={pf:.2f} | "
          f"PnL=Rs {total_pnl:.0f} | Trades/day={tpd:.2f} | "
          f"W={wins} L={losses}")
    return {"trades": len(df), "wr": wr, "pf": pf, "pnl": total_pnl, "tpd": tpd}


def main():
    years = list(range(2015, 2025))

    print("=" * 80)
    print("PHASE A VALIDATION: ADX Filter + Shock Detector + Combined")
    print("=" * 80)

    all_results = {}

    for year in years:
        try:
            df = load_year(year)
        except FileNotFoundError:
            print(f"\n{year}: data not found, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"YEAR {year} ({len(df)} bars)")
        print(f"{'=' * 60}")

        baseline = run_backtest(df, use_adx=False, use_shock=False, use_dynamic_theta=True)
        b = summarize(baseline, "BASELINE (dynamic theta only)")

        adx_only = run_backtest(df, use_adx=True, use_shock=False, use_dynamic_theta=True)
        a = summarize(adx_only, "ADX filter (ADX>=20)")

        shock_only = run_backtest(df, use_adx=False, use_shock=True, use_dynamic_theta=True)
        s = summarize(shock_only, "SHOCK detector")

        combined = run_backtest(df, use_adx=True, use_shock=True, use_dynamic_theta=True)
        c = summarize(combined, "COMBINED (ADX+Shock)")

        all_results[year] = {"baseline": b, "adx": a, "shock": s, "combined": c}

    # 2026 in-sample
    print(f"\n{'=' * 60}")
    print("2026 IN-SAMPLE (nifty_5m_real.csv)")
    print(f"{'=' * 60}")
    try:
        df26 = load_2026()
        baseline26 = run_backtest(df26, use_adx=False, use_shock=False, use_dynamic_theta=True)
        b26 = summarize(baseline26, "BASELINE")

        adx26 = run_backtest(df26, use_adx=True, use_shock=False, use_dynamic_theta=True)
        a26 = summarize(adx26, "ADX filter")

        shock26 = run_backtest(df26, use_adx=False, use_shock=True, use_dynamic_theta=True)
        s26 = summarize(shock26, "SHOCK detector")

        combined26 = run_backtest(df26, use_adx=True, use_shock=True, use_dynamic_theta=True)
        c26 = summarize(combined26, "COMBINED")

        all_results[2026] = {"baseline": b26, "adx": a26, "shock": s26, "combined": c26}
    except FileNotFoundError:
        print("  2026 data not found")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE: Win Rates by Year and Configuration")
    print(f"{'=' * 80}")
    print(f"{'Year':>6} | {'Baseline WR':>12} | {'ADX WR':>8} | {'Shock WR':>9} | {'Combined WR':>12} | {'Comb Trades':>12}")
    print("-" * 80)
    for year in sorted(all_results.keys()):
        r = all_results[year]
        bwr = f"{r['baseline'].get('wr', 0):.0f}%" if r['baseline'] else "N/A"
        awr = f"{r['adx'].get('wr', 0):.0f}%" if r['adx'] else "N/A"
        swr = f"{r['shock'].get('wr', 0):.0f}%" if r['shock'] else "N/A"
        cwr = f"{r['combined'].get('wr', 0):.0f}%" if r['combined'] else "N/A"
        ct = f"{r['combined'].get('tpd', 0):.2f}/day" if r['combined'] else "N/A"
        print(f"{year:>6} | {bwr:>12} | {awr:>8} | {swr:>9} | {cwr:>12} | {ct:>12}")

    # Decision gate
    print(f"\n{'=' * 80}")
    print("DECISION GATE")
    print(f"{'=' * 80}")

    if 2026 in all_results and all_results[2026].get('combined'):
        c = all_results[2026]['combined']
        wr_ok = c.get('wr', 0) >= 85
        tpd_ok = c.get('tpd', 0) >= 0.4
        print(f"  2026 Combined WR >= 85%: {'PASS' if wr_ok else 'FAIL'} ({c.get('wr', 0):.0f}%)")
        print(f"  2026 Trades/day >= 0.4:  {'PASS' if tpd_ok else 'FAIL'} ({c.get('tpd', 0):.2f})")

    oos_improved = 0
    oos_total = 0
    for year in range(2015, 2025):
        if year in all_results:
            b = all_results[year].get('baseline', {})
            c = all_results[year].get('combined', {})
            if b and c:
                oos_total += 1
                if c.get('wr', 0) >= b.get('wr', 0):
                    oos_improved += 1
    print(f"  OOS years improved or held: {oos_improved}/{oos_total}")

    if 2026 in all_results and all_results[2026].get('combined'):
        c = all_results[2026]['combined']
        if c.get('wr', 0) >= 85 and c.get('tpd', 0) >= 0.4:
            print("\n  >>> VERDICT: PROCEED to Phase B implementation <<<")
        else:
            print("\n  >>> VERDICT: DO NOT proceed -- adjust parameters <<<")


if __name__ == "__main__":
    logger.remove()
    main()
