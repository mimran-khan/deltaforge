"""Comprehensive backtest for the Pullback-in-Trend engine.

Tests the PullbackEngine with:
  - Realistic ATM options model (delta 0.45, theta, costs)
  - Bar-by-bar SL/TP simulation
  - Walk-forward validation (train/test split)
  - Monte Carlo robustness check
  - Parameter sweep for optimization
  - Daily compounding simulation

Usage:
  python backtest/pullback_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.pullback_engine import PullbackEngine, PullbackSignal


# ═══════════════════════════════════════════════════════════════════
#  OPTIONS MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════
ATM_PREMIUM = 100.0
ATM_DELTA = 0.45
THETA_PER_CANDLE = 0.30     # theta decay per 5-min candle
LOT_SIZE = 75
BROKERAGE_RT = 40.0          # round-trip brokerage
STT_RATE = 0.000625
STAMP_RATE = 0.00003
SLIPPAGE = 1.5               # Rs per unit

STARTING_CAPITAL = 10000.0


def simulate_option_trade(
    candles: pd.DataFrame,
    entry_idx: int,
    direction: str,
    sl_pct: float = 30.0,
    tp_pct: float = 30.0,
    max_hold: int = 24,
) -> dict:
    """Simulate a single ATM option trade bar-by-bar.

    Returns dict with trade results.
    """
    entry_price = candles['close'].iloc[entry_idx]
    entry_prem = ATM_PREMIUM + SLIPPAGE

    sl_prem = entry_prem * (1 - sl_pct / 100)
    tp_prem = entry_prem * (1 + tp_pct / 100)

    exit_prem = None
    exit_reason = None
    hold_candles = 0
    peak_prem = entry_prem
    trough_prem = entry_prem

    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(candles))):
        hold_candles += 1
        cur_price = candles['close'].iloc[j]

        if direction == "LONG":
            nifty_move = cur_price - entry_price
        else:
            nifty_move = entry_price - cur_price

        cur_prem = entry_prem + (nifty_move * ATM_DELTA) - (hold_candles * THETA_PER_CANDLE)
        cur_prem = max(cur_prem, 0.5)

        peak_prem = max(peak_prem, cur_prem)
        trough_prem = min(trough_prem, cur_prem)

        if cur_prem <= sl_prem:
            exit_prem = sl_prem
            exit_reason = "SL"
            break

        if cur_prem >= tp_prem:
            exit_prem = tp_prem
            exit_reason = "TP"
            break

    if exit_prem is None:
        exit_prem = cur_prem
        exit_reason = "TIME"

    exit_prem -= SLIPPAGE

    raw_pnl = (exit_prem - entry_prem) * LOT_SIZE
    stt = exit_prem * LOT_SIZE * STT_RATE
    stamp = entry_prem * LOT_SIZE * STAMP_RATE
    costs = BROKERAGE_RT + stt + stamp
    net_pnl = raw_pnl - costs

    return {
        'entry_idx': entry_idx,
        'direction': direction,
        'entry_price': entry_price,
        'entry_prem': entry_prem,
        'exit_prem': exit_prem + SLIPPAGE,
        'exit_reason': exit_reason,
        'hold_candles': hold_candles,
        'nifty_move': nifty_move,
        'peak_prem': peak_prem,
        'raw_pnl': raw_pnl,
        'costs': costs,
        'net_pnl': net_pnl,
    }


def run_backtest(
    df: pd.DataFrame,
    trading_days: list,
    sl_pct: float = 30.0,
    tp_pct: float = 30.0,
    max_hold: int = 24,
    htf_bull: int = 55,
    htf_bear: int = 45,
    ltf_os: int = 40,
    ltf_ob: int = 60,
    vwap_dev: float = 1.0,
    max_trades: int = 3,
    verbose: bool = False,
) -> dict:
    """Run full backtest across trading days."""
    engine = PullbackEngine()
    engine.HTF_BULL_RSI = htf_bull
    engine.HTF_BEAR_RSI = htf_bear
    engine.LTF_OVERSOLD = ltf_os
    engine.LTF_OVERBOUGHT = ltf_ob
    engine.VWAP_DEV = vwap_dev
    engine.MAX_SIGNALS_PER_DAY = max_trades

    all_trades = []
    daily_pnls = []
    daily_details = []
    capital = STARTING_CAPITAL

    for day in trading_days:
        day_df = df[pd.Series(df.index.date).values == day].copy()
        if len(day_df) < 30:
            continue

        engine.reset_day()
        indicators = engine.precompute(day_df)

        day_trades = []
        used_bars = set()

        for i in range(8, len(day_df) - 1):
            if i in used_bars:
                continue
            if len(day_trades) >= max_trades:
                break

            t = day_df.index[i].strftime('%H:%M')
            signals = engine.scan(indicators, i, t)

            for sig in signals:
                remaining = len(day_df) - i - 1
                if remaining < 3:
                    continue

                result = simulate_option_trade(
                    day_df, i, sig.direction,
                    sl_pct=sl_pct, tp_pct=tp_pct,
                    max_hold=min(max_hold, remaining),
                )
                result['signal_type'] = sig.signal_type
                result['confidence'] = sig.confidence
                result['day'] = day

                day_trades.append(result)
                used_bars.update(range(i, i + result['hold_candles'] + 1))
                break

        day_pnl = sum(t['net_pnl'] for t in day_trades)
        daily_pnls.append(day_pnl)
        capital += day_pnl
        all_trades.extend(day_trades)

        daily_details.append({
            'day': day,
            'trades': len(day_trades),
            'pnl': day_pnl,
            'capital': capital,
        })

    return {
        'trades': all_trades,
        'daily_pnls': daily_pnls,
        'daily_details': daily_details,
        'final_capital': capital,
    }


def monte_carlo(daily_pnls: list, n_sims: int = 5000) -> dict:
    """Monte Carlo simulation by shuffling daily P&L sequence."""
    if not daily_pnls:
        return {}

    pnls = np.array(daily_pnls)
    final_capitals = []

    for _ in range(n_sims):
        shuffled = np.random.permutation(pnls)
        equity = STARTING_CAPITAL + np.cumsum(shuffled)
        final_capitals.append(equity[-1])

    final_capitals = np.array(final_capitals)
    return {
        'prob_profit': (final_capitals > STARTING_CAPITAL).mean() * 100,
        'median_capital': np.median(final_capitals),
        'p5_capital': np.percentile(final_capitals, 5),
        'p95_capital': np.percentile(final_capitals, 95),
        'worst': np.min(final_capitals),
        'best': np.max(final_capitals),
    }


def print_results(label: str, result: dict, mc: dict = None):
    """Pretty-print backtest results."""
    trades = result['trades']
    daily = result['daily_pnls']

    if not trades:
        print(f"\n  {label}: NO TRADES")
        return

    wins = [t for t in trades if t['net_pnl'] > 0]
    losses = [t for t in trades if t['net_pnl'] <= 0]
    n = len(trades)
    wr = len(wins) / n * 100

    total_pnl = sum(t['net_pnl'] for t in trades)
    avg_pnl = total_pnl / n

    gross_profit = sum(t['net_pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['net_pnl'] for t in losses)) if losses else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([t['net_pnl'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['net_pnl'] for t in losses])) if losses else 0

    profit_days = sum(1 for p in daily if p > 0)
    total_days = len(daily)

    print(f"\n  {'═'*60}")
    print(f"  {label}")
    print(f"  {'═'*60}")
    print(f"  Trades: {n} over {total_days} days ({n/total_days:.1f}/day)")
    print(f"  Win Rate: {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg Win: Rs {avg_win:+,.0f} | Avg Loss: Rs {avg_loss:,.0f}")
    print(f"  Total P&L: Rs {total_pnl:+,.0f}")
    print(f"  Return: {total_pnl/STARTING_CAPITAL*100:+.1f}%")
    print(f"  Final Capital: Rs {result['final_capital']:,.0f}")

    print(f"\n  Daily:")
    print(f"    Profit days: {profit_days}/{total_days} ({profit_days/total_days*100:.0f}%)")
    print(f"    Avg daily: Rs {np.mean(daily):+,.0f}")
    print(f"    Best day: Rs {max(daily):+,.0f}")
    print(f"    Worst day: Rs {min(daily):+,.0f}")
    print(f"    Days ≥ Rs 500: {sum(1 for p in daily if p >= 500)}")
    print(f"    Days ≥ Rs 1000: {sum(1 for p in daily if p >= 1000)}")

    # Exit breakdown
    by_exit = {}
    for t in trades:
        r = t['exit_reason']
        by_exit.setdefault(r, []).append(t['net_pnl'])
    print(f"\n  Exit breakdown:")
    for r, pnls in sorted(by_exit.items()):
        wr_e = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"    {r}: {len(pnls)} trades, WR={wr_e:.0f}%, avg Rs {np.mean(pnls):+,.0f}")

    # Signal type breakdown
    by_sig = {}
    for t in trades:
        s = t.get('signal_type', 'UNKNOWN')
        by_sig.setdefault(s, []).append(t['net_pnl'])
    print(f"\n  Signal types:")
    for s, pnls in sorted(by_sig.items()):
        wr_s = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"    {s}: {len(pnls)} trades, WR={wr_s:.0f}%, avg Rs {np.mean(pnls):+,.0f}")

    # Direction breakdown
    by_dir = {}
    for t in trades:
        d = t['direction']
        by_dir.setdefault(d, []).append(t['net_pnl'])
    print(f"\n  Direction:")
    for d, pnls in sorted(by_dir.items()):
        wr_d = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"    {d}: {len(pnls)} trades, WR={wr_d:.0f}%, avg Rs {np.mean(pnls):+,.0f}")

    if mc:
        print(f"\n  Monte Carlo ({len(daily)} day sequences, 5000 shuffles):")
        print(f"    Prob profit: {mc['prob_profit']:.1f}%")
        print(f"    Median final: Rs {mc['median_capital']:,.0f}")
        print(f"    5th pct: Rs {mc['p5_capital']:,.0f}")
        print(f"    95th pct: Rs {mc['p95_capital']:,.0f}")

    # Compounding simulation
    cap = STARTING_CAPITAL
    for dp in daily:
        scale = min(cap / STARTING_CAPITAL, 3.0)
        cap += dp * scale
        cap = max(cap, 3000)  # minimum capital floor
    print(f"\n  Compounding simulation:")
    print(f"    Rs {STARTING_CAPITAL:,.0f} → Rs {cap:,.0f} ({(cap-STARTING_CAPITAL)/STARTING_CAPITAL*100:+.1f}%)")


def main():
    print("=" * 72)
    print("  PULLBACK-IN-TREND ENGINE -- COMPREHENSIVE BACKTEST")
    print("=" * 72)

    csv_path = ROOT / "data" / "nifty_5m_real.csv"
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found")
        return

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    days = sorted(set(df.index.date))
    non_tues = [d for d in days if d.weekday() != 1]

    print(f"\n  Data: {len(df)} candles, {len(non_tues)} trading days (excl. Tuesday)")

    # ── FULL PERIOD BACKTEST ──
    result = run_backtest(
        df, non_tues,
        sl_pct=30, tp_pct=30, max_hold=24,
        max_trades=3,
    )
    mc = monte_carlo(result['daily_pnls'])
    print_results("FULL PERIOD (default params)", result, mc)

    # ── WALK-FORWARD: Train on first 70%, test on last 30% ──
    split = int(len(non_tues) * 0.7)
    train_days = non_tues[:split]
    test_days = non_tues[split:]

    print(f"\n\n  {'─'*60}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  Train: {len(train_days)} days | Test: {len(test_days)} days")
    print(f"  {'─'*60}")

    # Train
    train_result = run_backtest(df, train_days, sl_pct=30, tp_pct=30, max_trades=3)
    print_results(f"TRAIN ({len(train_days)} days)", train_result)

    # Test (out-of-sample)
    test_result = run_backtest(df, test_days, sl_pct=30, tp_pct=30, max_trades=3)
    test_mc = monte_carlo(test_result['daily_pnls'])
    print_results(f"TEST / OUT-OF-SAMPLE ({len(test_days)} days)", test_result, test_mc)

    # ── PARAMETER SWEEP ──
    print(f"\n\n  {'─'*60}")
    print(f"  PARAMETER SWEEP (finding optimal SL/TP)")
    print(f"  {'─'*60}")

    best_pf = 0
    best_params = {}

    for sl in [20, 25, 30, 35]:
        for tp in [20, 25, 30, 35, 40]:
            for mh in [12, 18, 24]:
                r = run_backtest(df, non_tues, sl_pct=sl, tp_pct=tp, max_hold=mh, max_trades=3)
                if not r['trades']:
                    continue
                trades = r['trades']
                wins = [t for t in trades if t['net_pnl'] > 0]
                losses = [t for t in trades if t['net_pnl'] <= 0]
                gp = sum(t['net_pnl'] for t in wins) if wins else 0
                gl = abs(sum(t['net_pnl'] for t in losses)) if losses else 1
                pf = gp / gl if gl > 0 else 0
                total = sum(t['net_pnl'] for t in trades)
                wr = len(wins)/len(trades)*100

                if pf > best_pf and len(trades) >= 10:
                    best_pf = pf
                    best_params = {'sl': sl, 'tp': tp, 'max_hold': mh,
                                   'pf': pf, 'wr': wr, 'pnl': total,
                                   'trades': len(trades)}

                if pf >= 2.0 and len(trades) >= 10:
                    print(f"    SL={sl}% TP={tp}% Hold={mh}: "
                          f"{len(trades)}t WR={wr:.0f}% PF={pf:.2f} "
                          f"PnL=Rs{total:+,.0f}")

    if best_params:
        print(f"\n  BEST: SL={best_params['sl']}% TP={best_params['tp']}% "
              f"Hold={best_params['max_hold']}: "
              f"PF={best_params['pf']:.2f} WR={best_params['wr']:.0f}% "
              f"PnL=Rs{best_params['pnl']:+,.0f} ({best_params['trades']} trades)")

        # Run best params with MC
        best_r = run_backtest(
            df, non_tues,
            sl_pct=best_params['sl'], tp_pct=best_params['tp'],
            max_hold=best_params['max_hold'], max_trades=3,
        )
        best_mc = monte_carlo(best_r['daily_pnls'])
        print_results("OPTIMAL PARAMETERS", best_r, best_mc)


if __name__ == "__main__":
    main()
