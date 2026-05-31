"""Walk-Forward V4 -- Production-grade multi-indicator confluence.

Key principles:
  1. 200+ indicator signals per candle (12 categories)
  2. ZERO random noise -- deterministic premium model
  3. Same code paths for backtest and live trading
  4. Progressive position sizing (smaller % as capital grows)
  5. Walk-forward: optimize on train, validate on unseen test data

Usage:
    python -m backtest.walk_forward_v4
"""

from __future__ import annotations
import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.confluence import ConfluenceEngine
from engine.premium_model import create_premium_state


def backtest_confluence(
    df: pd.DataFrame,
    lot_size: int = 75,
    starting_capital: float = 10000,
    deploy_pct: float = 80.0,
    entry_threshold: float = 40.0,
    min_strength: str = "MODERATE",
    max_trades_day: int = 4,
    max_consec_loss: int = 3,
    daily_loss_pct: float = 15,
    delta: float = 0.45,
    base_premium: float = 95,
    theta_per_candle: float = 0.15,
    premium_sl_pct: float = 35,
    trail_trigger_pct: float = 50,
    trail_pct: float = 30,
    dd_reduce_threshold: float = 30,
    dd_reduce_factor: float = 0.5,
    entry_start: str = "09:30",
    entry_end: str = "14:00",
    eod_exit: str = "15:15",
    weight_overrides: dict | None = None,
) -> dict:
    """Production-parity backtest with 200+ indicator confluence."""

    strength_order = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3, "EXTREME": 4}
    min_str_val = strength_order.get(min_strength, 2)

    engine = ConfluenceEngine(weight_overrides=weight_overrides)

    unique_days = sorted(set(df.index.date))
    capital = starting_capital
    peak = starting_capital
    trades = []
    equity_curve = []

    for day in unique_days:
        dates_arr = pd.Series(df.index.date)
        day_mask = (dates_arr == day).values
        day_df = df[day_mask].copy()
        if len(day_df) < 15:
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0, "lots": 0,
                                  "best_score": 0, "signals": 0})
            continue

        try:
            indicators = engine.precompute(day_df)
        except Exception:
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0, "lots": 0,
                                  "best_score": 0, "signals": 0})
            continue

        day_start_cap = capital
        day_pnl = 0.0
        day_trades = 0
        consec_loss = 0
        loss_limit = day_start_cap * (daily_loss_pct / 100)

        # Conservative progressive position sizing
        dd_from_peak = (peak - capital) / peak * 100 if peak > 0 else 0
        sizing_mult = dd_reduce_factor if dd_from_peak > dd_reduce_threshold else 1.0
        growth = capital / starting_capital
        if growth > 5:
            eff_deploy = deploy_pct * 0.25
        elif growth > 3:
            eff_deploy = deploy_pct * 0.4
        elif growth > 1.5:
            eff_deploy = deploy_pct * 0.6
        else:
            eff_deploy = deploy_pct
        cost_per_lot = base_premium * lot_size
        deployable = capital * (eff_deploy / 100) * sizing_mult
        day_lots = max(1, int(deployable / cost_per_lot))

        best_score = 0
        signal_count = 0

        for i in range(5, len(day_df)):
            ts = day_df.index[i]
            t_str = ts.strftime("%H:%M")

            if t_str < entry_start or t_str > entry_end:
                continue
            if day_trades >= max_trades_day:
                break
            if consec_loss >= max_consec_loss:
                break
            if day_pnl < 0 and abs(day_pnl) >= loss_limit:
                break

            result = engine.score(indicators, i)
            abs_score = abs(result.score)
            if abs_score > abs(best_score):
                best_score = result.score

            if abs_score < entry_threshold:
                continue
            if strength_order.get(result.strength, 0) < min_str_val:
                continue

            signal_count += 1

            entry_idx = day_df["close"].iloc[i]
            prem_state = create_premium_state(
                entry_index_price=entry_idx,
                direction=result.direction,
                base_premium=base_premium,
                delta=delta,
                theta_per_candle=theta_per_candle,
                sl_pct=premium_sl_pct,
                confluence_score=result.score,
            )
            qty = day_lots * lot_size

            exit_prem = prem_state.entry_premium
            exit_reason = "EOD"
            exit_time = day_df.index[-1]

            for k in range(i + 1, len(day_df)):
                candles_elapsed = k - i
                current_idx = day_df["close"].iloc[k]
                current_prem = prem_state.current_premium(current_idx, candles_elapsed)

                trail_floor = prem_state.update_trail(
                    current_prem, trail_trigger_pct, trail_pct)

                reason = prem_state.check_exit(current_prem, trail_floor)
                if reason:
                    exit_prem = (prem_state.sl_premium if reason == "SL"
                                 else prem_state.target_premium if reason == "TGT"
                                 else trail_floor if reason == "TRAIL"
                                 else current_prem)
                    exit_reason = reason
                    exit_time = day_df.index[k]
                    break

                if day_df.index[k].strftime("%H:%M") >= eod_exit:
                    exit_prem = current_prem
                    exit_reason = "EOD"
                    exit_time = day_df.index[k]
                    break

            pnl = (exit_prem - prem_state.entry_premium) * qty
            capital += pnl
            day_pnl += pnl
            day_trades += 1
            consec_loss = consec_loss + 1 if pnl < 0 else 0

            trades.append({
                "dir": result.direction,
                "entry_time": str(day_df.index[i]),
                "exit_time": str(exit_time),
                "entry_prem": prem_state.entry_premium,
                "exit_prem": round(exit_prem, 2),
                "qty": qty, "lots": day_lots,
                "pnl": round(pnl, 0), "reason": exit_reason,
                "capital": round(capital, 0),
                "confluence": result.score,
                "strength": result.strength,
                "bullish": result.bullish_count,
                "bearish": result.bearish_count,
                "total_ind": result.total_indicators,
            })

            if capital <= 0:
                break

        if capital > peak:
            peak = capital

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades, "lots": day_lots,
            "best_score": round(best_score, 1),
            "signals": signal_count,
        })

        if capital <= 0:
            break

    return _stats(trades, equity_curve, starting_capital, capital, peak, unique_days)


def _stats(trades, equity_curve, start_cap, capital, peak, days):
    total_pnl = capital - start_cap
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gp / gl if gl > 0 else float("inf")

    eq = [e["capital"] for e in equity_curve]
    max_dd = 0
    if eq:
        pa = np.maximum.accumulate(eq)
        dd = [(p - v) / p * 100 if p > 0 else 0 for p, v in zip(pa, eq)]
        max_dd = max(dd) if dd else 0

    active = [e for e in equity_curve if e["trades"] > 0]
    prof_days = sum(1 for e in equity_curve if e["daily_pnl"] > 0)
    loss_days = sum(1 for e in equity_curve if e["daily_pnl"] < 0)

    exits = {}
    for t in trades:
        er = t["reason"]
        exits.setdefault(er, {"count": 0, "pnl": 0})
        exits[er]["count"] += 1
        exits[er]["pnl"] += t["pnl"]

    strengths = {}
    for t in trades:
        st = t.get("strength", "?")
        strengths.setdefault(st, {"w": 0, "l": 0, "pnl": 0})
        if t["pnl"] > 0:
            strengths[st]["w"] += 1
        else:
            strengths[st]["l"] += 1
        strengths[st]["pnl"] += t["pnl"]

    return {
        "capital": round(capital, 0),
        "pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / start_cap * 100, 1),
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1),
        "avg_win": round(np.mean([t["pnl"] for t in wins]), 0) if wins else 0,
        "avg_loss": round(np.mean([t["pnl"] for t in losses]), 0) if losses else 0,
        "profit_factor": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "prof_days": prof_days, "loss_days": loss_days,
        "trading_days": len(days), "active_days": len(active),
        "equity_curve": equity_curve, "trade_list": trades,
        "exit_reasons": exits, "strength_stats": strengths,
    }


def optimize(train_df, start_cap=10000):
    total = 0

    def sc(r):
        if r["trades"] < 3:
            return -100
        # Heavy DD penalty + consistency focus (not raw returns)
        pf_capped = min(r["profit_factor"], 3)
        wr_capped = min(r["win_rate"], 70)
        dd_penalty = r["max_dd"] * 1.5  # heavy DD penalty
        consistency_bonus = min(r["prof_days"], r["loss_days"] + 1) * 5
        return pf_capped * wr_capped - dd_penalty + consistency_bonus

    # Phase 1: Threshold + strength (conservative range to avoid overfit)
    print("\n  Phase 1: Confluence threshold...")
    p1 = []
    for thresh, ms in itertools.product(
        [40, 45, 50, 55, 60, 65, 70],
        ["MODERATE", "STRONG"]
    ):
        r = backtest_confluence(train_df, starting_capital=start_cap,
                                entry_threshold=thresh, min_strength=ms)
        p1.append({"p": {"entry_threshold": thresh, "min_strength": ms},
                    "s": sc(r), "pf": r["profit_factor"], "wr": r["win_rate"],
                    "t": r["trades"], "dd": r["max_dd"], "ret": r["return_pct"]})
        total += 1
    p1.sort(key=lambda x: x["s"], reverse=True)
    b1 = p1[0]["p"]
    print(f"    Best: thresh={b1['entry_threshold']}, str={b1['min_strength']} "
          f"(PF={p1[0]['pf']}, WR={p1[0]['wr']}%, {p1[0]['t']}t, "
          f"DD={p1[0]['dd']}%, score={p1[0]['s']:.0f})")

    # Phase 2: Premium SL + trailing (tighter range for robustness)
    print("  Phase 2: Premium SL + trailing...")
    p2 = []
    for sl, tt, tp in itertools.product(
        [15, 20, 25, 30], [40, 60, 80], [25, 35, 45]
    ):
        r = backtest_confluence(train_df, starting_capital=start_cap,
                                **b1, premium_sl_pct=sl,
                                trail_trigger_pct=tt, trail_pct=tp)
        p2.append({"p": {"premium_sl_pct": sl, "trail_trigger_pct": tt,
                          "trail_pct": tp},
                    "s": sc(r), "pf": r["profit_factor"], "wr": r["win_rate"],
                    "t": r["trades"], "dd": r["max_dd"]})
        total += 1
    p2.sort(key=lambda x: x["s"], reverse=True)
    b2 = p2[0]["p"]
    print(f"    Best: SL={b2['premium_sl_pct']}%, trail_trig={b2['trail_trigger_pct']}%, "
          f"trail={b2['trail_pct']}% (PF={p2[0]['pf']}, score={p2[0]['s']:.0f})")

    # Phase 3: Risk mgmt (conservative: fewer trades/day)
    print("  Phase 3: Risk management...")
    p3 = []
    for mt, mc, dd in itertools.product(
        [1, 2, 3], [2, 3], [15, 25, 35]
    ):
        r = backtest_confluence(train_df, starting_capital=start_cap,
                                **b1, **b2,
                                max_trades_day=mt, max_consec_loss=mc,
                                dd_reduce_threshold=dd)
        p3.append({"p": {"max_trades_day": mt, "max_consec_loss": mc,
                          "dd_reduce_threshold": dd},
                    "s": sc(r), "pf": r["profit_factor"], "wr": r["win_rate"],
                    "t": r["trades"], "dd": r["max_dd"]})
        total += 1
    p3.sort(key=lambda x: x["s"], reverse=True)
    b3 = p3[0]["p"]
    print(f"    Best: trades/d={b3['max_trades_day']}, consec={b3['max_consec_loss']}, "
          f"dd_reduce={b3['dd_reduce_threshold']}% (score={p3[0]['s']:.0f})")

    # Phase 4: Fixed realistic premium model (no optimization to prevent overfit)
    # ATM Nifty option delta is typically 0.45-0.50
    # Theta decay per 5-min candle is ~0.15 for weekly options
    b4 = {"delta": 0.45, "theta_per_candle": 0.15}
    print(f"  Phase 4: Fixed premium model (delta={b4['delta']}, theta={b4['theta_per_candle']})")
    print(f"    (Fixed to realistic ATM values to prevent overfit)")

    print(f"\n  Total combos tested: {total}")
    all_p = {**b1, **b2, **b3, **b4}
    final = backtest_confluence(train_df, starting_capital=start_cap, **all_p)
    return {"best_params": all_p, "train_result": final}


def walk_forward(df, train_pct=0.6, start_cap=10000):
    unique_days = sorted(set(df.index.date))
    split = int(len(unique_days) * train_pct)
    train_days, test_days = unique_days[:split], unique_days[split:]

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD V4 -- PRODUCTION-GRADE CONFLUENCE")
    print(f"  200+ indicator signals | 12 categories | deterministic premium")
    print(f"  Categories: trend, momentum, volatility, volume, trend_strength,")
    print(f"  structure, candlestick, statistical, divergence, HTF, derivative")
    print(f"{'='*72}")
    print(f"  Data    : {unique_days[0]} to {unique_days[-1]} ({len(unique_days)} days)")
    print(f"  Train   : {train_days[0]} to {train_days[-1]} ({len(train_days)} days)")
    print(f"  Test OOS: {test_days[0]} to {test_days[-1]} ({len(test_days)} days)")

    dates_arr = pd.Series(df.index.date)
    train_df = df[dates_arr.isin(set(train_days)).values].copy()
    test_df = df[dates_arr.isin(set(test_days)).values].copy()

    print(f"\n  PHASE 1: Optimizing on train ({len(train_days)} days)...")
    opt = optimize(train_df, start_cap)
    bp = opt["best_params"]
    tr = opt["train_result"]

    print(f"\n{'='*72}")
    print(f"  TRAIN RESULTS ({train_days[0]} to {train_days[-1]})")
    print(f"{'='*72}")
    _print(tr)

    print(f"\n  PHASE 2: OUT-OF-SAMPLE ({len(test_days)} days)...")
    te = backtest_confluence(test_df, starting_capital=start_cap, **bp)

    print(f"\n{'='*72}")
    print(f"  TEST RESULTS (OOS) ({test_days[0]} to {test_days[-1]})")
    print(f"{'='*72}")
    _print(te)

    # Stability
    print(f"\n{'='*72}")
    print(f"  STABILITY & VERDICT")
    print(f"{'='*72}")
    pf_r = min(tr["profit_factor"], te["profit_factor"]) / \
           max(tr["profit_factor"], te["profit_factor"]) \
           if max(tr["profit_factor"], te["profit_factor"]) > 0 else 0
    wr_d = abs(tr["win_rate"] - te["win_rate"])
    print(f"  {'':25s} {'TRAIN':>10} {'TEST(OOS)':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Profit Factor':25s} {tr['profit_factor']:>10.2f} {te['profit_factor']:>10.2f}")
    print(f"  {'Win Rate':25s} {tr['win_rate']:>9.0f}% {te['win_rate']:>9.0f}%")
    print(f"  {'Max Drawdown':25s} {tr['max_dd']:>9.0f}% {te['max_dd']:>9.0f}%")
    print(f"  {'Return':25s} {tr['return_pct']:>9.0f}% {te['return_pct']:>9.0f}%")
    print(f"  {'Trades':25s} {tr['trades']:>10d} {te['trades']:>10d}")
    print(f"  {'PF Stability':25s} {pf_r:>10.2f}")

    if te["profit_factor"] > 1.3 and te["win_rate"] > 50 and te["max_dd"] < 40:
        v = "STRONG PASS"
    elif te["profit_factor"] > 1.15 and te["win_rate"] > 45:
        v = "PASS"
    elif te["profit_factor"] > 1.0 and te["win_rate"] > 40:
        v = "MARGINAL"
    else:
        v = "FAIL"
    print(f"\n  VERDICT: {v}")

    print(f"\n{'='*72}")
    print(f"  OPTIMIZED PARAMETERS")
    print(f"{'='*72}")
    for k, val in bp.items():
        print(f"    {k:<30} = {val}")
    print(f"{'='*72}")

    return {"best_params": bp, "train": tr, "test": te, "verdict": v}


def _print(r, pfx="  "):
    pnl = r["pnl"]
    s = "+" if pnl >= 0 else ""
    print(f"{pfx}Capital      : Rs {r['capital']:>12,.0f}")
    print(f"{pfx}P&L          : Rs {s}{pnl:>11,.0f} ({r['return_pct']}%)")
    print(f"{pfx}Trades       : {r['trades']} ({r['wins']}W / {r['losses']}L)")
    print(f"{pfx}Win Rate     : {r['win_rate']}%")
    print(f"{pfx}Profit Factor: {r['profit_factor']}")
    print(f"{pfx}Max Drawdown : {r['max_dd']}%")
    print(f"{pfx}Avg Win/Loss : Rs {r['avg_win']:>7,.0f} / Rs {r['avg_loss']:>7,.0f}")
    print(f"{pfx}Days (W/L)   : {r['prof_days']}W / {r['loss_days']}L / "
          f"{r['trading_days'] - r['active_days']} skip")

    ex = r.get("exit_reasons", {})
    if ex:
        print(f"{pfx}Exits        :", end="")
        parts = []
        for er in sorted(ex):
            d = ex[er]
            parts.append(f"{er}={d['count']}(Rs{d['pnl']:+,.0f})")
        print(" " + " | ".join(parts))

    ss = r.get("strength_stats", {})
    if ss:
        print(f"{pfx}By strength  :", end="")
        parts = []
        for nm in ["WEAK", "MODERATE", "STRONG", "EXTREME"]:
            if nm in ss:
                d = ss[nm]
                tot = d["w"] + d["l"]
                wr = d["w"] / tot * 100 if tot else 0
                parts.append(f"{nm}={tot}t/{wr:.0f}%W/Rs{d['pnl']:+,.0f}")
        print(" " + " | ".join(parts))

    ec = r.get("equity_curve", [])
    if ec and len(ec) > 3:
        print(f"{pfx}Equity curve :")
        n = len(ec)
        for idx in sorted(set(min(x, n-1) for x in [0, n//4, n//2, 3*n//4, n-1])):
            e = ec[idx]
            print(f"{pfx}  Day {idx+1:>3d} ({e['date']}): "
                  f"Rs {e['capital']:>10,.0f} [{e['trades']}t, Rs{e['daily_pnl']:>+8,.0f}, "
                  f"conf={e.get('best_score',0):>+.0f}, {e.get('signals',0)} sigs]")


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    data_path = settings.DATA_DIR / "nifty_5m_real.csv"
    if not data_path.exists():
        print("ERROR: No data. Run: python -m backtest.data_fetcher")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    days = len(set(df.index.date))
    print(f"Real Nifty 5-min data: {len(df)} candles, {days} trading days")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    result = walk_forward(df, train_pct=0.6, start_cap=10000)

    if result["test"]["trade_list"]:
        pd.DataFrame(result["test"]["trade_list"]).to_csv(
            settings.DATA_DIR / "v4_test_trades.csv", index=False)
    if result["test"]["equity_curve"]:
        pd.DataFrame(result["test"]["equity_curve"]).to_csv(
            settings.DATA_DIR / "v4_test_equity.csv", index=False)

    return result


if __name__ == "__main__":
    main()
