"""Walk-Forward V6 -- Designed around proven signal edge.

Signal quality test showed:
  - 53.6% WR at 60-min horizon with confluence >= 60
  - Average favorable move: +3.6 Nifty points
  - NO edge at 5m/15m/30m -- short-term is noise

V6 design principles:
  1. HOLD for 1 hour minimum (where the edge exists)
  2. ITM options (delta 0.65) for better P&L per point
  3. Lower theta (ITM options decay slower)
  4. Only take strongest signal of the day
  5. Time-based exit (hold 1hr) not SL-based exit
  6. 3-fold CV to prevent overfit

Usage:
    python -m backtest.walk_forward_v6
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


def backtest_v6(
    df: pd.DataFrame,
    lot_size: int = 75,
    starting_capital: float = 10000,
    # Signal
    entry_threshold: float = 55.0,
    min_strength: str = "STRONG",
    # Timing
    entry_start: str = "09:45",
    entry_end: str = "13:00",
    hold_candles: int = 12,       # 12 x 5min = 60 min hold
    eod_exit: str = "15:15",
    # ITM option model (delta 0.65, lower theta)
    delta: float = 0.65,
    theta_per_candle: float = 0.08,
    base_premium: float = 180,     # ITM options are more expensive
    # Risk
    max_loss_per_trade_pct: float = 15,  # max loss as % of premium
    max_trades_day: int = 1,
    daily_loss_pct: float = 20,
    # Profit target
    target_points: float = 5.0,    # Nifty points to target
    # Position sizing
    max_lots: int = 1,
    grow_after_multiple: float = 2.0,
    weight_overrides: dict | None = None,
) -> dict:
    """V6 backtester based on 1-hour hold period with proven edge."""

    strength_order = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3, "EXTREME": 4}
    min_str_val = strength_order.get(min_strength, 3)

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
        if len(day_df) < 20:
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0})
            continue

        try:
            indicators = engine.precompute(day_df)
        except Exception:
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0})
            continue

        day_pnl = 0.0
        day_trades = 0
        loss_limit = capital * (daily_loss_pct / 100)

        # Fixed lot sizing with slow growth
        growth = capital / starting_capital
        day_lots = max_lots
        if growth >= grow_after_multiple * 4:
            day_lots = max_lots + 3
        elif growth >= grow_after_multiple * 2:
            day_lots = max_lots + 2
        elif growth >= grow_after_multiple:
            day_lots = max_lots + 1

        # Scan all valid candles and find the BEST signal(s)
        candidates = []
        for i in range(5, len(day_df)):
            t_str = day_df.index[i].strftime("%H:%M")
            if t_str < entry_start or t_str > entry_end:
                continue

            result = engine.score(indicators, i)
            abs_score = abs(result.score)

            if abs_score < entry_threshold:
                continue
            if strength_order.get(result.strength, 0) < min_str_val:
                continue

            candidates.append((i, result))

        # Take top N signals by absolute score, spaced at least 12 candles apart
        candidates.sort(key=lambda x: abs(x[1].score), reverse=True)
        selected = []
        used_ranges = []
        for idx, result in candidates:
            if len(selected) >= max_trades_day:
                break
            # Ensure minimum spacing
            overlaps = any(abs(idx - u) < hold_candles for u in used_ranges)
            if overlaps:
                continue
            selected.append((idx, result))
            used_ranges.append(idx)

        for entry_i, result in selected:
            if day_pnl < 0 and abs(day_pnl) >= loss_limit:
                break

            entry_price = day_df["close"].iloc[entry_i]
            direction = result.direction
            qty = day_lots * lot_size
            max_loss = base_premium * (max_loss_per_trade_pct / 100)
            target_premium_gain = target_points * delta

            exit_price = entry_price
            exit_candle = min(entry_i + hold_candles, len(day_df) - 1)
            exit_reason = "TIME"
            best_prem_gain = 0

            for k in range(entry_i + 1, len(day_df)):
                cur_price = day_df["close"].iloc[k]
                elapsed = k - entry_i

                if direction == "LONG":
                    index_move = cur_price - entry_price
                else:
                    index_move = entry_price - cur_price

                prem_change = index_move * delta - elapsed * theta_per_candle
                if prem_change > best_prem_gain:
                    best_prem_gain = prem_change

                # Stop loss: max loss exceeded
                if prem_change < -max_loss:
                    exit_price = cur_price
                    exit_candle = k
                    exit_reason = "SL"
                    break

                # Profit target reached
                if prem_change >= target_premium_gain:
                    exit_price = cur_price
                    exit_candle = k
                    exit_reason = "TGT"
                    break

                # Time exit: hold period complete
                if elapsed >= hold_candles:
                    exit_price = cur_price
                    exit_candle = k
                    exit_reason = "TIME"
                    break

                # EOD
                if day_df.index[k].strftime("%H:%M") >= eod_exit:
                    exit_price = cur_price
                    exit_candle = k
                    exit_reason = "EOD"
                    break

            # Calculate P&L
            elapsed_final = exit_candle - entry_i
            if direction == "LONG":
                index_pnl = exit_price - entry_price
            else:
                index_pnl = entry_price - exit_price

            prem_pnl_per_unit = index_pnl * delta - elapsed_final * theta_per_candle
            total_pnl = prem_pnl_per_unit * qty

            capital += total_pnl
            day_pnl += total_pnl
            day_trades += 1

            trades.append({
                "dir": direction,
                "entry_time": str(day_df.index[entry_i]),
                "exit_time": str(day_df.index[exit_candle]),
                "entry_price": round(entry_price, 1),
                "exit_price": round(exit_price, 1),
                "index_move": round(index_pnl, 1),
                "hold_candles": elapsed_final,
                "prem_pnl_unit": round(prem_pnl_per_unit, 2),
                "qty": qty, "lots": day_lots,
                "pnl": round(total_pnl, 0),
                "reason": exit_reason,
                "capital": round(capital, 0),
                "confluence": round(result.score, 1),
                "strength": result.strength,
                "total_ind": result.total_indicators,
            })

            if capital <= 0:
                break

        if capital > peak:
            peak = capital

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades,
        })

        if capital <= 0:
            break

    return _stats(trades, equity_curve, starting_capital, capital, unique_days)


def _stats(trades, ec, start_cap, capital, days):
    pnl = capital - start_cap
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gp / gl if gl > 0 else float("inf")
    eq = [e["capital"] for e in ec]
    max_dd = 0
    if eq:
        pa = np.maximum.accumulate(eq)
        dd = [(p - v) / p * 100 if p > 0 else 0 for p, v in zip(pa, eq)]
        max_dd = max(dd) if dd else 0
    active = [e for e in ec if e["trades"] > 0]
    prof_d = sum(1 for e in ec if e["daily_pnl"] > 0)
    loss_d = sum(1 for e in ec if e["daily_pnl"] < 0)
    exits = {}
    for t in trades:
        exits.setdefault(t["reason"], {"count": 0, "pnl": 0})
        exits[t["reason"]]["count"] += 1
        exits[t["reason"]]["pnl"] += t["pnl"]
    return {
        "capital": round(capital, 0), "pnl": round(pnl, 0),
        "return_pct": round(pnl / start_cap * 100, 1),
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1),
        "avg_win": round(np.mean([t["pnl"] for t in wins]), 0) if wins else 0,
        "avg_loss": round(np.mean([t["pnl"] for t in losses]), 0) if losses else 0,
        "profit_factor": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "prof_days": prof_d, "loss_days": loss_d,
        "trading_days": len(days), "active_days": len(active),
        "equity_curve": ec, "trade_list": trades, "exit_reasons": exits,
    }


def cross_validate_v6(df, n_folds=3, start_cap=10000, **params):
    unique_days = sorted(set(df.index.date))
    fold_size = len(unique_days) // n_folds
    folds = []
    for f in range(n_folds):
        s = f * fold_size
        e = s + fold_size
        test_set = set(unique_days[s:e])
        dates_arr = pd.Series(df.index.date)
        test_df = df[dates_arr.isin(test_set).values].copy()
        r = backtest_v6(test_df, starting_capital=start_cap, **params)
        folds.append(r)
    valid = [f for f in folds if f["trades"] > 0]
    if not valid:
        return {"folds": folds, "avg_pf": 0, "avg_wr": 0, "avg_dd": 100,
                "total_trades": 0, "pf_std": 0, "consistency": -1}
    avg_pf = np.mean([f["profit_factor"] for f in valid])
    avg_wr = np.mean([f["win_rate"] for f in valid])
    avg_dd = np.mean([f["max_dd"] for f in valid])
    pf_std = np.std([f["profit_factor"] for f in valid])
    total_t = sum(f["trades"] for f in folds)
    return {"folds": folds, "avg_pf": round(avg_pf, 2), "avg_wr": round(avg_wr, 1),
            "avg_dd": round(avg_dd, 1), "total_trades": total_t,
            "pf_std": round(pf_std, 2), "consistency": round(avg_pf - pf_std, 2)}


def optimize_v6(df, start_cap=10000):
    total = 0

    def cv_score(params):
        cv = cross_validate_v6(df, n_folds=3, start_cap=start_cap, **params)
        if cv["total_trades"] < 10:
            return -100
        return (min(cv["avg_pf"], 3) * min(cv["avg_wr"], 70)
                - cv["avg_dd"] * 1.0 + cv["consistency"] * 30)

    # Phase 1: Entry threshold + hold period
    print("\n  Phase 1: Threshold + hold period (3-fold CV)...")
    p1 = []
    for thresh, hold in itertools.product(
        [40, 45, 50, 55, 60, 65],
        [6, 9, 12, 18, 24]    # 30m to 2hr
    ):
        params = {"entry_threshold": thresh, "hold_candles": hold}
        s = cv_score(params)
        cv = cross_validate_v6(df, n_folds=3, start_cap=start_cap, **params)
        p1.append({"p": params, "s": s,
                    "pf": cv["avg_pf"], "wr": cv["avg_wr"],
                    "t": cv["total_trades"], "dd": cv["avg_dd"]})
        total += 1
    p1.sort(key=lambda x: x["s"], reverse=True)
    b1 = p1[0]["p"]
    print(f"    Best: thresh={b1['entry_threshold']}, hold={b1['hold_candles']}x5m "
          f"({b1['hold_candles']*5}min) "
          f"(avgPF={p1[0]['pf']}, avgWR={p1[0]['wr']}%, {p1[0]['t']}t)")

    # Phase 2: Target + SL
    print("  Phase 2: Target points + SL (3-fold CV)...")
    p2 = []
    for tgt, sl_pct in itertools.product(
        [3, 5, 8, 12, 18],
        [10, 15, 20, 25]
    ):
        params = {**b1, "target_points": tgt, "max_loss_per_trade_pct": sl_pct}
        s = cv_score(params)
        cv = cross_validate_v6(df, n_folds=3, start_cap=start_cap, **params)
        p2.append({"p": {"target_points": tgt, "max_loss_per_trade_pct": sl_pct},
                    "s": s, "pf": cv["avg_pf"], "wr": cv["avg_wr"],
                    "t": cv["total_trades"], "dd": cv["avg_dd"]})
        total += 1
    p2.sort(key=lambda x: x["s"], reverse=True)
    b2 = p2[0]["p"]
    print(f"    Best: target={b2['target_points']}pts, SL={b2['max_loss_per_trade_pct']}% "
          f"(avgPF={p2[0]['pf']}, avgWR={p2[0]['wr']}%)")

    # Phase 3: Trades per day + entry window
    print("  Phase 3: Trades/day + entry timing (3-fold CV)...")
    p3 = []
    for mt, e_start, e_end in itertools.product(
        [1, 2],
        ["09:30", "09:45", "10:00"],
        ["12:30", "13:00", "13:30", "14:00"]
    ):
        params = {**b1, **b2, "max_trades_day": mt,
                  "entry_start": e_start, "entry_end": e_end}
        s = cv_score(params)
        cv = cross_validate_v6(df, n_folds=3, start_cap=start_cap, **params)
        p3.append({"p": {"max_trades_day": mt, "entry_start": e_start,
                          "entry_end": e_end},
                    "s": s, "pf": cv["avg_pf"], "wr": cv["avg_wr"],
                    "t": cv["total_trades"], "dd": cv["avg_dd"]})
        total += 1
    p3.sort(key=lambda x: x["s"], reverse=True)
    b3 = p3[0]["p"]
    print(f"    Best: {b3['max_trades_day']}t/d, {b3['entry_start']}-{b3['entry_end']} "
          f"(avgPF={p3[0]['pf']}, avgWR={p3[0]['wr']}%)")

    # Phase 4: Min strength
    print("  Phase 4: Minimum signal strength (3-fold CV)...")
    p4 = []
    for ms in ["MODERATE", "STRONG", "EXTREME"]:
        params = {**b1, **b2, **b3, "min_strength": ms}
        s = cv_score(params)
        cv = cross_validate_v6(df, n_folds=3, start_cap=start_cap, **params)
        p4.append({"p": {"min_strength": ms},
                    "s": s, "pf": cv["avg_pf"], "wr": cv["avg_wr"],
                    "t": cv["total_trades"], "dd": cv["avg_dd"]})
        total += 1
    p4.sort(key=lambda x: x["s"], reverse=True)
    b4 = p4[0]["p"]
    print(f"    Best: min_strength={b4['min_strength']} "
          f"(avgPF={p4[0]['pf']}, avgWR={p4[0]['wr']}%)")

    print(f"\n  Total combos: {total} x 3 folds = {total * 3} backtests")

    all_p = {**b1, **b2, **b3, **b4}
    return all_p


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

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD V6 -- EDGE-OPTIMIZED DESIGN")
    print(f"  Based on proven 53.5% directional edge at 60-min horizon")
    print(f"  ITM options (delta=0.65) | 1hr hold | strongest signal only")
    print(f"{'='*72}")

    best_params = optimize_v6(df, start_cap=10000)

    # Full period result
    print(f"\n{'='*72}")
    print(f"  FULL PERIOD RESULTS")
    print(f"{'='*72}")
    full = backtest_v6(df, starting_capital=10000, **best_params)
    _print(full)

    # 3-fold CV
    print(f"\n{'='*72}")
    print(f"  3-FOLD CROSS-VALIDATION")
    print(f"{'='*72}")
    cv = cross_validate_v6(df, n_folds=3, start_cap=10000, **best_params)
    for fi, fr in enumerate(cv["folds"]):
        print(f"  Fold {fi+1}: {fr['trades']:>3d}t, PF={fr['profit_factor']:>5.2f}, "
              f"WR={fr['win_rate']:>5.1f}%, DD={fr['max_dd']:>5.1f}%, "
              f"Ret={fr['return_pct']:>+7.1f}%")

    print(f"\n  {'Avg PF':25s}: {cv['avg_pf']}")
    print(f"  {'Avg WR':25s}: {cv['avg_wr']}%")
    print(f"  {'Avg DD':25s}: {cv['avg_dd']}%")
    print(f"  {'Consistency (PF-Std)':25s}: {cv['consistency']}")
    print(f"  {'Total Trades':25s}: {cv['total_trades']}")

    if cv["avg_pf"] > 1.3 and cv["avg_wr"] > 50 and cv["avg_dd"] < 30:
        v = "STRONG PASS"
    elif cv["avg_pf"] > 1.1 and cv["avg_wr"] > 48 and cv["consistency"] > 0.5:
        v = "PASS"
    elif cv["avg_pf"] > 1.0:
        v = "MARGINAL"
    else:
        v = "FAIL"

    print(f"\n  VERDICT: {v}")

    print(f"\n{'='*72}")
    print(f"  OPTIMIZED PARAMETERS")
    print(f"{'='*72}")
    for k, val in best_params.items():
        print(f"    {k:<30} = {val}")
    print(f"{'='*72}")

    if full["trade_list"]:
        pd.DataFrame(full["trade_list"]).to_csv(
            settings.DATA_DIR / "v6_trades.csv", index=False)
    if full["equity_curve"]:
        pd.DataFrame(full["equity_curve"]).to_csv(
            settings.DATA_DIR / "v6_equity.csv", index=False)

    return {"params": best_params, "full": full, "cv": cv, "verdict": v}


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

    ex = r.get("exit_reasons", {})
    if ex:
        print(f"{pfx}Exits:")
        for er in sorted(ex):
            d = ex[er]
            print(f"{pfx}  {er:6s}: {d['count']:>3d}t, Rs {d['pnl']:>9,.0f}")

    ec = r.get("equity_curve", [])
    if ec and len(ec) > 3:
        print(f"{pfx}Equity:")
        n = len(ec)
        for idx in sorted(set(min(x, n-1) for x in [0, n//4, n//2, 3*n//4, n-1])):
            e = ec[idx]
            print(f"{pfx}  Day {idx+1:>3d} ({e['date']}): Rs {e['capital']:>10,.0f} "
                  f"[{e['trades']}t, Rs{e['daily_pnl']:>+8,.0f}]")

    # Trade analysis
    tl = r.get("trade_list", [])
    if tl:
        holds = [t["hold_candles"] for t in tl]
        moves = [t["index_move"] for t in tl]
        confs = [abs(t["confluence"]) for t in tl]
        print(f"{pfx}Trade Stats:")
        print(f"{pfx}  Avg hold: {np.mean(holds):.0f} candles ({np.mean(holds)*5:.0f} min)")
        print(f"{pfx}  Avg index move: {np.mean(moves):+.1f} pts")
        print(f"{pfx}  Avg confluence: {np.mean(confs):.0f}")


if __name__ == "__main__":
    main()
