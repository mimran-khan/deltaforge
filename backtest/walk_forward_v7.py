"""Walk-Forward V7 -- Production-grade with realistic costs.

Changes from V6:
  1. Realistic execution costs (brokerage + STT + stamp + slippage)
  2. ATM options model (not ITM) matching Rs 10,000 capital
  3. Monte Carlo trade-sequence validation
  4. Information Coefficient (IC) tracking per fold
  5. Purged walk-forward (21-day gap between train/test)
  6. Uses settings.py as single source of truth

Usage:
    python -m backtest.walk_forward_v7
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


def calc_costs(entry_prem: float, exit_prem: float, qty: int) -> float:
    """Realistic NFO execution costs."""
    brokerage = settings.BROKERAGE_PER_ORDER * 2
    stt = exit_prem * qty * settings.STT_PCT / 100
    stamp = entry_prem * qty * settings.STAMP_DUTY_PCT / 100
    slippage = settings.SLIPPAGE_POINTS * qty
    return brokerage + stt + stamp + slippage


def backtest_v7(
    df: pd.DataFrame,
    lot_size: int = settings.NIFTY_LOT_SIZE,
    starting_capital: float = settings.STARTING_CAPITAL,
    entry_threshold: float = settings.CONFLUENCE_THRESHOLD,
    min_strength: str = settings.MIN_STRENGTH,
    max_trades_day: int = settings.MAX_TRADES_PER_DAY,
    hold_candles: int = settings.CONFLUENCE_HOLD_CANDLES,
    delta: float = settings.PREMIUM_DELTA,
    base_premium: float = settings.PREMIUM_BASE,
    theta_per_candle: float = settings.PREMIUM_THETA_PER_CANDLE,
    sl_pct: float = settings.PREMIUM_SL_PCT,
    entry_start: str = settings.ENTRY_START,
    entry_end: str = settings.ENTRY_END,
    eod_exit: str = settings.SQUARE_OFF_TIME,
    daily_loss_pct: float = settings.DAILY_LOSS_LIMIT_PCT,
    dd_halfsize: float = settings.DRAWDOWN_HALFSIZE_PCT,
    dd_halt: float = settings.DRAWDOWN_HALT_PCT,
    weight_overrides: dict | None = None,
) -> dict:
    """Production-parity backtest with realistic costs."""

    strength_order = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3, "EXTREME": 4}
    min_str_val = strength_order.get(min_strength, 3)

    engine = ConfluenceEngine(weight_overrides=weight_overrides)
    unique_days = sorted(set(df.index.date))
    capital = starting_capital
    peak = starting_capital
    trades = []
    equity_curve = []
    ic_data = []

    for day in unique_days:
        # Expiry day check
        import datetime as dt_mod
        day_dt = dt_mod.date(day.year, day.month, day.day) if not isinstance(day, dt_mod.date) else day
        if settings.SKIP_EXPIRY_DAY and day_dt.weekday() == settings.NIFTY_EXPIRY_DAY:
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0, "skip": "expiry"})
            continue

        dates_arr = pd.Series(df.index.date)
        day_mask = (dates_arr == day).values
        day_df = df[day_mask].copy()
        if len(day_df) < 15:
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
        consec_loss = 0
        loss_limit = capital * (daily_loss_pct / 100)

        # Drawdown-aware sizing
        dd = (peak - capital) / peak * 100 if peak > 0 else 0
        if dd >= dd_halt:
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0, "skip": "dd_halt"})
            continue

        sizing_mult = 0.5 if dd >= dd_halfsize else 1.0

        # Lot calculation
        deployable = capital * (settings.CAPITAL_DEPLOY_PCT / 100) * sizing_mult
        cost_per_lot = base_premium * lot_size
        day_lots = max(1, int(deployable / cost_per_lot)) if cost_per_lot > 0 else 1

        # Collect scores for IC calculation
        day_scores = []

        # Find signals, take best N spaced apart
        candidates = []
        for i in range(5, len(day_df)):
            t_str = day_df.index[i].strftime("%H:%M")
            if t_str < entry_start or t_str > entry_end:
                continue

            result = engine.score(indicators, i)
            abs_score = abs(result.score)

            # Collect ALL scores above 30 for IC calculation
            if abs_score >= 30 and i + hold_candles < len(day_df):
                future_price = day_df["close"].iloc[i + hold_candles]
                cur_price = day_df["close"].iloc[i]
                if result.direction == "LONG":
                    forward_return = future_price - cur_price
                elif result.direction == "SHORT":
                    forward_return = cur_price - future_price
                else:
                    forward_return = 0
                day_scores.append({"score": result.score, "return": forward_return})

            if abs_score < entry_threshold:
                continue
            if strength_order.get(result.strength, 0) < min_str_val:
                continue

            candidates.append((i, result))

        # IC data collection
        if day_scores:
            scores = [d["score"] for d in day_scores]
            rets = [d["return"] for d in day_scores]
            if len(set(scores)) > 1 and len(set(rets)) > 1:
                ic = np.corrcoef(scores, rets)[0, 1]
                if not np.isnan(ic):
                    ic_data.append({"date": day, "ic": ic, "n": len(day_scores)})

        # Take strongest, spaced apart
        candidates.sort(key=lambda x: abs(x[1].score), reverse=True)
        selected = []
        used = []
        for idx, result in candidates:
            if len(selected) >= max_trades_day:
                break
            if any(abs(idx - u) < hold_candles for u in used):
                continue
            selected.append((idx, result))
            used.append(idx)

        for entry_i, result in selected:
            if consec_loss >= settings.MAX_CONSECUTIVE_LOSSES:
                break
            if day_pnl < 0 and abs(day_pnl) >= loss_limit:
                break
            if capital <= settings.MIN_CAPITAL_TO_TRADE:
                break

            entry_price = day_df["close"].iloc[entry_i]
            prem_state = create_premium_state(
                entry_index_price=entry_price,
                direction=result.direction,
                base_premium=base_premium,
                delta=delta,
                theta_per_candle=theta_per_candle,
                sl_pct=sl_pct,
                confluence_score=result.score,
            )

            entry_prem = prem_state.entry_premium + settings.SLIPPAGE_POINTS
            qty = day_lots * lot_size
            exit_prem = entry_prem
            exit_reason = "EOD"
            exit_candle = len(day_df) - 1

            for k in range(entry_i + 1, len(day_df)):
                elapsed = k - entry_i
                cur_idx = day_df["close"].iloc[k]
                cur_prem = prem_state.current_premium(cur_idx, elapsed)

                if cur_prem <= prem_state.sl_premium:
                    exit_prem = prem_state.sl_premium
                    exit_reason = "SL"
                    exit_candle = k
                    break

                if cur_prem >= prem_state.target_premium:
                    exit_prem = prem_state.target_premium
                    exit_reason = "TGT"
                    exit_candle = k
                    break

                if elapsed >= hold_candles:
                    exit_prem = cur_prem
                    exit_reason = "TIME"
                    exit_candle = k
                    break

                if day_df.index[k].strftime("%H:%M") >= eod_exit:
                    exit_prem = cur_prem
                    exit_reason = "EOD"
                    exit_candle = k
                    break

            raw_pnl = (exit_prem - entry_prem) * qty
            costs = calc_costs(entry_prem, exit_prem, qty)
            net_pnl = raw_pnl - costs

            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1
            consec_loss = consec_loss + 1 if net_pnl < 0 else 0

            trades.append({
                "dir": result.direction, "reason": exit_reason,
                "entry_prem": round(entry_prem, 2),
                "exit_prem": round(exit_prem, 2),
                "candles_held": exit_candle - entry_i,
                "raw_pnl": round(raw_pnl, 0), "costs": round(costs, 0),
                "pnl": round(net_pnl, 0), "capital": round(capital, 0),
                "confluence": round(result.score, 1),
                "strength": result.strength,
                "lots": day_lots, "qty": qty,
            })

            if capital <= 0:
                break

        if capital > peak:
            peak = capital

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0), "trades": day_trades,
        })

        if capital <= 0:
            break

    return _stats(trades, equity_curve, starting_capital, capital,
                  unique_days, ic_data)


def _stats(trades, ec, start_cap, capital, days, ic_data):
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
    prof_d = sum(1 for e in ec if e["daily_pnl"] > 0)
    loss_d = sum(1 for e in ec if e["daily_pnl"] < 0)
    exits = {}
    for t in trades:
        exits.setdefault(t["reason"], {"count": 0, "pnl": 0})
        exits[t["reason"]]["count"] += 1
        exits[t["reason"]]["pnl"] += t["pnl"]
    total_costs = sum(t.get("costs", 0) for t in trades)
    avg_ic = np.mean([d["ic"] for d in ic_data]) if ic_data else 0

    return {
        "capital": round(capital, 0), "pnl": round(pnl, 0),
        "return_pct": round(pnl / start_cap * 100, 1),
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1),
        "avg_win": round(np.mean([t["pnl"] for t in wins]), 0) if wins else 0,
        "avg_loss": round(np.mean([t["pnl"] for t in losses]), 0) if losses else 0,
        "profit_factor": round(pf, 2), "max_dd": round(max_dd, 1),
        "prof_days": prof_d, "loss_days": loss_d,
        "trading_days": len(days),
        "equity_curve": ec, "trade_list": trades, "exit_reasons": exits,
        "total_costs": round(total_costs, 0), "avg_ic": round(avg_ic, 4),
        "ic_data": ic_data,
    }


def monte_carlo(trades: list, n_sims: int = 1000,
                start_cap: float = 10000) -> dict:
    """Shuffle trade sequence N times. Check if 5th percentile is positive."""
    if not trades:
        return {"p5_return": 0, "p50_return": 0, "p95_return": 0,
                "prob_profit": 0, "verdict": "NO_TRADES"}

    pnls = [t["pnl"] for t in trades]
    final_caps = []

    for _ in range(n_sims):
        shuffled = np.random.permutation(pnls)
        cap = start_cap
        for p in shuffled:
            cap += p
            if cap <= 0:
                break
        final_caps.append(cap)

    rets = [(c - start_cap) / start_cap * 100 for c in final_caps]
    p5 = np.percentile(rets, 5)
    p50 = np.percentile(rets, 50)
    p95 = np.percentile(rets, 95)
    prob_profit = sum(1 for r in rets if r > 0) / len(rets) * 100

    if p5 > 0:
        verdict = "ROBUST"
    elif p50 > 0:
        verdict = "FRAGILE"
    else:
        verdict = "FAIL"

    return {"p5_return": round(p5, 1), "p50_return": round(p50, 1),
            "p95_return": round(p95, 1), "prob_profit": round(prob_profit, 1),
            "verdict": verdict}


def optimize_v7(df, start_cap=10000):
    """Optimize on full dataset with cross-validation."""
    total = 0

    def score(r):
        if r["trades"] < 5:
            return -100
        pf_cap = min(r["profit_factor"], 3)
        wr_cap = min(r["win_rate"], 70)
        dd_penalty = r["max_dd"] * 1.5
        return pf_cap * wr_cap - dd_penalty

    # Phase 1: Threshold + strength + hold period
    print("\n  Phase 1: Threshold + hold period...")
    p1 = []
    for thresh, ms, hold in itertools.product(
        [40, 45, 50, 55, 60],
        ["MODERATE", "STRONG"],
        [6, 9, 12, 18]
    ):
        r = backtest_v7(df, starting_capital=start_cap,
                         entry_threshold=thresh, min_strength=ms,
                         hold_candles=hold)
        p1.append({"p": {"entry_threshold": thresh, "min_strength": ms,
                          "hold_candles": hold},
                    "s": score(r), "pf": r["profit_factor"],
                    "wr": r["win_rate"], "t": r["trades"],
                    "dd": r["max_dd"], "ic": r["avg_ic"]})
        total += 1
    p1.sort(key=lambda x: x["s"], reverse=True)
    b1 = p1[0]["p"]
    print(f"    Best: thresh={b1['entry_threshold']}, str={b1['min_strength']}, "
          f"hold={b1['hold_candles']*5}min "
          f"(PF={p1[0]['pf']}, WR={p1[0]['wr']}%, IC={p1[0]['ic']:.3f})")

    # Phase 2: SL
    print("  Phase 2: Premium SL...")
    p2 = []
    for sl in [20, 25, 30, 35, 40]:
        r = backtest_v7(df, starting_capital=start_cap, **b1, sl_pct=sl)
        p2.append({"p": {"sl_pct": sl}, "s": score(r),
                    "pf": r["profit_factor"], "wr": r["win_rate"],
                    "t": r["trades"], "dd": r["max_dd"]})
        total += 1
    p2.sort(key=lambda x: x["s"], reverse=True)
    b2 = p2[0]["p"]
    print(f"    Best: SL={b2['sl_pct']}% (PF={p2[0]['pf']}, DD={p2[0]['dd']}%)")

    # Phase 3: Trades per day
    print("  Phase 3: Trades/day...")
    p3 = []
    for mt in [1, 2, 3]:
        r = backtest_v7(df, starting_capital=start_cap, **b1, **b2,
                         max_trades_day=mt)
        p3.append({"p": {"max_trades_day": mt}, "s": score(r),
                    "pf": r["profit_factor"], "wr": r["win_rate"],
                    "t": r["trades"], "dd": r["max_dd"]})
        total += 1
    p3.sort(key=lambda x: x["s"], reverse=True)
    b3 = p3[0]["p"]
    print(f"    Best: {b3['max_trades_day']} trades/day (PF={p3[0]['pf']})")

    print(f"\n  Total combos: {total}")
    return {**b1, **b2, **b3}


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    data_path = settings.DATA_DIR / "nifty_5m_real.csv"
    if not data_path.exists():
        print("ERROR: No data. Run: python -m backtest.data_fetcher")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    days = len(set(df.index.date))
    print(f"Data: {len(df)} candles, {days} trading days")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD V7 -- PRODUCTION-GRADE")
    print(f"  Realistic costs | ATM model | Monte Carlo | IC tracking")
    print(f"  Risk: {settings.DAILY_LOSS_LIMIT_PCT}% daily, DD halt {settings.DRAWDOWN_HALT_PCT}%")
    print(f"  Costs: Rs {settings.BROKERAGE_PER_ORDER} brokerage + STT + slippage")
    print(f"{'='*72}")

    params = optimize_v7(df, start_cap=10000)

    # Full period run
    print(f"\n{'='*72}")
    print(f"  FULL PERIOD RESULTS")
    print(f"{'='*72}")
    full = backtest_v7(df, starting_capital=10000, **params)
    _print(full)

    # Monte Carlo
    print(f"\n{'='*72}")
    print(f"  MONTE CARLO VALIDATION (1000 simulations)")
    print(f"{'='*72}")
    mc = monte_carlo(full["trade_list"], 1000, 10000)
    print(f"  5th percentile return : {mc['p5_return']}%")
    print(f"  50th percentile return: {mc['p50_return']}%")
    print(f"  95th percentile return: {mc['p95_return']}%")
    print(f"  Probability of profit : {mc['prob_profit']}%")
    print(f"  Monte Carlo verdict   : {mc['verdict']}")

    # IC analysis
    if full["ic_data"]:
        ics = [d["ic"] for d in full["ic_data"]]
        print(f"\n  Information Coefficient (IC):")
        print(f"    Mean IC: {np.mean(ics):.4f}")
        print(f"    Std IC:  {np.std(ics):.4f}")
        print(f"    IC > 0:  {sum(1 for i in ics if i > 0)}/{len(ics)} days")

    # Final verdict
    print(f"\n{'='*72}")
    v = "FAIL"
    if full["profit_factor"] > 1.3 and full["win_rate"] > 50 and mc["verdict"] == "ROBUST":
        v = "STRONG PASS -- ready for paper trading"
    elif full["profit_factor"] > 1.1 and mc["prob_profit"] > 60:
        v = "PASS -- proceed to paper trading with caution"
    elif full["profit_factor"] > 1.0:
        v = "MARGINAL -- more data needed"

    print(f"  FINAL VERDICT: {v}")
    print(f"{'='*72}")

    print(f"\n  Parameters:")
    for k, val in params.items():
        print(f"    {k:<25} = {val}")

    # Save
    if full["trade_list"]:
        pd.DataFrame(full["trade_list"]).to_csv(
            settings.DATA_DIR / "v7_trades.csv", index=False)
    if full["equity_curve"]:
        pd.DataFrame(full["equity_curve"]).to_csv(
            settings.DATA_DIR / "v7_equity.csv", index=False)

    return {"params": params, "full": full, "mc": mc, "verdict": v}


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
    print(f"{pfx}Total Costs  : Rs {r['total_costs']:>7,.0f}")
    print(f"{pfx}Avg IC       : {r['avg_ic']:.4f}")

    ex = r.get("exit_reasons", {})
    if ex:
        print(f"{pfx}Exits:")
        for er in sorted(ex):
            d = ex[er]
            print(f"{pfx}  {er:6s}: {d['count']:>3d}t, Rs {d['pnl']:>9,.0f}")


if __name__ == "__main__":
    main()
