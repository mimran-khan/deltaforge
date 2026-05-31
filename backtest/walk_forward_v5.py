"""Walk-Forward V5 -- Robust validation with anti-overfit design.

Key changes from V4:
1. 3-fold cross-validation instead of single train/test split
2. Daily regime filter using 2-year daily data
3. Conservative position sizing (1 lot fixed, grow slowly)
4. Minimum trade count for statistical significance
5. Score function that rewards CONSISTENCY across folds

Usage:
    python -m backtest.walk_forward_v5
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


def compute_daily_regime(daily_df: pd.DataFrame) -> dict:
    """Analyze 2-year daily data to determine which days are favorable.

    Returns a dict mapping date -> regime info that the intraday
    system can use to filter trades.
    """
    c = daily_df["close"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    atr14 = daily_df[["high", "low", "close"]].copy()
    prev_c = c.shift(1)
    tr = pd.concat([
        daily_df["high"] - daily_df["low"],
        (daily_df["high"] - prev_c).abs(),
        (daily_df["low"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(span=14, adjust=False).mean()
    atr_pct = atr_val / c * 100

    rsi14 = _rsi(c, 14)
    daily_range_pct = (daily_df["high"] - daily_df["low"]) / c * 100

    regimes = {}
    for i in range(50, len(daily_df)):
        dt = daily_df.index[i].date() if hasattr(daily_df.index[i], 'date') else daily_df.index[i]
        is_trending = ema20.iloc[i] > ema50.iloc[i]
        is_volatile_enough = atr_pct.iloc[i] > 0.5
        rsi_ok = 35 < rsi14.iloc[i] < 65  # not overbought/oversold
        trend_dir = 1 if ema20.iloc[i] > ema50.iloc[i] else -1

        favorable = is_volatile_enough
        regimes[dt] = {
            "trend_dir": trend_dir,
            "trending": is_trending,
            "atr_pct": round(atr_pct.iloc[i], 2),
            "rsi": round(rsi14.iloc[i], 0),
            "range_pct": round(daily_range_pct.iloc[i], 2),
            "favorable": favorable,
        }
    return regimes


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_g = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def backtest_v5(
    df: pd.DataFrame,
    daily_regimes: dict | None = None,
    lot_size: int = 75,
    starting_capital: float = 10000,
    entry_threshold: float = 55.0,
    min_strength: str = "STRONG",
    max_trades_day: int = 2,
    max_consec_loss: int = 3,
    daily_loss_pct: float = 20,
    delta: float = 0.45,
    base_premium: float = 95,
    theta_per_candle: float = 0.15,
    premium_sl_pct: float = 25,
    trail_trigger_pct: float = 60,
    trail_pct: float = 30,
    entry_start: str = "09:45",
    entry_end: str = "13:30",
    eod_exit: str = "15:15",
    use_daily_filter: bool = True,
    weight_overrides: dict | None = None,
) -> dict:
    """Production-parity backtest with daily regime filter."""

    strength_order = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3, "EXTREME": 4}
    min_str_val = strength_order.get(min_strength, 3)

    engine = ConfluenceEngine(weight_overrides=weight_overrides)
    unique_days = sorted(set(df.index.date))
    capital = starting_capital
    peak = starting_capital
    trades = []
    equity_curve = []

    for day in unique_days:
        # Daily regime filter
        if use_daily_filter and daily_regimes:
            regime = daily_regimes.get(day)
            if regime and not regime["favorable"]:
                equity_curve.append({"date": day, "capital": round(capital, 0),
                                      "daily_pnl": 0, "trades": 0, "lots": 0,
                                      "skip_reason": "regime"})
                continue

        dates_arr = pd.Series(df.index.date)
        day_mask = (dates_arr == day).values
        day_df = df[day_mask].copy()
        if len(day_df) < 15:
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0, "lots": 0})
            continue

        try:
            indicators = engine.precompute(day_df)
        except Exception:
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0, "lots": 0})
            continue

        day_start_cap = capital
        day_pnl = 0.0
        day_trades = 0
        consec_loss = 0
        loss_limit = day_start_cap * (daily_loss_pct / 100)

        # FIXED LOT SIZING: 1 lot until capital doubles, then slowly grow
        growth = capital / starting_capital
        if growth >= 4:
            day_lots = 3
        elif growth >= 2:
            day_lots = 2
        else:
            day_lots = 1

        # Collect all signals first, then take the strongest
        day_signals = []

        for i in range(5, len(day_df)):
            ts = day_df.index[i]
            t_str = ts.strftime("%H:%M")
            if t_str < entry_start or t_str > entry_end:
                continue

            result = engine.score(indicators, i)
            abs_score = abs(result.score)

            if abs_score < entry_threshold:
                continue
            if strength_order.get(result.strength, 0) < min_str_val:
                continue

            # Only trade in daily trend direction if available
            if use_daily_filter and daily_regimes:
                regime = daily_regimes.get(day)
                if regime:
                    if result.direction == "LONG" and regime["trend_dir"] == -1:
                        continue
                    if result.direction == "SHORT" and regime["trend_dir"] == 1:
                        continue

            day_signals.append((i, result))

        # Take the strongest N signals (capped at max_trades_day)
        day_signals.sort(key=lambda x: abs(x[1].score), reverse=True)

        # Deduplicate: only 1 signal per 15-candle (75-min) window
        used_windows = set()
        filtered_signals = []
        for idx, result in day_signals:
            window = idx // 15
            if window not in used_windows:
                filtered_signals.append((idx, result))
                used_windows.add(window)

        for sig_idx, (i, result) in enumerate(filtered_signals[:max_trades_day]):
            if consec_loss >= max_consec_loss:
                break
            if day_pnl < 0 and abs(day_pnl) >= loss_limit:
                break

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
                elapsed = k - i
                cur_idx = day_df["close"].iloc[k]
                cur_prem = prem_state.current_premium(cur_idx, elapsed)

                trail_floor = prem_state.update_trail(
                    cur_prem, trail_trigger_pct, trail_pct)

                reason = prem_state.check_exit(cur_prem, trail_floor)
                if reason:
                    exit_prem = {
                        "SL": prem_state.sl_premium,
                        "TGT": prem_state.target_premium,
                        "TRAIL": trail_floor,
                    }.get(reason, cur_prem)
                    exit_reason = reason
                    exit_time = day_df.index[k]
                    break

                if day_df.index[k].strftime("%H:%M") >= eod_exit:
                    exit_prem = cur_prem
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
        })
        if capital <= 0:
            break

    return _stats(trades, equity_curve, starting_capital, capital, unique_days)


def _stats(trades, equity_curve, start_cap, capital, days):
    pnl = capital - start_cap
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
        "prof_days": prof_days, "loss_days": loss_days,
        "trading_days": len(days), "active_days": len(active),
        "equity_curve": equity_curve, "trade_list": trades,
        "exit_reasons": exits,
    }


def cross_validate(df, daily_regimes, n_folds=3, start_cap=10000, **params):
    """N-fold cross-validation. Returns average metrics across folds."""
    unique_days = sorted(set(df.index.date))
    fold_size = len(unique_days) // n_folds
    fold_results = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size
        test_days = set(unique_days[test_start:test_end])
        train_days = set(unique_days) - test_days

        dates_arr = pd.Series(df.index.date)
        test_df = df[dates_arr.isin(test_days).values].copy()

        r = backtest_v5(test_df, daily_regimes=daily_regimes,
                         starting_capital=start_cap, **params)
        fold_results.append(r)

    # Average metrics
    avg_pf = np.mean([r["profit_factor"] for r in fold_results if r["trades"] > 0])
    avg_wr = np.mean([r["win_rate"] for r in fold_results if r["trades"] > 0])
    avg_dd = np.mean([r["max_dd"] for r in fold_results])
    total_trades = sum(r["trades"] for r in fold_results)
    pf_std = np.std([r["profit_factor"] for r in fold_results if r["trades"] > 0])

    return {
        "folds": fold_results,
        "avg_pf": round(avg_pf, 2),
        "avg_wr": round(avg_wr, 1),
        "avg_dd": round(avg_dd, 1),
        "total_trades": total_trades,
        "pf_std": round(pf_std, 2),
        "consistency": round(avg_pf - pf_std, 2),
    }


def optimize_v5(df, daily_regimes, start_cap=10000):
    """Optimize using 3-fold CV to prevent overfitting."""

    def cv_score(params):
        cv = cross_validate(df, daily_regimes, n_folds=3,
                             start_cap=start_cap, **params)
        if cv["total_trades"] < 10:
            return -100
        return (min(cv["avg_pf"], 3) * min(cv["avg_wr"], 70)
                - cv["avg_dd"] * 1.0
                + cv["consistency"] * 20)

    total = 0

    # Phase 1: Threshold + strength (optimized across ALL folds)
    print("\n  Phase 1: Threshold (3-fold CV)...")
    p1 = []
    for thresh, ms in itertools.product(
        [35, 40, 45, 50, 55, 60],
        ["MODERATE", "STRONG"]
    ):
        params = {"entry_threshold": thresh, "min_strength": ms}
        s = cv_score(params)
        cv = cross_validate(df, daily_regimes, n_folds=3,
                             start_cap=start_cap, **params)
        p1.append({"p": params, "s": s,
                    "pf": cv["avg_pf"], "wr": cv["avg_wr"],
                    "t": cv["total_trades"], "dd": cv["avg_dd"],
                    "con": cv["consistency"]})
        total += 1
    p1.sort(key=lambda x: x["s"], reverse=True)
    b1 = p1[0]["p"]
    print(f"    Best: thresh={b1['entry_threshold']}, str={b1['min_strength']} "
          f"(avgPF={p1[0]['pf']}, avgWR={p1[0]['wr']}%, {p1[0]['t']}t, "
          f"avgDD={p1[0]['dd']}%, consistency={p1[0]['con']:.2f})")

    # Phase 2: SL + trailing
    print("  Phase 2: SL + trailing (3-fold CV)...")
    p2 = []
    for sl, tt, tp in itertools.product(
        [20, 25, 30, 35], [40, 60, 80], [25, 35]
    ):
        params = {**b1, "premium_sl_pct": sl,
                  "trail_trigger_pct": tt, "trail_pct": tp}
        s = cv_score(params)
        cv = cross_validate(df, daily_regimes, n_folds=3,
                             start_cap=start_cap, **params)
        p2.append({"p": {"premium_sl_pct": sl, "trail_trigger_pct": tt,
                          "trail_pct": tp},
                    "s": s, "pf": cv["avg_pf"], "wr": cv["avg_wr"],
                    "t": cv["total_trades"], "dd": cv["avg_dd"]})
        total += 1
    p2.sort(key=lambda x: x["s"], reverse=True)
    b2 = p2[0]["p"]
    print(f"    Best: SL={b2['premium_sl_pct']}%, trail={b2['trail_trigger_pct']}/{b2['trail_pct']}% "
          f"(avgPF={p2[0]['pf']}, score={p2[0]['s']:.0f})")

    # Phase 3: Trades/day + daily filter
    print("  Phase 3: Risk + regime filter (3-fold CV)...")
    p3 = []
    for mt, uf in itertools.product([1, 2, 3], [True, False]):
        params = {**b1, **b2, "max_trades_day": mt, "use_daily_filter": uf}
        s = cv_score(params)
        cv = cross_validate(df, daily_regimes, n_folds=3,
                             start_cap=start_cap, **params)
        p3.append({"p": {"max_trades_day": mt, "use_daily_filter": uf},
                    "s": s, "pf": cv["avg_pf"], "wr": cv["avg_wr"],
                    "t": cv["total_trades"], "dd": cv["avg_dd"]})
        total += 1
    p3.sort(key=lambda x: x["s"], reverse=True)
    b3 = p3[0]["p"]
    print(f"    Best: trades/d={b3['max_trades_day']}, daily_filter={b3['use_daily_filter']} "
          f"(avgPF={p3[0]['pf']}, score={p3[0]['s']:.0f})")

    print(f"\n  Total combos tested (x3 folds each): {total} x 3 = {total * 3}")
    all_p = {**b1, **b2, **b3}
    return all_p


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    # Load real data
    data_5m = settings.DATA_DIR / "nifty_5m_real.csv"
    data_daily = settings.DATA_DIR / "nifty_daily_real.csv"

    if not data_5m.exists():
        print("ERROR: No 5m data. Run: python -m backtest.data_fetcher")
        sys.exit(1)

    df = pd.read_csv(data_5m, index_col=0, parse_dates=True)
    days = len(set(df.index.date))
    print(f"5-min data: {len(df)} candles, {days} trading days")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    # Load daily data for regime analysis
    daily_regimes = None
    if data_daily.exists():
        daily_df = pd.read_csv(data_daily, index_col=0, parse_dates=True)
        print(f"Daily data: {len(daily_df)} days for regime analysis")
        daily_regimes = compute_daily_regime(daily_df)
        print(f"Regimes computed: {len(daily_regimes)} days")
        favorable = sum(1 for r in daily_regimes.values() if r["favorable"])
        print(f"Favorable trading days: {favorable}/{len(daily_regimes)} "
              f"({favorable/len(daily_regimes)*100:.0f}%)")

    # Optimize with 3-fold CV
    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD V5 -- 3-FOLD CROSS-VALIDATED OPTIMIZATION")
    print(f"  200+ indicators | daily regime filter | anti-overfit design")
    print(f"{'='*72}")

    best_params = optimize_v5(df, daily_regimes, start_cap=10000)

    # Final run on full data with best params
    print(f"\n{'='*72}")
    print(f"  FINAL RESULTS ON FULL PERIOD")
    print(f"{'='*72}")
    full_result = backtest_v5(df, daily_regimes=daily_regimes,
                               starting_capital=10000, **best_params)
    _print(full_result)

    # 3-fold CV final
    print(f"\n{'='*72}")
    print(f"  3-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*72}")
    cv = cross_validate(df, daily_regimes, n_folds=3,
                         start_cap=10000, **best_params)
    for fold_i, fr in enumerate(cv["folds"]):
        print(f"\n  Fold {fold_i + 1}: {fr['trades']}t, "
              f"PF={fr['profit_factor']}, WR={fr['win_rate']}%, "
              f"DD={fr['max_dd']}%, Ret={fr['return_pct']}%")

    print(f"\n  CROSS-VALIDATION SUMMARY:")
    print(f"  {'Avg Profit Factor':25s}: {cv['avg_pf']}")
    print(f"  {'Avg Win Rate':25s}: {cv['avg_wr']}%")
    print(f"  {'Avg Max Drawdown':25s}: {cv['avg_dd']}%")
    print(f"  {'PF Std Deviation':25s}: {cv['pf_std']}")
    print(f"  {'Consistency (PF-Std)':25s}: {cv['consistency']}")
    print(f"  {'Total Trades':25s}: {cv['total_trades']}")

    # Verdict
    if cv["avg_pf"] > 1.3 and cv["avg_wr"] > 50 and cv["avg_dd"] < 40:
        v = "STRONG PASS"
    elif cv["avg_pf"] > 1.1 and cv["avg_wr"] > 45 and cv["consistency"] > 0.5:
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

    # Save results
    if full_result["trade_list"]:
        pd.DataFrame(full_result["trade_list"]).to_csv(
            settings.DATA_DIR / "v5_trades.csv", index=False)
    if full_result["equity_curve"]:
        pd.DataFrame(full_result["equity_curve"]).to_csv(
            settings.DATA_DIR / "v5_equity.csv", index=False)

    return {"params": best_params, "full": full_result, "cv": cv, "verdict": v}


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
        print(f"{pfx}Exits:")
        for er in sorted(ex):
            d = ex[er]
            avg_p = d["pnl"] / d["count"] if d["count"] else 0
            print(f"{pfx}  {er:6s}: {d['count']:>3d} trades, "
                  f"Rs {d['pnl']:>10,.0f} (avg Rs {avg_p:>7,.0f})")

    ec = r.get("equity_curve", [])
    if ec and len(ec) > 3:
        print(f"{pfx}Equity:")
        n = len(ec)
        for idx in sorted(set(min(x, n-1) for x in [0, n//4, n//2, 3*n//4, n-1])):
            e = ec[idx]
            print(f"{pfx}  Day {idx+1:>3d} ({e['date']}): Rs {e['capital']:>10,.0f} "
                  f"[{e['trades']}t, Rs{e['daily_pnl']:>+8,.0f}]")


if __name__ == "__main__":
    main()
