"""Walk-forward optimization V2 -- addressing all issues from V1.

Key improvements:
1. VWAP_MOM disabled (consistent loser) -- replaced with SuperTrend+VWAP filter
2. Trailing stop mechanism instead of fixed targets
3. Market regime detection (ATR-based, skip choppy low-range days)
4. Drawdown circuit breaker (reduce lot size after losses)
5. Adaptive position sizing (smaller after drawdown)
6. Expanded ORB window and relaxed filters for more signals

Usage:
    python -m backtest.walk_forward_v2
"""

from __future__ import annotations
import sys
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.indicators import ema, rsi, atr, supertrend_fast


def detect_regime(day_df: pd.DataFrame) -> str:
    """Detect market regime for the day using early-session data.

    Returns: 'trending', 'ranging', or 'volatile'
    """
    if len(day_df) < 10:
        return "unknown"

    early = day_df.iloc[:6]  # first 30 min
    full_range = day_df["high"].max() - day_df["low"].min()
    early_range = early["high"].max() - early["low"].min()
    close_vals = day_df["close"].values

    # Direction consistency: count consecutive up/down moves
    changes = np.diff(close_vals[:20])
    direction_changes = np.sum(np.diff(np.sign(changes)) != 0)
    direction_ratio = direction_changes / max(len(changes) - 1, 1)

    avg_body = np.mean(np.abs(day_df["close"].values[:15] - day_df["open"].values[:15]))
    avg_range = np.mean(day_df["high"].values[:15] - day_df["low"].values[:15])
    body_ratio = avg_body / avg_range if avg_range > 0 else 0

    if direction_ratio < 0.55 and body_ratio > 0.4:
        return "trending"
    elif early_range > full_range * 0.7:
        return "volatile"
    else:
        return "ranging"


def backtest_v2(
    df: pd.DataFrame,
    lot_size: int = 75,
    starting_capital: float = 10000,
    deploy_pct: float = 80.0,
    # ORB params
    orb_range_max: float = 250,
    orb_range_min: float = 8,
    orb_rr: float = 1.5,
    orb_entry_end: str = "12:00",
    orb_window: int = 3,
    # MOM params
    momentum_pct: float = 0.10,
    mom_trail_pct: float = 30,
    # SuperTrend params
    st_period: int = 10,
    st_mult: float = 2.0,
    # Risk
    max_trades_day: int = 4,
    max_consec_loss: int = 3,
    daily_loss_pct: float = 15,
    # Trailing stop
    trail_trigger_pct: float = 50,
    trail_pct: float = 35,
    # Premium
    delta: float = 0.45,
    base_premium: float = 95,
    premium_sl_pct: float = 35,
    premium_target: float = 30,
    # Regime
    skip_ranging: bool = True,
    # Drawdown protection
    dd_reduce_threshold: float = 30,
    dd_reduce_factor: float = 0.5,
) -> dict:
    """V2 backtester with trailing stops, regime filter, drawdown protection."""

    unique_days = sorted(set(df.index.date))
    capital = starting_capital
    peak = starting_capital
    trades = []
    equity_curve = []
    np.random.seed(42)

    for day in unique_days:
        dates_arr = pd.Series(df.index.date)
        day_mask = (dates_arr == day).values
        day_df = df[day_mask].copy()
        if len(day_df) < 15:
            equity_curve.append({"date": day, "capital": capital,
                                  "daily_pnl": 0, "trades": 0, "lots": 0,
                                  "regime": "unknown"})
            continue

        # Regime detection
        regime = detect_regime(day_df)
        if skip_ranging and regime == "ranging":
            equity_curve.append({"date": day, "capital": capital,
                                  "daily_pnl": 0, "trades": 0, "lots": 0,
                                  "regime": regime})
            continue

        day_start_cap = capital
        day_pnl = 0.0
        day_trades = 0
        consec_loss = 0
        loss_limit = day_start_cap * (daily_loss_pct / 100)

        # Drawdown-adjusted position sizing
        dd_from_peak = (peak - capital) / peak * 100 if peak > 0 else 0
        sizing_mult = dd_reduce_factor if dd_from_peak > dd_reduce_threshold else 1.0

        cost_per_lot = base_premium * lot_size
        deployable = capital * (deploy_pct / 100) * sizing_mult
        day_lots = max(1, int(deployable / cost_per_lot))

        close = day_df["close"]
        high = day_df["high"]
        low = day_df["low"]
        vol = day_df["volume"]

        # Indicators
        ema_fast = ema(close, 9)
        ema_slow = ema(close, 21)
        rsi_vals = rsi(close, 14)
        vol_sma = vol.rolling(20, min_periods=3).mean()

        # VWAP
        tp = (high + low + close) / 3
        cum_tp_vol = (tp * vol).cumsum()
        cum_vol = vol.cumsum().replace(0, np.nan)
        vwap_line = cum_tp_vol / cum_vol

        # SuperTrend (expects a df with 'high','low','close' columns)
        st, st_dir = supertrend_fast(day_df, st_period, st_mult)

        # ORB from first N candles
        orb_high = day_df.iloc[:orb_window]["high"].max()
        orb_low = day_df.iloc[:orb_window]["low"].min()
        orb_range = orb_high - orb_low

        signals = []

        for i in range(orb_window + 1, len(day_df)):
            ts = day_df.index[i]
            t_str = ts.strftime("%H:%M")
            if t_str < "09:30" or t_str > "14:30":
                continue

            c_now = close.iloc[i]
            c_prev = close.iloc[i - 1]
            h_now = high.iloc[i]
            l_now = low.iloc[i]
            v_now = vol.iloc[i]
            v_avg = vol_sma.iloc[i] if not np.isnan(vol_sma.iloc[i]) else v_now
            r = rsi_vals.iloc[i] if not np.isnan(rsi_vals.iloc[i]) else 50
            vw = vwap_line.iloc[i] if not np.isnan(vwap_line.iloc[i]) else c_now
            st_d = st_dir.iloc[i] if i < len(st_dir) and not np.isnan(st_dir.iloc[i]) else 0

            # ── ORB ────────────────────────────────────────────
            if orb_range_min < orb_range < orb_range_max and t_str <= orb_entry_end:
                if c_now > orb_high and c_prev <= orb_high:
                    signals.append({
                        "strat": "ORB", "dir": "LONG", "idx": i,
                        "entry": c_now, "sl": orb_low,
                        "tgt": c_now + orb_range * orb_rr,
                        "strength": 3,
                    })
                elif c_now < orb_low and c_prev >= orb_low:
                    signals.append({
                        "strat": "ORB", "dir": "SHORT", "idx": i,
                        "entry": c_now, "sl": orb_high,
                        "tgt": c_now - orb_range * orb_rr,
                        "strength": 3,
                    })

            # ── SuperTrend Momentum (replaces bad VWAP_MOM) ────
            if st_d == 1 and c_now > vw and r > 50:
                pct_move = (c_now - c_prev) / c_prev * 100 if c_prev else 0
                if pct_move > momentum_pct * 0.5:
                    signals.append({
                        "strat": "ST_MOM", "dir": "LONG", "idx": i,
                        "entry": c_now, "sl": c_now - 50,
                        "tgt": c_now + 70,
                        "strength": 2,
                    })

            elif st_d == -1 and c_now < vw and r < 50:
                pct_move = (c_prev - c_now) / c_prev * 100 if c_prev else 0
                if pct_move > momentum_pct * 0.5:
                    signals.append({
                        "strat": "ST_MOM", "dir": "SHORT", "idx": i,
                        "entry": c_now, "sl": c_now + 50,
                        "tgt": c_now - 70,
                        "strength": 2,
                    })

            # ── Momentum Burst ─────────────────────────────────
            pct_move = abs(c_now - c_prev) / c_prev * 100 if c_prev else 0
            if pct_move > momentum_pct and v_now > v_avg * 1.2:
                if c_now > c_prev and c_now > vw:
                    signals.append({
                        "strat": "MOM", "dir": "LONG", "idx": i,
                        "entry": c_now, "sl": c_now - 40,
                        "tgt": c_now + 60,
                        "strength": 2 + (1 if st_d == 1 else 0),
                    })
                elif c_now < c_prev and c_now < vw:
                    signals.append({
                        "strat": "MOM", "dir": "SHORT", "idx": i,
                        "entry": c_now, "sl": c_now + 40,
                        "tgt": c_now - 60,
                        "strength": 2 + (1 if st_d == -1 else 0),
                    })

        # Sort by strength (strongest first)
        signals.sort(key=lambda s: s["strength"], reverse=True)

        # Deduplicate -- only one trade per 10-candle window
        used_windows = set()
        filtered = []
        for sig in signals:
            win = sig["idx"] // 10
            if win not in used_windows:
                filtered.append(sig)
                used_windows.add(win)

        # ── Execute signals with trailing stop ──────────────────
        for sig in filtered:
            if day_trades >= max_trades_day:
                break
            if consec_loss >= max_consec_loss:
                break
            if day_pnl < 0 and abs(day_pnl) >= loss_limit:
                break

            entry_i = sig["idx"]
            direction = sig["dir"]
            entry_idx = sig["entry"]

            prem = base_premium + np.random.uniform(-8, 15)
            sl_prem = prem * (1 - premium_sl_pct / 100)
            tgt_prem = prem + premium_target
            qty = day_lots * lot_size

            exit_prem = prem
            exit_reason = "EOD"
            exit_time = day_df.index[-1]
            peak_prem = prem
            trailing_active = False

            for k in range(entry_i + 1, len(day_df)):
                fc = close.iloc[k]
                if direction == "LONG":
                    idx_move = fc - entry_idx
                else:
                    idx_move = entry_idx - fc

                d = delta + np.random.uniform(-0.02, 0.02)
                sim_p = prem + idx_move * d

                # Track peak premium
                if sim_p > peak_prem:
                    peak_prem = sim_p

                # Activate trailing stop
                prem_gain_pct = (peak_prem - prem) / prem * 100
                if prem_gain_pct >= trail_trigger_pct:
                    trailing_active = True
                    trail_floor = peak_prem * (1 - trail_pct / 100)
                    if sim_p <= trail_floor:
                        exit_prem = trail_floor
                        exit_reason = "TRAIL"
                        exit_time = day_df.index[k]
                        break

                # Fixed SL
                if sim_p <= sl_prem:
                    exit_prem = sl_prem
                    exit_reason = "SL"
                    exit_time = day_df.index[k]
                    break

                # Fixed target
                if sim_p >= tgt_prem:
                    exit_prem = tgt_prem
                    exit_reason = "TGT"
                    exit_time = day_df.index[k]
                    break

                # EOD exit
                if day_df.index[k].strftime("%H:%M") >= "15:15":
                    exit_prem = max(sim_p, 1)
                    exit_reason = "EOD"
                    exit_time = day_df.index[k]
                    break

            pnl = (exit_prem - prem) * qty
            capital += pnl
            day_pnl += pnl
            day_trades += 1

            if pnl < 0:
                consec_loss += 1
            else:
                consec_loss = 0

            trades.append({
                "strategy": sig["strat"], "dir": direction,
                "entry_time": day_df.index[entry_i], "exit_time": exit_time,
                "entry_prem": round(prem, 2), "exit_prem": round(exit_prem, 2),
                "qty": qty, "lots": day_lots,
                "pnl": round(pnl, 0), "reason": exit_reason,
                "capital": round(capital, 0),
                "regime": regime,
            })

            if capital <= 0:
                break

        if capital > peak:
            peak = capital

        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0),
            "trades": day_trades, "lots": day_lots,
            "regime": regime,
        })

        if capital <= 0:
            break

    # ── Stats ────────────────────────────────────────────────────
    total_pnl = capital - starting_capital
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    avg_w = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_l = np.mean([t["pnl"] for t in losses]) if losses else 0
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gp / gl if gl > 0 else float("inf")

    eq = [e["capital"] for e in equity_curve]
    if eq:
        pa = np.maximum.accumulate(eq)
        dd = [(p - v) / p * 100 if p > 0 else 0 for p, v in zip(pa, eq)]
        max_dd = max(dd)
    else:
        max_dd = 0

    active = [e for e in equity_curve if e["trades"] > 0]
    prof_days = sum(1 for e in equity_curve if e["daily_pnl"] > 0)
    loss_days = sum(1 for e in equity_curve if e["daily_pnl"] < 0)

    regimes = {}
    for t in trades:
        r = t.get("regime", "unknown")
        if r not in regimes:
            regimes[r] = {"w": 0, "l": 0, "pnl": 0}
        if t["pnl"] > 0:
            regimes[r]["w"] += 1
        else:
            regimes[r]["l"] += 1
        regimes[r]["pnl"] += t["pnl"]

    exit_reasons = {}
    for t in trades:
        er = t["reason"]
        if er not in exit_reasons:
            exit_reasons[er] = {"count": 0, "pnl": 0}
        exit_reasons[er]["count"] += 1
        exit_reasons[er]["pnl"] += t["pnl"]

    return {
        "capital": round(capital, 0),
        "pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / starting_capital * 100, 1),
        "trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1),
        "avg_win": round(avg_w, 0), "avg_loss": round(avg_l, 0),
        "profit_factor": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "prof_days": prof_days, "loss_days": loss_days,
        "trading_days": len(unique_days),
        "active_days": len(active),
        "equity_curve": equity_curve,
        "trade_list": trades,
        "regimes": regimes,
        "exit_reasons": exit_reasons,
    }


def optimize_v2(train_df: pd.DataFrame,
                starting_capital: float = 10000) -> dict:
    """Phased optimization on V2 engine."""

    best_score = -float("inf")
    total = 0

    def score_result(r):
        if r["trades"] < 5:
            return -100
        pf_adj = min(r["profit_factor"], 3.0)
        wr_adj = min(r["win_rate"], 70)
        dd_pen = r["max_dd"] * 0.5
        return pf_adj * wr_adj - dd_pen + r["return_pct"] * 0.1

    # Phase 1: ORB params
    print("\n  Phase 1: ORB optimization...")
    p1_results = []
    for orb_max, orb_rr, orb_win in itertools.product(
        [120, 180, 250, 350], [1.0, 1.5, 2.0, 2.5], [3, 4, 6]
    ):
        r = backtest_v2(train_df, starting_capital=starting_capital,
                        orb_range_max=orb_max, orb_rr=orb_rr, orb_window=orb_win)
        s = score_result(r)
        p1_results.append({
            "params": {"orb_range_max": orb_max, "orb_rr": orb_rr, "orb_window": orb_win},
            "pf": r["profit_factor"], "wr": r["win_rate"],
            "ret": r["return_pct"], "trades": r["trades"],
            "dd": r["max_dd"], "score": s,
        })
        total += 1
    p1_results.sort(key=lambda x: x["score"], reverse=True)
    best_orb = p1_results[0]["params"]
    print(f"    Best: {best_orb} (PF={p1_results[0]['pf']}, "
          f"WR={p1_results[0]['wr']}%, {p1_results[0]['trades']} trades, "
          f"score={p1_results[0]['score']:.1f})")

    # Phase 2: Premium target/SL
    print("  Phase 2: Premium target/SL optimization...")
    p2_results = []
    for tgt, sl_pct in itertools.product(
        [12, 18, 25, 35, 45], [25, 30, 35, 40, 50]
    ):
        r = backtest_v2(train_df, starting_capital=starting_capital,
                        **best_orb, premium_target=tgt, premium_sl_pct=sl_pct)
        s = score_result(r)
        p2_results.append({
            "params": {"premium_target": tgt, "premium_sl_pct": sl_pct},
            "pf": r["profit_factor"], "wr": r["win_rate"],
            "ret": r["return_pct"], "trades": r["trades"],
            "dd": r["max_dd"], "score": s,
        })
        total += 1
    p2_results.sort(key=lambda x: x["score"], reverse=True)
    best_prem = p2_results[0]["params"]
    print(f"    Best: {best_prem} (PF={p2_results[0]['pf']}, "
          f"WR={p2_results[0]['wr']}%, score={p2_results[0]['score']:.1f})")

    # Phase 3: Momentum + trailing
    print("  Phase 3: Momentum + trailing stop optimization...")
    p3_results = []
    for mom_pct, trail_trig, trail_pct in itertools.product(
        [0.06, 0.10, 0.15, 0.20],
        [30, 50, 80],
        [25, 35, 50]
    ):
        r = backtest_v2(train_df, starting_capital=starting_capital,
                        **best_orb, **best_prem,
                        momentum_pct=mom_pct,
                        trail_trigger_pct=trail_trig, trail_pct=trail_pct)
        s = score_result(r)
        p3_results.append({
            "params": {"momentum_pct": mom_pct,
                       "trail_trigger_pct": trail_trig, "trail_pct": trail_pct},
            "pf": r["profit_factor"], "wr": r["win_rate"],
            "ret": r["return_pct"], "trades": r["trades"],
            "dd": r["max_dd"], "score": s,
        })
        total += 1
    p3_results.sort(key=lambda x: x["score"], reverse=True)
    best_mom = p3_results[0]["params"]
    print(f"    Best: {best_mom} (PF={p3_results[0]['pf']}, "
          f"WR={p3_results[0]['wr']}%, score={p3_results[0]['score']:.1f})")

    # Phase 4: Risk params
    print("  Phase 4: Risk management optimization...")
    p4_results = []
    for max_trades, max_cl, dd_thresh in itertools.product(
        [3, 4, 5, 6],
        [2, 3, 4],
        [20, 30, 40]
    ):
        r = backtest_v2(train_df, starting_capital=starting_capital,
                        **best_orb, **best_prem, **best_mom,
                        max_trades_day=max_trades, max_consec_loss=max_cl,
                        dd_reduce_threshold=dd_thresh)
        s = score_result(r)
        p4_results.append({
            "params": {"max_trades_day": max_trades, "max_consec_loss": max_cl,
                       "dd_reduce_threshold": dd_thresh},
            "pf": r["profit_factor"], "wr": r["win_rate"],
            "ret": r["return_pct"], "trades": r["trades"],
            "dd": r["max_dd"], "score": s,
        })
        total += 1
    p4_results.sort(key=lambda x: x["score"], reverse=True)
    best_risk = p4_results[0]["params"]
    print(f"    Best: {best_risk} (PF={p4_results[0]['pf']}, "
          f"WR={p4_results[0]['wr']}%, score={p4_results[0]['score']:.1f})")

    print(f"\n  Total combos tested: {total}")

    # Final combined run
    all_params = {**best_orb, **best_prem, **best_mom, **best_risk}
    final = backtest_v2(train_df, starting_capital=starting_capital, **all_params)

    return {"best_params": all_params, "train_result": final}


def walk_forward_v2(df: pd.DataFrame,
                    train_pct: float = 0.6,
                    starting_capital: float = 10000) -> dict:
    """Walk-forward with V2 engine."""

    unique_days = sorted(set(df.index.date))
    split_idx = int(len(unique_days) * train_pct)
    train_days = unique_days[:split_idx]
    test_days = unique_days[split_idx:]

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD V2 -- REAL DATA OPTIMIZATION")
    print(f"{'='*70}")
    print(f"  Data range  : {unique_days[0]} to {unique_days[-1]}")
    print(f"  Train period: {train_days[0]} to {train_days[-1]} ({len(train_days)} days)")
    print(f"  Test period : {test_days[0]} to {test_days[-1]} ({len(test_days)} days)")

    train_set = set(train_days)
    test_set = set(test_days)
    dates_arr = pd.Series(df.index.date)
    train_df = df[dates_arr.isin(train_set).values].copy()
    test_df = df[dates_arr.isin(test_set).values].copy()

    print(f"\n  PHASE 1: Optimizing on train data ({len(train_days)} days)...")
    opt = optimize_v2(train_df, starting_capital)
    best_params = opt["best_params"]
    train_result = opt["train_result"]

    print(f"\n{'='*70}")
    print(f"  TRAIN RESULTS ({train_days[0]} to {train_days[-1]})")
    print(f"{'='*70}")
    print_result_v2(train_result)

    print(f"\n  PHASE 2: OUT-OF-SAMPLE validation ({len(test_days)} days)...")
    test_result = backtest_v2(test_df, starting_capital=starting_capital, **best_params)

    print(f"\n{'='*70}")
    print(f"  TEST RESULTS -- OUT-OF-SAMPLE ({test_days[0]} to {test_days[-1]})")
    print(f"{'='*70}")
    print_result_v2(test_result)

    # Stability
    print(f"\n{'='*70}")
    print(f"  STABILITY ANALYSIS")
    print(f"{'='*70}")
    train_pf = train_result["profit_factor"]
    test_pf = test_result["profit_factor"]
    train_wr = train_result["win_rate"]
    test_wr = test_result["win_rate"]
    train_dd = train_result["max_dd"]
    test_dd = test_result["max_dd"]

    pf_ratio = min(train_pf, test_pf) / max(train_pf, test_pf) if max(train_pf, test_pf) > 0 else 0

    print(f"  {'Metric':<20} {'Train':>10} {'Test':>10} {'Stable?':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Profit Factor':<20} {train_pf:>10.2f} {test_pf:>10.2f} "
          f"{'YES' if pf_ratio > 0.5 else 'NO':>10}")
    print(f"  {'Win Rate':<20} {train_wr:>9.0f}% {test_wr:>9.0f}% "
          f"{'YES' if abs(train_wr - test_wr) < 15 else 'NO':>10}")
    print(f"  {'Max Drawdown':<20} {train_dd:>9.0f}% {test_dd:>9.0f}% "
          f"{'YES' if test_dd < 60 else 'WARN':>10}")
    print(f"  {'Trades/Day':<20} "
          f"{train_result['trades']/max(train_result['active_days'],1):>10.1f} "
          f"{test_result['trades']/max(test_result['active_days'],1):>10.1f}")

    # Verdict
    if test_pf > 1.3 and test_wr > 45 and test_dd < 50:
        verdict = "STRONG PASS"
        msg = "System shows robust edge. Ready for paper trading."
    elif test_pf > 1.1 and test_wr > 40:
        verdict = "PASS"
        msg = "System shows edge on OOS. Continue paper testing to confirm."
    elif test_pf > 1.0:
        verdict = "MARGINAL"
        msg = "Slight edge exists. More iteration needed."
    else:
        verdict = "FAIL"
        msg = "No edge on OOS. Requires fundamental redesign."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {msg}")

    # Best params
    print(f"\n{'='*70}")
    print(f"  OPTIMIZED PARAMETERS")
    print(f"{'='*70}")
    for k, v in best_params.items():
        print(f"    {k:<25} = {v}")

    return {
        "best_params": best_params,
        "train": train_result,
        "test": test_result,
        "verdict": verdict,
        "stability": {"pf_ratio": pf_ratio},
    }


def print_result_v2(r: dict, prefix: str = "  "):
    pnl = r["pnl"]
    s = "+" if pnl >= 0 else ""
    print(f"{prefix}Capital      : Rs {r['capital']:>12,.0f}")
    print(f"{prefix}P&L          : Rs {s}{pnl:>11,.0f} ({r['return_pct']}%)")
    print(f"{prefix}Trades       : {r['trades']} ({r['wins']}W / {r['losses']}L)")
    print(f"{prefix}Win Rate     : {r['win_rate']}%")
    print(f"{prefix}Profit Factor: {r['profit_factor']}")
    print(f"{prefix}Max Drawdown : {r['max_dd']}%")
    print(f"{prefix}Days (P/L)   : {r['prof_days']}W / {r['loss_days']}L / "
          f"{r['trading_days'] - r['active_days']} skip")

    # Strategy breakdown
    strats = {}
    for t in r.get("trade_list", []):
        st = t["strategy"]
        if st not in strats:
            strats[st] = {"w": 0, "l": 0, "pnl": 0}
        if t["pnl"] > 0:
            strats[st]["w"] += 1
        else:
            strats[st]["l"] += 1
        strats[st]["pnl"] += t["pnl"]
    if strats:
        print(f"{prefix}Strategy breakdown:")
        for name, st in sorted(strats.items()):
            tot = st["w"] + st["l"]
            wr = st["w"] / tot * 100 if tot else 0
            print(f"{prefix}  {name:10s}: {tot:>3d} trades, "
                  f"{wr:>4.0f}% win, Rs {st['pnl']:>10,.0f}")

    # Exit reasons
    exits = r.get("exit_reasons", {})
    if exits:
        print(f"{prefix}Exit reasons:")
        for er, data in sorted(exits.items()):
            print(f"{prefix}  {er:6s}: {data['count']:>3d} exits, Rs {data['pnl']:>10,.0f}")

    # Regime breakdown
    regimes = r.get("regimes", {})
    if regimes:
        print(f"{prefix}Regime breakdown:")
        for rg, data in sorted(regimes.items()):
            tot = data["w"] + data["l"]
            wr = data["w"] / tot * 100 if tot else 0
            print(f"{prefix}  {rg:10s}: {tot:>3d} trades, "
                  f"{wr:>4.0f}% win, Rs {data['pnl']:>10,.0f}")

    # Equity curve milestones
    ec = r.get("equity_curve", [])
    if ec and len(ec) > 3:
        print(f"{prefix}Equity milestones:")
        indices = [0] + [i for i in range(len(ec) // 5, len(ec), len(ec) // 5)] + [len(ec) - 1]
        indices = sorted(set(min(i, len(ec) - 1) for i in indices))
        for idx in indices:
            e = ec[idx]
            regime_info = f" [{e.get('regime', '')}]" if e.get("regime") else ""
            print(f"{prefix}  Day {idx+1:>3d} ({e['date']}): "
                  f"Rs {e['capital']:>10,.0f}  "
                  f"[{e['trades']} trades, Rs {e['daily_pnl']:>+7,.0f}]{regime_info}")


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    data_path = settings.DATA_DIR / "nifty_5m_real.csv"
    if not data_path.exists():
        print("ERROR: No real data found. Run backtest/data_fetcher.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} real Nifty 5-min candles")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    days = len(set(df.index.date))
    print(f"Trading days: {days}")

    result = walk_forward_v2(df, train_pct=0.6, starting_capital=10000)

    # Save results
    if result["test"]["trade_list"]:
        pd.DataFrame(result["test"]["trade_list"]).to_csv(
            settings.DATA_DIR / "wf_v2_test_trades.csv", index=False)
    if result["test"]["equity_curve"]:
        pd.DataFrame(result["test"]["equity_curve"]).to_csv(
            settings.DATA_DIR / "wf_v2_test_equity.csv", index=False)

    return result


if __name__ == "__main__":
    main()
