"""Walk-forward optimization and validation on REAL Nifty data.

This is the core system perfection engine:
1. Loads real 5-min Nifty data
2. Splits into train (first 60%) and test (last 40%) periods
3. Optimizes strategy parameters on train period
4. Validates on test period (out-of-sample)
5. Iterates until consistent edge found

Usage:
    python -m backtest.walk_forward
"""

from __future__ import annotations
import sys
import itertools
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.indicators import ema, rsi, atr, supertrend_fast


# ── Core backtester on real 5-min data ──────────────────────────────

def backtest_on_real_data(
    df: pd.DataFrame,
    lot_size: int = 75,
    starting_capital: float = 10000,
    deploy_pct: float = 80.0,
    # ORB params
    orb_range_max: float = 250,
    orb_range_min: float = 10,
    orb_vol_mult: float = 1.3,
    orb_rr: float = 1.5,
    orb_sl_pct: float = 40,
    orb_entry_end: str = "11:30",
    # VWAP/EMA params
    ema_fast: int = 9,
    ema_slow: int = 21,
    rsi_period: int = 14,
    rsi_long_thresh: float = 48,
    rsi_short_thresh: float = 52,
    vwap_sl_points: float = 40,
    vwap_target_points: float = 55,
    # Momentum params
    momentum_pct: float = 0.12,
    momentum_vol_mult: float = 1.3,
    mom_sl_points: float = 35,
    mom_target_points: float = 50,
    # Risk
    max_trades_day: int = 5,
    max_consec_loss: int = 2,
    daily_loss_pct: float = 20,
    # Premium simulation
    delta: float = 0.45,
    base_premium: float = 95,
    premium_sl_pct: float = 40,
    premium_target: float = 25,
) -> dict:
    """Run a single backtest pass with given parameters on real data."""

    unique_days = sorted(set(df.index.date))
    capital = starting_capital
    peak = capital
    trades = []
    equity_curve = []

    for day in unique_days:
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 12:
            equity_curve.append({"date": day, "capital": capital,
                                  "daily_pnl": 0, "trades": 0, "lots": 0})
            continue

        day_start_cap = capital
        day_pnl = 0.0
        day_trades = 0
        consec_loss = 0
        loss_limit = day_start_cap * (daily_loss_pct / 100)

        # Dynamic lot sizing
        cost_per_lot = base_premium * lot_size
        deployable = capital * (deploy_pct / 100)
        day_lots = max(1, int(deployable / cost_per_lot))

        close = day_df["close"]
        high = day_df["high"]
        low = day_df["low"]
        vol = day_df["volume"]

        # Indicators
        ema_f = ema(close, ema_fast)
        ema_s = ema(close, ema_slow)
        rsi_vals = rsi(close, rsi_period)
        vol_sma = vol.rolling(20, min_periods=5).mean()

        # VWAP
        tp = (high + low + close) / 3
        cum_tp_vol = (tp * vol).cumsum()
        cum_vol = vol.cumsum().replace(0, np.nan)
        vwap_line = cum_tp_vol / cum_vol

        # ORB from first 3 candles (15 min of 5-min data)
        orb_count = 3
        if len(day_df) < orb_count + 2:
            equity_curve.append({"date": day, "capital": capital,
                                  "daily_pnl": 0, "trades": 0, "lots": day_lots})
            continue

        orb_high = day_df.iloc[:orb_count]["high"].max()
        orb_low = day_df.iloc[:orb_count]["low"].min()
        orb_range = orb_high - orb_low

        signals = []

        for i in range(orb_count + 1, len(day_df)):
            ts = day_df.index[i]
            t_str = ts.strftime("%H:%M")
            if t_str < "09:30" or t_str > "14:30":
                continue

            c_now = close.iloc[i]
            c_prev = close.iloc[i - 1]
            h_now = high.iloc[i]
            l_now = low.iloc[i]
            v_now = vol.iloc[i]
            v_avg = vol_sma.iloc[i] if i < len(vol_sma) and not np.isnan(vol_sma.iloc[i]) else v_now
            ef = ema_f.iloc[i] if not np.isnan(ema_f.iloc[i]) else c_now
            es = ema_s.iloc[i] if not np.isnan(ema_s.iloc[i]) else c_now
            ef_p = ema_f.iloc[i-1] if not np.isnan(ema_f.iloc[i-1]) else c_prev
            es_p = ema_s.iloc[i-1] if not np.isnan(ema_s.iloc[i-1]) else c_prev
            r = rsi_vals.iloc[i] if not np.isnan(rsi_vals.iloc[i]) else 50
            vw = vwap_line.iloc[i] if not np.isnan(vwap_line.iloc[i]) else c_now

            # ── ORB ────────────────────────────────────────────
            if (orb_range_min < orb_range < orb_range_max and t_str <= orb_entry_end):
                if c_now > orb_high and c_prev <= orb_high:
                    if v_now > v_avg * orb_vol_mult or orb_vol_mult == 0:
                        sl_idx = orb_low
                        tgt_idx = c_now + orb_range * orb_rr
                        signals.append(("ORB", "LONG", i, c_now, sl_idx, tgt_idx))

                elif c_now < orb_low and c_prev >= orb_low:
                    if v_now > v_avg * orb_vol_mult or orb_vol_mult == 0:
                        sl_idx = orb_high
                        tgt_idx = c_now - orb_range * orb_rr
                        signals.append(("ORB", "SHORT", i, c_now, sl_idx, tgt_idx))

            # ── EMA Crossover ──────────────────────────────────
            cross_up = ef_p <= es_p and ef > es
            cross_down = ef_p >= es_p and ef < es

            if cross_up and c_now > vw and r > rsi_long_thresh:
                signals.append(("VWAP_MOM", "LONG", i, c_now,
                                c_now - vwap_sl_points, c_now + vwap_target_points))

            elif cross_down and c_now < vw and r < rsi_short_thresh:
                signals.append(("VWAP_MOM", "SHORT", i, c_now,
                                c_now + vwap_sl_points, c_now - vwap_target_points))

            # ── Momentum burst ─────────────────────────────────
            pct_move = (c_now - c_prev) / c_prev * 100 if c_prev != 0 else 0

            if abs(pct_move) > momentum_pct and v_now > v_avg * momentum_vol_mult:
                if pct_move > 0 and c_now > vw:
                    signals.append(("MOM", "LONG", i, c_now,
                                    c_now - mom_sl_points, c_now + mom_target_points))
                elif pct_move < 0 and c_now < vw:
                    signals.append(("MOM", "SHORT", i, c_now,
                                    c_now + mom_sl_points, c_now - mom_target_points))

        # ── Execute signals ─────────────────────────────────────
        for sig in signals:
            strat, direction, entry_i, entry_idx, sl_idx, tgt_idx = sig

            if day_trades >= max_trades_day:
                break
            if consec_loss >= max_consec_loss:
                break
            if day_pnl < 0 and abs(day_pnl) >= loss_limit:
                break

            prem = base_premium + np.random.uniform(-10, 20)
            sl_prem = prem * (1 - premium_sl_pct / 100)
            tgt_prem = prem + premium_target
            qty = day_lots * lot_size

            exit_prem = prem
            exit_reason = "EOD"
            exit_time = day_df.index[-1]

            for k in range(entry_i + 1, len(day_df)):
                fc = close.iloc[k]
                if direction == "LONG":
                    idx_move = fc - entry_idx
                else:
                    idx_move = entry_idx - fc

                d = delta + np.random.uniform(-0.03, 0.03)
                sim_p = prem + idx_move * d

                if sim_p <= sl_prem:
                    exit_prem = sl_prem
                    exit_reason = "SL"
                    exit_time = day_df.index[k]
                    break
                elif sim_p >= tgt_prem:
                    exit_prem = tgt_prem
                    exit_reason = "TGT"
                    exit_time = day_df.index[k]
                    break

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
                "strategy": strat, "dir": direction,
                "entry_time": day_df.index[entry_i], "exit_time": exit_time,
                "entry_prem": round(prem, 2), "exit_prem": round(exit_prem, 2),
                "qty": qty, "lots": day_lots,
                "pnl": round(pnl, 0), "reason": exit_reason,
                "capital": round(capital, 0),
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

    # Stats
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
        dd = [(p - v) / p * 100 for p, v in zip(pa, eq)]
        max_dd = max(dd)
    else:
        max_dd = 0

    active = [e for e in equity_curve if e["trades"] > 0]
    drs = [(e["daily_pnl"] / max(e["capital"] - e["daily_pnl"], 1)) * 100
           for e in active]
    avg_dr = np.mean(drs) if drs else 0

    prof_days = sum(1 for e in equity_curve if e["daily_pnl"] > 0)
    loss_days = sum(1 for e in equity_curve if e["daily_pnl"] < 0)

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
        "avg_daily_ret": round(avg_dr, 1),
        "prof_days": prof_days, "loss_days": loss_days,
        "trading_days": len(unique_days),
        "active_days": len(active),
        "equity_curve": equity_curve,
        "trade_list": trades,
    }


# ── Parameter Grid Optimization ─────────────────────────────────────

def optimize_parameters(train_df: pd.DataFrame,
                        starting_capital: float = 10000) -> dict:
    """Grid search over key parameters on training data."""

    param_grid = {
        "orb_range_max": [150, 200, 250, 300],
        "orb_rr": [1.0, 1.5, 2.0],
        "ema_fast": [5, 9, 13],
        "ema_slow": [15, 21, 30],
        "rsi_long_thresh": [45, 50, 55],
        "momentum_pct": [0.08, 0.12, 0.18],
        "premium_target": [15, 20, 25, 30],
        "premium_sl_pct": [30, 40, 50],
    }

    # Run a reasonable subset: key combos only
    best_result = None
    best_params = {}
    best_score = -float("inf")
    results = []
    total = 0

    # Phase 1: Optimize ORB params (fix others at defaults)
    print("\n  Phase 1: Optimizing ORB parameters...")
    for orb_max, orb_rr in itertools.product(
        param_grid["orb_range_max"], param_grid["orb_rr"]
    ):
        r = backtest_on_real_data(
            train_df, starting_capital=starting_capital,
            orb_range_max=orb_max, orb_rr=orb_rr,
        )
        score = r["profit_factor"] * min(r["win_rate"], 70) - r["max_dd"] * 0.3
        if r["trades"] < 5:
            score -= 50
        results.append({"params": {"orb_max": orb_max, "orb_rr": orb_rr},
                         "pf": r["profit_factor"], "wr": r["win_rate"],
                         "ret": r["return_pct"], "trades": r["trades"],
                         "score": score})
        total += 1

    orb_results = sorted(results, key=lambda x: x["score"], reverse=True)
    best_orb = orb_results[0]["params"] if orb_results else {"orb_max": 250, "orb_rr": 1.5}
    print(f"    Best ORB: max={best_orb['orb_max']}, RR={best_orb['orb_rr']} "
          f"(score={orb_results[0]['score']:.1f})")

    # Phase 2: Optimize EMA/RSI params
    print("  Phase 2: Optimizing EMA/RSI parameters...")
    results2 = []
    for ef, es, rsi_t in itertools.product(
        param_grid["ema_fast"], param_grid["ema_slow"], param_grid["rsi_long_thresh"]
    ):
        if ef >= es:
            continue
        r = backtest_on_real_data(
            train_df, starting_capital=starting_capital,
            orb_range_max=best_orb["orb_max"], orb_rr=best_orb["orb_rr"],
            ema_fast=ef, ema_slow=es,
            rsi_long_thresh=rsi_t, rsi_short_thresh=100 - rsi_t,
        )
        score = r["profit_factor"] * min(r["win_rate"], 70) - r["max_dd"] * 0.3
        if r["trades"] < 5:
            score -= 50
        results2.append({"params": {"ema_fast": ef, "ema_slow": es, "rsi_thresh": rsi_t},
                          "pf": r["profit_factor"], "wr": r["win_rate"],
                          "ret": r["return_pct"], "trades": r["trades"],
                          "score": score})
        total += 1

    ema_results = sorted(results2, key=lambda x: x["score"], reverse=True)
    best_ema = ema_results[0]["params"] if ema_results else {"ema_fast": 9, "ema_slow": 21, "rsi_thresh": 48}
    print(f"    Best EMA: fast={best_ema['ema_fast']}, slow={best_ema['ema_slow']}, "
          f"RSI={best_ema['rsi_thresh']} (score={ema_results[0]['score']:.1f})")

    # Phase 3: Optimize premium targets and SL
    print("  Phase 3: Optimizing premium target/SL...")
    results3 = []
    for tgt, sl_pct in itertools.product(
        param_grid["premium_target"], param_grid["premium_sl_pct"]
    ):
        r = backtest_on_real_data(
            train_df, starting_capital=starting_capital,
            orb_range_max=best_orb["orb_max"], orb_rr=best_orb["orb_rr"],
            ema_fast=best_ema["ema_fast"], ema_slow=best_ema["ema_slow"],
            rsi_long_thresh=best_ema["rsi_thresh"],
            rsi_short_thresh=100 - best_ema["rsi_thresh"],
            premium_target=tgt, premium_sl_pct=sl_pct,
        )
        score = r["profit_factor"] * min(r["win_rate"], 70) - r["max_dd"] * 0.3
        if r["trades"] < 5:
            score -= 50
        results3.append({"params": {"target": tgt, "sl_pct": sl_pct},
                          "pf": r["profit_factor"], "wr": r["win_rate"],
                          "ret": r["return_pct"], "trades": r["trades"],
                          "score": score})
        total += 1

    prem_results = sorted(results3, key=lambda x: x["score"], reverse=True)
    best_prem = prem_results[0]["params"] if prem_results else {"target": 25, "sl_pct": 40}
    print(f"    Best Premium: target={best_prem['target']}, SL={best_prem['sl_pct']}% "
          f"(score={prem_results[0]['score']:.1f})")

    # Phase 4: Optimize momentum
    print("  Phase 4: Optimizing momentum parameters...")
    results4 = []
    for mom_pct in param_grid["momentum_pct"]:
        r = backtest_on_real_data(
            train_df, starting_capital=starting_capital,
            orb_range_max=best_orb["orb_max"], orb_rr=best_orb["orb_rr"],
            ema_fast=best_ema["ema_fast"], ema_slow=best_ema["ema_slow"],
            rsi_long_thresh=best_ema["rsi_thresh"],
            rsi_short_thresh=100 - best_ema["rsi_thresh"],
            premium_target=best_prem["target"], premium_sl_pct=best_prem["sl_pct"],
            momentum_pct=mom_pct,
        )
        score = r["profit_factor"] * min(r["win_rate"], 70) - r["max_dd"] * 0.3
        if r["trades"] < 5:
            score -= 50
        results4.append({"params": {"momentum_pct": mom_pct},
                          "pf": r["profit_factor"], "wr": r["win_rate"],
                          "ret": r["return_pct"], "trades": r["trades"],
                          "score": score})
        total += 1

    mom_results = sorted(results4, key=lambda x: x["score"], reverse=True)
    best_mom = mom_results[0]["params"] if mom_results else {"momentum_pct": 0.12}
    print(f"    Best Momentum: threshold={best_mom['momentum_pct']}% "
          f"(score={mom_results[0]['score']:.1f})")

    print(f"\n  Total combinations tested: {total}")

    best_params = {
        "orb_range_max": best_orb["orb_max"],
        "orb_rr": best_orb["orb_rr"],
        "ema_fast": best_ema["ema_fast"],
        "ema_slow": best_ema["ema_slow"],
        "rsi_long_thresh": best_ema["rsi_thresh"],
        "rsi_short_thresh": 100 - best_ema["rsi_thresh"],
        "momentum_pct": best_mom["momentum_pct"],
        "premium_target": best_prem["target"],
        "premium_sl_pct": best_prem["sl_pct"],
    }

    # Final run with best params
    final = backtest_on_real_data(train_df, starting_capital=starting_capital, **best_params)

    return {"best_params": best_params, "train_result": final}


# ── Walk-Forward Validation ─────────────────────────────────────────

def walk_forward_test(df: pd.DataFrame,
                       train_pct: float = 0.6,
                       starting_capital: float = 10000) -> dict:
    """Split data into train/test, optimize on train, validate on test."""

    unique_days = sorted(set(df.index.date))
    split_idx = int(len(unique_days) * train_pct)
    train_days = unique_days[:split_idx]
    test_days = unique_days[split_idx:]

    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"{'='*65}")
    print(f"  Total days  : {len(unique_days)}")
    print(f"  Train period: {train_days[0]} to {train_days[-1]} ({len(train_days)} days)")
    print(f"  Test period : {test_days[0]} to {test_days[-1]} ({len(test_days)} days)")

    train_set = set(train_days)
    test_set = set(test_days)
    dates_arr = pd.Series(df.index.date)
    train_mask = dates_arr.isin(train_set).values
    test_mask = dates_arr.isin(test_set).values
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # Phase 1: Optimize on train data
    print(f"\n  PHASE 1: Optimizing on training data...")
    opt = optimize_parameters(train_df, starting_capital)
    best_params = opt["best_params"]
    train_result = opt["train_result"]

    print(f"\n  TRAIN RESULTS:")
    print_result(train_result, "  ")

    # Phase 2: Validate on test data (out-of-sample)
    print(f"\n  PHASE 2: Validating on test data (OUT-OF-SAMPLE)...")
    test_result = backtest_on_real_data(test_df, starting_capital=starting_capital,
                                         **best_params)

    print(f"\n  TEST RESULTS (OUT-OF-SAMPLE):")
    print_result(test_result, "  ")

    # Stability check
    print(f"\n  STABILITY CHECK:")
    train_pf = train_result["profit_factor"]
    test_pf = test_result["profit_factor"]
    train_wr = train_result["win_rate"]
    test_wr = test_result["win_rate"]

    pf_ratio = min(train_pf, test_pf) / max(train_pf, test_pf) if max(train_pf, test_pf) > 0 else 0
    wr_diff = abs(train_wr - test_wr)

    print(f"    PF stability  : {pf_ratio:.2f} (1.0 = perfectly stable)")
    print(f"    WR difference : {wr_diff:.1f}% between train/test")
    print(f"    Train PF={train_pf:.2f}, Test PF={test_pf:.2f}")
    print(f"    Train WR={train_wr:.0f}%, Test WR={test_wr:.0f}%")

    if test_pf > 1.2 and test_wr > 40 and pf_ratio > 0.5:
        verdict = "PASS -- System shows consistent edge in-sample AND out-of-sample"
    elif test_pf > 1.0 and test_wr > 35:
        verdict = "MARGINAL -- Edge exists but weak on out-of-sample. Paper trade to confirm."
    else:
        verdict = "FAIL -- System does not generalize. Needs redesign."

    print(f"\n  VERDICT: {verdict}")

    return {
        "best_params": best_params,
        "train": train_result,
        "test": test_result,
        "stability": {
            "pf_ratio": pf_ratio,
            "wr_diff": wr_diff,
        },
        "verdict": verdict,
    }


def print_result(r: dict, prefix: str = ""):
    pnl = r["pnl"]
    s = "+" if pnl >= 0 else ""
    print(f"{prefix}Capital     : Rs {r['capital']:>12,.0f} (from Rs 10,000)")
    print(f"{prefix}P&L         : Rs {s}{pnl:>11,.0f} ({r['return_pct']}%)")
    print(f"{prefix}Trades      : {r['trades']} ({r['wins']}W / {r['losses']}L)")
    print(f"{prefix}Win Rate    : {r['win_rate']}%")
    print(f"{prefix}Profit Fctr : {r['profit_factor']}")
    print(f"{prefix}Max DD      : {r['max_dd']}%")
    print(f"{prefix}Avg Daily   : {r['avg_daily_ret']}%")
    print(f"{prefix}Active Days : {r['active_days']} / {r['trading_days']}")

    strats = {}
    for t in r.get("trade_list", []):
        s = t["strategy"]
        if s not in strats:
            strats[s] = {"w": 0, "l": 0, "pnl": 0}
        if t["pnl"] > 0:
            strats[s]["w"] += 1
        else:
            strats[s]["l"] += 1
        strats[s]["pnl"] += t["pnl"]
    if strats:
        print(f"{prefix}Strategies  :")
        for name, st in sorted(strats.items()):
            tot = st["w"] + st["l"]
            wr = st["w"] / tot * 100 if tot else 0
            print(f"{prefix}  {name:10s}: {tot:>3d} trades, {wr:>4.0f}% win, Rs {st['pnl']:>10,.0f}")

    # Equity milestones
    ec = r.get("equity_curve", [])
    if ec and len(ec) > 5:
        print(f"{prefix}Equity curve:")
        show = [0, len(ec)//4, len(ec)//2, 3*len(ec)//4, len(ec)-1]
        for idx in show:
            e = ec[idx]
            print(f"{prefix}  Day {idx+1:>3d} ({e['date']}): Rs {e['capital']:>10,.0f}")


def print_optimized_settings(params: dict):
    """Print the optimized settings ready to paste into config/settings.py."""
    print("\n" + "=" * 65)
    print("  OPTIMIZED SETTINGS (paste into config/settings.py)")
    print("=" * 65)
    print(f"  ORB_MAX_RANGE_POINTS = {params['orb_range_max']}")
    print(f"  ORB_RR_RATIO = {params['orb_rr']}")
    print(f"  VWAP_EMA_FAST = {params['ema_fast']}")
    print(f"  VWAP_EMA_SLOW = {params['ema_slow']}")
    print(f"  VWAP_RSI_LONG_THRESHOLD = {params['rsi_long_thresh']}")
    print(f"  VWAP_RSI_SHORT_THRESHOLD = {params['rsi_short_thresh']}")
    print(f"  PREMIUM_TARGET_POINTS = {params['premium_target']}")
    print(f"  PREMIUM_SL_PCT = {params['premium_sl_pct']}")
    print(f"  # Momentum threshold: {params['momentum_pct']}%")
    print("=" * 65)


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    # Load real data
    data_path = settings.DATA_DIR / "nifty_5m_real.csv"
    if not data_path.exists():
        print("No real data found. Downloading...")
        from backtest.data_fetcher import download_all
        download_all()

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} real Nifty 5-min candles")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Trading days: {len(set(df.index.date))}")

    # Run walk-forward
    result = walk_forward_test(df, train_pct=0.6, starting_capital=10000)

    # Print optimized params
    print_optimized_settings(result["best_params"])

    # Save results
    pd.DataFrame(result["test"]["trade_list"]).to_csv(
        settings.DATA_DIR / "wf_test_trades.csv", index=False)
    pd.DataFrame(result["test"]["equity_curve"]).to_csv(
        settings.DATA_DIR / "wf_test_equity.csv", index=False)

    return result


if __name__ == "__main__":
    main()
