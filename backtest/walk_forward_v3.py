"""Walk-Forward V3 -- Multi-Indicator Confluence System.

Uses 40+ indicators voting on every candle. Only trades when
confluence exceeds threshold. Walk-forward optimizes the
threshold and indicator weights.

Usage:
    python -m backtest.walk_forward_v3
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
from engine.confluence import ConfluenceEngine, ConfluenceResult


def backtest_confluence(
    df: pd.DataFrame,
    lot_size: int = 75,
    starting_capital: float = 10000,
    deploy_pct: float = 80.0,
    # Confluence thresholds
    entry_threshold: float = 40.0,
    exit_reversal: float = -10.0,
    # Indicator weights
    trend_weight: float = 1.0,
    momentum_weight: float = 1.2,
    volatility_weight: float = 0.8,
    volume_weight: float = 0.9,
    trend_strength_weight: float = 1.1,
    structure_weight: float = 0.7,
    # Risk
    max_trades_day: int = 4,
    max_consec_loss: int = 3,
    daily_loss_pct: float = 15,
    # Premium sim
    delta: float = 0.45,
    base_premium: float = 95,
    premium_sl_pct: float = 35,
    # Trailing stop
    trail_trigger_pct: float = 50,
    trail_pct: float = 30,
    # Drawdown protection
    dd_reduce_threshold: float = 30,
    dd_reduce_factor: float = 0.5,
    # Time filters
    entry_start: str = "09:30",
    entry_end: str = "14:00",
    exit_time: str = "15:15",
    # Minimum strength
    min_strength: str = "MODERATE",
) -> dict:
    """Backtest using multi-indicator confluence scoring."""

    strength_order = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3, "EXTREME": 4}
    min_str_val = strength_order.get(min_strength, 2)

    engine = ConfluenceEngine(
        trend_weight=trend_weight,
        momentum_weight=momentum_weight,
        volatility_weight=volatility_weight,
        volume_weight=volume_weight,
        trend_strength_weight=trend_strength_weight,
        structure_weight=structure_weight,
    )

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
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0, "lots": 0,
                                  "best_score": 0, "signals_found": 0})
            continue

        # Precompute all 40+ indicators for the day
        try:
            indicators = engine.precompute(day_df)
        except Exception:
            equity_curve.append({"date": day, "capital": round(capital, 0),
                                  "daily_pnl": 0, "trades": 0, "lots": 0,
                                  "best_score": 0, "signals_found": 0})
            continue

        day_start_cap = capital
        day_pnl = 0.0
        day_trades = 0
        consec_loss = 0
        loss_limit = day_start_cap * (daily_loss_pct / 100)

        # Drawdown-adjusted sizing with progressive scaling
        dd_from_peak = (peak - capital) / peak * 100 if peak > 0 else 0
        sizing_mult = dd_reduce_factor if dd_from_peak > dd_reduce_threshold else 1.0
        # Progressive deploy: as capital grows, deploy a smaller %
        # This prevents one massive loss from wiping months of gains
        if capital > starting_capital * 10:
            effective_deploy = deploy_pct * 0.4
        elif capital > starting_capital * 5:
            effective_deploy = deploy_pct * 0.6
        elif capital > starting_capital * 2:
            effective_deploy = deploy_pct * 0.8
        else:
            effective_deploy = deploy_pct
        cost_per_lot = base_premium * lot_size
        deployable = capital * (effective_deploy / 100) * sizing_mult
        day_lots = max(1, int(deployable / cost_per_lot))

        signals_found = 0
        best_score = 0
        in_trade = False

        for i in range(3, len(day_df)):
            ts = day_df.index[i]
            t_str = ts.strftime("%H:%M")

            if in_trade:
                continue

            if t_str < entry_start or t_str > entry_end:
                continue

            if day_trades >= max_trades_day:
                break
            if consec_loss >= max_consec_loss:
                break
            if day_pnl < 0 and abs(day_pnl) >= loss_limit:
                break

            # Score this candle with ALL indicators
            result = engine.score(indicators, i)
            abs_score = abs(result.score)
            if abs_score > abs(best_score):
                best_score = result.score

            # Check if confluence exceeds threshold
            if abs_score < entry_threshold:
                continue

            str_val = strength_order.get(result.strength, 0)
            if str_val < min_str_val:
                continue

            signals_found += 1
            direction = result.direction
            entry_idx = day_df["close"].iloc[i]

            # Dynamic premium based on confluence strength
            prem_boost = (abs_score - entry_threshold) / 100 * 10
            prem = base_premium + np.random.uniform(-5, 10) + prem_boost
            sl_prem = prem * (1 - premium_sl_pct / 100)
            qty = day_lots * lot_size

            # Dynamic target based on confluence strength
            if abs_score >= 70:
                target_mult = 1.5
            elif abs_score >= 50:
                target_mult = 1.2
            else:
                target_mult = 1.0
            tgt_prem = prem + (abs_score / 100 * 40) * target_mult

            exit_prem = prem
            exit_reason = "EOD"
            exit_time_val = day_df.index[-1]
            peak_prem = prem

            for k in range(i + 1, len(day_df)):
                fc = day_df["close"].iloc[k]
                if direction == "LONG":
                    idx_move = fc - entry_idx
                else:
                    idx_move = entry_idx - fc

                d = delta + np.random.uniform(-0.02, 0.02)
                sim_p = prem + idx_move * d

                if sim_p > peak_prem:
                    peak_prem = sim_p

                # Trailing stop
                prem_gain_pct = (peak_prem - prem) / prem * 100
                if prem_gain_pct >= trail_trigger_pct:
                    trail_floor = peak_prem * (1 - trail_pct / 100)
                    if sim_p <= trail_floor:
                        exit_prem = trail_floor
                        exit_reason = "TRAIL"
                        exit_time_val = day_df.index[k]
                        break

                # Fixed SL
                if sim_p <= sl_prem:
                    exit_prem = sl_prem
                    exit_reason = "SL"
                    exit_time_val = day_df.index[k]
                    break

                # Fixed target
                if sim_p >= tgt_prem:
                    exit_prem = tgt_prem
                    exit_reason = "TGT"
                    exit_time_val = day_df.index[k]
                    break

                # EOD
                if day_df.index[k].strftime("%H:%M") >= exit_time:
                    exit_prem = max(sim_p, 1)
                    exit_reason = "EOD"
                    exit_time_val = day_df.index[k]
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
                "dir": direction, "entry_time": day_df.index[i],
                "exit_time": exit_time_val,
                "entry_prem": round(prem, 2), "exit_prem": round(exit_prem, 2),
                "qty": qty, "lots": day_lots,
                "pnl": round(pnl, 0), "reason": exit_reason,
                "capital": round(capital, 0),
                "confluence": round(result.score, 1),
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
            "signals_found": signals_found,
        })

        if capital <= 0:
            break

    return _compute_stats(trades, equity_curve, starting_capital, capital, peak, unique_days)


def _compute_stats(trades, equity_curve, starting_capital, capital, peak, unique_days):
    total_pnl = capital - starting_capital
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
        max_dd = max(dd)

    active = [e for e in equity_curve if e["trades"] > 0]
    prof_days = sum(1 for e in equity_curve if e["daily_pnl"] > 0)
    loss_days = sum(1 for e in equity_curve if e["daily_pnl"] < 0)

    exit_reasons = {}
    for t in trades:
        er = t["reason"]
        if er not in exit_reasons:
            exit_reasons[er] = {"count": 0, "pnl": 0}
        exit_reasons[er]["count"] += 1
        exit_reasons[er]["pnl"] += t["pnl"]

    strength_stats = {}
    for t in trades:
        st = t.get("strength", "UNKNOWN")
        if st not in strength_stats:
            strength_stats[st] = {"w": 0, "l": 0, "pnl": 0}
        if t["pnl"] > 0:
            strength_stats[st]["w"] += 1
        else:
            strength_stats[st]["l"] += 1
        strength_stats[st]["pnl"] += t["pnl"]

    return {
        "capital": round(capital, 0),
        "pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / starting_capital * 100, 1),
        "trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1),
        "avg_win": round(np.mean([t["pnl"] for t in wins]), 0) if wins else 0,
        "avg_loss": round(np.mean([t["pnl"] for t in losses]), 0) if losses else 0,
        "profit_factor": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "prof_days": prof_days, "loss_days": loss_days,
        "trading_days": len(unique_days),
        "active_days": len(active),
        "equity_curve": equity_curve,
        "trade_list": trades,
        "exit_reasons": exit_reasons,
        "strength_stats": strength_stats,
    }


def optimize_v3(train_df: pd.DataFrame, starting_capital: float = 10000) -> dict:
    """Optimize confluence parameters on training data."""

    def score_result(r):
        if r["trades"] < 3:
            return -100
        pf_adj = min(r["profit_factor"], 4.0)
        wr_adj = min(r["win_rate"], 75)
        dd_pen = r["max_dd"] * 0.5
        return pf_adj * wr_adj - dd_pen + r["return_pct"] * 0.15

    total = 0

    # Phase 1: Entry threshold + minimum strength
    print("\n  Phase 1: Confluence threshold optimization...")
    p1 = []
    for thresh, min_str in itertools.product(
        [25, 30, 35, 40, 45, 50, 55, 60],
        ["WEAK", "MODERATE", "STRONG"]
    ):
        r = backtest_confluence(train_df, starting_capital=starting_capital,
                                entry_threshold=thresh, min_strength=min_str)
        s = score_result(r)
        p1.append({"params": {"entry_threshold": thresh, "min_strength": min_str},
                    "pf": r["profit_factor"], "wr": r["win_rate"],
                    "ret": r["return_pct"], "trades": r["trades"],
                    "dd": r["max_dd"], "score": s})
        total += 1
    p1.sort(key=lambda x: x["score"], reverse=True)
    best_thresh = p1[0]["params"]
    print(f"    Best: threshold={best_thresh['entry_threshold']}, "
          f"min_strength={best_thresh['min_strength']} "
          f"(PF={p1[0]['pf']}, WR={p1[0]['wr']}%, "
          f"{p1[0]['trades']} trades, score={p1[0]['score']:.1f})")

    # Phase 2: Indicator category weights
    print("  Phase 2: Category weight optimization...")
    p2 = []
    for tw, mw, vw in itertools.product(
        [0.8, 1.0, 1.3],
        [0.8, 1.0, 1.2, 1.5],
        [0.5, 0.8, 1.0]
    ):
        r = backtest_confluence(train_df, starting_capital=starting_capital,
                                **best_thresh,
                                trend_weight=tw, momentum_weight=mw,
                                volatility_weight=vw)
        s = score_result(r)
        p2.append({"params": {"trend_weight": tw, "momentum_weight": mw,
                               "volatility_weight": vw},
                    "pf": r["profit_factor"], "wr": r["win_rate"],
                    "ret": r["return_pct"], "trades": r["trades"],
                    "dd": r["max_dd"], "score": s})
        total += 1
    p2.sort(key=lambda x: x["score"], reverse=True)
    best_weights = p2[0]["params"]
    print(f"    Best: trend={best_weights['trend_weight']}, "
          f"momentum={best_weights['momentum_weight']}, "
          f"vol={best_weights['volatility_weight']} "
          f"(score={p2[0]['score']:.1f})")

    # Phase 3: Premium/SL optimization
    print("  Phase 3: Premium SL + trailing optimization...")
    p3 = []
    for sl_pct, trail_trig, trail_pct in itertools.product(
        [25, 30, 35, 40, 50],
        [30, 50, 70],
        [20, 30, 40]
    ):
        r = backtest_confluence(train_df, starting_capital=starting_capital,
                                **best_thresh, **best_weights,
                                premium_sl_pct=sl_pct,
                                trail_trigger_pct=trail_trig, trail_pct=trail_pct)
        s = score_result(r)
        p3.append({"params": {"premium_sl_pct": sl_pct,
                               "trail_trigger_pct": trail_trig, "trail_pct": trail_pct},
                    "pf": r["profit_factor"], "wr": r["win_rate"],
                    "ret": r["return_pct"], "trades": r["trades"],
                    "dd": r["max_dd"], "score": s})
        total += 1
    p3.sort(key=lambda x: x["score"], reverse=True)
    best_sl = p3[0]["params"]
    print(f"    Best: SL={best_sl['premium_sl_pct']}%, "
          f"trail_trigger={best_sl['trail_trigger_pct']}%, "
          f"trail={best_sl['trail_pct']}% "
          f"(score={p3[0]['score']:.1f})")

    # Phase 4: Risk management
    print("  Phase 4: Risk management optimization...")
    p4 = []
    for max_t, max_cl, dd_thresh in itertools.product(
        [2, 3, 4, 5],
        [2, 3],
        [20, 30, 40]
    ):
        r = backtest_confluence(train_df, starting_capital=starting_capital,
                                **best_thresh, **best_weights, **best_sl,
                                max_trades_day=max_t, max_consec_loss=max_cl,
                                dd_reduce_threshold=dd_thresh)
        s = score_result(r)
        p4.append({"params": {"max_trades_day": max_t, "max_consec_loss": max_cl,
                               "dd_reduce_threshold": dd_thresh},
                    "pf": r["profit_factor"], "wr": r["win_rate"],
                    "ret": r["return_pct"], "trades": r["trades"],
                    "dd": r["max_dd"], "score": s})
        total += 1
    p4.sort(key=lambda x: x["score"], reverse=True)
    best_risk = p4[0]["params"]
    print(f"    Best: max_trades={best_risk['max_trades_day']}, "
          f"max_consec_loss={best_risk['max_consec_loss']}, "
          f"dd_reduce={best_risk['dd_reduce_threshold']}% "
          f"(score={p4[0]['score']:.1f})")

    print(f"\n  Total combinations tested: {total}")

    all_params = {**best_thresh, **best_weights, **best_sl, **best_risk}
    final = backtest_confluence(train_df, starting_capital=starting_capital, **all_params)

    return {"best_params": all_params, "train_result": final}


def walk_forward_v3(df: pd.DataFrame, train_pct: float = 0.6,
                     starting_capital: float = 10000) -> dict:
    """Full walk-forward with multi-indicator confluence."""

    unique_days = sorted(set(df.index.date))
    split_idx = int(len(unique_days) * train_pct)
    train_days = unique_days[:split_idx]
    test_days = unique_days[split_idx:]

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD V3 -- MULTI-INDICATOR CONFLUENCE")
    print(f"  Using 40+ indicators: EMA, MACD, Bollinger, Stochastic,")
    print(f"  ADX, CCI, Ichimoku, SuperTrend, PSAR, OBV, MFI, etc.")
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
    opt = optimize_v3(train_df, starting_capital)
    best_params = opt["best_params"]
    train_result = opt["train_result"]

    print(f"\n{'='*70}")
    print(f"  TRAIN RESULTS ({train_days[0]} to {train_days[-1]})")
    print(f"{'='*70}")
    print_result_v3(train_result)

    print(f"\n  PHASE 2: OUT-OF-SAMPLE validation ({len(test_days)} days)...")
    test_result = backtest_confluence(test_df, starting_capital=starting_capital,
                                      **best_params)

    print(f"\n{'='*70}")
    print(f"  TEST RESULTS -- OUT-OF-SAMPLE ({test_days[0]} to {test_days[-1]})")
    print(f"{'='*70}")
    print_result_v3(test_result)

    # Stability analysis
    print(f"\n{'='*70}")
    print(f"  STABILITY ANALYSIS")
    print(f"{'='*70}")
    tr_pf = train_result["profit_factor"]
    te_pf = test_result["profit_factor"]
    tr_wr = train_result["win_rate"]
    te_wr = test_result["win_rate"]
    tr_dd = train_result["max_dd"]
    te_dd = test_result["max_dd"]

    pf_ratio = min(tr_pf, te_pf) / max(tr_pf, te_pf) if max(tr_pf, te_pf) > 0 else 0
    wr_diff = abs(tr_wr - te_wr)

    print(f"  {'Metric':<20} {'Train':>10} {'Test':>10} {'Stable?':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Profit Factor':<20} {tr_pf:>10.2f} {te_pf:>10.2f} "
          f"{'YES' if pf_ratio > 0.5 else 'NO':>10}")
    print(f"  {'Win Rate':<20} {tr_wr:>9.0f}% {te_wr:>9.0f}% "
          f"{'YES' if wr_diff < 15 else 'NO':>10}")
    print(f"  {'Max Drawdown':<20} {tr_dd:>9.0f}% {te_dd:>9.0f}% "
          f"{'OK' if te_dd < 50 else 'HIGH':>10}")
    print(f"  {'PF Stability':<20} {pf_ratio:>10.2f} {'':>10} "
          f"{'OK' if pf_ratio > 0.4 else 'WARN':>10}")

    # Verdict
    if te_pf > 1.3 and te_wr > 50 and te_dd < 40:
        verdict = "STRONG PASS"
        msg = "System shows robust edge. Ready for paper trading."
    elif te_pf > 1.15 and te_wr > 45 and te_dd < 50:
        verdict = "PASS"
        msg = "System shows consistent edge. Paper trade to confirm."
    elif te_pf > 1.0 and te_wr > 40:
        verdict = "MARGINAL"
        msg = "Slight edge. More iteration needed before going live."
    else:
        verdict = "FAIL"
        msg = "No edge on out-of-sample data."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {msg}")

    print(f"\n{'='*70}")
    print(f"  OPTIMIZED PARAMETERS")
    print(f"{'='*70}")
    for k, v in best_params.items():
        print(f"    {k:<30} = {v}")

    return {
        "best_params": best_params,
        "train": train_result,
        "test": test_result,
        "verdict": verdict,
        "stability": {"pf_ratio": pf_ratio, "wr_diff": wr_diff},
    }


def print_result_v3(r: dict, prefix: str = "  "):
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
    print(f"{prefix}Avg Win      : Rs {r['avg_win']:>8,.0f}")
    print(f"{prefix}Avg Loss     : Rs {r['avg_loss']:>8,.0f}")

    # Exit reasons
    exits = r.get("exit_reasons", {})
    if exits:
        print(f"{prefix}Exit reasons:")
        for er, data in sorted(exits.items()):
            avg = data["pnl"] / data["count"] if data["count"] else 0
            print(f"{prefix}  {er:10s}: {data['count']:>3d} exits, "
                  f"Rs {data['pnl']:>10,.0f} (avg Rs {avg:>7,.0f})")

    # Confluence strength breakdown
    ss = r.get("strength_stats", {})
    if ss:
        print(f"{prefix}Signal strength breakdown:")
        for name in ["WEAK", "MODERATE", "STRONG", "EXTREME"]:
            if name in ss:
                data = ss[name]
                tot = data["w"] + data["l"]
                wr = data["w"] / tot * 100 if tot else 0
                print(f"{prefix}  {name:10s}: {tot:>3d} trades, "
                      f"{wr:>4.0f}% win, Rs {data['pnl']:>10,.0f}")

    # Equity milestones
    ec = r.get("equity_curve", [])
    if ec and len(ec) > 3:
        print(f"{prefix}Equity curve:")
        n = len(ec)
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        for idx in sorted(set(min(i, n - 1) for i in indices)):
            e = ec[idx]
            print(f"{prefix}  Day {idx+1:>3d} ({e['date']}): "
                  f"Rs {e['capital']:>10,.0f}  "
                  f"[{e['trades']} trades, Rs {e['daily_pnl']:>+7,.0f}, "
                  f"best_conf={e.get('best_score', 0):>+.0f}]")


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    data_path = settings.DATA_DIR / "nifty_5m_real.csv"
    if not data_path.exists():
        print("ERROR: No real data. Run: python -m backtest.data_fetcher")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    days = len(set(df.index.date))
    print(f"Loaded {len(df)} real Nifty 5-min candles ({days} trading days)")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    result = walk_forward_v3(df, train_pct=0.6, starting_capital=10000)

    # Save
    if result["test"]["trade_list"]:
        pd.DataFrame(result["test"]["trade_list"]).to_csv(
            settings.DATA_DIR / "wf_v3_trades.csv", index=False)
    if result["test"]["equity_curve"]:
        pd.DataFrame(result["test"]["equity_curve"]).to_csv(
            settings.DATA_DIR / "wf_v3_equity.csv", index=False)

    return result


if __name__ == "__main__":
    main()
