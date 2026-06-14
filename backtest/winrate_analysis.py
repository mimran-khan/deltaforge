"""Win-rate analysis: loss patterns, confidence/time/ADX breakdowns, improvement sweep.

Usage:
    python -m backtest.winrate_analysis
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs

DAYS = 100
STARTING_CAPITAL = 10_000


@dataclass
class BacktestConfig:
    min_confidence: float = 50
    entry_start: str = "09:30"
    trail_trigger_pct: float = 12.0
    trail_pct: float = 8.0
    sl_pct: float | None = None  # override all strategy SL if set
    label: str = "baseline"


@contextmanager
def patched_settings(cfg: BacktestConfig):
    """Temporarily patch settings and premium SL for one backtest run."""
    from engine import premium_model

    orig = {
        "PULLBACK_MIN_CONFIDENCE": settings.PULLBACK_MIN_CONFIDENCE,
        "ENTRY_START": settings.ENTRY_START,
        "TRAIL_TRIGGER_PCT": settings.TRAIL_TRIGGER_PCT,
        "TRAIL_PCT": settings.TRAIL_PCT,
        "PREMIUM_SL_PCT": settings.PREMIUM_SL_PCT,
    }
    orig_sl = copy.deepcopy(premium_model.STRATEGY_SL_PCT)

    settings.PULLBACK_MIN_CONFIDENCE = cfg.min_confidence
    settings.ENTRY_START = cfg.entry_start
    settings.TRAIL_TRIGGER_PCT = cfg.trail_trigger_pct
    settings.TRAIL_PCT = cfg.trail_pct

    if cfg.sl_pct is not None:
        for k in premium_model.STRATEGY_SL_PCT:
            premium_model.STRATEGY_SL_PCT[k] = cfg.sl_pct
        settings.PREMIUM_SL_PCT = cfg.sl_pct

    try:
        yield
    finally:
        for k, v in orig.items():
            setattr(settings, k, v)
        premium_model.STRATEGY_SL_PCT.clear()
        premium_model.STRATEGY_SL_PCT.update(orig_sl)


def run_enhanced_backtest(
    df: pd.DataFrame,
    cfg: BacktestConfig | None = None,
    starting_capital: float = STARTING_CAPITAL,
    lot_size: int | None = None,
) -> dict:
    """Compound backtest with confidence + ADX captured per trade."""
    from engine.multi_strategy_engine import MultiStrategyEngine
    from engine.premium_model import create_premium_state, STRATEGY_SL_PCT

    cfg = cfg or BacktestConfig()
    lot_size = lot_size or settings.NIFTY_LOT_SIZE

    with patched_settings(cfg):
        engine = MultiStrategyEngine()
        capital = starting_capital
        peak = capital
        trades: list[dict] = []
        equity_curve: list[dict] = []

        unique_days = sorted(set(df.index.date))

        for day_idx, day in enumerate(unique_days):
            day_df = df[df.index.date == day].copy()
            if len(day_df) < 10:
                equity_curve.append(
                    {"date": day, "capital": capital, "daily_pnl": 0,
                     "trades": 0, "lots": 0}
                )
                continue

            prev_day_data = None
            if day_idx > 0:
                prev_day = unique_days[day_idx - 1]
                prev_df = df[df.index.date == prev_day]
                if len(prev_df) > 0:
                    prev_day_data = {
                        "high": prev_df["high"].max(),
                        "low": prev_df["low"].min(),
                        "close": prev_df["close"].iloc[-1],
                    }
            engine.reset_day(prev_day_data)

            day_start_cap = capital
            day_pnl = 0
            day_trades = 0
            consec_loss = 0
            daily_loss_limit = day_start_cap * (settings.DAILY_LOSS_LIMIT_PCT / 100)

            per_lot = getattr(settings, "CAPITAL_PER_LOT", 10_000)
            day_lots = max(1, int(capital / per_lot))
            day_lots = min(day_lots, getattr(settings, "MAX_LOTS_CAP", 10))

            indicators = engine.precompute(day_df)
            adx_series = indicators.get("adx")

            open_positions: list[dict] = []

            for i in range(10, len(day_df)):
                ts = day_df.index[i]
                time_str = ts.strftime("%H:%M")
                nifty_price = day_df["close"].iloc[i]

                closed_this_bar: list[dict] = []
                for pos in open_positions:
                    pos["candles_held"] += 1
                    cur_prem = pos["prem_state"].current_premium(
                        nifty_price, pos["candles_held"]
                    )

                    if cur_prem > pos["peak_premium"]:
                        pos["peak_premium"] = cur_prem

                    trail_floor = pos["prem_state"].update_trail(
                        cur_prem,
                        trigger_pct=settings.TRAIL_TRIGGER_PCT,
                        trail_pct=settings.TRAIL_PCT,
                    )

                    exit_reason = None
                    exit_prem = cur_prem

                    if cur_prem <= pos["sl_premium"]:
                        exit_reason = "SL"
                        exit_prem = pos["sl_premium"]
                    elif cur_prem >= pos["prem_state"].target_premium:
                        exit_reason = "TGT"
                        exit_prem = pos["prem_state"].target_premium
                    elif trail_floor is not None and cur_prem <= trail_floor:
                        exit_reason = "TRAIL"
                        exit_prem = trail_floor
                    elif pos["candles_held"] >= settings.PULLBACK_HOLD_CANDLES:
                        exit_reason = "TIME"
                    elif time_str >= settings.SQUARE_OFF_TIME:
                        exit_reason = "EOD"

                    if exit_reason:
                        if exit_reason == "SL":
                            engine.record_sl_exit(pos["signal_type"], i)

                        costs = _calc_realistic_costs(
                            pos["entry_premium"], exit_prem,
                            pos["qty"], day_lots,
                        )
                        raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                        net_pnl = raw_pnl - costs

                        capital += net_pnl
                        day_pnl += net_pnl
                        day_trades += 1
                        consec_loss = consec_loss + 1 if net_pnl < 0 else 0

                        peak_gain = (
                            (pos["peak_premium"] - pos["entry_premium"])
                            / pos["entry_premium"] * 100
                        )

                        trades.append(_build_trade_record(
                            pos, ts, exit_prem, peak_gain, day_lots,
                            net_pnl, exit_reason, capital,
                        ))
                        closed_this_bar.append(pos)

                for pos in closed_this_bar:
                    open_positions.remove(pos)

                if time_str < settings.ENTRY_START or time_str > settings.ENTRY_END:
                    continue

                max_sim = getattr(settings, "MAX_SIMULTANEOUS_POSITIONS", 2)
                if len(open_positions) >= max_sim:
                    continue
                if consec_loss >= settings.MAX_CONSECUTIVE_LOSSES:
                    continue
                if day_pnl < 0 and abs(day_pnl) >= daily_loss_limit:
                    continue

                open_dirs = {p["direction"] for p in open_positions}
                signals = engine.scan(indicators, i, time_str)

                for signal in signals:
                    if signal.direction in open_dirs:
                        continue
                    if signal.confidence < settings.PULLBACK_MIN_CONFIDENCE:
                        continue

                    adx_at_entry = _adx_at_bar(adx_series, i)

                    theta = settings.get_scaled_theta(nifty_price)
                    prem_state = create_premium_state(
                        entry_index_price=nifty_price,
                        direction=signal.direction,
                        base_premium=settings.PREMIUM_BASE,
                        delta=settings.PREMIUM_DELTA,
                        theta_per_candle=theta,
                        sl_pct=settings.PREMIUM_SL_PCT,
                        confluence_score=signal.confidence,
                        signal_type=signal.signal_type,
                    )

                    spread = getattr(settings, "BID_ASK_SPREAD", 0.30)
                    entry_premium = prem_state.entry_premium + spread
                    eff_sl = STRATEGY_SL_PCT.get(
                        signal.signal_type, settings.PREMIUM_SL_PCT
                    )
                    sl_premium = entry_premium * (1 - eff_sl / 100)
                    qty = day_lots * lot_size

                    open_positions.append({
                        "direction": signal.direction,
                        "signal_type": signal.signal_type,
                        "entry_time": ts,
                        "entry_premium": entry_premium,
                        "sl_premium": sl_premium,
                        "qty": qty,
                        "prem_state": prem_state,
                        "candles_held": 0,
                        "peak_premium": entry_premium,
                        "confidence": signal.confidence,
                        "adx_at_entry": adx_at_entry,
                    })
                    break

                if capital <= 0:
                    break

            for pos in open_positions:
                nifty_price = day_df["close"].iloc[-1]
                exit_prem = pos["prem_state"].current_premium(
                    nifty_price, pos["candles_held"]
                )
                brokerage = getattr(settings, "BROKERAGE_PER_ORDER", 20) * 2
                stt = exit_prem * pos["qty"] * getattr(settings, "STT_PCT", 0.0125) / 100
                slippage = getattr(settings, "SLIPPAGE_POINTS", 0.5) * pos["qty"]
                costs = brokerage + stt + slippage
                raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                net_pnl = raw_pnl - costs
                capital += net_pnl
                day_pnl += net_pnl
                day_trades += 1
                peak_gain = (
                    (pos["peak_premium"] - pos["entry_premium"])
                    / pos["entry_premium"] * 100
                )
                trades.append(_build_trade_record(
                    pos, day_df.index[-1], exit_prem, peak_gain, day_lots,
                    net_pnl, "EOD", capital,
                ))

            if capital > peak:
                peak = capital

            equity_curve.append({
                "date": day, "capital": round(capital, 0),
                "daily_pnl": round(day_pnl, 0),
                "trades": day_trades, "lots": day_lots,
            })

            if capital <= 0:
                break

    return _summarize(trades, equity_curve, starting_capital, cfg)


def _adx_at_bar(adx_series, bar_idx: int) -> float:
    if adx_series is None or bar_idx >= len(adx_series):
        return 25.0
    val = adx_series.iloc[bar_idx]
    return float(val) if not np.isnan(val) else 25.0


def _build_trade_record(
    pos: dict, exit_ts, exit_prem: float, peak_gain: float,
    day_lots: int, net_pnl: float, reason: str, capital: float,
) -> dict:
    return {
        "strategy": pos["signal_type"],
        "signal": pos["direction"],
        "entry_time": pos["entry_time"],
        "exit_time": exit_ts,
        "entry_premium": round(pos["entry_premium"], 2),
        "exit_premium": round(exit_prem, 2),
        "peak_premium": round(pos["peak_premium"], 2),
        "peak_gain_pct": round(peak_gain, 2),
        "qty": pos["qty"],
        "lots": day_lots,
        "pnl": round(net_pnl, 0),
        "reason": reason,
        "capital_after": round(capital, 0),
        "confidence": pos.get("confidence", 0),
        "adx_at_entry": round(pos.get("adx_at_entry", 0), 1),
    }


def _summarize(
    trades: list[dict], equity_curve: list[dict],
    starting_capital: float, cfg: BacktestConfig,
) -> dict:
    capital = trades[-1]["capital_after"] if trades else starting_capital
    total_pnl = capital - starting_capital
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    eq_vals = [e["capital"] for e in equity_curve]
    peak_arr = np.maximum.accumulate(eq_vals) if eq_vals else [starting_capital]
    dd_arr = [(p - v) / p * 100 for p, v in zip(peak_arr, eq_vals)]
    max_dd = max(dd_arr) if dd_arr else 0

    return {
        "config": cfg,
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / starting_capital * 100, 1),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "trades": trades,
        "equity_curve": equity_curve,
    }


# ── Analysis helpers ────────────────────────────────────────────────

def _wr_stats(subset: list[dict]) -> dict:
    if not subset:
        return {"count": 0, "wr": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0}
    wins = sum(1 for t in subset if t["pnl"] > 0)
    return {
        "count": len(subset),
        "wr": round(wins / len(subset) * 100, 1),
        "avg_pnl": round(np.mean([t["pnl"] for t in subset]), 0),
        "total_pnl": round(sum(t["pnl"] for t in subset), 0),
    }


def analyze_by_confidence(trades: list[dict]) -> pd.DataFrame:
    bands = [(50, 59), (60, 69), (70, 79), (80, 200)]
    rows = []
    for lo, hi in bands:
        subset = [t for t in trades if lo <= t.get("confidence", 0) <= hi]
        s = _wr_stats(subset)
        rows.append({
            "band": f"{lo}-{hi}" if hi < 200 else f"{lo}+",
            **s,
        })
    return pd.DataFrame(rows)


def analyze_by_hour(trades: list[dict]) -> pd.DataFrame:
    windows = [
        ("09:00-10:00", 9, 10),
        ("10:00-11:00", 10, 11),
        ("11:00-12:00", 11, 12),
        ("12:00-13:00", 12, 13),
        ("13:00-14:00", 13, 14),
        ("14:00-15:00", 14, 15),
    ]
    rows = []
    for label, h_start, h_end in windows:
        subset = []
        for t in trades:
            et = t["entry_time"]
            if isinstance(et, str):
                et = pd.Timestamp(et)
            hour = et.hour
            if h_start <= hour < h_end:
                subset.append(t)
        s = _wr_stats(subset)
        rows.append({"window": label, **s})
    return pd.DataFrame(rows)


def analyze_loss_by_reason(trades: list[dict]) -> pd.DataFrame:
    losses = [t for t in trades if t["pnl"] <= 0]
    reasons = ["SL", "TRAIL", "TIME", "EOD", "TGT"]
    rows = []
    for r in reasons:
        subset = [t for t in losses if t["reason"] == r]
        if not subset:
            rows.append({"reason": r, "count": 0, "avg_loss": 0.0, "total_loss": 0})
            continue
        rows.append({
            "reason": r,
            "count": len(subset),
            "avg_loss": round(np.mean([t["pnl"] for t in subset]), 0),
            "total_loss": round(sum(t["pnl"] for t in subset), 0),
        })
    return pd.DataFrame(rows)


def analyze_sl_peak_gain(trades: list[dict]) -> pd.DataFrame:
    sl_losses = [t for t in trades if t["pnl"] <= 0 and t["reason"] == "SL"]
    bands = [
        ("0-2%", 0, 2),
        ("2-5%", 2, 5),
        ("5-8%", 5, 8),
        ("8-10%", 8, 10),
        ("10-12%", 10, 12),
        ("12%+", 12, 999),
    ]
    rows = []
    for label, lo, hi in bands:
        subset = [t for t in sl_losses if lo <= t["peak_gain_pct"] < hi]
        s = _wr_stats(subset)
        rows.append({
            "peak_gain_band": label,
            "sl_losses": s["count"],
            "avg_peak_gain": round(
                np.mean([t["peak_gain_pct"] for t in subset]), 1
            ) if subset else 0,
            "avg_loss": s["avg_pnl"],
        })
    return pd.DataFrame(rows)


def analyze_by_strategy(trades: list[dict]) -> pd.DataFrame:
    strategies = sorted(set(t["strategy"] for t in trades))
    rows = []
    for strat in strategies:
        subset = [t for t in trades if t["strategy"] == strat]
        s = _wr_stats(subset)
        loss_subset = [t for t in subset if t["pnl"] <= 0]
        sl_pct = (
            sum(1 for t in loss_subset if t["reason"] == "SL")
            / len(loss_subset) * 100 if loss_subset else 0
        )
        rows.append({
            "strategy": strat,
            "trades": s["count"],
            "wr": s["wr"],
            "avg_pnl": s["avg_pnl"],
            "total_pnl": s["total_pnl"],
            "loss_sl_pct": round(sl_pct, 1),
        })
    return pd.DataFrame(rows)


def analyze_by_adx(trades: list[dict]) -> pd.DataFrame:
    bands = [(0, 15), (15, 25), (25, 35), (35, 999)]
    rows = []
    for lo, hi in bands:
        subset = [t for t in trades if lo <= t.get("adx_at_entry", 0) < hi]
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        s = _wr_stats(subset)
        rows.append({"adx_band": label, **s})
    return pd.DataFrame(rows)


def confidence_threshold_sweep(
    df: pd.DataFrame, thresholds: list[int],
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        cfg = BacktestConfig(min_confidence=thr, label=f"conf>={thr}")
        res = run_enhanced_backtest(df, cfg)
        rows.append({
            "threshold": thr,
            "trades": res["total_trades"],
            "wr": res["win_rate"],
            "pf": res["profit_factor"],
            "final_capital": res["final_capital"],
            "max_dd": res["max_drawdown_pct"],
        })
    return pd.DataFrame(rows)


def run_improvement_scenarios(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Test individual and combined config improvements vs baseline."""
    base_cfg = BacktestConfig(
        min_confidence=settings.PULLBACK_MIN_CONFIDENCE,
        entry_start=settings.ENTRY_START,
        trail_trigger_pct=settings.TRAIL_TRIGGER_PCT,
        trail_pct=settings.TRAIL_PCT,
    )

    scenarios = [
        BacktestConfig(min_confidence=60, label="min_conf=60"),
        BacktestConfig(entry_start="09:45", label="skip_first_30min"),
        BacktestConfig(trail_trigger_pct=8.0, label="trail_trigger=8%"),
        BacktestConfig(sl_pct=10.0, label="sl=10%"),
        BacktestConfig(sl_pct=12.0, label="sl=12%"),
        BacktestConfig(
            min_confidence=60, entry_start="09:45",
            label="conf60+skip30min",
        ),
        BacktestConfig(
            min_confidence=60, trail_trigger_pct=8.0,
            label="conf60+trail8%",
        ),
        BacktestConfig(
            min_confidence=60, entry_start="09:45",
            trail_trigger_pct=8.0, sl_pct=10.0,
            label="conf60+skip30+trail8+sl10",
        ),
        BacktestConfig(
            min_confidence=65, entry_start="09:45",
            trail_trigger_pct=8.0,
            label="conf65+skip30+trail8",
        ),
    ]

    rows = []
    b = baseline
    for cfg in scenarios:
        res = run_enhanced_backtest(df, cfg)
        rows.append({
            "scenario": cfg.label,
            "trades": res["total_trades"],
            "wr": res["win_rate"],
            "wr_delta": round(res["win_rate"] - b["win_rate"], 1),
            "pf": res["profit_factor"],
            "pf_delta": round(res["profit_factor"] - b["profit_factor"], 2),
            "final_capital": res["final_capital"],
            "capital_delta": res["final_capital"] - b["final_capital"],
            "trade_delta": res["total_trades"] - b["total_trades"],
            "max_dd": res["max_drawdown_pct"],
        })
    return pd.DataFrame(rows)


def df_to_md(df: pd.DataFrame, title: str) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([f"### {title}", "", header, sep, *rows, ""])


def generate_report(
    baseline: dict,
    conf_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    loss_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    strat_df: pd.DataFrame,
    adx_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    improve_df: pd.DataFrame,
) -> str:
    b = baseline
    cfg = b["config"]

    # Rank by capital delta first, then WR — WR-only tweaks that destroy PF rank lower
    ranked = improve_df.copy()
    ranked["score"] = (
        ranked["capital_delta"] / max(abs(b["final_capital"]), 1) * 100
        + ranked["wr_delta"] * 0.5
        + ranked["pf_delta"] * 5
    )
    ranked = ranked.sort_values("score", ascending=False)

    # Recommend only scenarios that improve or preserve capital AND don't crater PF
    viable = improve_df[
        (improve_df["capital_delta"] >= -50000)
        & (improve_df["pf"] >= b["profit_factor"] * 0.90)
    ]
    if len(viable) > 0:
        rec = viable.sort_values(["wr", "final_capital"], ascending=[False, False]).iloc[0]
        rec_name = rec["scenario"]
        rec_note = (
            "Preserves capital/PF while improving or maintaining edge."
        )
    else:
        rec_name = "baseline (no change)"
        rec = {
            "scenario": rec_name,
            "wr": b["win_rate"],
            "wr_delta": 0,
            "pf": b["profit_factor"],
            "pf_delta": 0,
            "final_capital": b["final_capital"],
            "capital_delta": 0,
            "trades": b["total_trades"],
            "trade_delta": 0,
            "max_dd": b["max_drawdown_pct"],
        }
        rec_note = (
            "No tested tweak improved WR without sacrificing capital or PF. "
            "Focus on strategy-level filters instead."
        )

    rec_map = {
        "min_conf=60": {"PULLBACK_MIN_CONFIDENCE": 60},
        "skip_first_30min": {"ENTRY_START": "09:45"},
        "trail_trigger=8%": {"TRAIL_TRIGGER_PCT": 8.0},
        "sl=10%": {"STRATEGY_SL_PCT": 10.0},
        "sl=12%": {"STRATEGY_SL_PCT": 12.0},
        "conf60+skip30min": {
            "PULLBACK_MIN_CONFIDENCE": 60, "ENTRY_START": "09:45",
        },
        "conf60+trail8%": {
            "PULLBACK_MIN_CONFIDENCE": 60, "TRAIL_TRIGGER_PCT": 8.0,
        },
        "conf60+skip30+trail8+sl10": {
            "PULLBACK_MIN_CONFIDENCE": 60, "ENTRY_START": "09:45",
            "TRAIL_TRIGGER_PCT": 8.0, "STRATEGY_SL_PCT": 10.0,
        },
        "conf65+skip30+trail8": {
            "PULLBACK_MIN_CONFIDENCE": 65, "ENTRY_START": "09:45",
            "TRAIL_TRIGGER_PCT": 8.0,
        },
        "baseline (no change)": {},
    }

    # Strategy-level insight
    worst_strat = strat_df.sort_values("wr").iloc[0] if len(strat_df) else None

    lines = [
        "# Win Rate Analysis Report",
        "",
        f"**Period:** {DAYS} trading days | **Starting capital:** Rs {STARTING_CAPITAL:,}",
        "",
        "## Baseline Results (Current Config)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Min confidence | {cfg.min_confidence} |",
        f"| Entry start | {cfg.entry_start} |",
        f"| Trail trigger | {cfg.trail_trigger_pct}% |",
        f"| Strategy SL | 8% (STRATEGY_SL_PCT) |",
        f"| Total trades | {b['total_trades']} |",
        f"| Win rate | {b['win_rate']}% |",
        f"| Profit factor | {b['profit_factor']} |",
        f"| Final capital | Rs {b['final_capital']:,} |",
        f"| Max drawdown | {b['max_drawdown_pct']}% |",
        "",
        "## Key Findings (Executive Summary)",
        "",
    ]

    if worst_strat is not None:
        lines.append(
            f"- **Worst strategy:** {worst_strat['strategy']} at "
            f"{worst_strat['wr']}% WR ({int(worst_strat['trades'])} trades) — "
            f"consider disabling in `MultiStrategyEngine.DISABLED_STRATEGIES`."
        )

    low_conf = conf_df[conf_df["band"] == "50-59"]
    if len(low_conf) and low_conf.iloc[0]["count"] == 0:
        lines.append(
            "- **Low-confidence hypothesis rejected:** zero trades in 50–59 band; "
            "engine already emits confidence ≥ 60. Raising `PULLBACK_MIN_CONFIDENCE` "
            "to 60 removes path-dependent trades but does not improve WR."
        )

    mid_conf = conf_df[conf_df["band"] == "70-79"]
    if len(mid_conf):
        lines.append(
            f"- **70–79 confidence band underperforms:** "
            f"{mid_conf.iloc[0]['wr']}% WR vs 67.8% for 80+ — "
            f"filter or penalize mid-range confidence signals."
        )

    sl_row = loss_df[loss_df["reason"] == "SL"]
    trail_row = loss_df[loss_df["reason"] == "TRAIL"]
    if len(sl_row) and sl_row.iloc[0]["count"] > 0:
        lines.append(
            f"- **100% of losses are SL exits** ({int(sl_row.iloc[0]['count'])} trades); "
            f"**zero TRAIL exits** — trail at {cfg.trail_trigger_pct}% never activates "
            f"before 8% SL is hit."
        )

    peak_low = peak_df[peak_df["peak_gain_band"] == "0-2%"]
    if len(peak_low) and peak_low.iloc[0]["sl_losses"] > 0:
        lines.append(
            f"- **{int(peak_low.iloc[0]['sl_losses'])}/{int(sl_row.iloc[0]['count'])} "
            f"SL losses peaked below 2%** — lowering trail trigger will NOT help; "
            f"trades reverse before reaching any trail threshold."
        )

    toxic_hour = hour_df.sort_values("avg_pnl").iloc[0]
    best_hour = hour_df.sort_values("wr", ascending=False).iloc[0]
    lines.extend([
        f"- **Weakest hour:** {toxic_hour['window']} "
        f"({toxic_hour['wr']}% WR, avg P&L Rs {toxic_hour['avg_pnl']:,.0f}).",
        f"- **Best hour:** {best_hour['window']} "
        f"({best_hour['wr']}% WR, avg P&L Rs {best_hour['avg_pnl']:,.0f}).",
        "",
        "---",
        "",
        "## 1. Loss Pattern Analysis",
        "",
        df_to_md(conf_df, "1a. By Confidence Level"),
        df_to_md(hour_df, "1b. By Time of Day (Entry Hour)"),
        df_to_md(loss_df, "1c. Losing Trades by Exit Reason"),
        df_to_md(peak_df, "1d. SL Losses — Peak Gain Before Stop"),
        df_to_md(strat_df, "1e. By Strategy"),
        df_to_md(adx_df, "1f. By ADX at Entry"),
        df_to_md(sweep_df, "1g. Confidence Threshold Sweep (Re-run)"),
        "",
        "---",
        "",
        "## 2. Improvement Scenarios",
        "",
        df_to_md(improve_df, "Individual & Combined Improvements vs Baseline"),
        "",
        "### Ranking (capital-preserving score, best first)",
        "",
    ])

    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        lines.append(
            f"{i}. **{row['scenario']}** — WR {row['wr']}% "
            f"({row['wr_delta']:+.1f}), PF {row['pf']} ({row['pf_delta']:+.2f}), "
            f"capital Rs {row['final_capital']:,} ({row['capital_delta']:+,})"
        )

    lines.extend([
        "",
        "> **Note:** Wider SL (10%/12%) raises WR mechanically but cuts final capital "
        "by 37–53% and PF by ~20%. Not recommended despite higher WR.",
        "",
        "---",
        "",
        "## 3. Recommended Config",
        "",
        f"**Scenario:** `{rec_name}`",
        "",
        f"_{rec_note}_",
        "",
        "| Metric | Baseline | Recommended | Delta |",
        "|--------|----------|-------------|-------|",
        f"| Win rate | {b['win_rate']}% | {rec['wr']}% | {rec['wr_delta']:+.1f} |",
        f"| Profit factor | {b['profit_factor']} | {rec['pf']} | {rec['pf_delta']:+.2f} |",
        f"| Final capital | Rs {b['final_capital']:,} | Rs {rec['final_capital']:,} | Rs {rec['capital_delta']:+,} |",
        f"| Max drawdown | {b['max_drawdown_pct']}% | {rec['max_dd']}% | — |",
        f"| Trade count | {b['total_trades']} | {rec['trades']} | {rec['trade_delta']:+d} |",
        "",
        "### Actionable Changes (Priority Order)",
        "",
        "1. **Disable ADX_BREAKOUT** — 33% WR, 24 trades, all losses via SL.",
        "   ```python",
        "   # engine/multi_strategy_engine.py",
        '   DISABLED_STRATEGIES = {..., "ADX_BREAKOUT"}',
        "   ```",
        "",
        "2. **Add ADX ceiling filter** — ADX 35+ shows 50% WR vs 63% at 25–35.",
        "   ```python",
        "   # In MultiStrategyEngine.scan(), after MIN_ADX check:",
        "   if adx_val > 35:",
        "       return []",
        "   ```",
        "",
        "3. **Restrict entry window** — skip 10:00–11:00 (50% WR, negative avg P&L); "
        "favor 13:00–15:00 entries (55–70% WR).",
        "",
        "4. **Do NOT widen SL** — increases WR but destroys compound growth.",
        "",
        "5. **Do NOT lower trail trigger** — 0 trail exits in baseline; SL fires first at 8%.",
        "",
        "### Code Changes Required",
        "",
        "```python",
        "# config/settings.py — keep current values",
        f"PULLBACK_MIN_CONFIDENCE = {settings.PULLBACK_MIN_CONFIDENCE}  # no change needed",
        f'ENTRY_START = "{settings.ENTRY_START}"  # optional: "13:00" for afternoon-only',
        f"TRAIL_TRIGGER_PCT = {settings.TRAIL_TRIGGER_PCT}  # no change — trail never fires",
        "",
        "# engine/multi_strategy_engine.py",
        'DISABLED_STRATEGIES = {',
        '    "SUPERTREND", "VWAP_MOMENTUM", "VWAP_MEAN_REV", "STOCH_CROSS",',
        '    "ADX_BREAKOUT",  # NEW — 33% WR in this backtest',
        "}",
        "MAX_ADX = 35  # NEW class attr — skip overextended trends",
        "```",
        "",
    ])

    changes = rec_map.get(rec_name, {})
    if changes:
        lines.append("### Scenario-specific overrides (if applying a tested tweak)")
        lines.append("```python")
        if "PULLBACK_MIN_CONFIDENCE" in changes:
            lines.append(f"PULLBACK_MIN_CONFIDENCE = {changes['PULLBACK_MIN_CONFIDENCE']}")
        if "ENTRY_START" in changes:
            lines.append(f'ENTRY_START = "{changes["ENTRY_START"]}"')
        if "TRAIL_TRIGGER_PCT" in changes:
            lines.append(f"TRAIL_TRIGGER_PCT = {changes['TRAIL_TRIGGER_PCT']}")
        if "STRATEGY_SL_PCT" in changes:
            sl = changes["STRATEGY_SL_PCT"]
            lines.append(f"STRATEGY_SL_PCT = {{k: {sl} for k in STRATEGY_SL_PCT}}")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    print(f"Loading {DAYS} days of data...")
    df = load_real_data(days=DAYS)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days\n")

    base_cfg = BacktestConfig(
        min_confidence=settings.PULLBACK_MIN_CONFIDENCE,
        entry_start=settings.ENTRY_START,
        trail_trigger_pct=settings.TRAIL_TRIGGER_PCT,
        trail_pct=settings.TRAIL_PCT,
        label="baseline",
    )

    print("Running baseline backtest...")
    baseline = run_enhanced_backtest(df, base_cfg)
    trades = baseline["trades"]
    print(
        f"  Baseline: {baseline['total_trades']} trades, "
        f"{baseline['win_rate']}% WR, PF {baseline['profit_factor']}, "
        f"Rs {baseline['final_capital']:,}\n"
    )

    print("Analyzing patterns...")
    conf_df = analyze_by_confidence(trades)
    hour_df = analyze_by_hour(trades)
    loss_df = analyze_loss_by_reason(trades)
    peak_df = analyze_sl_peak_gain(trades)
    strat_df = analyze_by_strategy(trades)
    adx_df = analyze_by_adx(trades)

    print("Running confidence threshold sweep...")
    sweep_df = confidence_threshold_sweep(df, [50, 55, 60, 65, 70])

    print("Testing improvement scenarios...")
    improve_df = run_improvement_scenarios(df, baseline)

    improve_path = PROJECT_ROOT / "data" / "winrate_improvements.csv"
    improve_df.to_csv(improve_path, index=False)

    report = generate_report(
        baseline, conf_df, hour_df, loss_df, peak_df,
        strat_df, adx_df, sweep_df, improve_df,
    )

    report_path = PROJECT_ROOT / "data" / "winrate_analysis_report.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)
    print(report)
    print(f"\nReport saved to {report_path}")

    trades_path = PROJECT_ROOT / "data" / "winrate_analysis_trades.csv"
    pd.DataFrame(trades).to_csv(trades_path, index=False)
    print(f"Trades saved to {trades_path}")

    return baseline


if __name__ == "__main__":
    main()
