"""CDGR leak analysis — daily return distribution, drag factors, parameter sweep.

Usage:
    python -m backtest.cdgr_leak_analysis
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, run_compound_backtest
from engine.multi_strategy_engine import MultiStrategyEngine

EQUITY_CSV = PROJECT_ROOT / "data" / "equity_curve.csv"
DAYS = 100
STARTING_CAPITAL = 10_000
LOT_SIZE = settings.NIFTY_LOT_SIZE
DEPLOY_PCT = getattr(settings, "CAPITAL_DEPLOY_PCT", 80.0)
MAX_LOTS_CAP = settings.MAX_LOTS_CAP

# Baseline values for reset
BASELINE = {
    "max_total": MultiStrategyEngine.MAX_TOTAL_PER_DAY,
    "cooldown": MultiStrategyEngine.COOLDOWN_BARS,
    "max_sim": settings.MAX_SIMULTANEOUS_POSITIONS,
}

BUCKETS = [
    ("big_win", ">15%", lambda r: r > 15),
    ("good_win", "7-15%", lambda r: 7 <= r <= 15),
    ("small_win", "0-7%", lambda r: 0 < r < 7),
    ("small_loss", "0 to -7%", lambda r: -7 <= r < 0),
    ("big_loss", "<-7%", lambda r: r < -7),
]


def daily_return_pct(row: pd.Series) -> float:
    start_cap = row["capital"] - row["daily_pnl"]
    if start_cap <= 0:
        return 0.0
    return row["daily_pnl"] / start_cap * 100


def categorize_return(ret: float) -> str:
    for name, _, pred in BUCKETS:
        if pred(ret):
            return name
    if ret == 0:
        return "flat"
    return "other"


@dataclass
class ScenarioResult:
    label: str
    max_total: int
    cooldown: int
    max_sim: int
    trades: int
    win_rate: float
    profit_factor: float
    final_capital: float
    max_dd: float
    cdgr: float
    active_days: int
    equity_curve: list


@contextmanager
def patched_scenario(*, max_total: int | None = None, cooldown: int | None = None, max_sim: int | None = None):
    orig_max_total = MultiStrategyEngine.MAX_TOTAL_PER_DAY
    orig_cooldown = MultiStrategyEngine.COOLDOWN_BARS
    orig_max_sim = settings.MAX_SIMULTANEOUS_POSITIONS
    try:
        if max_total is not None:
            MultiStrategyEngine.MAX_TOTAL_PER_DAY = max_total
        if cooldown is not None:
            MultiStrategyEngine.COOLDOWN_BARS = cooldown
        if max_sim is not None:
            settings.MAX_SIMULTANEOUS_POSITIONS = max_sim
        yield
    finally:
        MultiStrategyEngine.MAX_TOTAL_PER_DAY = orig_max_total
        MultiStrategyEngine.COOLDOWN_BARS = orig_cooldown
        settings.MAX_SIMULTANEOUS_POSITIONS = orig_max_sim


def calc_cdgr(final: float, initial: float, active_days: int) -> float:
    if active_days <= 0 or initial <= 0 or final <= 0:
        return 0.0
    return (final / initial) ** (1 / active_days) - 1


def run_scenario(df: pd.DataFrame, label: str, max_total: int, cooldown: int, max_sim: int) -> ScenarioResult:
    with patched_scenario(max_total=max_total, cooldown=cooldown, max_sim=max_sim):
        result = run_compound_backtest(
            df,
            starting_capital=STARTING_CAPITAL,
            lot_size=LOT_SIZE,
            deploy_pct=DEPLOY_PCT,
        )
    active = result["active_trading_days"]
    cdgr = calc_cdgr(result["final_capital"], STARTING_CAPITAL, active)
    return ScenarioResult(
        label=label,
        max_total=max_total,
        cooldown=cooldown,
        max_sim=max_sim,
        trades=result["total_trades"],
        win_rate=result["win_rate"],
        profit_factor=result["profit_factor"],
        final_capital=result["final_capital"],
        max_dd=result["max_drawdown_pct"],
        cdgr=cdgr,
        active_days=active,
        equity_curve=result["equity_curve"],
    )


def risk_metrics(equity_curve: list) -> dict:
    rows = []
    for e in equity_curve:
        start_cap = e["capital"] - e["daily_pnl"]
        ret_pct = (e["daily_pnl"] / start_cap * 100) if start_cap > 0 else 0.0
        rows.append({"daily_pnl": e["daily_pnl"], "ret_pct": ret_pct, "trades": e["trades"]})

    worst_pnl = min(r["daily_pnl"] for r in rows)
    worst_idx = next(i for i, r in enumerate(rows) if r["daily_pnl"] == worst_pnl)
    worst_start = rows[worst_idx]["daily_pnl"]
    # recompute worst pct from equity curve entry
    e = equity_curve[worst_idx]
    start_cap = e["capital"] - e["daily_pnl"]
    worst_pct = e["daily_pnl"] / start_cap * 100 if start_cap > 0 else 0.0

    # consecutive losing streak (days with daily_pnl < 0)
    max_streak = 0
    streak = 0
    for r in rows:
        if r["daily_pnl"] < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "worst_day_pnl": worst_pnl,
        "worst_day_pct": worst_pct,
        "max_losing_streak": max_streak,
    }


def part1_distribution(ec: pd.DataFrame) -> pd.DataFrame:
    active = ec[ec["trades"] > 0].copy()
    active["daily_ret_pct"] = active.apply(daily_return_pct, axis=1)
    active["bucket"] = active["daily_ret_pct"].apply(categorize_return)

    total_pnl = ec["daily_pnl"].sum()
    rows = []
    for name, label, _ in BUCKETS:
        sub = active[active["bucket"] == name]
        pnl_share = sub["daily_pnl"].sum() / total_pnl * 100 if total_pnl else 0.0
        rows.append({
            "bucket": label,
            "days": len(sub),
            "avg_return_pct": sub["daily_ret_pct"].mean() if len(sub) else 0.0,
            "pnl_share_pct": pnl_share,
            "total_pnl": sub["daily_pnl"].sum(),
        })
    return pd.DataFrame(rows), active


def part2_drag(ec: pd.DataFrame, active: pd.DataFrame) -> dict:
    ec = ec.copy()
    ec["daily_ret_pct"] = ec.apply(
        lambda r: daily_return_pct(r) if r["trades"] > 0 else 0.0, axis=1
    )

    no_trade = ec[ec["trades"] == 0]
    loss_days = ec[(ec["daily_pnl"] < 0) & (ec["trades"] > 0)]
    small_win = active[(active["daily_ret_pct"] > 0) & (active["daily_ret_pct"] <= 3)]
    profitable = ec[ec["daily_pnl"] > 0]

    at_max_lots = ec[ec["lots"] >= MAX_LOTS_CAP]
    active_with_lots = ec[ec["trades"] > 0]

    return {
        "calendar_days": len(ec),
        "no_trade_days": len(no_trade),
        "no_trade_pct": len(no_trade) / len(ec) * 100,
        "loss_day_count": len(loss_days),
        "avg_loss_day_pnl": loss_days["daily_pnl"].mean() if len(loss_days) else 0,
        "avg_loss_day_ret_pct": loss_days.apply(daily_return_pct, axis=1).mean() if len(loss_days) else 0,
        "avg_trades_on_loss_days": loss_days["trades"].mean() if len(loss_days) else 0,
        "small_win_days_0_3": len(small_win),
        "avg_trades_small_win": small_win["trades"].mean() if len(small_win) else 0,
        "avg_ret_small_win": small_win["daily_ret_pct"].mean() if len(small_win) else 0,
        "avg_lots_small_win": small_win["lots"].mean() if len(small_win) else 0,
        "days_at_max_lots": len(at_max_lots),
        "active_days_at_max_lots": len(active_with_lots[active_with_lots["lots"] >= MAX_LOTS_CAP]),
        "pct_active_at_max": (
            len(active_with_lots[active_with_lots["lots"] >= MAX_LOTS_CAP]) / len(active_with_lots) * 100
            if len(active_with_lots) else 0
        ),
        "avg_lots_all_days": ec["lots"].mean(),
        "avg_lots_active": active_with_lots["lots"].mean() if len(active_with_lots) else 0,
        "avg_trades_profitable": profitable[profitable["trades"] > 0]["trades"].mean()
        if len(profitable[profitable["trades"] > 0]) else 0,
        "avg_trades_loss": loss_days["trades"].mean() if len(loss_days) else 0,
        "avg_trades_all_active": active_with_lots["trades"].mean() if len(active_with_lots) else 0,
    }


def print_report(
    dist_df: pd.DataFrame,
    active: pd.DataFrame,
    drag: dict,
    scenarios: list[ScenarioResult],
    risk_rows: list[dict],
):
    ec = pd.read_csv(EQUITY_CSV)
    initial = ec.iloc[0]["capital"]
    final = ec.iloc[-1]["capital"]
    active_days = (ec["trades"] > 0).sum()
    baseline_cdgr = calc_cdgr(final, initial, active_days)

    print("\n" + "=" * 72)
    print("  DELTAFORGE CDGR LEAK ANALYSIS")
    print("=" * 72)

    print("\n## Part 1: Daily Return Distribution (active trading days)\n")
    print(f"{'Bucket':<14} {'Days':>6} {'Avg Ret%':>10} {'P&L Share%':>12} {'Total P&L':>14}")
    print("-" * 60)
    for _, r in dist_df.iterrows():
        print(
            f"{r['bucket']:<14} {int(r['days']):>6} {r['avg_return_pct']:>9.2f}% "
            f"{r['pnl_share_pct']:>11.1f}% {r['total_pnl']:>14,.0f}"
        )

    print(f"\nBaseline CDGR from equity curve: {baseline_cdgr * 100:.2f}% "
          f"({active_days} active days, Rs {initial:,.0f} -> Rs {final:,.0f})")

    print("\n## Part 2: What's Dragging Daily Returns Below 10%\n")
    print(f"a) No-trade days: {drag['no_trade_days']} / {drag['calendar_days']} "
          f"({drag['no_trade_pct']:.1f}%) — 0% return days in calendar")
    print(f"b) Loss days: {drag['loss_day_count']} days, "
          f"avg loss Rs {drag['avg_loss_day_pnl']:,.0f} ({drag['avg_loss_day_ret_pct']:.2f}%), "
          f"avg {drag['avg_trades_on_loss_days']:.1f} trades/day")
    print(f"c) Small win days (0-3%): {drag['small_win_days_0_3']} days, "
          f"avg ret {drag['avg_ret_small_win']:.2f}%, "
          f"avg {drag['avg_trades_small_win']:.1f} trades, "
          f"avg {drag['avg_lots_small_win']:.1f} lots")
    print(f"d) Lot utilization: {drag['days_at_max_lots']} calendar days at MAX_LOTS_CAP ({MAX_LOTS_CAP}), "
          f"{drag['active_days_at_max_lots']} active days ({drag['pct_active_at_max']:.1f}% of active). "
          f"Avg lots active days: {drag['avg_lots_active']:.1f}")
    print(f"e) Trade count: profitable days avg {drag['avg_trades_profitable']:.2f} trades, "
          f"loss days avg {drag['avg_trades_loss']:.2f}, all active avg {drag['avg_trades_all_active']:.2f}")

    print("\n## Part 3: Parameter Sweep (100-day backtest)\n")
    print(
        f"{'Scenario':<42} {'Trades':>6} {'WR%':>6} {'PF':>5} "
        f"{'Final Cap':>12} {'MaxDD%':>7} {'CDGR%':>7} {'Active':>6}"
    )
    print("-" * 95)
    for s in scenarios:
        print(
            f"{s.label:<42} {s.trades:>6} {s.win_rate:>5.1f} {s.profit_factor:>5.2f} "
            f"{s.final_capital:>12,.0f} {s.max_dd:>6.1f} {s.cdgr * 100:>6.2f} {s.active_days:>6}"
        )

    print("\n## Part 4: Risk Check (top 3 by CDGR)\n")
    for row in risk_rows:
        print(f"**{row['label']}**")
        print(f"  CDGR: {row['cdgr'] * 100:.2f}% | PF: {row['pf']:.2f} | Max DD: {row['max_dd']:.1f}%")
        print(f"  Worst day: Rs {row['worst_day_pnl']:,.0f} ({row['worst_day_pct']:.2f}%)")
        print(f"  Longest losing streak: {row['max_losing_streak']} days")
        print(f"  PF >= 2.0: {'YES' if row['pf'] >= 2.0 else 'NO'}")
        print()

    # Recommendation
    eligible = [
        s for s in scenarios
        if s.profit_factor >= 2.0 and s.max_dd <= 40
    ]
    eligible.sort(key=lambda x: x.cdgr, reverse=True)
    print("## Recommendation\n")
    if eligible:
        best = eligible[0]
        print(
            f"Best config meeting PF >= 2.0 and Max DD <= 40%: **{best.label}**\n"
            f"- CDGR: {best.cdgr * 100:.2f}% | PF: {best.profit_factor:.2f} | "
            f"Max DD: {best.max_dd:.1f}% | Trades: {best.trades} | Active days: {best.active_days}\n"
            f"- Params: MAX_TOTAL={best.max_total}, COOLDOWN={best.cooldown}, MAX_SIM={best.max_sim}"
        )
        if best.cdgr * 100 < 10:
            print(
                f"\nNote: Even the best eligible config reaches {best.cdgr * 100:.2f}% CDGR, "
                f"below the 10% target. Primary leaks: no-trade days, loss days, and "
                f"small-win days with limited trade throughput."
            )
    else:
        print("No config met both PF >= 2.0 and Max DD <= 40%. Review risk-adjusted tradeoffs above.")


def main():
    ec = pd.read_csv(EQUITY_CSV)
    ec["date"] = pd.to_datetime(ec["date"])

    dist_df, active = part1_distribution(ec)
    drag = part2_drag(ec, active)

    print("Loading 100-day market data...")
    df = load_real_data(days=DAYS)

    scenario_defs = [
        ("Baseline (T=4, CD=3, Sim=2)", 4, 3, 2),
        ("MAX_TOTAL=6", 6, 3, 2),
        ("MAX_TOTAL=8", 8, 3, 2),
        ("MAX_SIM=3", 4, 3, 3),
        ("COOLDOWN=2", 4, 2, 2),
        ("COOLDOWN=1", 4, 1, 2),
        ("Combo T6+CD2+Sim2", 6, 2, 2),
        ("Combo T6+CD2+Sim3", 6, 2, 3),
        ("Combo T8+CD1+Sim3", 8, 1, 3),
    ]

    scenarios: list[ScenarioResult] = []
    for label, mt, cd, sim in scenario_defs:
        print(f"  Running: {label}...")
        scenarios.append(run_scenario(df, label, mt, cd, sim))

    top3 = sorted(scenarios, key=lambda s: s.cdgr, reverse=True)[:3]
    risk_rows = []
    for s in top3:
        rm = risk_metrics(s.equity_curve)
        risk_rows.append({
            "label": s.label,
            "cdgr": s.cdgr,
            "pf": s.profit_factor,
            "max_dd": s.max_dd,
            **rm,
        })

    print_report(dist_df, active, drag, scenarios, risk_rows)


if __name__ == "__main__":
    main()
