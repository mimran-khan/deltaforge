"""Deep backtest analysis — 100-day trade log, strategy/exit breakdown, parameter sweep.

Usage:
    python -m backtest.deep_analysis
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.comparison_study import load_study_data
from backtest.run_backtest import run_compound_backtest
from engine.multi_strategy_engine import MultiStrategyEngine
from engine import premium_model

STUDY_DAYS = 100
STARTING_CAPITAL = 10_000
LOT_SIZE = getattr(settings, "NIFTY_LOT_SIZE", 65)
DEPLOY_PCT = getattr(settings, "CAPITAL_DEPLOY_PCT", 80.0)

# Current baseline from .env / defaults
BASELINE = {
    "sl_pct": getattr(settings, "PREMIUM_SL_PCT", 10.0),
    "trail_trigger": settings.TRAIL_TRIGGER_PCT,
    "trail_pct": settings.TRAIL_PCT,
    "max_sim": settings.MAX_SIMULTANEOUS_POSITIONS,
    "max_total": MultiStrategyEngine.MAX_TOTAL_PER_DAY,
}

SL_VALUES = [8, 10, 12, 15]
TRAIL_TRIGGERS = [10, 12, 15, 20]
TRAIL_PCTS = [5, 6, 8, 10]
MAX_SIM_VALUES = [1, 2, 3]
MAX_TOTAL_VALUES = [4, 6, 8, 10, 15]

EXIT_TYPES = ["SL", "TGT", "TRAIL", "TIME", "EOD"]


@dataclass
class SweepResult:
    label: str
    sl_pct: float
    trail_trigger: float
    trail_pct: float
    max_sim: int
    max_total: int
    final_capital: float
    profit_factor: float
    win_rate: float
    max_drawdown_pct: float
    total_trades: int
    avg_win: float
    avg_loss: float
    total_pnl: float
    return_pct: float
    stage: str = ""


@contextmanager
def patched_config(
    *,
    sl_pct: float | None = None,
    trail_trigger: float | None = None,
    trail_pct: float | None = None,
    max_sim: int | None = None,
    max_total: int | None = None,
):
    """Monkey-patch settings and engine caps for a single backtest run."""
    orig_premium_sl = settings.PREMIUM_SL_PCT
    orig_trail_trigger = settings.TRAIL_TRIGGER_PCT
    orig_trail_pct = settings.TRAIL_PCT
    orig_max_sim = settings.MAX_SIMULTANEOUS_POSITIONS
    orig_strategy_sl = copy.deepcopy(premium_model.STRATEGY_SL_PCT)
    orig_max_total = MultiStrategyEngine.MAX_TOTAL_PER_DAY

    try:
        if sl_pct is not None:
            settings.PREMIUM_SL_PCT = sl_pct
            for key in premium_model.STRATEGY_SL_PCT:
                premium_model.STRATEGY_SL_PCT[key] = sl_pct
        if trail_trigger is not None:
            settings.TRAIL_TRIGGER_PCT = trail_trigger
        if trail_pct is not None:
            settings.TRAIL_PCT = trail_pct
        if max_sim is not None:
            settings.MAX_SIMULTANEOUS_POSITIONS = max_sim
        if max_total is not None:
            MultiStrategyEngine.MAX_TOTAL_PER_DAY = max_total
        yield
    finally:
        settings.PREMIUM_SL_PCT = orig_premium_sl
        settings.TRAIL_TRIGGER_PCT = orig_trail_trigger
        settings.TRAIL_PCT = orig_trail_pct
        settings.MAX_SIMULTANEOUS_POSITIONS = orig_max_sim
        premium_model.STRATEGY_SL_PCT.clear()
        premium_model.STRATEGY_SL_PCT.update(orig_strategy_sl)
        MultiStrategyEngine.MAX_TOTAL_PER_DAY = orig_max_total


def run_config(
    df: pd.DataFrame,
    *,
    sl_pct: float,
    trail_trigger: float,
    trail_pct: float,
    max_sim: int,
    max_total: int,
    label: str = "",
    stage: str = "",
) -> tuple[dict, SweepResult]:
    with patched_config(
        sl_pct=sl_pct,
        trail_trigger=trail_trigger,
        trail_pct=trail_pct,
        max_sim=max_sim,
        max_total=max_total,
    ):
        result = run_compound_backtest(
            df,
            starting_capital=STARTING_CAPITAL,
            lot_size=LOT_SIZE,
            deploy_pct=DEPLOY_PCT,
        )

    sr = SweepResult(
        label=label or f"SL={sl_pct} T={trail_trigger}/{trail_pct} sim={max_sim} tot={max_total}",
        sl_pct=sl_pct,
        trail_trigger=trail_trigger,
        trail_pct=trail_pct,
        max_sim=max_sim,
        max_total=max_total,
        final_capital=result["final_capital"],
        profit_factor=result["profit_factor"],
        win_rate=result["win_rate"],
        max_drawdown_pct=result["max_drawdown_pct"],
        total_trades=result["total_trades"],
        avg_win=result["avg_win"],
        avg_loss=result["avg_loss"],
        total_pnl=result["total_pnl"],
        return_pct=result["return_pct"],
        stage=stage,
    )
    return result, sr


def strategy_breakdown(trades: list[dict]) -> pd.DataFrame:
    rows = []
    df = pd.DataFrame(trades)
    if df.empty:
        return pd.DataFrame()

    for strat in sorted(df["strategy"].unique()):
        st = df[df["strategy"] == strat]
        wins = st[st["pnl"] > 0]
        losses = st[st["pnl"] <= 0]
        gross_profit = wins["pnl"].sum() if len(wins) else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        rows.append({
            "strategy": strat,
            "trades": len(st),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(st) * 100, 1),
            "avg_win": round(wins["pnl"].mean(), 0) if len(wins) else 0,
            "avg_loss": round(losses["pnl"].mean(), 0) if len(losses) else 0,
            "profit_factor": round(pf, 2) if pf != float("inf") else 999.0,
            "net_pnl": round(st["pnl"].sum(), 0),
            "pnl_per_trade": round(st["pnl"].mean(), 0),
        })
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False)


def exit_breakdown(trades: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(trades)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for reason in EXIT_TYPES:
        rt = df[df["reason"] == reason]
        if rt.empty:
            rows.append({
                "exit_type": reason,
                "count": 0,
                "avg_pnl": 0,
                "total_pnl": 0,
            })
        else:
            rows.append({
                "exit_type": reason,
                "count": len(rt),
                "avg_pnl": round(rt["pnl"].mean(), 0),
                "total_pnl": round(rt["pnl"].sum(), 0),
            })
    return pd.DataFrame(rows)


def sweep_table(results: list[SweepResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def pick_best(results: list[SweepResult], key: str = "profit_factor") -> SweepResult:
    return max(results, key=lambda r: (r.profit_factor, r.final_capital))


def collect_strategy_pnl_across_configs(
    df: pd.DataFrame,
    configs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate per-strategy net P&L across all sweep configs."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    config_wins: dict[str, int] = {}

    for cfg in configs:
        result, _ = run_config(df, **cfg)
        strat_pnl = {}
        for t in result["trades"]:
            strat_pnl[t["strategy"]] = strat_pnl.get(t["strategy"], 0) + t["pnl"]
        for strat, pnl in strat_pnl.items():
            totals[strat] = totals.get(strat, 0) + pnl
            counts[strat] = counts.get(strat, 0) + 1
            if pnl > 0:
                config_wins[strat] = config_wins.get(strat, 0) + 1

    rows = []
    for strat in sorted(totals.keys()):
        n_cfg = counts[strat]
        rows.append({
            "strategy": strat,
            "configs_tested": n_cfg,
            "configs_profitable": config_wins.get(strat, 0),
            "total_net_pnl": round(totals[strat], 0),
            "avg_pnl_per_config": round(totals[strat] / n_cfg, 0),
        })
    return pd.DataFrame(rows).sort_values("total_net_pnl")


def print_df(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)
    if frame.empty:
        print("  (no data)")
        return
    print(frame.to_string(index=False))


def print_sweep_stage(title: str, results: list[SweepResult], best: SweepResult) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)
    hdr = (
        f"{'Config':<42} {'Capital':>10} {'PF':>6} {'WR%':>6} "
        f"{'MaxDD%':>7} {'Trades':>6} {'AvgW':>8} {'AvgL':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        marker = " *" if r.label == best.label else "  "
        print(
            f"{marker}{r.label:<40} "
            f"Rs {r.final_capital:>8,.0f} {r.profit_factor:>6.2f} {r.win_rate:>6.1f} "
            f"{r.max_drawdown_pct:>7.1f} {r.total_trades:>6} "
            f"{r.avg_win:>8,.0f} {r.avg_loss:>8,.0f}"
        )
    print(f"\n  >> Best: {best.label} (PF={best.profit_factor}, WR={best.win_rate}%)")


def main() -> None:
    print("=" * 80)
    print("  DELTAFORGE DEEP ANALYSIS — 100-DAY BACKTEST")
    print("=" * 80)
    print(f"\nBaseline config:")
    print(f"  SL={BASELINE['sl_pct']}%  Trail={BASELINE['trail_trigger']}/{BASELINE['trail_pct']}%  "
          f"MaxSim={BASELINE['max_sim']}  MaxTotal={BASELINE['max_total']}")
    print(f"  Capital=Rs {STARTING_CAPITAL:,}  Lot={LOT_SIZE}  Deploy={DEPLOY_PCT}%\n")

    print(f"Loading {STUDY_DAYS} days of Nifty 5m data...")
    df = load_study_data(STUDY_DAYS)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} trading days")
    print(f"Range: {df.index[0].date()} to {df.index[-1].date()}\n")

    # ── Baseline 100-day backtest ───────────────────────────────────────────
    print("Running baseline backtest (current config)...")
    baseline_result, baseline_sr = run_config(
        df,
        sl_pct=BASELINE["sl_pct"],
        trail_trigger=BASELINE["trail_trigger"],
        trail_pct=BASELINE["trail_pct"],
        max_sim=BASELINE["max_sim"],
        max_total=BASELINE["max_total"],
        label="BASELINE",
        stage="baseline",
    )

    trades = baseline_result["trades"]
    print(f"\nBaseline: {baseline_sr.total_trades} trades | "
          f"WR {baseline_sr.win_rate}% | PF {baseline_sr.profit_factor} | "
          f"Final Rs {baseline_sr.final_capital:,.0f} ({baseline_sr.return_pct}%) | "
          f"MaxDD {baseline_sr.max_drawdown_pct}%")

    strat_df = strategy_breakdown(trades)
    exit_df = exit_breakdown(trades)
    print_df("PER-STRATEGY BREAKDOWN (BASELINE)", strat_df)
    print_df("EXIT-TYPE BREAKDOWN (BASELINE)", exit_df)

    all_sweep_results: list[SweepResult] = [baseline_sr]
    all_sweep_configs: list[dict[str, Any]] = [{
        "sl_pct": BASELINE["sl_pct"],
        "trail_trigger": BASELINE["trail_trigger"],
        "trail_pct": BASELINE["trail_pct"],
        "max_sim": BASELINE["max_sim"],
        "max_total": BASELINE["max_total"],
        "label": "BASELINE",
        "stage": "baseline",
    }]
    cross_strat_totals: dict[str, float] = {}
    cross_strat_config_counts: dict[str, int] = {}
    cross_strat_config_wins: dict[str, int] = {}

    def record_cross_strat(result: dict) -> None:
        strat_pnl: dict[str, float] = {}
        for t in result["trades"]:
            strat_pnl[t["strategy"]] = strat_pnl.get(t["strategy"], 0) + t["pnl"]
        for strat, pnl in strat_pnl.items():
            cross_strat_totals[strat] = cross_strat_totals.get(strat, 0) + pnl
            cross_strat_config_counts[strat] = cross_strat_config_counts.get(strat, 0) + 1
            if pnl > 0:
                cross_strat_config_wins[strat] = cross_strat_config_wins.get(strat, 0) + 1

    record_cross_strat(baseline_result)

    # ── Stage A: SL sweep ───────────────────────────────────────────────────
    print("\n\nStage A: SL sweep (4 runs)...")
    best_sl = BASELINE["sl_pct"]
    best_trail_trigger = BASELINE["trail_trigger"]
    best_trail_pct = BASELINE["trail_pct"]
    best_max_sim = BASELINE["max_sim"]
    best_max_total = BASELINE["max_total"]

    sl_results: list[SweepResult] = []
    for sl in SL_VALUES:
        label = f"SL={sl}%"
        cfg = {
            "sl_pct": sl,
            "trail_trigger": best_trail_trigger,
            "trail_pct": best_trail_pct,
            "max_sim": best_max_sim,
            "max_total": best_max_total,
            "label": label,
            "stage": "sl_sweep",
        }
        result, sr = run_config(df, **cfg)
        sl_results.append(sr)
        all_sweep_results.append(sr)
        all_sweep_configs.append(cfg)
        record_cross_strat(result)
        print(f"  {label}: PF={sr.profit_factor} WR={sr.win_rate}% "
              f"Final=Rs {sr.final_capital:,.0f} Trades={sr.total_trades}")

    best_sl_result = pick_best(sl_results)
    best_sl = best_sl_result.sl_pct
    print_sweep_stage("STAGE A — SL SWEEP RESULTS", sl_results, best_sl_result)

    # ── Stage B: Trail trigger + pct sweep (12 combos) ───────────────────────
    print("\n\nStage B: Trail trigger/pct sweep (12 combos, SL fixed at best)...")
    trail_results: list[SweepResult] = []
    for trig in TRAIL_TRIGGERS:
        for tpct in TRAIL_PCTS:
            label = f"SL={best_sl}% T={trig}/{tpct}%"
            cfg = {
                "sl_pct": best_sl,
                "trail_trigger": trig,
                "trail_pct": tpct,
                "max_sim": best_max_sim,
                "max_total": best_max_total,
                "label": label,
                "stage": "trail_sweep",
            }
            result, sr = run_config(df, **cfg)
            trail_results.append(sr)
            all_sweep_results.append(sr)
            all_sweep_configs.append(cfg)
            record_cross_strat(result)

    best_trail_result = pick_best(trail_results)
    best_trail_trigger = best_trail_result.trail_trigger
    best_trail_pct = best_trail_result.trail_pct
    print_sweep_stage(
        f"STAGE B — TRAIL SWEEP (SL={best_sl}%)",
        trail_results,
        best_trail_result,
    )

    # ── Stage C: Max simultaneous positions ─────────────────────────────────
    print(f"\n\nStage C: MAX_SIMULTANEOUS_POSITIONS sweep (SL={best_sl}, "
          f"Trail={best_trail_trigger}/{best_trail_pct})...")
    sim_results: list[SweepResult] = []
    for sim in MAX_SIM_VALUES:
        label = f"MaxSim={sim}"
        cfg = {
            "sl_pct": best_sl,
            "trail_trigger": best_trail_trigger,
            "trail_pct": best_trail_pct,
            "max_sim": sim,
            "max_total": best_max_total,
            "label": label,
            "stage": "sim_sweep",
        }
        result, sr = run_config(df, **cfg)
        sim_results.append(sr)
        all_sweep_results.append(sr)
        all_sweep_configs.append(cfg)
        record_cross_strat(result)

    best_sim_result = pick_best(sim_results)
    best_max_sim = best_sim_result.max_sim
    print_sweep_stage(
        f"STAGE C — MAX SIM SWEEP (SL={best_sl}, T={best_trail_trigger}/{best_trail_pct})",
        sim_results,
        best_sim_result,
    )

    # ── Stage D: Max total per day ────────────────────────────────────────────
    print(f"\n\nStage D: MAX_TOTAL_PER_DAY sweep (SL={best_sl}, "
          f"Trail={best_trail_trigger}/{best_trail_pct}, Sim={best_max_sim})...")
    total_results: list[SweepResult] = []
    for tot in MAX_TOTAL_VALUES:
        label = f"MaxTotal={tot}"
        cfg = {
            "sl_pct": best_sl,
            "trail_trigger": best_trail_trigger,
            "trail_pct": best_trail_pct,
            "max_sim": best_max_sim,
            "max_total": tot,
            "label": label,
            "stage": "total_sweep",
        }
        result, sr = run_config(df, **cfg)
        total_results.append(sr)
        all_sweep_results.append(sr)
        all_sweep_configs.append(cfg)
        record_cross_strat(result)

    best_total_result = pick_best(total_results)
    best_max_total = best_total_result.max_total
    print_sweep_stage(
        f"STAGE D — MAX TOTAL SWEEP (SL={best_sl}, T={best_trail_trigger}/{best_trail_pct}, Sim={best_max_sim})",
        total_results,
        best_total_result,
    )

    # ── Top 5 configs by PF (all stages) ─────────────────────────────────────
    sweep_only = [r for r in all_sweep_results if r.stage != "baseline"]
    top5 = sorted(sweep_only, key=lambda r: (r.profit_factor, r.final_capital), reverse=True)[:5]
    print(f"\n{'=' * 80}")
    print("  TOP 5 CONFIGS BY PROFIT FACTOR (all sweep runs)")
    print("=" * 80)
    for i, r in enumerate(top5, 1):
        print(
            f"  {i}. {r.label} [{r.stage}] | PF={r.profit_factor} WR={r.win_rate}% | "
            f"Final Rs {r.final_capital:,.0f} | MaxDD {r.max_drawdown_pct}% | "
            f"Trades={r.total_trades} AvgW={r.avg_win:,.0f} AvgL={r.avg_loss:,.0f}"
        )

    # ── Recommended config final run ──────────────────────────────────────────
    print(f"\n\nRunning recommended config validation...")
    recommended_result, recommended_sr = run_config(
        df,
        sl_pct=best_sl,
        trail_trigger=best_trail_trigger,
        trail_pct=best_trail_pct,
        max_sim=best_max_sim,
        max_total=best_max_total,
        label="RECOMMENDED",
        stage="recommended",
    )
    rec_strat_df = strategy_breakdown(recommended_result["trades"])
    rec_exit_df = exit_breakdown(recommended_result["trades"])

    print(f"\n{'=' * 80}")
    print("  RECOMMENDED CONFIG — PROJECTED PERFORMANCE")
    print("=" * 80)
    print(f"  SL %              : {best_sl}")
    print(f"  Trail Trigger %   : {best_trail_trigger}")
    print(f"  Trail %           : {best_trail_pct}")
    print(f"  Max Sim Positions : {best_max_sim}")
    print(f"  Max Total/Day     : {best_max_total}")
    print(f"  Final Capital     : Rs {recommended_sr.final_capital:,.0f}")
    print(f"  Return            : {recommended_sr.return_pct}%")
    print(f"  Profit Factor     : {recommended_sr.profit_factor}")
    print(f"  Win Rate          : {recommended_sr.win_rate}%")
    print(f"  Max Drawdown      : {recommended_sr.max_drawdown_pct}%")
    print(f"  Total Trades      : {recommended_sr.total_trades}")
    print(f"  Avg Win / Avg Loss: Rs {recommended_sr.avg_win:,.0f} / Rs {recommended_sr.avg_loss:,.0f}")

    print_df("PER-STRATEGY (RECOMMENDED CONFIG)", rec_strat_df)
    print_df("EXIT-TYPE (RECOMMENDED CONFIG)", rec_exit_df)

    # ── Net loser strategies across all configs ───────────────────────────────
    cross_rows = []
    for strat in sorted(cross_strat_totals.keys()):
        n_cfg = cross_strat_config_counts[strat]
        cross_rows.append({
            "strategy": strat,
            "configs_tested": n_cfg,
            "configs_profitable": cross_strat_config_wins.get(strat, 0),
            "total_net_pnl": round(cross_strat_totals[strat], 0),
            "avg_pnl_per_config": round(cross_strat_totals[strat] / n_cfg, 0),
        })
    cross_strat_df = pd.DataFrame(cross_rows).sort_values("total_net_pnl")
    print_df(
        f"STRATEGY P&L ACROSS ALL {len(all_sweep_configs)} SWEEP CONFIGS",
        cross_strat_df,
    )

    losers = cross_strat_df[cross_strat_df["total_net_pnl"] < 0]
    consistent_losers = cross_strat_df[
        (cross_strat_df["total_net_pnl"] < 0)
        & (cross_strat_df["configs_profitable"] <= cross_strat_df["configs_tested"] // 2)
    ]

    print(f"\n{'=' * 80}")
    print("  RECOMMENDATIONS")
    print("=" * 80)

    print("\n  a) STRATEGIES TO DISABLE (net losers across sweep):")
    if losers.empty:
        print("     None — all strategies net positive across tested configs.")
    else:
        for _, row in losers.iterrows():
            flag = "DISABLE" if row["strategy"] in consistent_losers["strategy"].values else "REVIEW"
            print(
                f"     [{flag}] {row['strategy']}: total Rs {row['total_net_pnl']:+,.0f} "
                f"({row['configs_profitable']}/{row['configs_tested']} configs profitable)"
            )

    print(f"\n  b) Optimal SL %: {best_sl}% "
          f"(baseline was {BASELINE['sl_pct']}%, PF {best_sl_result.profit_factor} vs baseline {baseline_sr.profit_factor})")
    print(f"  c) Optimal Trail: trigger={best_trail_trigger}%, trail={best_trail_pct}% "
          f"(baseline was {BASELINE['trail_trigger']}/{BASELINE['trail_pct']}%)")
    print(f"  d) Optimal Max Sim Positions: {best_max_sim} "
          f"(baseline was {BASELINE['max_sim']})")
    print(f"  e) Optimal Max Trades/Day: {best_max_total} "
          f"(baseline was {BASELINE['max_total']})")

    print(f"\n  Baseline vs Recommended:")
    print(f"    Baseline  : PF={baseline_sr.profit_factor} WR={baseline_sr.win_rate}% "
          f"Final=Rs {baseline_sr.final_capital:,.0f} MaxDD={baseline_sr.max_drawdown_pct}%")
    print(f"    Recommended: PF={recommended_sr.profit_factor} WR={recommended_sr.win_rate}% "
          f"Final=Rs {recommended_sr.final_capital:,.0f} MaxDD={recommended_sr.max_drawdown_pct}%")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
