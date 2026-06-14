"""Win-rate scenario comparison — monkey-patch only, no production edits.

Usage:
    python backtest/test_wr_scenarios.py
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, run_compound_backtest
from engine.multi_strategy_engine import MultiStrategyEngine

DAYS = 100
STARTING_CAPITAL = 10_000
LOT_SIZE = 75  # user-specified; settings.NIFTY_LOT_SIZE is 65
DEPLOY_PCT = 80.0

# Snapshot baseline class state
_BASELINE_DISABLED = MultiStrategyEngine.DISABLED_STRATEGIES.copy()
_BASELINE_SCAN = MultiStrategyEngine.scan
_BASELINE_MIN_CONF = settings.PULLBACK_MIN_CONFIDENCE


def _restore_all() -> None:
    MultiStrategyEngine.DISABLED_STRATEGIES.clear()
    MultiStrategyEngine.DISABLED_STRATEGIES.update(_BASELINE_DISABLED)
    MultiStrategyEngine.scan = _BASELINE_SCAN
    settings.PULLBACK_MIN_CONFIDENCE = _BASELINE_MIN_CONF


@contextmanager
def scenario_patches(
    *,
    disable_adx_breakout: bool = False,
    adx_ceiling: float | None = None,
    min_confidence: float | None = None,
):
    """Apply monkey-patches for one scenario; restore on exit."""
    orig_disabled = MultiStrategyEngine.DISABLED_STRATEGIES.copy()
    orig_scan = MultiStrategyEngine.scan
    orig_min_conf = settings.PULLBACK_MIN_CONFIDENCE

    try:
        if disable_adx_breakout:
            MultiStrategyEngine.DISABLED_STRATEGIES.add("ADX_BREAKOUT")

        if adx_ceiling is not None:
            ceiling = adx_ceiling

            def filtered_scan(self, indicators, bar_idx, time_str=""):
                adx_val = self._sv(indicators.get("adx", pd.Series()), bar_idx, 25.0)
                if adx_val > ceiling:
                    return []
                return orig_scan(self, indicators, bar_idx, time_str)

            MultiStrategyEngine.scan = filtered_scan

        if min_confidence is not None:
            settings.PULLBACK_MIN_CONFIDENCE = min_confidence

        yield
    finally:
        MultiStrategyEngine.DISABLED_STRATEGIES.clear()
        MultiStrategyEngine.DISABLED_STRATEGIES.update(orig_disabled)
        MultiStrategyEngine.scan = orig_scan
        settings.PULLBACK_MIN_CONFIDENCE = orig_min_conf


def run_scenario(name: str, df: pd.DataFrame, **patch_kwargs) -> dict:
    with scenario_patches(**patch_kwargs):
        result = run_compound_backtest(
            df,
            starting_capital=STARTING_CAPITAL,
            lot_size=LOT_SIZE,
            deploy_pct=DEPLOY_PCT,
        )
    result["scenario"] = name
    return result


def format_row(r: dict) -> dict:
    return {
        "scenario": r["scenario"],
        "trades": r["total_trades"],
        "win_rate": r["win_rate"],
        "profit_factor": r["profit_factor"],
        "final_capital": r["final_capital"],
        "max_drawdown": r["max_drawdown_pct"],
        "avg_win": r["avg_win"],
        "avg_loss": r["avg_loss"],
    }


def verdict(row: dict, baseline: dict) -> str:
    """IMPLEMENT if WR improves without destroying capital/PF."""
    wr_delta = row["win_rate"] - baseline["win_rate"]
    cap_ratio = row["final_capital"] / max(baseline["final_capital"], 1)
    pf_delta = row["profit_factor"] - baseline["profit_factor"]

    if wr_delta >= 2 and cap_ratio >= 0.85 and pf_delta >= -0.15:
        return "IMPLEMENT"
    if wr_delta >= 1 and cap_ratio >= 0.95 and pf_delta >= 0:
        return "IMPLEMENT"
    return "DON'T IMPLEMENT"


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("Loading 100-day backtest data...")
    df = load_real_data(days=DAYS)
    print(f"  {len(df)} bars, {len(set(df.index.date))} calendar days\n")

    scenarios = [
        ("1 Baseline", {}),
        ("2 Disable ADX_BREAKOUT", {"disable_adx_breakout": True}),
        ("3 ADX ceiling 35", {"adx_ceiling": 35}),
        ("4 Confidence >= 75", {"min_confidence": 75}),
        (
            "5 All three combined",
            {"disable_adx_breakout": True, "adx_ceiling": 35, "min_confidence": 75},
        ),
        (
            "6 ADX_BREAKOUT off + conf>=75",
            {"disable_adx_breakout": True, "min_confidence": 75},
        ),
    ]

    results = []
    for name, kwargs in scenarios:
        print(f"Running {name}...")
        r = run_scenario(name, df, **kwargs)
        results.append(format_row(r))

    _restore_all()

    baseline = results[0]
    print("\n" + "=" * 110)
    print("COMPARISON TABLE (100-day compound backtest, lot_size=75, capital=Rs 10,000)")
    print("=" * 110)
    header = (
        f"{'Scenario':<32} {'Trades':>6} {'WR%':>6} {'PF':>6} "
        f"{'Final Cap':>12} {'MaxDD%':>7} {'Avg Win':>9} {'Avg Loss':>9} {'Verdict':>16}"
    )
    print(header)
    print("-" * 110)

    for row in results:
        v = verdict(row, baseline) if row["scenario"] != baseline["scenario"] else "—"
        print(
            f"{row['scenario']:<32} {row['trades']:>6} {row['win_rate']:>6.1f} "
            f"{row['profit_factor']:>6.2f} {row['final_capital']:>12,.0f} "
            f"{row['max_drawdown']:>7.1f} {row['avg_win']:>9,.0f} {row['avg_loss']:>9,.0f} "
            f"{v:>16}"
        )

    print("\n" + "=" * 110)
    print("VERDICTS (vs baseline)")
    print("=" * 110)
    for row in results[1:]:
        v = verdict(row, baseline)
        wr_d = row["win_rate"] - baseline["win_rate"]
        cap_d = row["final_capital"] - baseline["final_capital"]
        pf_d = row["profit_factor"] - baseline["profit_factor"]
        reason_parts = [
            f"WR {wr_d:+.1f}pp",
            f"capital {cap_d:+,.0f}",
            f"PF {pf_d:+.2f}",
        ]
        if row["win_rate"] > baseline["win_rate"] and row["final_capital"] < baseline["final_capital"] * 0.9:
            reason_parts.append("WR up but capital materially down")
        if row["profit_factor"] < baseline["profit_factor"] * 0.9:
            reason_parts.append("PF degraded")
        print(f"  {row['scenario']}: {v} — {', '.join(reason_parts)}")


if __name__ == "__main__":
    main()
