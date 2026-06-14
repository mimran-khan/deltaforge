"""Individual strategy backtest sweep -- 100 days, realistic adaptive mode.

Runs each strategy in isolation plus an all-active combined portfolio.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data
from backtest.realistic_backtest import run_realistic_backtest
from engine.multi_strategy_engine import MultiStrategyEngine

ALL_STRATEGIES = frozenset({
    "STOCH_CROSS", "PULLBACK", "EMA_MOMENTUM", "SUPERTREND", "RSI_REVERSION",
    "VWAP_MOMENTUM", "VWAP_MEAN_REV", "CPR_RANGE", "GAP_TRADE", "CPR_BREAKOUT",
    "ADX_BREAKOUT", "TREND_RIDE", "ORB_BREAKOUT", "BB_SQUEEZE", "VWAP_BOUNCE",
    "RSI_DIVERGENCE",
})

ACTIVE_COMBINED = frozenset({
    "PULLBACK", "TREND_RIDE", "CPR_BREAKOUT", "CPR_RANGE",
    "ORB_BREAKOUT", "BB_SQUEEZE", "VWAP_BOUNCE", "RSI_DIVERGENCE",
    "STOCH_CROSS", "ADX_BREAKOUT",
})


def _engine_for(enabled: frozenset) -> MultiStrategyEngine:
    """Return engine with only `enabled` strategies active."""

    class _FilteredEngine(MultiStrategyEngine):
        DISABLED_STRATEGIES = ALL_STRATEGIES - enabled

    return _FilteredEngine()


def _run_label(enabled: frozenset, label: str, df, capital: int, lot_size: int) -> dict:
    engine = _engine_for(enabled)
    result = run_realistic_backtest(
        df, starting_capital=capital, lot_size=lot_size, engine_override=engine,
    )
    result["label"] = label
    return result


def _print_table(rows: list[dict]):
    print()
    print(f"{'Strategy':<22} {'Trades':>6} {'WR%':>6} {'PF':>6} "
          f"{'Total PnL':>14} {'MaxDD%':>7} {'AvgRet%':>7}")
    print("-" * 75)
    for r in rows:
        pnl = r["total_pnl"]
        pnl_str = f"Rs {pnl:,.0f}"
        print(f"{r['label']:<22} {r['total_trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['profit_factor']:>6.2f} {pnl_str:>14} "
              f"{r['max_drawdown_pct']:>6.1f}% {r['avg_daily_return_pct']:>6.1f}%")
    print()


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    days = 100
    capital = 10_000
    lot_size = settings.NIFTY_LOT_SIZE

    print(f"\nLoading {days} trading days of Nifty data...")
    df = load_real_data(days=days)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days\n")

    configs = [
        (frozenset({"PULLBACK"}), "PULLBACK (ref)"),
        (frozenset({"TREND_RIDE"}), "TREND_RIDE (ref)"),
        (frozenset({"ORB_BREAKOUT"}), "ORB_BREAKOUT"),
        (frozenset({"BB_SQUEEZE"}), "BB_SQUEEZE"),
        (frozenset({"VWAP_BOUNCE"}), "VWAP_BOUNCE"),
        (frozenset({"RSI_DIVERGENCE"}), "RSI_DIVERGENCE"),
        (frozenset({"STOCH_CROSS"}), "STOCH_CROSS"),
        (frozenset({"ADX_BREAKOUT"}), "ADX_BREAKOUT"),
        (ACTIVE_COMBINED, "ALL COMBINED"),
    ]

    rows = []
    for enabled, label in configs:
        print(f"  Running {label}...", flush=True)
        rows.append(_run_label(enabled, label, df, capital, lot_size))

    _print_table(rows)
    return rows


if __name__ == "__main__":
    main()
