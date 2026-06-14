"""Isolate impact of strategy code changes on 100-day and Jun 9-12 backtests.

Runtime monkeypatching only — does not modify production source files.
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")

from config import settings
from backtest.run_backtest import load_real_data, run_compound_backtest
from engine.multi_strategy_engine import MultiStrategyEngine, TradeSignal

STARTING_CAPITAL = 10_000

_ORIG_DISABLED = MultiStrategyEngine.DISABLED_STRATEGIES.copy()
_ORIG_CHECK_PULLBACK = MultiStrategyEngine._check_pullback
_ORIG_CHECK_TREND_RIDE = MultiStrategyEngine._check_trend_ride


def calc_cdgr(final: float, initial: float, trading_days: int) -> float:
    if trading_days <= 0 or initial <= 0 or final <= 0:
        return 0.0
    return (final / initial) ** (1 / trading_days) - 1


def _check_pullback_no_bull_trend_cont(
    self: MultiStrategyEngine, ind_dict: dict, idx: int
) -> TradeSignal | None:
    """Original pullback without bull trend-continuation entry path."""
    sig = _ORIG_CHECK_PULLBACK(self, ind_dict, idx)
    if sig is None:
        return None
    if sig.direction == "LONG" and "TrendCont" in sig.reason:
        return None
    return sig


def _check_trend_ride_fixed72(
    self: MultiStrategyEngine, ind_dict: dict, idx: int
) -> TradeSignal | None:
    """Trend ride with fixed RSI5 cap of 72 (no ADX>40 widening to 78)."""
    if self._trend_ride_count >= self.MAX_TREND_RIDE_PER_DAY:
        return None
    if idx < 5:
        return None

    close = self._sv(ind_dict["close"], idx)
    ema_9 = self._sv(ind_dict["ema_9"], idx, 0)
    ema_20 = self._sv(ind_dict["ema_20"], idx, close)
    rsi_5m = self._sv(ind_dict["rsi_5m"], idx, 50)
    adx_val = self._sv(ind_dict.get("adx", pd.Series()), idx, 0)
    adx_prev = self._sv(ind_dict.get("adx", pd.Series()), idx - 1, 0)
    plus_di = self._sv(ind_dict.get("plus_di", pd.Series()), idx, 0)
    minus_di = self._sv(ind_dict.get("minus_di", pd.Series()), idx, 0)
    st_dir = self._sv(ind_dict["supertrend_dir"], idx, 0)

    if pd.isna(close) or ema_9 == 0 or ema_20 == 0:
        return None

    if adx_val < 25 or adx_val <= adx_prev:
        return None

    di_spread = abs(plus_di - minus_di)
    if di_spread < 5:
        return None

    rsi_15m = self._htf_rsi(ind_dict, idx, 50)
    direction = None
    reasons: list[str] = []

    rsi5_cap = 72  # fixed — no ADX>40 widening
    if (
        ema_9 > ema_20
        and close > ema_20
        and plus_di > minus_di
        and rsi_15m > 52
        and 50 <= rsi_5m <= rsi5_cap
    ):
        direction = "LONG"
        reasons = [
            "TrendRide↑",
            f"ADX={adx_val:.0f}↑",
            f"+DI={plus_di:.0f}>{minus_di:.0f}",
            f"RSI15={rsi_15m:.0f}↑",
            f"RSI5={rsi_5m:.0f}",
        ]
    elif (
        ema_9 < ema_20
        and close < ema_20
        and minus_di > plus_di
        and rsi_15m < 48
        and 28 <= rsi_5m <= 50
    ):
        direction = "SHORT"
        reasons = [
            "TrendRide↓",
            f"ADX={adx_val:.0f}↑",
            f"-DI={minus_di:.0f}>{plus_di:.0f}",
            f"RSI15={rsi_15m:.0f}↓",
            f"RSI5={rsi_5m:.0f}",
        ]

    if not direction:
        return None

    conf = 68
    if adx_val > 35:
        conf += 5
    if adx_val > 45:
        conf += 3
    htf_strength = abs(rsi_15m - 50)
    if htf_strength > 10:
        conf += 4
    if di_spread > 12:
        conf += 3
    if (direction == "LONG" and st_dir == 1) or (
        direction == "SHORT" and st_dir == -1
    ):
        conf += 3
    conf = min(conf, 100)

    return TradeSignal(
        direction=direction,
        signal_type="TREND_RIDE",
        confidence=conf,
        htf_rsi=rsi_15m,
        ltf_rsi=rsi_5m,
        nifty_price=close,
        reason=" | ".join(reasons),
        pullback_count=0,
    )


def restore_patches() -> None:
    MultiStrategyEngine.DISABLED_STRATEGIES.clear()
    MultiStrategyEngine.DISABLED_STRATEGIES.update(_ORIG_DISABLED)
    MultiStrategyEngine._check_pullback = _ORIG_CHECK_PULLBACK  # type: ignore[method-assign]
    MultiStrategyEngine._check_trend_ride = _ORIG_CHECK_TREND_RIDE  # type: ignore[method-assign]


@dataclass
class PatchConfig:
    disable_ema_momentum: bool = False
    fixed_rsi5_cap_72: bool = False
    no_bull_pullback_trend_cont: bool = False


@contextmanager
def apply_patches(cfg: PatchConfig):
    restore_patches()
    try:
        if cfg.disable_ema_momentum:
            MultiStrategyEngine.DISABLED_STRATEGIES.add("EMA_MOMENTUM")
        if cfg.fixed_rsi5_cap_72:
            MultiStrategyEngine._check_trend_ride = _check_trend_ride_fixed72  # type: ignore[method-assign]
        if cfg.no_bull_pullback_trend_cont:
            MultiStrategyEngine._check_pullback = _check_pullback_no_bull_trend_cont  # type: ignore[method-assign]
        yield
    finally:
        restore_patches()


def run_backtest(df: pd.DataFrame, cfg: PatchConfig) -> dict:
    with apply_patches(cfg):
        return run_compound_backtest(
            df,
            starting_capital=STARTING_CAPITAL,
            use_adaptive=False,
            use_risk_gates=False,
        )


def format_row(label: str, result: dict) -> str:
    td = result["trading_days"]
    cdgr = calc_cdgr(result["final_capital"], STARTING_CAPITAL, td) * 100
    pnl = result["total_pnl"]
    return (
        f"{label:35s} | {result['total_trades']:>4d} | "
        f"{result['win_rate']:>5.1f}% | Rs {pnl:>8,.0f} | "
        f"{result['profit_factor']:>4.2f} | {cdgr:>4.1f}%"
    )


SCENARIOS: list[tuple[str, PatchConfig]] = [
    ("All new strategy code (current)", PatchConfig()),
    ("Revert EMA_MOMENTUM (disable it)", PatchConfig(disable_ema_momentum=True)),
    ("Revert RSI5 cap (fixed 72)", PatchConfig(fixed_rsi5_cap_72=True)),
    ("Revert PULLBACK trend-cont", PatchConfig(no_bull_pullback_trend_cont=True)),
    (
        "Revert ALL strategy changes",
        PatchConfig(
            disable_ema_momentum=True,
            fixed_rsi5_cap_72=True,
            no_bull_pullback_trend_cont=True,
        ),
    ),
]


def run_suite(label: str, df: pd.DataFrame) -> dict[str, dict]:
    print(f"\n{label}")
    print("=" * 90)
    print(f"{'Config':35s} | Trades | WR    | P&L        | PF   | CDGR")
    print("-" * 90)
    results: dict[str, dict] = {}
    for name, cfg in SCENARIOS:
        result = run_backtest(df, cfg)
        results[name] = result
        print(format_row(name, result))
    return results


def main() -> None:
    old_cpl = settings.CAPITAL_PER_LOT
    settings.CAPITAL_PER_LOT = 6000

    try:
        df_100 = load_real_data(days=100)
        run_suite(
            "STRATEGY CODE CHANGE ISOLATION (CPL=6k, no adaptive, no risk gates) — 100-DAY",
            df_100,
        )
        print(
            f"{'Baseline target':35s} |  516 | 56.0% | Rs 3,320,000 | 1.85 |  6.0%"
        )

        df_30 = load_real_data(days=30)
        unique_days = sorted(set(df_30.index.date))
        last_4 = set(unique_days[-4:])
        df_jun = df_30[df_30.index.map(lambda t: t.date() in last_4)]
        print(f"\nJun 9-12 dates: {sorted(last_4)}")

        run_suite(
            "STRATEGY CODE CHANGE ISOLATION (CPL=6k, no adaptive, no risk gates) — JUN 9-12",
            df_jun,
        )
        print(
            f"{'Baseline target':35s} |   18 | 72.2% | Rs   26,386 |  —   |  —"
        )
    finally:
        settings.CAPITAL_PER_LOT = old_cpl
        restore_patches()


if __name__ == "__main__":
    main()
