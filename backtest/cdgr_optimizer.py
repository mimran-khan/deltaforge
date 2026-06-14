"""CDGR optimizer — strategy-specific exits, breakeven stop, re-enabled strategies.

Targets >= 8% compound daily growth rate (CDGR) on 100-day backtest while
keeping PF >= 2.0 and Max DD <= 40%.

Usage:
    python -m backtest.cdgr_optimizer
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs
from engine.multi_strategy_engine import MultiStrategyEngine, TradeSignal
from engine.premium_model import create_premium_state, STRATEGY_SL_PCT

DAYS = 100
STARTING_CAPITAL = 10_000
LOT_SIZE = settings.NIFTY_LOT_SIZE
DEPLOY_PCT = getattr(settings, "CAPITAL_DEPLOY_PCT", 80.0)

# Baseline that reproduces ~6.46% CDGR from prior sweeps
BASELINE_TRAIL_TRIGGER = 10.0
BASELINE_TRAIL_PCT = 5.0
BASELINE_SL = 8.0


@dataclass
class StrategyExitParams:
    sl: float | None = None
    trail_trigger: float | None = None
    trail_pct: float | None = None
    target_mult: float | None = None


@dataclass
class OptimizerConfig:
    """Full backtest configuration for one scenario."""

    label: str = "baseline"
    max_total_per_day: int = 8
    cooldown_bars: int = 3
    max_sim: int = 2
    warmup_bars: int = 10
    default_sl: float = BASELINE_SL
    default_trail_trigger: float = BASELINE_TRAIL_TRIGGER
    default_trail_pct: float = BASELINE_TRAIL_PCT
    strategy_exits: dict[str, StrategyExitParams] = field(default_factory=dict)
    breakeven_trigger_pct: float | None = None
    reenable_strategies: set[str] = field(default_factory=set)
    strategy_min_confidence: dict[str, float] = field(default_factory=dict)
    adx_breakout_range: tuple[float, float] | None = None


@dataclass
class ScenarioResult:
    label: str
    config: OptimizerConfig
    trades: int
    win_rate: float
    profit_factor: float
    final_capital: float
    max_dd: float
    cdgr: float
    active_days: int
    flat_days: int
    avg_trades_per_active_day: float
    meets_target: bool


def calc_cdgr(final: float, initial: float, active_days: int) -> float:
    if active_days <= 0 or initial <= 0 or final <= 0:
        return 0.0
    return (final / initial) ** (1 / active_days) - 1


def _resolve_exit(signal_type: str, cfg: OptimizerConfig) -> dict[str, float]:
    """Merge default and per-strategy exit params."""
    over = cfg.strategy_exits.get(signal_type)
    return {
        "sl": over.sl if over and over.sl is not None else cfg.default_sl,
        "trail_trigger": (
            over.trail_trigger
            if over and over.trail_trigger is not None
            else cfg.default_trail_trigger
        ),
        "trail_pct": (
            over.trail_pct if over and over.trail_pct is not None else cfg.default_trail_pct
        ),
        "target_mult": over.target_mult if over and over.target_mult is not None else None,
    }


class OptimizedEngine(MultiStrategyEngine):
    """Engine with re-enabled strategies and optional scan filters."""

    def __init__(self, opt_cfg: OptimizerConfig):
        super().__init__()
        self._opt_cfg = opt_cfg
        self.MAX_TOTAL_PER_DAY = opt_cfg.max_total_per_day
        self.COOLDOWN_BARS = opt_cfg.cooldown_bars
        self.DISABLED_STRATEGIES = set(MultiStrategyEngine.DISABLED_STRATEGIES)
        self.DISABLED_STRATEGIES -= opt_cfg.reenable_strategies

        if opt_cfg.adx_breakout_range and "ADX_BREAKOUT" in opt_cfg.reenable_strategies:
            lo, hi = opt_cfg.adx_breakout_range
            orig = self._check_adx_breakout

            def _filtered_adx(ind_dict: dict, idx: int):
                sig = orig(ind_dict, idx)
                if sig is None:
                    return None
                adx_val = self._sv(ind_dict.get("adx", pd.Series()), idx, 0)
                if adx_val < lo or adx_val > hi:
                    return None
                return sig

            self._check_adx_breakout = _filtered_adx  # type: ignore[method-assign]

    def scan(self, indicators: dict, bar_idx: int, time_str: str = "") -> list[TradeSignal]:
        signals = super().scan(indicators, bar_idx, time_str)
        if not signals or not self._opt_cfg.strategy_min_confidence:
            return signals
        filtered = []
        for sig in signals:
            min_conf = self._opt_cfg.strategy_min_confidence.get(sig.signal_type)
            if min_conf is not None and sig.confidence < min_conf:
                continue
            filtered.append(sig)
        return filtered


def run_optimized_backtest(
    df: pd.DataFrame,
    cfg: OptimizerConfig,
    starting_capital: float = STARTING_CAPITAL,
    lot_size: int = LOT_SIZE,
) -> dict:
    """Compound backtest with per-strategy exits, breakeven stop, and engine overrides."""
    engine = OptimizedEngine(cfg)
    capital = starting_capital
    peak = capital
    trades: list[dict] = []
    equity_curve: list[dict] = []

    unique_days = sorted(set(df.index.date))
    warmup = cfg.warmup_bars

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < warmup + 1:
            equity_curve.append(
                {"date": day, "capital": capital, "daily_pnl": 0, "trades": 0, "lots": 0}
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
        open_positions: list[dict] = []

        for i in range(warmup, len(day_df)):
            ts = day_df.index[i]
            time_str = ts.strftime("%H:%M")
            nifty_price = day_df["close"].iloc[i]

            closed_this_bar: list[dict] = []
            for pos in open_positions:
                pos["candles_held"] += 1
                cur_prem = pos["prem_state"].current_premium(nifty_price, pos["candles_held"])

                if cur_prem > pos["peak_premium"]:
                    pos["peak_premium"] = cur_prem

                peak_gain_pct = (
                    (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100
                )
                if cfg.breakeven_trigger_pct is not None and peak_gain_pct >= cfg.breakeven_trigger_pct:
                    pos["sl_premium"] = max(pos["sl_premium"], pos["entry_premium"])
                    pos["breakeven_active"] = True

                trail_floor = pos["prem_state"].update_trail(
                    cur_prem,
                    trigger_pct=pos["trail_trigger"],
                    trail_pct=pos["trail_pct"],
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
                        pos["entry_premium"], exit_prem, pos["qty"], day_lots
                    )
                    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                    net_pnl = raw_pnl - costs

                    capital += net_pnl
                    day_pnl += net_pnl
                    day_trades += 1
                    consec_loss = consec_loss + 1 if net_pnl < 0 else 0

                    trades.append(
                        {
                            "strategy": pos["signal_type"],
                            "signal": pos["direction"],
                            "entry_time": pos["entry_time"],
                            "exit_time": ts,
                            "entry_premium": round(pos["entry_premium"], 2),
                            "exit_premium": round(exit_prem, 2),
                            "peak_premium": round(pos["peak_premium"], 2),
                            "peak_gain_pct": round(peak_gain_pct, 2),
                            "qty": pos["qty"],
                            "lots": day_lots,
                            "pnl": round(net_pnl, 0),
                            "reason": exit_reason,
                            "capital_after": round(capital, 0),
                            "breakeven": pos.get("breakeven_active", False),
                        }
                    )
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

            if len(open_positions) >= cfg.max_sim:
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
                if signal.confidence < getattr(settings, "PULLBACK_MIN_CONFIDENCE", 50):
                    continue

                exit_p = _resolve_exit(signal.signal_type, cfg)
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

                if exit_p["target_mult"] is not None:
                    prem_state.target_premium = round(entry_premium * exit_p["target_mult"], 2)

                sl_premium = entry_premium * (1 - exit_p["sl"] / 100)
                qty = day_lots * lot_size

                open_positions.append(
                    {
                        "direction": signal.direction,
                        "signal_type": signal.signal_type,
                        "entry_time": ts,
                        "entry_premium": entry_premium,
                        "sl_premium": sl_premium,
                        "qty": qty,
                        "prem_state": prem_state,
                        "candles_held": 0,
                        "peak_premium": entry_premium,
                        "trail_trigger": exit_p["trail_trigger"],
                        "trail_pct": exit_p["trail_pct"],
                        "breakeven_active": False,
                    }
                )
                break

            if capital <= 0:
                break

        for pos in open_positions:
            nifty_price = day_df["close"].iloc[-1]
            exit_prem = pos["prem_state"].current_premium(nifty_price, pos["candles_held"])
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
                (pos["peak_premium"] - pos["entry_premium"]) / pos["entry_premium"] * 100
            )
            trades.append(
                {
                    "strategy": pos["signal_type"],
                    "signal": pos["direction"],
                    "entry_time": pos["entry_time"],
                    "exit_time": day_df.index[-1],
                    "entry_premium": round(pos["entry_premium"], 2),
                    "exit_premium": round(exit_prem, 2),
                    "peak_premium": round(pos["peak_premium"], 2),
                    "peak_gain_pct": round(peak_gain, 2),
                    "qty": pos["qty"],
                    "lots": day_lots,
                    "pnl": round(net_pnl, 0),
                    "reason": "EOD",
                    "capital_after": round(capital, 0),
                    "breakeven": pos.get("breakeven_active", False),
                }
            )

        if capital > peak:
            peak = capital

        equity_curve.append(
            {
                "date": day,
                "capital": round(capital, 0),
                "daily_pnl": round(day_pnl, 0),
                "trades": day_trades,
                "lots": day_lots,
            }
        )

        if capital <= 0:
            break

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

    active_days_list = [e for e in equity_curve if e["trades"] > 0]
    flat_days = sum(1 for e in equity_curve if e["trades"] == 0)

    return {
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_pnl": round(total_pnl, 0),
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "active_trading_days": len(active_days_list),
        "flat_days": flat_days,
        "trading_days": len(equity_curve),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def run_scenario(df: pd.DataFrame, cfg: OptimizerConfig) -> ScenarioResult:
    with patched_settings(cfg.max_sim):
        result = run_optimized_backtest(df, cfg)
    active = result["active_trading_days"]
    cdgr = calc_cdgr(result["final_capital"], STARTING_CAPITAL, active)
    avg_trades = result["total_trades"] / active if active else 0
    pf = result["profit_factor"]
    max_dd = result["max_drawdown_pct"]
    meets = cdgr >= 0.08 and pf >= 2.0 and max_dd <= 40
    return ScenarioResult(
        label=cfg.label,
        config=cfg,
        trades=result["total_trades"],
        win_rate=result["win_rate"],
        profit_factor=pf,
        final_capital=result["final_capital"],
        max_dd=max_dd,
        cdgr=cdgr,
        active_days=active,
        flat_days=result["flat_days"],
        avg_trades_per_active_day=round(avg_trades, 2),
        meets_target=meets,
    )


@contextmanager
def patched_settings(max_sim: int):
    orig = settings.MAX_SIMULTANEOUS_POSITIONS
    try:
        settings.MAX_SIMULTANEOUS_POSITIONS = max_sim
        yield
    finally:
        settings.MAX_SIMULTANEOUS_POSITIONS = orig


def baseline_config() -> OptimizerConfig:
    return OptimizerConfig(
        label="Baseline (T=8, trail 10/5, SL 8%)",
        max_total_per_day=8,
        cooldown_bars=3,
        max_sim=2,
        warmup_bars=settings.SCAN_WARMUP_BARS,
    )


def approach1_pullback_sweep() -> list[OptimizerConfig]:
    configs = []
    for sl, trig, trail in product([6, 8, 10], [8, 10, 12], [4, 5, 6]):
        configs.append(
            OptimizerConfig(
                label=f"PULLBACK SL={sl}% trail {trig}/{trail}%",
                max_total_per_day=8,
                strategy_exits={
                    "PULLBACK": StrategyExitParams(sl=sl, trail_trigger=trig, trail_pct=trail),
                },
            )
        )
    return configs


def approach1_ema_momentum_sweep() -> list[OptimizerConfig]:
    configs = []
    for mult in [1.25, 1.30, 1.35, 1.40, 1.50]:
        configs.append(
            OptimizerConfig(
                label=f"EMA_MOMENTUM target_mult={mult}",
                max_total_per_day=8,
                strategy_exits={
                    "EMA_MOMENTUM": StrategyExitParams(target_mult=mult),
                },
            )
        )
    return configs


def approach2_breakeven_sweep() -> list[OptimizerConfig]:
    return [
        OptimizerConfig(label="Breakeven @ +3%", max_total_per_day=8, breakeven_trigger_pct=3),
        OptimizerConfig(label="Breakeven @ +5%", max_total_per_day=8, breakeven_trigger_pct=5),
        OptimizerConfig(label="Breakeven @ +7%", max_total_per_day=8, breakeven_trigger_pct=7),
    ]


def approach3_reenable_strategies() -> list[OptimizerConfig]:
    return [
        OptimizerConfig(
            label="STOCH_CROSS SL=6% conf>=75",
            max_total_per_day=8,
            reenable_strategies={"STOCH_CROSS"},
            strategy_min_confidence={"STOCH_CROSS": 75},
            strategy_exits={"STOCH_CROSS": StrategyExitParams(sl=6)},
        ),
        OptimizerConfig(
            label="ADX_BREAKOUT ADX 20-30",
            max_total_per_day=8,
            reenable_strategies={"ADX_BREAKOUT"},
            adx_breakout_range=(20, 30),
        ),
        OptimizerConfig(
            label="STOCH+ADX combined filters",
            max_total_per_day=8,
            reenable_strategies={"STOCH_CROSS", "ADX_BREAKOUT"},
            strategy_min_confidence={"STOCH_CROSS": 75},
            strategy_exits={"STOCH_CROSS": StrategyExitParams(sl=6)},
            adx_breakout_range=(20, 30),
        ),
    ]


def approach4_warmup_sweep() -> list[OptimizerConfig]:
    return [
        OptimizerConfig(label=f"SCAN_WARMUP_BARS={w}", max_total_per_day=8, warmup_bars=w)
        for w in [5, 7, 8, 10, 12]
    ]


def analyze_no_trade_days(df: pd.DataFrame, equity_curve: list[dict]) -> dict:
    """Day-of-week and data-gap analysis for zero-trade days."""
    no_trade = [e for e in equity_curve if e["trades"] == 0]
    dow_counts: dict[str, int] = {}
    for e in no_trade:
        d = e["date"]
        if isinstance(d, str):
            d = datetime.strptime(d[:10], "%Y-%m-%d").date()
        name = d.strftime("%A")
        dow_counts[name] = dow_counts.get(name, 0) + 1

    unique_days = sorted(set(df.index.date))
    gaps = []
    for i in range(1, len(unique_days)):
        delta = (unique_days[i] - unique_days[i - 1]).days
        if delta > 3:
            gaps.append((unique_days[i - 1], unique_days[i], delta))

    short_days = []
    for day in unique_days:
        n = len(df[df.index.date == day])
        if n < settings.SCAN_WARMUP_BARS + 5:
            short_days.append((day, n))

    return {
        "no_trade_count": len(no_trade),
        "calendar_days": len(equity_curve),
        "dow_distribution": dow_counts,
        "calendar_gaps": gaps,
        "short_candle_days": short_days[:10],
        "data_range": (min(unique_days), max(unique_days)),
    }


def approach5_combinations(top_pullback: StrategyExitParams | None) -> list[OptimizerConfig]:
    """Smart combinations of best levers."""
    pb = top_pullback or StrategyExitParams(sl=8, trail_trigger=10, trail_pct=5)
    combos = []

    def _cfg(label: str, **kwargs) -> OptimizerConfig:
        base = dict(
            max_total_per_day=8,
            cooldown_bars=3,
            max_sim=2,
            strategy_exits={"PULLBACK": copy.copy(pb)},
        )
        base.update(kwargs)
        return OptimizerConfig(label=label, **base)

    combos.append(_cfg("Combo: PULLBACK opt + breakeven 5%", breakeven_trigger_pct=5))
    combos.append(
        _cfg(
            "Combo: PULLBACK opt + BE5 + EMA tgt 1.35",
            strategy_exits={
                "PULLBACK": copy.copy(pb),
                "EMA_MOMENTUM": StrategyExitParams(target_mult=1.35),
            },
            breakeven_trigger_pct=5,
        )
    )
    combos.append(
        _cfg(
            "Combo: PULLBACK opt + BE5 + STOCH",
            breakeven_trigger_pct=5,
            reenable_strategies={"STOCH_CROSS"},
            strategy_min_confidence={"STOCH_CROSS": 75},
            strategy_exits={
                "PULLBACK": copy.copy(pb),
                "STOCH_CROSS": StrategyExitParams(sl=6),
            },
        )
    )
    combos.append(
        _cfg(
            "Combo: PULLBACK opt + BE5 + ADX 20-30",
            breakeven_trigger_pct=5,
            reenable_strategies={"ADX_BREAKOUT"},
            adx_breakout_range=(20, 30),
            strategy_exits={"PULLBACK": copy.copy(pb)},
        )
    )
    combos.append(
        _cfg(
            "Combo: FULL (PB+BE5+EMA+STOCH+ADX)",
            breakeven_trigger_pct=5,
            reenable_strategies={"STOCH_CROSS", "ADX_BREAKOUT"},
            strategy_min_confidence={"STOCH_CROSS": 75},
            adx_breakout_range=(20, 30),
            strategy_exits={
                "PULLBACK": copy.copy(pb),
                "EMA_MOMENTUM": StrategyExitParams(target_mult=1.35),
                "STOCH_CROSS": StrategyExitParams(sl=6),
            },
        )
    )
    combos.append(
        OptimizerConfig(
            label="Combo: T8 CD1 Sim3 + PB opt + BE5",
            max_total_per_day=8,
            cooldown_bars=1,
            max_sim=3,
            breakeven_trigger_pct=5,
            strategy_exits={"PULLBACK": copy.copy(pb)},
        )
    )
    return combos


def print_results_table(title: str, results: list[ScenarioResult], limit: int | None = None):
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(
        f"{'Scenario':<48} {'Trades':>6} {'WR%':>6} {'PF':>5} "
        f"{'Final Cap':>12} {'MaxDD%':>7} {'CDGR%':>7} {'Flat':>5} {'OK':>3}"
    )
    print("-" * 100)
    sorted_r = sorted(results, key=lambda r: r.cdgr, reverse=True)
    if limit:
        sorted_r = sorted_r[:limit]
    for r in sorted_r:
        ok = "Y" if r.meets_target else ""
        print(
            f"{r.label:<48} {r.trades:>6} {r.win_rate:>5.1f} {r.profit_factor:>5.2f} "
            f"{r.final_capital:>12,.0f} {r.max_dd:>6.1f} {r.cdgr * 100:>6.2f} "
            f"{r.flat_days:>5} {ok:>3}"
        )


def format_recommended(cfg: OptimizerConfig, result: ScenarioResult) -> str:
    lines = [
        f"Label: {cfg.label}",
        f"CDGR: {result.cdgr * 100:.2f}% | PF: {result.profit_factor:.2f} | "
        f"WR: {result.win_rate:.1f}% | Max DD: {result.max_dd:.1f}%",
        f"Final Capital: Rs {result.final_capital:,.0f} | Trades: {result.trades} | "
        f"Active days: {result.active_days} | No-trade days: {result.flat_days}",
        "",
        "Engine params:",
        f"  MAX_TOTAL_PER_DAY = {cfg.max_total_per_day}",
        f"  COOLDOWN_BARS = {cfg.cooldown_bars}",
        f"  MAX_SIMULTANEOUS_POSITIONS = {cfg.max_sim}",
        f"  SCAN_WARMUP_BARS = {cfg.warmup_bars}",
        f"  Default SL = {cfg.default_sl}% | Trail = {cfg.default_trail_trigger}%/{cfg.default_trail_pct}%",
    ]
    if cfg.breakeven_trigger_pct is not None:
        lines.append(f"  Breakeven stop @ +{cfg.breakeven_trigger_pct}% peak gain")
    if cfg.reenable_strategies:
        lines.append(f"  Re-enabled strategies: {sorted(cfg.reenable_strategies)}")
    if cfg.strategy_min_confidence:
        lines.append(f"  Min confidence filters: {cfg.strategy_min_confidence}")
    if cfg.adx_breakout_range:
        lo, hi = cfg.adx_breakout_range
        lines.append(f"  ADX_BREAKOUT range: {lo}-{hi}")
    if cfg.strategy_exits:
        lines.append("  Per-strategy exits:")
        for name, p in sorted(cfg.strategy_exits.items()):
            parts = []
            if p.sl is not None:
                parts.append(f"SL={p.sl}%")
            if p.trail_trigger is not None:
                parts.append(f"trail={p.trail_trigger}/{p.trail_pct}%")
            if p.target_mult is not None:
                parts.append(f"target_mult={p.target_mult}")
            lines.append(f"    {name}: {', '.join(parts) if parts else 'defaults'}")
    return "\n".join(lines)


def main():
    print("\nLoading 100-day market data...")
    df = load_real_data(days=DAYS)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} calendar days")
    print(f"Date range: {unique_days[0]} → {unique_days[-1]}")
    print(f"SCAN_WARMUP_BARS (settings): {settings.SCAN_WARMUP_BARS}")

    all_results: list[ScenarioResult] = []

    # Baseline
    print("\nRunning baseline...")
    baseline = baseline_config()
    baseline_result = run_scenario(df, baseline)
    all_results.append(baseline_result)

    # Approach 1a: PULLBACK sweep
    print("\nApproach 1a: PULLBACK SL/trail sweep (27 configs)...")
    pb_results = []
    for cfg in approach1_pullback_sweep():
        pb_results.append(run_scenario(df, cfg))
    all_results.extend(pb_results)
    pb_results.sort(key=lambda r: r.cdgr, reverse=True)
    best_pb = pb_results[0]
    best_pb_params = best_pb.config.strategy_exits.get("PULLBACK")

    print_results_table("Approach 1a: Top 10 PULLBACK exit configs", pb_results, limit=10)
    print(f"\n  Best PULLBACK: {best_pb.label} → CDGR {best_pb.cdgr * 100:.2f}%")

    # Approach 1b: EMA_MOMENTUM targets
    print("\nApproach 1b: EMA_MOMENTUM target sweep...")
    ema_results = [run_scenario(df, c) for c in approach1_ema_momentum_sweep()]
    all_results.extend(ema_results)
    print_results_table("Approach 1b: EMA_MOMENTUM wider targets", ema_results)

    # Approach 2: Breakeven
    print("\nApproach 2: Breakeven stop...")
    be_results = [run_scenario(df, c) for c in approach2_breakeven_sweep()]
    all_results.extend(be_results)
    print_results_table("Approach 2: Breakeven stop", be_results)

    # Approach 3: Re-enable strategies
    print("\nApproach 3: Re-enabled strategies with filters...")
    reenable_results = [run_scenario(df, c) for c in approach3_reenable_strategies()]
    all_results.extend(reenable_results)
    print_results_table("Approach 3: Re-enabled strategies", reenable_results)

    # Approach 4: No-trade analysis + warmup
    print("\nApproach 4: No-trade day analysis...")
    with patched_settings(baseline.max_sim):
        baseline_full = run_optimized_backtest(df, baseline)
    nt = analyze_no_trade_days(df, baseline_full["equity_curve"])
    print(f"  No-trade days: {nt['no_trade_count']} / {nt['calendar_days']}")
    print(f"  Data range: {nt['data_range'][0]} → {nt['data_range'][1]}")
    print("  No-trade by day-of-week:")
    for dow, cnt in sorted(nt["dow_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {dow}: {cnt}")
    if nt["calendar_gaps"]:
        print(f"  Calendar gaps (>3 days): {len(nt['calendar_gaps'])}")
        for g in nt["calendar_gaps"][:5]:
            print(f"    {g[0]} → {g[1]} ({g[2]} days)")
    else:
        print("  No large calendar gaps in selected 100-day window")

    print("\n  Warmup bar sweep...")
    warmup_results = [run_scenario(df, c) for c in approach4_warmup_sweep()]
    all_results.extend(warmup_results)
    print_results_table("Approach 4: SCAN_WARMUP_BARS sweep", warmup_results)

    # Approach 5: Combinations
    print("\nApproach 5: Smart combinations...")
    combo_cfgs = approach5_combinations(best_pb_params)
    combo_results = [run_scenario(df, c) for c in combo_cfgs]
    all_results.extend(combo_results)
    print_results_table("Approach 5: Combined configs", combo_results)

    # Final ranking
    eligible = [r for r in all_results if r.profit_factor >= 2.0 and r.max_dd <= 40]
    eligible.sort(key=lambda r: r.cdgr, reverse=True)
    target_met = [r for r in eligible if r.cdgr >= 0.08]

    print("\n" + "=" * 100)
    print("  OVERALL TOP 15 (PF >= 2.0, Max DD <= 40%)")
    print("=" * 100)
    print_results_table("", eligible, limit=15)

    print("\n" + "=" * 100)
    print("  RECOMMENDED CONFIG")
    print("=" * 100)
    if target_met:
        best = target_met[0]
        print(f"\n✓ Target met: CDGR >= 8%\n")
        print(format_recommended(best.config, best))
    elif eligible:
        best = eligible[0]
        print(f"\n✗ No config reached 8% CDGR with PF>=2 and DD<=40%. Closest:\n")
        print(format_recommended(best.config, best))
        print(f"\n  Gap to 8% CDGR: {(0.08 - best.cdgr) * 100:.2f} percentage points")
    else:
        best = max(all_results, key=lambda r: r.cdgr)
        print("\n✗ No config met risk gates (PF>=2, DD<=40%). Highest CDGR regardless:\n")
        print(format_recommended(best.config, best))

    print(f"\n  Baseline reference: CDGR {baseline_result.cdgr * 100:.2f}%, "
          f"PF {baseline_result.profit_factor:.2f}, "
          f"Max DD {baseline_result.max_dd:.1f}%, "
          f"{baseline_result.trades} trades, "
          f"{baseline_result.flat_days} no-trade days")

    # Strategy breakdown for best config
    with patched_settings(best.config.max_sim):
        best_full = run_optimized_backtest(df, best.config)
    strat_stats: dict[str, dict] = {}
    for t in best_full["trades"]:
        s = t["strategy"]
        if s not in strat_stats:
            strat_stats[s] = {"n": 0, "w": 0, "pnl": 0}
        strat_stats[s]["n"] += 1
        if t["pnl"] > 0:
            strat_stats[s]["w"] += 1
        strat_stats[s]["pnl"] += t["pnl"]

    print("\n  Best config strategy breakdown:")
    for name, st in sorted(strat_stats.items()):
        wr = st["w"] / st["n"] * 100 if st["n"] else 0
        print(f"    {name:14s}: {st['n']:>3d} trades | {wr:>4.0f}% WR | Rs {st['pnl']:>10,.0f}")

    be_trades = [t for t in best_full["trades"] if t.get("breakeven")]
    if be_trades:
        be_saved = sum(1 for t in be_trades if t["pnl"] >= 0 and t["reason"] != "SL")
        print(f"\n  Breakeven-active trades: {len(be_trades)} "
              f"(non-SL exits after BE: {be_saved})")

    return all_results


if __name__ == "__main__":
    main()
