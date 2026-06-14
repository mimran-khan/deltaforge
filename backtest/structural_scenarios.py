"""Structural change scenarios — unlock more trades via monkey-patching only.

Tests scan/backtest bottlenecks without modifying production files.

Usage:
    python backtest/structural_scenarios.py
"""

from __future__ import annotations

import copy
import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs
from engine import premium_model
from engine.multi_strategy_engine import MultiStrategyEngine, TradeSignal

DAYS = 100
STARTING_CAPITAL = 10_000
LOT_SIZE = settings.NIFTY_LOT_SIZE
DEPLOY_PCT = getattr(settings, "CAPITAL_DEPLOY_PCT", 80.0)

# ── Snapshots for restore ────────────────────────────────────────────────────
_ORIG_MAX_TOTAL = MultiStrategyEngine.MAX_TOTAL_PER_DAY
_ORIG_SCAN = MultiStrategyEngine.scan
_ORIG_CHECK_PULLBACK = MultiStrategyEngine._check_pullback
_ORIG_SL_PCT = copy.deepcopy(premium_model.STRATEGY_SL_PCT)
_ORIG_TARGET_MULT = copy.deepcopy(premium_model.STRATEGY_TARGET_MULT)

SCENARIO_A_SL = {"PULLBACK": 6.0}
SCENARIO_A_EMA_TARGETS = {70: 1.50, 50: 1.40, 0: 1.30}


def calc_cdgr(final: float, initial: float, active_days: int) -> float:
    if active_days <= 0 or initial <= 0 or final <= 0:
        return 0.0
    return (final / initial) ** (1 / active_days) - 1


def restore_all_modules() -> None:
    """Reset class attrs and premium_model dicts to production values."""
    MultiStrategyEngine.MAX_TOTAL_PER_DAY = _ORIG_MAX_TOTAL
    MultiStrategyEngine.scan = _ORIG_SCAN
    MultiStrategyEngine._check_pullback = _ORIG_CHECK_PULLBACK
    premium_model.STRATEGY_SL_PCT.clear()
    premium_model.STRATEGY_SL_PCT.update(_ORIG_SL_PCT)
    premium_model.STRATEGY_TARGET_MULT.clear()
    premium_model.STRATEGY_TARGET_MULT.update(_ORIG_TARGET_MULT)
    importlib.reload(premium_model)


def apply_scenario_a_patches() -> None:
    MultiStrategyEngine.MAX_TOTAL_PER_DAY = 8
    premium_model.STRATEGY_SL_PCT["PULLBACK"] = 6.0
    premium_model.STRATEGY_TARGET_MULT["EMA_MOMENTUM"] = copy.deepcopy(
        SCENARIO_A_EMA_TARGETS
    )


def scan_return_all_candidates(
    self: MultiStrategyEngine,
    indicators: dict,
    bar_idx: int,
    time_str: str = "",
) -> list[TradeSignal]:
    """Like scan() but returns every valid candidate, not just the best."""
    self._current_time = time_str or "12:00"

    total = (
        self._pullback_count
        + self._stoch_count
        + self._momentum_count
        + self._supertrd_count
        + self._rsi_rev_count
        + self._vwap_count
        + self._vwap_mr_count
        + self._cpr_range_count
        + self._gap_count
        + self._cpr_breakout_count
        + self._adx_breakout_count
    )
    if total >= self.MAX_TOTAL_PER_DAY:
        return []

    if bar_idx in self._used_bars:
        return []

    if time_str:
        win_start, win_end = getattr(self, "_time_window", ("09:30", "14:30"))
        if time_str < win_start or time_str > win_end:
            return []

    adx_val = self._sv(indicators.get("adx", pd.Series()), bar_idx, 25.0)
    if adx_val < self.MIN_ADX:
        return []

    if not self.shock.check(indicators["close"], bar_idx):
        return []

    vwap_val = self._sv(indicators.get("vwap", pd.Series()), bar_idx, np.nan)
    close_val = self._sv(indicators["close"], bar_idx)

    def _vwap_ok(sig: TradeSignal) -> bool:
        if not self.USE_VWAP_FILTER or np.isnan(vwap_val):
            return True
        if sig.direction == "LONG" and close_val < vwap_val:
            return False
        if sig.direction == "SHORT" and close_val > vwap_val:
            return False
        return True

    def _vwap_boost(sig: TradeSignal) -> None:
        if np.isnan(vwap_val):
            return
        if (sig.direction == "LONG" and close_val > vwap_val) or (
            sig.direction == "SHORT" and close_val < vwap_val
        ):
            sig.confidence = min(sig.confidence + 5, 100)

    candidates: list[tuple[TradeSignal, str]] = []
    for check_fn, counter_attr in [
        (self._check_stoch_cross, "_stoch_count"),
        (self._check_pullback, "_pullback_count"),
        (self._check_ema_momentum, "_momentum_count"),
        (self._check_supertrend_flip, "_supertrd_count"),
        (self._check_rsi_reversion, "_rsi_rev_count"),
        (self._check_vwap_momentum, "_vwap_count"),
        (self._check_vwap_mean_reversion, "_vwap_mr_count"),
        (self._check_cpr_range, "_cpr_range_count"),
        (self._check_gap_trade, "_gap_count"),
        (self._check_cpr_breakout, "_cpr_breakout_count"),
        (self._check_adx_breakout, "_adx_breakout_count"),
    ]:
        sig = check_fn(indicators, bar_idx)
        if sig and _vwap_ok(sig):
            if sig.signal_type in self.DISABLED_STRATEGIES:
                continue
            if self._is_strategy_cooled(sig.signal_type, bar_idx):
                continue
            _vwap_boost(sig)
            candidates.append((sig, counter_attr))

    if not candidates:
        return []

    if not self._bar_quality_ok(indicators, bar_idx):
        return []

    results: list[TradeSignal] = []
    for sig, counter_attr in candidates:
        self._signals_today.append(sig)
        setattr(self, counter_attr, getattr(self, counter_attr) + 1)
        results.append(sig)

    self._used_bars.update(range(bar_idx, bar_idx + self.COOLDOWN_BARS))
    return results


def _patch_scan_time_window(start: str, end: str) -> None:
    orig = MultiStrategyEngine.scan

    def _wrapped(
        self: MultiStrategyEngine,
        indicators: dict,
        bar_idx: int,
        time_str: str = "",
    ) -> list[TradeSignal]:
        if time_str:
            if time_str < start or time_str > end:
                return []
        return orig(self, indicators, bar_idx, time_str)

    MultiStrategyEngine.scan = _wrapped  # type: ignore[method-assign]


def _check_pullback_with_long_trend_cont(
    self: MultiStrategyEngine, ind_dict: dict, idx: int
) -> TradeSignal | None:
    sig = _ORIG_CHECK_PULLBACK(self, ind_dict, idx)
    if sig is not None:
        return sig

    rsi_5m = self._sv(ind_dict["rsi_5m"], idx, 50)
    rsi_15m = self._htf_rsi(ind_dict, idx, 50)
    if rsi_15m == 50.0:
        if rsi_5m < 35:
            rsi_15m = 40
        elif rsi_5m > 65:
            rsi_15m = 60

    if rsi_15m <= self.HTF_BULL_RSI:
        return None

    htf_strength = abs(rsi_15m - 50)
    if self.HTF_DEAD_ZONE_LO <= htf_strength < self.HTF_DEAD_ZONE_HI:
        return None

    if self._pullback_count >= self.MAX_PULLBACK_PER_DAY:
        return None

    close = self._sv(ind_dict["close"], idx)
    ema_9 = self._sv(ind_dict["ema_9"], idx, close)
    ema_20 = self._sv(ind_dict["ema_20"], idx, close)
    st_dir = self._sv(ind_dict["supertrend_dir"], idx, 0)
    adx_val = self._sv(ind_dict.get("adx", pd.Series()), idx, 0)
    rsi_prev = self._sv(ind_dict["rsi_5m"], idx - 1, rsi_5m) if idx >= 1 else rsi_5m

    if np.isnan(close):
        return None

    pb_count = 0
    trend_cont = False
    reasons = [f"15m_RSI={rsi_15m:.0f}↑"]

    if pb_count < 1 and adx_val > 35 and st_dir == 1 and ema_9 > ema_20:
        rsi_dip = rsi_5m < rsi_prev - 2
        bull_candles = 0
        if idx >= 2:
            for lb in range(3):
                c = self._sv(ind_dict["close"], idx - lb)
                o = self._sv(ind_dict["open"], idx - lb)
                if not np.isnan(c) and not np.isnan(o) and c > o:
                    bull_candles += 1
        if rsi_dip or bull_candles >= 3:
            trend_cont = True
            pb_count = 1
            if rsi_dip:
                reasons.append(f"TrendCont RSI dip {rsi_prev:.0f}→{rsi_5m:.0f}")
            if bull_candles >= 3:
                reasons.append(f"TrendCont {bull_candles} bull candles")

    if pb_count < 1:
        return None

    conf = 60 if trend_cont else 55 + (pb_count * 10)
    conf += min(htf_strength * 0.3, 8)
    if close > ema_20:
        conf += 3
    if st_dir == 1:
        conf += 3
    conf = min(conf, 100)

    return TradeSignal(
        direction="LONG",
        signal_type="PULLBACK",
        confidence=conf,
        htf_rsi=rsi_15m,
        ltf_rsi=rsi_5m,
        nifty_price=close,
        reason=" | ".join(reasons),
        pullback_count=pb_count,
    )


def run_compound_backtest_variant(
    df: pd.DataFrame,
    *,
    starting_capital: float = STARTING_CAPITAL,
    lot_size: int = LOT_SIZE,
    deploy_pct: float = DEPLOY_PCT,
    engine_override: MultiStrategyEngine | None = None,
    break_after_first: bool = True,
    allow_same_direction: bool = False,
) -> dict:
    """Copy of run_compound_backtest with optional multi-entry / same-dir."""
    from engine.premium_model import create_premium_state, STRATEGY_SL_PCT

    engine = engine_override if engine_override is not None else MultiStrategyEngine()

    capital = starting_capital
    peak = capital
    trades: list[dict] = []
    equity_curve: list[dict] = []

    unique_days = sorted(set(df.index.date))

    for day_idx, day in enumerate(unique_days):
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            equity_curve.append(
                {
                    "date": day,
                    "capital": capital,
                    "daily_pnl": 0,
                    "trades": 0,
                    "lots": 0,
                }
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
                        pos["entry_premium"], exit_prem, pos["qty"], day_lots
                    )
                    raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
                    net_pnl = raw_pnl - costs

                    capital += net_pnl
                    day_pnl += net_pnl
                    day_trades += 1
                    consec_loss = consec_loss + 1 if net_pnl < 0 else 0

                    peak_gain = (
                        (pos["peak_premium"] - pos["entry_premium"])
                        / pos["entry_premium"]
                        * 100
                    )

                    trades.append(
                        {
                            "strategy": pos["signal_type"],
                            "signal": pos["direction"],
                            "entry_time": pos["entry_time"],
                            "exit_time": ts,
                            "entry_premium": round(pos["entry_premium"], 2),
                            "exit_premium": round(exit_prem, 2),
                            "peak_premium": round(pos["peak_premium"], 2),
                            "peak_gain_pct": round(peak_gain, 2),
                            "qty": pos["qty"],
                            "lots": day_lots,
                            "pnl": round(net_pnl, 0),
                            "reason": exit_reason,
                            "capital_after": round(capital, 0),
                        }
                    )
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                open_positions.remove(pos)

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
                if len(open_positions) >= max_sim:
                    break
                if not allow_same_direction and signal.direction in open_dirs:
                    continue
                if signal.confidence < getattr(
                    settings, "PULLBACK_MIN_CONFIDENCE", 50
                ):
                    continue

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
                    }
                )
                open_dirs.add(signal.direction)
                if break_after_first:
                    break

            if capital <= 0:
                break

        for pos in open_positions:
            nifty_price = day_df["close"].iloc[-1]
            exit_prem = pos["prem_state"].current_premium(
                nifty_price, pos["candles_held"]
            )
            brokerage = getattr(settings, "BROKERAGE_PER_ORDER", 20) * 2
            stt = (
                exit_prem
                * pos["qty"]
                * getattr(settings, "STT_PCT", 0.0125)
                / 100
            )
            slippage = getattr(settings, "SLIPPAGE_POINTS", 0.5) * pos["qty"]
            costs = brokerage + stt + slippage
            raw_pnl = (exit_prem - pos["entry_premium"]) * pos["qty"]
            net_pnl = raw_pnl - costs
            capital += net_pnl
            day_pnl += net_pnl
            day_trades += 1
            peak_gain = (
                (pos["peak_premium"] - pos["entry_premium"])
                / pos["entry_premium"]
                * 100
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
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    eq_vals = [e["capital"] for e in equity_curve]
    peak_arr = np.maximum.accumulate(eq_vals) if eq_vals else [starting_capital]
    dd_arr = [(p - v) / p * 100 for p, v in zip(peak_arr, eq_vals)]
    max_dd = max(dd_arr) if dd_arr else 0

    active_days = [e for e in equity_curve if e["trades"] > 0]
    daily_rets = [
        (e["daily_pnl"] / max(e["capital"] - e["daily_pnl"], 1)) * 100
        for e in active_days
    ]
    avg_daily_ret = np.mean(daily_rets) if daily_rets else 0

    profitable_days = sum(1 for e in equity_curve if e["daily_pnl"] > 0)
    loss_days = sum(1 for e in equity_curve if e["daily_pnl"] < 0)
    flat_days = sum(1 for e in equity_curve if e["daily_pnl"] == 0)

    active_count = len(active_days)
    cdgr = calc_cdgr(capital, starting_capital, active_count)

    return {
        "starting_capital": starting_capital,
        "final_capital": round(capital, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / starting_capital * 100, 1),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 0),
        "avg_loss": round(avg_loss, 0),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "profitable_days": profitable_days,
        "loss_days": loss_days,
        "flat_days": flat_days,
        "trading_days": len(equity_curve),
        "active_trading_days": active_count,
        "avg_daily_return_pct": round(avg_daily_ret, 1),
        "cdgr": cdgr,
        "trades": trades,
        "equity_curve": equity_curve,
    }


@dataclass
class ScenarioConfig:
    label: str
    scenario_a: bool = False
    multi_entry: bool = False
    all_candidates: bool = False
    allow_same_direction: bool = False
    long_trend_cont: bool = False
    time_window: tuple[str, str] | None = None


def _apply_config(cfg: ScenarioConfig) -> None:
    restore_all_modules()
    if cfg.scenario_a:
        apply_scenario_a_patches()
    if cfg.all_candidates:
        MultiStrategyEngine.scan = scan_return_all_candidates  # type: ignore
    elif cfg.time_window:
        start, end = cfg.time_window
        _patch_scan_time_window(start, end)
    if cfg.long_trend_cont:
        MultiStrategyEngine._check_pullback = _check_pullback_with_long_trend_cont  # type: ignore


@contextmanager
def scenario_context(cfg: ScenarioConfig):
    _apply_config(cfg)
    try:
        yield
    finally:
        restore_all_modules()


def run_scenario(df: pd.DataFrame, cfg: ScenarioConfig) -> dict:
    with scenario_context(cfg):
        result = run_compound_backtest_variant(
            df,
            break_after_first=not cfg.multi_entry,
            allow_same_direction=cfg.allow_same_direction,
        )
    result["scenario"] = cfg.label
    return result


def meets_risk_gates(r: dict) -> bool:
    return r["profit_factor"] >= 2.0 and r["max_drawdown_pct"] <= 40.0


def meets_target(r: dict) -> bool:
    return meets_risk_gates(r) and r["cdgr"] >= 0.08


def format_table_row(r: dict) -> str:
    ok = "Y" if meets_target(r) else ("~" if meets_risk_gates(r) else "")
    tpd = r["total_trades"] / max(r["active_trading_days"], 1)
    return (
        f"{r['scenario']:<42} {r['total_trades']:>6} {r['win_rate']:>5.1f} "
        f"{r['profit_factor']:>5.2f} {r['final_capital']:>12,.0f} "
        f"{r['max_drawdown_pct']:>6.1f} {r['cdgr'] * 100:>6.2f} "
        f"{r['active_trading_days']:>6} {tpd:>5.1f} {ok:>3}"
    )


def build_scenario_g(individual: list[tuple[ScenarioConfig, dict]]) -> ScenarioConfig:
    """Merge flags from scenarios that pass risk gates without CDGR regression vs A."""
    baseline_a_cdgr = next(
        r["cdgr"] for cfg, r in individual if cfg.label.startswith("A:")
    )
    flags = {
        "multi_entry": False,
        "all_candidates": False,
        "allow_same_direction": False,
        "long_trend_cont": False,
        "time_window": None,
    }
    for cfg, r in individual:
        if cfg.label.startswith("G:"):
            continue
        if not meets_risk_gates(r):
            continue
        if r["cdgr"] + 0.001 < baseline_a_cdgr:
            continue
        if cfg.multi_entry:
            flags["multi_entry"] = True
        if cfg.all_candidates:
            flags["all_candidates"] = True
        if cfg.allow_same_direction:
            flags["allow_same_direction"] = True
        if cfg.long_trend_cont:
            flags["long_trend_cont"] = True
        if cfg.time_window:
            flags["time_window"] = cfg.time_window

    parts = []
    if flags["multi_entry"]:
        parts.append("multi-entry")
    if flags["all_candidates"]:
        parts.append("all-candidates")
    if flags["allow_same_direction"]:
        parts.append("same-dir")
    if flags["long_trend_cont"]:
        parts.append("long-trend-cont")
    if flags["time_window"]:
        parts.append(f"window {flags['time_window'][0]}-{flags['time_window'][1]}")

    label = "G: Best combo (" + ", ".join(parts) + ")" if parts else "G: Best combo (A only)"

    return ScenarioConfig(
        label=label,
        scenario_a=True,
        multi_entry=flags["multi_entry"],
        all_candidates=flags["all_candidates"],
        allow_same_direction=flags["allow_same_direction"],
        long_trend_cont=flags["long_trend_cont"],
        time_window=flags["time_window"],
    )


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("Loading 100-day backtest data...")
    df = load_real_data(days=DAYS)
    cal_days = len(set(df.index.date))
    print(f"  {len(df)} bars, {cal_days} calendar days\n")

    scenarios: list[ScenarioConfig] = [
        ScenarioConfig("A: MAX8 + PB SL6% + EMA tgt 1.50x", scenario_a=True),
        ScenarioConfig(
            "B: A + multi-entry (no break)",
            scenario_a=True,
            multi_entry=True,
        ),
        ScenarioConfig(
            "C: B + scan all candidates",
            scenario_a=True,
            multi_entry=True,
            all_candidates=True,
        ),
        ScenarioConfig(
            "D: C + same-direction positions",
            scenario_a=True,
            multi_entry=True,
            all_candidates=True,
            allow_same_direction=True,
        ),
        ScenarioConfig(
            "E: D + PULLBACK LONG trend continuation",
            scenario_a=True,
            multi_entry=True,
            all_candidates=True,
            allow_same_direction=True,
            long_trend_cont=True,
        ),
        ScenarioConfig(
            "F: A + window 09:15-14:45",
            scenario_a=True,
            time_window=("09:15", "14:45"),
        ),
    ]

    results: list[tuple[ScenarioConfig, dict]] = []
    for cfg in scenarios:
        print(f"Running {cfg.label}...")
        r = run_scenario(df, cfg)
        results.append((cfg, r))
        restore_all_modules()

    g_cfg = build_scenario_g(results)
    print(f"\nRunning {g_cfg.label}...")
    g_result = run_scenario(df, g_cfg)
    results.append((g_cfg, g_result))
    restore_all_modules()

    print("\n" + "=" * 118)
    print("STRUCTURAL SCENARIO COMPARISON (100-day compound, monkey-patched)")
    print("=" * 118)
    header = (
        f"{'Scenario':<42} {'Trades':>6} {'WR%':>5} {'PF':>5} "
        f"{'Final Cap':>12} {'MaxDD%':>6} {'CDGR%':>6} {'Active':>6} "
        f"{'T/Day':>5} {'OK':>3}"
    )
    print(header)
    print("-" * 118)
    for _, r in results:
        print(format_table_row(r))

    target_hits = [r for _, r in results if meets_target(r)]
    eligible = [r for _, r in results if meets_risk_gates(r)]
    best = max(eligible, key=lambda x: x["cdgr"]) if eligible else None

    print("\n" + "=" * 118)
    print("TARGET: CDGR >= 8%, PF >= 2.0, Max DD <= 40%")
    print("=" * 118)
    if target_hits:
        print(f"✓ {len(target_hits)} scenario(s) met all gates:")
        for r in sorted(target_hits, key=lambda x: -x["cdgr"]):
            print(
                f"  {r['scenario']}: CDGR {r['cdgr'] * 100:.2f}%, "
                f"PF {r['profit_factor']:.2f}, DD {r['max_drawdown_pct']:.1f}%, "
                f"{r['total_trades']} trades"
            )
    else:
        print("✗ No scenario reached 8% CDGR with PF >= 2.0 and DD <= 40%.")
        if best:
            print(
                f"\n  Closest (risk gates OK): {best['scenario']}\n"
                f"    CDGR: {best['cdgr'] * 100:.2f}% "
                f"(gap {(0.08 - best['cdgr']) * 100:.2f} pp)\n"
                f"    PF: {best['profit_factor']:.2f} | "
                f"WR: {best['win_rate']:.1f}% | "
                f"Max DD: {best['max_drawdown_pct']:.1f}%\n"
                f"    Trades: {best['total_trades']} | "
                f"Active days: {best['active_trading_days']} | "
                f"Trades/active day: "
                f"{best['total_trades'] / max(best['active_trading_days'], 1):.1f}"
            )

    restore_all_modules()


if __name__ == "__main__":
    main()
