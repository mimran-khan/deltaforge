#!/usr/bin/env python3
"""100-day replay: baseline vs regime-conditional FADE mode (Rs 10k).

Fade regime (intraday only, hysteresis):
  Enter when fade_score >= 3, exit when <= 1.
  Score components (each +1):
    - close below session TWAP
    - lower highs after morning peak
    - >= 40 pts below session high
    - LTF RSI < 50 AND HTF RSI > 60
    - ADX rising vs 2 bars ago AND price lower vs 2 bars ago

When fade_active:
  - Block PULLBACK LONG and STOCH_CROSS LONG only
  - All other strategies / SHORT signals unchanged
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs, run_compound_backtest
from engine.premium_model import STRATEGY_SL_PCT, create_premium_state
from engine.multi_strategy_engine import MultiStrategyEngine
from risk.adaptive_mode import AdaptiveModeController


@dataclass
class FadeConfig:
    enabled: bool = False
    global_sl_cooldown_bars: int = 0
    fade_enter_score: int = 3
    fade_exit_score: int = 1
    pts_below_high: float = 40.0

    @property
    def label(self) -> str:
        if not self.enabled and not self.global_sl_cooldown_bars:
            return "BASELINE"
        parts = []
        if self.enabled:
            parts.append("FADE_REGIME")
        if self.global_sl_cooldown_bars:
            parts.append(f"SLcd{self.global_sl_cooldown_bars}")
        return "+".join(parts)


class DayState:
    def __init__(self):
        self.session_high = 0.0
        self.twap_sum = 0.0
        self.bar_count = 0
        self.last_swing_high = 0.0
        self.lower_highs = False
        self.last_sl_bar: int | None = None
        self.fade_active = False
        self.fade_bars = 0
        self.blocked_fade: list[dict] = []

    def update_bar(self, high: float, close: float):
        self.bar_count += 1
        self.twap_sum += (high + close + close) / 3
        self.session_high = max(self.session_high, high)
        if high >= self.session_high - 0.01:
            self.last_swing_high = high
        elif self.last_swing_high > 0 and high < self.last_swing_high:
            self.lower_highs = True

    def twap(self) -> float:
        return self.twap_sum / self.bar_count if self.bar_count else 0.0

    def update_fade_regime(
        self,
        cfg: FadeConfig,
        close: float,
        ltf_rsi: float,
        htf_rsi: float,
        adx: float,
        adx_prev: float,
        close_prev: float,
    ):
        if not cfg.enabled:
            return
        score = 0
        twap = self.twap()
        if twap > 0 and close < twap:
            score += 1
        if self.lower_highs:
            score += 1
        if self.session_high - close >= cfg.pts_below_high:
            score += 1
        if ltf_rsi < 50 and htf_rsi > 60:
            score += 1
        if adx > adx_prev and close < close_prev:
            score += 1

        if not self.fade_active and score >= cfg.fade_enter_score:
            self.fade_active = True
        elif self.fade_active and score <= cfg.fade_exit_score:
            self.fade_active = False
        if self.fade_active:
            self.fade_bars += 1


def _entry_blocked(cfg: FadeConfig, day: DayState, bar_i: int, signal) -> str | None:
    if cfg.global_sl_cooldown_bars and day.last_sl_bar is not None:
        if bar_i - day.last_sl_bar < cfg.global_sl_cooldown_bars:
            return "global SL cooldown"

    if cfg.enabled and day.fade_active:
        if signal.signal_type in ("PULLBACK", "STOCH_CROSS") and signal.direction == "LONG":
            return "fade regime blocks dip-buy LONG"
    return None


def run_backtest(df: pd.DataFrame, cfg: FadeConfig, starting_capital: float = 10_000) -> dict:
    lot_size = settings.NIFTY_LOT_SIZE
    engine = MultiStrategyEngine()
    adaptive = AdaptiveModeController()
    capital = starting_capital
    peak = capital
    trades: list[dict] = []
    equity_curve: list[dict] = []
    fade_days: set = set()
    blocked: list[dict] = []

    unique_days = sorted(set(df.index.date))
    warmup_n = 5

    for day_idx, day in enumerate(unique_days):
        day_st = DayState()
        day_df = df[df.index.date == day].copy()
        if len(day_df) < 10:
            equity_curve.append({"date": day, "capital": capital, "daily_pnl": 0, "trades": 0})
            continue

        prev_day_data = None
        if day_idx > 0:
            prev_df = df[df.index.date == unique_days[day_idx - 1]]
            if len(prev_df) > 0:
                prev_day_data = {
                    "high": prev_df["high"].max(),
                    "low": prev_df["low"].min(),
                    "close": prev_df["close"].iloc[-1],
                }
        engine.reset_day(prev_day_data)
        adaptive.reset()

        day_start = capital
        day_pnl = 0.0
        day_trades = 0
        day_wins = day_losses = consec_loss = 0
        daily_loss_limit = day_start * (settings.DAILY_LOSS_LIMIT_PCT / 100)
        daily_profit_target = day_start * (getattr(settings, "DAILY_PROFIT_TARGET_PCT", 35) / 100)
        day_lots = max(1, min(int(capital / getattr(settings, "CAPITAL_PER_LOT", 10_000)),
                              getattr(settings, "MAX_LOTS_CAP", 10)))

        warmup_start = max(0, day_idx - warmup_n)
        warmup_df = df[df.index.map(lambda t: t.date() in set(unique_days[warmup_start:day_idx + 1]))]
        indicators = engine.precompute(warmup_df)
        today_idx = [i for i, ts in enumerate(warmup_df.index) if ts.date() == day]
        open_pos: list[dict] = []

        for i in today_idx:
            if i < 10:
                continue
            ts = warmup_df.index[i]
            time_str = ts.strftime("%H:%M")
            h = float(warmup_df["high"].iloc[i])
            c = float(warmup_df["close"].iloc[i])
            day_st.update_bar(h, c)

            rsi5 = float(engine._sv(indicators["rsi_5m"], i, 50))
            htf = float(engine._htf_rsi(indicators, i, 50))
            adx = float(engine._sv(indicators.get("adx", pd.Series()), i, 0))
            adx_prev = float(engine._sv(indicators.get("adx", pd.Series()), i - 2, adx))
            close_prev = float(warmup_df["close"].iloc[i - 2]) if i >= 2 else c
            day_st.update_fade_regime(cfg, c, rsi5, htf, adx, adx_prev, close_prev)
            if day_st.fade_active:
                fade_days.add(day)

            adaptive.on_bar()
            ap = adaptive.profile

            closed = []
            for pos in open_pos:
                pos["candles_held"] += 1
                cur = pos["prem_state"].current_premium(c, pos["candles_held"])
                if cur > pos["peak_premium"]:
                    pos["peak_premium"] = cur
                trail = pos["prem_state"].update_trail(
                    cur, ap.trail_trigger_pct, ap.trail_pct
                )
                exit_reason = exit_prem = None
                if cur <= pos["sl_premium"]:
                    exit_reason, exit_prem = "SL", pos["sl_premium"]
                elif cur >= pos["prem_state"].target_premium:
                    exit_reason, exit_prem = "TGT", pos["prem_state"].target_premium
                elif trail is not None and cur <= trail:
                    exit_reason, exit_prem = "TRAIL", trail
                elif pos["candles_held"] >= settings.PULLBACK_HOLD_CANDLES:
                    exit_reason, exit_prem = "TIME", cur
                elif time_str >= settings.SQUARE_OFF_TIME:
                    exit_reason, exit_prem = "EOD", cur

                if exit_reason:
                    if exit_reason == "SL":
                        engine.record_sl_exit(pos["signal_type"], i)
                        day_st.last_sl_bar = i
                    costs = _calc_realistic_costs(pos["entry_premium"], exit_prem, pos["qty"], day_lots)
                    net = (exit_prem - pos["entry_premium"]) * pos["qty"] - costs
                    capital += net
                    day_pnl += net
                    day_trades += 1
                    won = net > 0
                    day_wins += int(won)
                    day_losses += int(not won)
                    consec_loss = 0 if won else consec_loss + 1
                    adaptive.update(
                        daily_pnl_pct=(day_pnl / day_start * 100) if day_start else 0,
                        wins=day_wins, losses=day_losses,
                        consecutive_losses=consec_loss, trades=day_trades,
                        last_trade_won=won,
                    )
                    trades.append({
                        "date": day, "time": time_str,
                        "strategy": pos["signal_type"], "direction": pos["direction"],
                        "pnl": round(net, 0), "reason": exit_reason,
                        "fade_active": pos.get("fade_active", False),
                    })
                    closed.append(pos)
            for p in closed:
                open_pos.remove(p)

            if len(open_pos) >= ap.max_simultaneous:
                continue
            if consec_loss >= settings.MAX_CONSECUTIVE_LOSSES:
                continue
            if day_pnl < 0 and abs(day_pnl) >= daily_loss_limit:
                continue
            if day_pnl >= daily_profit_target:
                continue
            if adaptive.mode.value == "HALT":
                continue
            if day_trades >= ap.max_trades_per_day:
                continue

            open_dirs = {p["direction"] for p in open_pos}
            signals = engine.scan(indicators, i, time_str, max_total_override=ap.max_trades_per_day)
            for sig in signals:
                if sig.direction in open_dirs:
                    continue
                if sig.confidence < ap.min_confidence:
                    continue
                reason = _entry_blocked(cfg, day_st, i, sig)
                if reason:
                    blocked.append({
                        "date": day, "time": time_str,
                        "strategy": sig.signal_type, "direction": sig.direction,
                        "reason": reason, "fade_active": day_st.fade_active,
                    })
                    continue

                theta = settings.get_scaled_theta(c)
                prem = create_premium_state(
                    entry_index_price=c, direction=sig.direction,
                    base_premium=settings.PREMIUM_BASE, delta=settings.PREMIUM_DELTA,
                    theta_per_candle=theta, sl_pct=settings.PREMIUM_SL_PCT,
                    confluence_score=sig.confidence, signal_type=sig.signal_type,
                )
                spread = getattr(settings, "BID_ASK_SPREAD", 0.30)
                entry_prem = prem.entry_premium + spread
                eff_sl = STRATEGY_SL_PCT.get(sig.signal_type, settings.PREMIUM_SL_PCT)
                sl_prem = entry_prem * (1 - eff_sl / 100)
                eff_lots = max(1, int(day_lots * ap.lot_multiplier))
                open_pos.append({
                    "direction": sig.direction, "signal_type": sig.signal_type,
                    "entry_premium": entry_prem, "sl_premium": sl_prem,
                    "qty": eff_lots * lot_size, "prem_state": prem,
                    "candles_held": 0, "peak_premium": entry_prem,
                    "fade_active": day_st.fade_active,
                })
                break

        for pos in open_pos:
            li = today_idx[-1]
            c = float(warmup_df["close"].iloc[li])
            exit_prem = pos["prem_state"].current_premium(c, pos["candles_held"])
            costs = getattr(settings, "BROKERAGE_PER_ORDER", 20) * 2
            costs += exit_prem * pos["qty"] * getattr(settings, "STT_PCT", 0.0125) / 100
            costs += getattr(settings, "SLIPPAGE_POINTS", 0.5) * pos["qty"]
            net = (exit_prem - pos["entry_premium"]) * pos["qty"] - costs
            capital += net
            day_pnl += net
            day_trades += 1
            trades.append({
                "date": day, "time": "EOD",
                "strategy": pos["signal_type"], "direction": pos["direction"],
                "pnl": round(net, 0), "reason": "EOD",
                "fade_active": pos.get("fade_active", False),
            })

        peak = max(peak, capital)
        equity_curve.append({
            "date": day, "capital": round(capital, 0),
            "daily_pnl": round(day_pnl, 0), "trades": day_trades,
            "fade_bars": day_st.fade_bars,
        })

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    eq = [e["capital"] for e in equity_curve] or [starting_capital]
    pk = np.maximum.accumulate(eq)
    max_dd = max((p - v) / p * 100 for p, v in zip(pk, eq)) if eq else 0

    return {
        "label": cfg.label,
        "final_capital": round(capital, 0),
        "total_pnl": round(capital - starting_capital, 0),
        "return_pct": round((capital - starting_capital) / starting_capital * 100, 1),
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "max_drawdown_pct": round(max_dd, 1),
        "blocked": len(blocked),
        "fade_days": len(fade_days),
        "trades": trades,
        "blocked_log": blocked,
        "equity_curve": equity_curve,
    }


def print_table(results: list[dict], days: int):
    print("\n" + "=" * 95)
    print(f"  FADE REGIME REPLAY — {days} days | Rs 10,000 start")
    print("=" * 95)
    print(f"{'Scenario':<24} {'Final':>11} {'P&L':>11} {'Trades':>7} {'WR%':>6} {'PF':>5} {'MaxDD':>7} {'Blocked':>8} {'FadeDays':>8}")
    print("-" * 95)
    for r in results:
        pnl = r["total_pnl"]
        s = "+" if pnl >= 0 else ""
        print(
            f"{r['label']:<24} Rs{r['final_capital']:>8,.0f} {s}Rs{pnl:>8,.0f} "
            f"{r['total_trades']:>7} {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
            f"{r['max_drawdown_pct']:>6.1f}% {r['blocked']:>8} {r['fade_days']:>8}"
        )
    base = results[0]
    for r in results[1:]:
        d = r["final_capital"] - base["final_capital"]
        print(f"\n  {r['label']} vs BASELINE: Rs {d:+,.0f} final | {r['total_trades']-base['total_trades']:+d} trades | {r['blocked']} blocked")


def analyze_day(results: list[dict], target):
    from datetime import date
    if isinstance(target, str):
        target = date.fromisoformat(target)
    print(f"\n--- Day drill-down: {target} ---")
    for r in results:
        day_trades = [t for t in r["trades"] if t["date"] == target]
        day_pnl = sum(t["pnl"] for t in day_trades)
        blocked = [b for b in r.get("blocked_log", []) if b["date"] == target]
        print(f"  {r['label']:<22} PnL Rs {day_pnl:>8,.0f} | trades {len(day_trades)} | blocked {len(blocked)}")
        for t in day_trades:
            print(f"    {t.get('time','?')} {t['strategy']:12} {t['direction']:5} Rs {t['pnl']:>7,.0f} {t['reason']}")
        for b in blocked:
            print(f"    BLOCK {b['time']} {b['strategy']:12} {b['direction']:5} — {b['reason']}")


def main():
    days = 100
    cap = 10_000
    print(f"Loading {days} days...")
    df = load_real_data(days=days)
    udays = sorted(set(df.index.date))
    print(f"  {len(df):,} bars | {udays[0]} → {udays[-1]}")

    print("  BASELINE (canonical)...")
    raw = run_compound_backtest(df, starting_capital=cap)
    baseline = {
        "label": "BASELINE", "final_capital": raw["final_capital"],
        "total_pnl": raw["total_pnl"], "return_pct": raw["return_pct"],
        "total_trades": raw["total_trades"], "win_rate": raw["win_rate"],
        "profit_factor": raw["profit_factor"], "max_drawdown_pct": raw["max_drawdown_pct"],
        "blocked": 0, "fade_days": 0,
        "trades": [{"date": t["entry_time"].date() if hasattr(t["entry_time"], "date") else t.get("date"),
                    "strategy": t["strategy"], "direction": t["signal"],
                    "pnl": t["pnl"], "reason": t["reason"], "time": ""} for t in raw["trades"]],
        "blocked_log": [],
    }

    scenarios = [
        FadeConfig(enabled=True),
        FadeConfig(enabled=True, global_sl_cooldown_bars=6),
        FadeConfig(global_sl_cooldown_bars=6),  # SL only, no fade
    ]
    results = [baseline]
    for sc in scenarios:
        print(f"  Running {sc.label}...")
        results.append(run_backtest(df, sc, cap))

    print_table(results, len(udays))

    # Jun 12 big win day + check if any day like fade
    analyze_day(results, "2026-06-12")
    if udays[-1] >= __import__("datetime").date(2026, 6, 15):
        analyze_day(results, "2026-06-15")

    # Sample: trades blocked only in fade regime
    fade_r = next(r for r in results if r["label"] == "FADE_REGIME")
    fade_blocks = fade_r["blocked_log"]
    pb_long = sum(1 for b in fade_blocks if b["strategy"] in ("PULLBACK", "STOCH_CROSS") and b["direction"] == "LONG")
    short_allowed = [t for t in fade_r["trades"] if t.get("fade_active") and t["direction"] == "SHORT"]
    print(f"\n  Fade regime blocked {pb_long} dip-buy LONGs over 100d")
    print(f"  SHORT trades taken during fade regime: {len(short_allowed)}")
    if short_allowed:
        print(f"  Fade SHORT PnL: Rs {sum(t['pnl'] for t in short_allowed):,.0f}")

    out = ROOT / "tmp" / "fade_regime_results.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("trades", "blocked_log", "equity_curve")} for r in results]).to_csv(out, index=False)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
