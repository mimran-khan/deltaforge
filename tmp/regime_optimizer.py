#!/usr/bin/env python3
"""Grid search: regime-conditional policies to profit on fade days without baseline drag.

Inspired by web research:
  - VWAP mean reversion only on rotational days (ADX < 25) [CrossTrade, TraderVerdict]
  - Block dip-buy LONG in fade; enable mean-rev strategies in fade only
  - Post-SL cooldown hybrid; time gate after morning drive
  - Tighter fade entry (score 4+) to reduce false positives on trend days
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from backtest.run_backtest import load_real_data, _calc_realistic_costs, run_compound_backtest
from engine.premium_model import STRATEGY_SL_PCT, create_premium_state
from engine.multi_strategy_engine import MultiStrategyEngine
from risk.adaptive_mode import AdaptiveModeController

DIP_BUY = frozenset({"PULLBACK", "STOCH_CROSS"})
MEAN_REV = frozenset({"VWAP_MEAN_REV", "RSI_REVERSION"})
TREND_STRATS = frozenset({"TREND_RIDE", "CPR_BREAKOUT", "ADX_BREAKOUT"})


@dataclass
class Policy:
  name: str = "baseline"
  fade_enabled: bool = False
  fade_enter: int = 3
  fade_exit: int = 1
  pts_below_high: float = 40.0
  adx_max_fade: float | None = None          # require ADX < X to enter fade
  time_gate_after: str | None = None         # e.g. "11:00"
  block_dip_long: bool = False               # block PULLBACK/STOCH LONG in fade
  block_stoch_only: bool = False             # block only STOCH_CROSS LONG in fade
  block_pullback_only: bool = False          # block only PULLBACK LONG in fade
  block_dip_after_sl: bool = False           # block dip LONG only after any SL today
  enable_mr_in_fade: bool = False            # allow VWAP_MEAN_REV + RSI_REVERSION in fade
  disable_mr_outside_fade: bool = False
  block_trend_in_fade: bool = False          # block TREND_RIDE etc in fade
  short_conf_boost: int = 0                  # +confidence for SHORT in fade
  sl_cooldown_bars: int = 0
  require_lower_highs: bool = False          # fade needs LH for block to apply


class DayState:
  def __init__(self):
    self.session_high = 0.0
    self.twap_sum = 0.0
    self.bar_count = 0
    self.last_swing_high = 0.0
    self.lower_highs = False
    self.last_sl_bar: int | None = None
    self.had_sl_today = False
    self.fade_active = False
    self.fade_score = 0

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

  def update_fade(self, pol: Policy, close: float, ltf: float, htf: float,
                  adx: float, adx_prev: float, close_prev: float):
    if not pol.fade_enabled:
      self.fade_active = False
      self.fade_score = 0
      return
    if pol.adx_max_fade is not None and adx > pol.adx_max_fade:
      self.fade_active = False
      self.fade_score = 0
      return
    score = 0
    twap = self.twap()
    if twap > 0 and close < twap:
      score += 1
    if self.lower_highs:
      score += 1
    if self.session_high - close >= pol.pts_below_high:
      score += 1
    if ltf < 50 and htf > 60:
      score += 1
    if adx > adx_prev and close < close_prev:
      score += 1
    self.fade_score = score
    if not self.fade_active and score >= pol.fade_enter:
      self.fade_active = True
    elif self.fade_active and score <= pol.fade_exit:
      self.fade_active = False


def _strategy_allowed(pol: Policy, day: DayState, sig, time_str: str) -> bool:
  """Return True if strategy is allowed (not disabled by regime policy)."""
  st = sig.signal_type
  fade = day.fade_active
  after_gate = (pol.time_gate_after is None) or (time_str >= pol.time_gate_after)

  if pol.block_dip_after_sl and day.had_sl_today and st in DIP_BUY and sig.direction == "LONG":
    return False
  if pol.block_stoch_only and fade and after_gate and st == "STOCH_CROSS" and sig.direction == "LONG":
    return False
  if pol.block_pullback_only and fade and after_gate and st == "PULLBACK" and sig.direction == "LONG":
    return False
  if pol.block_dip_long and fade and after_gate and st in DIP_BUY and sig.direction == "LONG":
    if pol.require_lower_highs and not day.lower_highs:
      pass
    else:
      return False
  if pol.block_trend_in_fade and fade and st in TREND_STRATS:
    return False
  if pol.disable_mr_outside_fade and st in MEAN_REV and not fade:
    return False
  if pol.enable_mr_in_fade and st in MEAN_REV and not fade:
    return False
  return True


def _entry_blocked(pol: Policy, day: DayState, bar_i: int, sig, time_str: str) -> str | None:
  if pol.sl_cooldown_bars and day.last_sl_bar is not None:
    if bar_i - day.last_sl_bar < pol.sl_cooldown_bars:
      return "SL cooldown"
  if not _strategy_allowed(pol, day, sig, time_str):
    if pol.block_dip_long and day.fade_active and sig.signal_type in DIP_BUY:
      return "fade blocks dip LONG"
    if pol.block_dip_after_sl and day.had_sl_today and sig.signal_type in DIP_BUY:
      return "post-SL blocks dip LONG"
    if pol.block_trend_in_fade and day.fade_active:
      return "fade blocks trend strat"
    if pol.enable_mr_in_fade and sig.signal_type in MEAN_REV and not day.fade_active:
      return "MR only in fade"
  return None


def _patch_engine_disabled(engine: MultiStrategyEngine, pol: Policy):
  """Temporarily adjust DISABLED_STRATEGIES for mean-rev enablement."""
  base_disabled = set(MultiStrategyEngine.DISABLED_STRATEGIES)
  if pol.enable_mr_in_fade:
    base_disabled -= MEAN_REV
  engine.DISABLED_STRATEGIES = base_disabled


def run_policy(df: pd.DataFrame, pol: Policy, starting_capital: float = 10_000) -> dict:
  lot_size = settings.NIFTY_LOT_SIZE
  engine = MultiStrategyEngine()
  _patch_engine_disabled(engine, pol)
  adaptive = AdaptiveModeController()
  capital = starting_capital
  trades: list[dict] = []
  daily: list[dict] = []
  blocked = 0
  fade_days: set = set()

  unique_days = sorted(set(df.index.date))
  warmup_n = 5

  for day_idx, day in enumerate(unique_days):
    day_st = DayState()
    day_df = df[df.index.date == day]
    if len(day_df) < 10:
      daily.append({"date": day, "pnl": 0, "fade": False, "fade_pnl": 0, "trend_pnl": 0})
      continue

    prev_day_data = None
    if day_idx > 0:
      prev_df = df[df.index.date == unique_days[day_idx - 1]]
      if len(prev_df) > 0:
        prev_day_data = {"high": prev_df["high"].max(), "low": prev_df["low"].min(),
                         "close": prev_df["close"].iloc[-1]}
    engine.reset_day(prev_day_data)
    adaptive.reset()
    _patch_engine_disabled(engine, pol)

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
    day_fade_pnl = 0.0
    day_trend_pnl = 0.0
    day_was_fade = False

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
      day_st.update_fade(pol, c, rsi5, htf, adx, adx_prev, close_prev)
      if day_st.fade_active:
        day_was_fade = True
        fade_days.add(day)

      adaptive.on_bar()
      ap = adaptive.profile
      closed = []
      for pos in open_pos:
        pos["candles_held"] += 1
        cur = pos["prem_state"].current_premium(c, pos["candles_held"])
        if cur > pos["peak_premium"]:
          pos["peak_premium"] = cur
        trail = pos["prem_state"].update_trail(cur, ap.trail_trigger_pct, ap.trail_pct)
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
            day_st.had_sl_today = True
          costs = _calc_realistic_costs(pos["entry_premium"], exit_prem, pos["qty"], day_lots)
          net = (exit_prem - pos["entry_premium"]) * pos["qty"] - costs
          capital += net
          day_pnl += net
          if pos.get("fade_active"):
            day_fade_pnl += net
          else:
            day_trend_pnl += net
          day_trades += 1
          won = net > 0
          day_wins += int(won)
          day_losses += int(not won)
          consec_loss = 0 if won else consec_loss + 1
          adaptive.update(
            daily_pnl_pct=(day_pnl / day_start * 100) if day_start else 0,
            wins=day_wins, losses=day_losses,
            consecutive_losses=consec_loss, trades=day_trades, last_trade_won=won,
          )
          trades.append({"date": day, "pnl": net, "fade": pos.get("fade_active", False),
                         "strategy": pos["signal_type"], "dir": pos["direction"]})
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
        conf = sig.confidence
        if pol.short_conf_boost and day_st.fade_active and sig.direction == "SHORT":
          conf = min(conf + pol.short_conf_boost, 100)
        if conf < ap.min_confidence:
          continue
        reason = _entry_blocked(pol, day_st, i, sig, time_str)
        if reason:
          blocked += 1
          continue
        if not _strategy_allowed(pol, day_st, sig, time_str):
          blocked += 1
          continue

        theta = settings.get_scaled_theta(c)
        prem = create_premium_state(
          entry_index_price=c, direction=sig.direction,
          base_premium=settings.PREMIUM_BASE, delta=settings.PREMIUM_DELTA,
          theta_per_candle=theta, sl_pct=settings.PREMIUM_SL_PCT,
          confluence_score=conf, signal_type=sig.signal_type,
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
        # bump counter for MR strategies
        if sig.signal_type == "VWAP_MEAN_REV":
          engine._vwap_mr_count += 1
        elif sig.signal_type == "RSI_REVERSION":
          engine._rsi_rev_count += 1
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
      if pos.get("fade_active"):
        day_fade_pnl += net
      else:
        day_trend_pnl += net
      trades.append({"date": day, "pnl": net, "fade": pos.get("fade_active", False),
                     "strategy": pos["signal_type"], "dir": pos["direction"]})

    daily.append({
      "date": day, "pnl": round(day_pnl, 0), "fade": day_was_fade,
      "fade_pnl": round(day_fade_pnl, 0), "trend_pnl": round(day_trend_pnl, 0),
    })

  wins = [t for t in trades if t["pnl"] > 0]
  losses = [t for t in trades if t["pnl"] <= 0]
  gp = sum(t["pnl"] for t in wins)
  gl = abs(sum(t["pnl"] for t in losses))
  fade_day_pnl = sum(d["pnl"] for d in daily if d["fade"])
  non_fade_pnl = sum(d["pnl"] for d in daily if not d["fade"])
  dip_on_fade_baseline = 0  # filled by caller

  return {
    "name": pol.name,
    "final": round(capital, 0),
    "pnl": round(capital - starting_capital, 0),
    "trades": len(trades),
    "wr": round(len(wins) / len(trades) * 100, 1) if trades else 0,
    "pf": round(gp / gl, 2) if gl > 0 else 99.0,
    "blocked": blocked,
    "fade_days": len(fade_days),
    "fade_day_pnl": round(fade_day_pnl, 0),
    "non_fade_pnl": round(non_fade_pnl, 0),
    "daily": daily,
    "trades_list": trades,
  }


def build_policies() -> list[Policy]:
  policies: list[Policy] = []

  def add(name: str, **kw):
    policies.append(Policy(name=name, **kw))

  # --- Phase A: post-SL only (minimal drag) ---
  add("postSL_only", block_dip_after_sl=True)
  add("slcd6_only", sl_cooldown_bars=6)
  add("postSL+slcd6", block_dip_after_sl=True, sl_cooldown_bars=6)

  # --- Phase B: fade block with tighter entry ---
  for enter in (4, 5):
    add(f"fade_blk_e{enter}", fade_enabled=True, fade_enter=enter, block_dip_long=True)
    add(f"fade_blk_LH_e{enter}", fade_enabled=True, fade_enter=enter,
        block_dip_long=True, require_lower_highs=True)
    add(f"fade_blk_t11_e{enter}", fade_enabled=True, fade_enter=enter,
        block_dip_long=True, time_gate_after="11:00")

  # --- Phase C: ADX-gated (web: MR when ADX < 25) ---
  for adx in (25, 28):
    add(f"adx{adx}_blk", fade_enabled=True, fade_enter=4, adx_max_fade=adx, block_dip_long=True)
    add(f"adx{adx}_mr", fade_enabled=True, fade_enter=4, adx_max_fade=adx,
        block_dip_long=True, enable_mr_in_fade=True)

  # --- Phase D: mean-rev in fade (VWAP 2σ + RSI reversion) ---
  add("mr_fade_e3", fade_enabled=True, fade_enter=3, block_dip_long=True, enable_mr_in_fade=True)
  add("mr_fade_e4", fade_enabled=True, fade_enter=4, block_dip_long=True, enable_mr_in_fade=True)
  add("mr_fade_adx25", fade_enabled=True, fade_enter=4, adx_max_fade=25,
      block_dip_long=True, enable_mr_in_fade=True)

  # --- Phase E: hybrids from web research ---
  add("hybrid_v1", fade_enabled=True, fade_enter=4, adx_max_fade=25, pts_below_high=40,
      block_dip_long=True, enable_mr_in_fade=True, time_gate_after="11:00", sl_cooldown_bars=6)
  add("hybrid_v2", fade_enabled=True, fade_enter=4, adx_max_fade=25, pts_below_high=50,
      block_dip_long=True, enable_mr_in_fade=True, require_lower_highs=True, sl_cooldown_bars=6)
  add("hybrid_v3", fade_enabled=True, fade_enter=5, adx_max_fade=25,
      block_dip_long=True, enable_mr_in_fade=True, short_conf_boost=10)
  add("hybrid_v4", fade_enabled=True, fade_enter=4, adx_max_fade=28,
      block_dip_after_sl=True, enable_mr_in_fade=True, sl_cooldown_bars=6)
  add("hybrid_v5", fade_enabled=True, fade_enter=4, adx_max_fade=25,
      block_dip_after_sl=True, enable_mr_in_fade=True, sl_cooldown_bars=6)
  add("hybrid_v6", fade_enabled=True, fade_enter=4, adx_max_fade=25,
      block_dip_long=True, enable_mr_in_fade=True, block_trend_in_fade=True, sl_cooldown_bars=6)

  return policies


def main():
  days = 100
  cap = 10_000
  print(f"Loading {days} days...")
  df = load_real_data(days=days)
  udays = sorted(set(df.index.date))
  print(f"  {len(df):,} bars | {udays[0]} → {udays[-1]}")

  print("Running BASELINE...")
  raw = run_compound_backtest(df, starting_capital=cap)
  baseline = {
    "name": "BASELINE", "final": raw["final_capital"], "pnl": raw["total_pnl"],
    "trades": raw["total_trades"], "wr": raw["win_rate"], "pf": raw["profit_factor"],
    "blocked": 0, "fade_days": 0, "fade_day_pnl": 0, "non_fade_pnl": raw["total_pnl"],
    "daily": [], "trades_list": [],
  }

  # Classify fade days from baseline dip-buy losses
  baseline_policy = run_policy(df, Policy(name="baseline_ref"), cap)
  fade_days_set = {d["date"] for d in baseline_policy["daily"] if d["fade"]}
  baseline_fade_pnl = sum(d["pnl"] for d in baseline_policy["daily"] if d["fade"])
  baseline_non_fade = sum(d["pnl"] for d in baseline_policy["daily"] if not d["fade"])
  print(f"  Baseline fade-day PnL (engine fade detect): Rs {baseline_fade_pnl:,.0f} on {len(fade_days_set)} days")

  policies = build_policies()
  results = [baseline]
  for i, pol in enumerate(policies):
    print(f"  [{i+1}/{len(policies)}] {pol.name}...")
    results.append(run_policy(df, pol, cap))

  # Score: prefer final >= baseline-1% AND improved fade-day PnL
  tol = baseline["final"] * 0.01
  scored = []
  for r in results[1:]:
    delta_final = r["final"] - baseline["final"]
    delta_fade = r["fade_day_pnl"] - baseline_fade_pnl
    # objective: maximize fade improvement with capital constraint
    capital_ok = delta_final >= -tol
    scored.append((r, delta_final, delta_fade, capital_ok))

  # Sort: capital_ok first, then fade improvement, then final delta
  scored.sort(key=lambda x: (x[3], x[2], x[1]), reverse=True)

  print("\n" + "=" * 110)
  print("  TOP 15 POLICIES (capital within 1% of baseline, best fade-day improvement)")
  print("=" * 110)
  print(f"{'Policy':<28} {'Final':>10} {'ΔFinal':>10} {'FadePnL':>10} {'ΔFade':>10} {'NonFade':>10} {'PF':>5} {'Blk':>5} {'OK':>3}")
  print("-" * 110)
  shown = 0
  for r, dfinal, dfade, ok in scored:
    if not ok and shown >= 5:
      continue
    flag = "✓" if ok else "✗"
    print(f"{r['name']:<28} Rs{r['final']:>7,.0f} {dfinal:>+9,.0f} Rs{r['fade_day_pnl']:>7,.0f} "
          f"{dfade:>+9,.0f} Rs{r['non_fade_pnl']:>7,.0f} {r['pf']:>5.2f} {r['blocked']:>5} {flag:>3}")
    shown += 1
    if shown >= 15:
      break

  # Best overall regardless of capital
  scored_all = sorted(scored, key=lambda x: (x[2] + x[1] * 0.3), reverse=True)
  best = scored_all[0]
  print(f"\n  BEST BALANCED: {best[0]['name']}")
  print(f"    Final Rs {best[0]['final']:,.0f} ({best[1]:+,.0f} vs baseline)")
  print(f"    Fade-day PnL Rs {best[0]['fade_day_pnl']:,.0f} ({best[2]:+,.0f} vs baseline fade days)")

  # Policies that beat baseline on BOTH metrics
  winners = [x for x in scored if x[3] and x[2] > 0]
  print(f"\n  Policies beating baseline on fade days AND within 1% capital: {len(winners)}")
  for r, dfinal, dfade, ok in winners[:10]:
    print(f"    {r['name']}: final {dfinal:+,.0f}, fade {dfade:+,.0f}")

  # Save full results
  rows = []
  for r in results:
    rows.append({k: v for k, v in r.items() if k not in ("daily", "trades_list")})
  out = ROOT / "tmp" / "regime_optimizer_results.csv"
  pd.DataFrame(rows).to_csv(out, index=False)
  print(f"\n  Saved: {out}")

  # Day-level comparison for top 3
  top3 = [x[0] for x in scored[:3]]
  if baseline_policy["daily"]:
    losing_fade_days = [d for d in baseline_policy["daily"] if d["fade"] and d["pnl"] < 0]
    print(f"\n  Baseline losing fade days: {len(losing_fade_days)}")
    for pol_r in top3[:1]:
      pol_daily = {d["date"]: d for d in next(x for x in results if x["name"] == pol_r["name"])["daily"]}
      fixed = 0
      for ld in losing_fade_days:
        pd_ = pol_daily.get(ld["date"], {})
        if pd_.get("pnl", ld["pnl"]) > ld["pnl"]:
          fixed += 1
      print(f"  {pol_r['name']} improved {fixed}/{len(losing_fade_days)} losing fade days")


if __name__ == "__main__":
  main()
