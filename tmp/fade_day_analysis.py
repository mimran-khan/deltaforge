#!/usr/bin/env python3
"""Round 2: objective fade-day analysis + refined policy search.

Classifies fade days once (score>=3), compares day PnL vs baseline on those days.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tmp.regime_optimizer import (
  Policy, DayState, run_policy, load_real_data, run_compound_backtest,
)


def classify_fade_days(df) -> dict:
  """Return {date: {fade_bars, max_score, lower_highs}} per day."""
  from engine.multi_strategy_engine import MultiStrategyEngine
  engine = MultiStrategyEngine()
  unique_days = sorted(set(df.index.date))
  warmup_n = 5
  fade_days = {}
  for day_idx, day in enumerate(unique_days):
    day_st = DayState()
    warmup_start = max(0, day_idx - warmup_n)
    warmup_df = df[df.index.map(lambda t: t.date() in set(unique_days[warmup_start:day_idx + 1]))]
    indicators = engine.precompute(warmup_df)
    today_idx = [i for i, ts in enumerate(warmup_df.index) if ts.date() == day]
    max_score = 0
    fade_bars = 0
    was_fade = False
    for i in today_idx:
      if i < 10:
        continue
      h = float(warmup_df["high"].iloc[i])
      c = float(warmup_df["close"].iloc[i])
      day_st.update_bar(h, c)
      rsi5 = float(engine._sv(indicators["rsi_5m"], i, 50))
      htf = float(engine._htf_rsi(indicators, i, 50))
      adx = float(engine._sv(indicators.get("adx", pd.Series()), i, 0))
      adx_prev = float(engine._sv(indicators.get("adx", pd.Series()), i - 2, adx))
      close_prev = float(warmup_df["close"].iloc[i - 2]) if i >= 2 else c
      pol = Policy(fade_enabled=True, fade_enter=3)
      day_st.update_fade(pol, c, rsi5, htf, adx, adx_prev, close_prev)
      max_score = max(max_score, day_st.fade_score)
      if day_st.fade_active:
        fade_bars += 1
        was_fade = True
    if was_fade:
      fade_days[day] = {"fade_bars": fade_bars, "max_score": max_score, "lh": day_st.lower_highs}
  return fade_days


def baseline_daily_pnl(df, cap=10_000) -> dict:
  raw = run_compound_backtest(df, starting_capital=cap)
  daily = {}
  for t in raw["trades"]:
    d = t["entry_time"].date() if hasattr(t["entry_time"], "date") else t.get("date")
    daily[d] = daily.get(d, 0) + t["pnl"]
  return daily


def analyze(name: str, pol: Policy, df, fade_days: dict, base_daily: dict, cap=10_000):
  r = run_policy(df, pol, cap)
  pol_daily = {d["date"]: d["pnl"] for d in r["daily"]}
  fade_set = set(fade_days.keys())
  base_fade = sum(base_daily.get(d, 0) for d in fade_set)
  pol_fade = sum(pol_daily.get(d, 0) for d in fade_set)
  base_non = sum(v for d, v in base_daily.items() if d not in fade_set)
  pol_non = sum(v for d, v in pol_daily.items() if d not in fade_set)
  losing_fade = [d for d in fade_set if base_daily.get(d, 0) < 0]
  improved = sum(1 for d in losing_fade if pol_daily.get(d, 0) > base_daily.get(d, 0))
  return {
    "name": name,
    "final": r["final"],
    "delta_final": r["final"] - 2_154_405,
    "base_fade_pnl": round(base_fade, 0),
    "pol_fade_pnl": round(pol_fade, 0),
    "delta_fade": round(pol_fade - base_fade, 0),
    "base_non_fade": round(base_non, 0),
    "pol_non_fade": round(pol_non, 0),
    "delta_non_fade": round(pol_non - base_non, 0),
    "losing_fade_days": len(losing_fade),
    "improved_losing": improved,
    "pf": r["pf"],
    "trades": r["trades"],
    "blocked": r["blocked"],
  }


def main():
  cap = 10_000
  df = load_real_data(days=100)
  print("Classifying fade days...")
  fade_days = classify_fade_days(df)
  print(f"  {len(fade_days)} fade days detected")

  print("Baseline daily PnL...")
  base_daily = baseline_daily_pnl(df, cap)
  base_fade = sum(base_daily.get(d, 0) for d in fade_days)
  losing = [(d, base_daily.get(d, 0)) for d in fade_days if base_daily.get(d, 0) < 0]
  losing.sort(key=lambda x: x[1])
  print(f"  Baseline PnL on fade days: Rs {base_fade:,.0f}")
  print(f"  Losing fade days: {len(losing)} / {len(fade_days)}")
  print("  Worst 5 fade days:")
  for d, p in losing[:5]:
    print(f"    {d}: Rs {p:,.0f}")

  policies = [
    ("BASELINE", Policy(name="baseline")),
    ("postSL", Policy(name="postSL", block_dip_after_sl=True)),
    ("slcd6", Policy(name="slcd6", sl_cooldown_bars=6)),
    ("postSL+slcd6", Policy(name="x", block_dip_after_sl=True, sl_cooldown_bars=6)),
    ("fade_e4_blk", Policy(name="x", fade_enabled=True, fade_enter=4, block_dip_long=True)),
    ("fade_e4_t11", Policy(name="x", fade_enabled=True, fade_enter=4, block_dip_long=True, time_gate_after="11:00")),
    ("fade_e5_blk", Policy(name="x", fade_enabled=True, fade_enter=5, block_dip_long=True)),
    ("adx25_blk", Policy(name="x", fade_enabled=True, fade_enter=4, adx_max_fade=25, block_dip_long=True)),
    ("adx25_mr", Policy(name="x", fade_enabled=True, fade_enter=4, adx_max_fade=25, block_dip_long=True, enable_mr_in_fade=True)),
    ("mr_e4", Policy(name="x", fade_enabled=True, fade_enter=4, block_dip_long=True, enable_mr_in_fade=True)),
    ("hybrid_v1", Policy(name="x", fade_enabled=True, fade_enter=4, adx_max_fade=25, block_dip_long=True,
                         enable_mr_in_fade=True, time_gate_after="11:00", sl_cooldown_bars=6)),
    ("hybrid_v4", Policy(name="x", fade_enabled=True, fade_enter=4, adx_max_fade=28,
                         block_dip_after_sl=True, enable_mr_in_fade=True, sl_cooldown_bars=6)),
    ("hybrid_v5", Policy(name="x", fade_enabled=True, fade_enter=4, adx_max_fade=25,
                         block_dip_after_sl=True, enable_mr_in_fade=True, sl_cooldown_bars=6)),
    # New candidates from web: post-SL + MR in fade only after SL
    ("postSL_mr", Policy(name="x", block_dip_after_sl=True, fade_enabled=True, fade_enter=4,
                         enable_mr_in_fade=True, adx_max_fade=30)),
    ("slcd6_mr", Policy(name="x", sl_cooldown_bars=6, fade_enabled=True, fade_enter=4,
                        enable_mr_in_fade=True, adx_max_fade=30)),
    ("fade_blk_postSL", Policy(name="x", fade_enabled=True, fade_enter=4, block_dip_long=True,
                               block_dip_after_sl=True, sl_cooldown_bars=6)),
    ("fade_e5_mr_adx25", Policy(name="x", fade_enabled=True, fade_enter=5, adx_max_fade=25,
                                block_dip_long=True, enable_mr_in_fade=True)),
    ("fade_e5_LH_mr", Policy(name="x", fade_enabled=True, fade_enter=5, block_dip_long=True,
                             enable_mr_in_fade=True, require_lower_highs=True)),
    ("only_postSL_slcd6_mr", Policy(name="x", block_dip_after_sl=True, sl_cooldown_bars=6,
                                    fade_enabled=True, fade_enter=5, enable_mr_in_fade=True, adx_max_fade=28)),
  ]

  rows = []
  for name, pol in policies:
    print(f"  Analyzing {name}...")
    rows.append(analyze(name, pol, df, fade_days, base_daily, cap))

  df_out = pd.DataFrame(rows)
  df_out = df_out.sort_values("delta_fade", ascending=False)
  print("\n" + "=" * 120)
  print("OBJECTIVE FADE-DAY ANALYSIS (same 97 fade days for all policies)")
  print("=" * 120)
  print(df_out.to_string(index=False))

  # Pareto: within 2% capital AND positive fade delta
  tol = 2_154_405 * 0.02
  winners = df_out[(df_out["delta_final"] >= -tol) & (df_out["delta_fade"] > 0)]
  print(f"\nWithin 2% of baseline AND improved fade days: {len(winners)}")
  if len(winners):
    print(winners[["name", "final", "delta_final", "delta_fade", "delta_non_fade", "improved_losing"]].to_string(index=False))

  out = ROOT / "tmp" / "fade_day_objective_analysis.csv"
  df_out.to_csv(out, index=False)
  print(f"\nSaved: {out}")


if __name__ == "__main__":
  main()
