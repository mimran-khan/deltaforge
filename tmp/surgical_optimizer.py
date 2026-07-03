#!/usr/bin/env python3
"""Round 3: surgical policies targeting LOSING fade days only (41/98)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tmp.fade_day_analysis import classify_fade_days, baseline_daily_pnl
from tmp.regime_optimizer import Policy, run_policy, load_real_data, run_compound_backtest


def score_policy(name, pol, df, fade_days, base_daily, base_final):
  r = run_policy(df, pol)
  pol_daily = {d["date"]: d["pnl"] for d in r["daily"]}
  losing = [d for d in fade_days if base_daily.get(d, 0) < 0]
  winning_fade = [d for d in fade_days if base_daily.get(d, 0) >= 0]

  lose_base = sum(base_daily.get(d, 0) for d in losing)
  lose_pol = sum(pol_daily.get(d, 0) for d in losing)
  win_base = sum(base_daily.get(d, 0) for d in winning_fade)
  win_pol = sum(pol_daily.get(d, 0) for d in winning_fade)

  fixed = sum(1 for d in losing if pol_daily.get(d, 0) > base_daily.get(d, 0))
  broken = sum(1 for d in winning_fade if pol_daily.get(d, 0) < base_daily.get(d, 0))

  return {
    "name": name,
    "final": r["final"],
    "delta_cap": r["final"] - base_final,
    "lose_base": round(lose_base, 0),
    "lose_pol": round(lose_pol, 0),
    "delta_lose": round(lose_pol - lose_base, 0),
    "win_fade_base": round(win_base, 0),
    "win_fade_pol": round(win_pol, 0),
    "delta_win_fade": round(win_pol - win_base, 0),
    "fixed_losing": fixed,
    "broken_winning": broken,
    "n_losing": len(losing),
    "pf": r["pf"],
    "blocked": r["blocked"],
    "trades": r["trades"],
  }


def main():
  cap = 10_000
  df = load_real_data(days=100)
  fade_days = classify_fade_days(df)
  fade_set = set(fade_days.keys())
  base_daily = baseline_daily_pnl(df, cap)
  raw = run_compound_backtest(df, starting_capital=cap)
  base_final = raw["final_capital"]

  losing = sorted([(d, base_daily.get(d, 0)) for d in fade_set if base_daily.get(d, 0) < 0], key=lambda x: x[1])
  winning_fade = sum(base_daily.get(d, 0) for d in fade_set if base_daily.get(d, 0) >= 0)
  losing_sum = sum(v for _, v in losing)

  print("=" * 80)
  print("KEY INSIGHT: Fade days are ALREADY your profit engine")
  print("=" * 80)
  print(f"  Baseline total PnL:     Rs {raw['total_pnl']:,.0f}")
  print(f"  PnL on fade days:       Rs {sum(base_daily.get(d,0) for d in fade_set):,.0f} ({len(fade_set)} days)")
  print(f"  PnL on non-fade days:   Rs {sum(v for d,v in base_daily.items() if d not in fade_set):,.0f}")
  print(f"  Losing fade days:       {len(losing)} days, Rs {losing_sum:,.0f} total losses")
  print(f"  Winning fade days:      {len(fade_set)-len(losing)} days, Rs {winning_fade:,.0f}")
  print(f"\n  Goal: fix Rs {abs(losing_sum):,.0f} on losing fade days without touching Rs {winning_fade:,.0f} winners")

  policies = [
    ("baseline", Policy()),
    ("slcd6", Policy(sl_cooldown_bars=6)),
    ("postSL", Policy(block_dip_after_sl=True)),
    ("postSL+slcd6", Policy(block_dip_after_sl=True, sl_cooldown_bars=6)),
    # Surgical fade blocks
    ("fade_e5_blk", Policy(fade_enabled=True, fade_enter=5, block_dip_long=True)),
    ("fade_e5_LH", Policy(fade_enabled=True, fade_enter=5, block_dip_long=True, require_lower_highs=True)),
    ("fade_e5_t1130", Policy(fade_enabled=True, fade_enter=5, block_dip_long=True, time_gate_after="11:30")),
    ("fade_e4_t11", Policy(fade_enabled=True, fade_enter=4, block_dip_long=True, time_gate_after="11:00")),
    ("fade_e4_pts50", Policy(fade_enabled=True, fade_enter=4, pts_below_high=50, block_dip_long=True)),
    # Post-SL surgical (only after pain)
    ("fade_e4_postSL", Policy(fade_enabled=True, fade_enter=4, block_dip_after_sl=True)),
    ("fade_e5_postSL", Policy(fade_enabled=True, fade_enter=5, block_dip_after_sl=True)),
    ("fade_e5_postSL_slcd6", Policy(fade_enabled=True, fade_enter=5, block_dip_after_sl=True, sl_cooldown_bars=6)),
    # ADX surgical
    ("adx28_e5_blk", Policy(fade_enabled=True, fade_enter=5, adx_max_fade=28, block_dip_long=True)),
    ("adx25_e5_blk", Policy(fade_enabled=True, fade_enter=5, adx_max_fade=25, block_dip_long=True)),
    # MR only after SL (don't block winners preemptively)
    ("postSL_mr_e5", Policy(block_dip_after_sl=True, fade_enabled=True, fade_enter=5, enable_mr_in_fade=True, adx_max_fade=30)),
    ("slcd6_mr_e5", Policy(sl_cooldown_bars=6, fade_enabled=True, fade_enter=5, enable_mr_in_fade=True, adx_max_fade=30)),
    # Minimal: only STOCH block in fade (keep PULLBACK)
    ("fade_e5_stoch_only", Policy(fade_enabled=True, fade_enter=5, block_stoch_only=True)),
    ("fade_e5_pb_only", Policy(fade_enabled=True, fade_enter=5, block_pullback_only=True)),
  ]

  rows = []
  for name, pol in policies:
    if name == "fade_e5_stoch_only":
      continue  # skip - needs code change
    print(f"  {name}...")
    rows.append(score_policy(name, pol, df, fade_set, base_daily, base_final))

  df_out = pd.DataFrame(rows)
  # Score: maximize delta_lose (fix losing days), minimize delta_win_fade damage, minimize delta_cap
  df_out["score"] = (
    df_out["delta_lose"] * 2
    + df_out["delta_win_fade"] * 1.5
    + df_out["delta_cap"] * 0.001
    + df_out["fixed_losing"] * 5000
  )
  df_out = df_out.sort_values("score", ascending=False)

  print("\n" + "=" * 120)
  print("SURGICAL POLICY RANKING (fix losing fade days, preserve winning fade days)")
  print("=" * 120)
  cols = ["name", "final", "delta_cap", "delta_lose", "delta_win_fade", "fixed_losing", "broken_winning", "pf", "blocked"]
  print(df_out[cols].to_string(index=False))

  # Pareto frontier: delta_cap > -150K
  good = df_out[df_out["delta_cap"] > -150_000].sort_values("delta_lose", ascending=False)
  print(f"\nWithin Rs 150K of baseline ({len(good)} policies):")
  print(good[cols].head(8).to_string(index=False))

  # Best at fixing losing days
  best_fix = df_out.sort_values("delta_lose", ascending=False).head(5)
  print("\nBest at improving LOSING fade days:")
  print(best_fix[cols].to_string(index=False))

  out = ROOT / "tmp" / "surgical_policy_results.csv"
  df_out.to_csv(out, index=False)
  print(f"\nSaved: {out}")


if __name__ == "__main__":
  main()
