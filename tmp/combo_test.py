#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tmp.fade_day_analysis import classify_fade_days, baseline_daily_pnl
from tmp.regime_optimizer import Policy, run_policy, load_real_data, run_compound_backtest

df = load_real_data(days=100)
fade_set = set(classify_fade_days(df).keys())
base_daily = baseline_daily_pnl(df)
base_final = run_compound_backtest(df, 10000)["final_capital"]
losing = [d for d in fade_set if base_daily.get(d, 0) < 0]

combos = [
  ("slcd6", Policy(sl_cooldown_bars=6)),
  ("slcd4", Policy(sl_cooldown_bars=4)),
  ("slcd3", Policy(sl_cooldown_bars=3)),
  ("fade_e5+slcd6", Policy(fade_enabled=True, fade_enter=5, block_dip_long=True, sl_cooldown_bars=6)),
  ("fade_e5_t1130+slcd6", Policy(fade_enabled=True, fade_enter=5, block_dip_long=True, time_gate_after="11:30", sl_cooldown_bars=6)),
  ("adx28_e5+slcd6", Policy(fade_enabled=True, fade_enter=5, adx_max_fade=28, block_dip_long=True, sl_cooldown_bars=6)),
  ("fade_e5_stoch+slcd6", Policy(fade_enabled=True, fade_enter=5, block_stoch_only=True, sl_cooldown_bars=6)),
]
print(f"Baseline final: Rs {base_final:,.0f}")
hdr = f"{'Policy':<22} {'Final':>10} {'dCap':>10} {'dLose':>10} {'dWinFade':>12} {'Fix':>7}"
print(hdr)
for name, pol in combos:
  r = run_policy(df, pol)
  pd_ = {d["date"]: d["pnl"] for d in r["daily"]}
  dl = sum(pd_.get(d, 0) - base_daily.get(d, 0) for d in losing)
  wf = [d for d in fade_set if base_daily.get(d, 0) >= 0]
  dw = sum(pd_.get(d, 0) - base_daily.get(d, 0) for d in wf)
  fix = sum(1 for d in losing if pd_.get(d, 0) > base_daily.get(d, 0))
  print(f"{name:<22} Rs{r['final']:>7,.0f} {r['final']-base_final:>+9,.0f} {dl:>+9,.0f} {dw:>+11,.0f} {fix:>3}/{len(losing)}")
