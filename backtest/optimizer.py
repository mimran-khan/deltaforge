"""Iterative strategy optimizer -- sweeps parameters to find 7%+ daily compound.

Usage:
    python -m backtest.optimizer [--iterations 25] [--capital 10000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.run_backtest import load_real_data, run_compound_backtest
from config import settings
from engine.multi_strategy_engine import MultiStrategyEngine
from engine.premium_model import STRATEGY_SL_PCT, STRATEGY_TARGET_MULT


def _geometric_daily_return(start_cap: float, end_cap: float, days: int) -> float:
    if end_cap <= 0 or days <= 0 or start_cap <= 0:
        return -100.0
    return ((end_cap / start_cap) ** (1 / days) - 1) * 100


def run_single(df, params: dict, starting_capital: float = 10000,
               lot_size: int = 65) -> dict:
    """Run one backtest with overridden parameters. Returns results + config."""

    orig_settings = {}
    for key, val in params.get("settings", {}).items():
        orig_settings[key] = getattr(settings, key, None)
        setattr(settings, key, val)

    orig_sl_pct = dict(STRATEGY_SL_PCT)
    if "sl_pct" in params:
        for strat, pct in params["sl_pct"].items():
            STRATEGY_SL_PCT[strat] = pct

    orig_target_mult = {k: dict(v) for k, v in STRATEGY_TARGET_MULT.items()}
    if "target_mult" in params:
        for strat, tiers in params["target_mult"].items():
            STRATEGY_TARGET_MULT[strat] = tiers

    engine = MultiStrategyEngine()

    if "disabled" in params:
        engine.DISABLED_STRATEGIES = set(params["disabled"])
    if "min_adx" in params:
        engine.MIN_ADX = params["min_adx"]
    if "max_adx" in params:
        engine.MAX_ADX = params["max_adx"]
    if "max_total_per_day" in params:
        engine.MAX_TOTAL_PER_DAY = params["max_total_per_day"]
    if "cooldown_bars" in params:
        engine.COOLDOWN_BARS = params["cooldown_bars"]
    if "sl_cooldown_bars" in params:
        engine.SL_COOLDOWN_BARS = params["sl_cooldown_bars"]

    results = run_compound_backtest(
        df, starting_capital=starting_capital,
        lot_size=lot_size, deploy_pct=params.get("deploy_pct", 100),
        engine_override=engine,
        use_adaptive=params.get("use_adaptive", True),
        use_risk_gates=params.get("use_risk_gates", True),
    )

    for key, val in orig_settings.items():
        if val is not None:
            setattr(settings, key, val)
    STRATEGY_SL_PCT.update(orig_sl_pct)
    for k, v in orig_target_mult.items():
        STRATEGY_TARGET_MULT[k] = v

    active_days = results.get("active_trading_days", 1)
    geo_daily = _geometric_daily_return(
        starting_capital, results["final_capital"], active_days)

    results["geo_daily_return_pct"] = round(geo_daily, 2)
    results["params"] = params
    return results


def generate_param_grid() -> list[dict]:
    """Generate parameter combinations to test."""
    configs = []

    base_disabled = {"VWAP_MOMENTUM", "VWAP_MEAN_REV", "RSI_REVERSION",
                     "ADX_BREAKOUT", "ORB_BREAKOUT", "BB_SQUEEZE",
                     "VWAP_BOUNCE", "RSI_DIVERGENCE"}

    target_wide = {
        "PULLBACK":      {70: 1.60, 50: 1.45, 0: 1.35},
        "TREND_RIDE":    {70: 2.00, 50: 1.75, 0: 1.50},
        "CPR_BREAKOUT":  {70: 1.60, 50: 1.45, 0: 1.35},
        "GAP_TRADE":     {70: 1.60, 50: 1.45, 0: 1.35},
        "CPR_RANGE":     {70: 1.40, 50: 1.30, 0: 1.20},
        "SUPERTREND":    {70: 1.60, 50: 1.45, 0: 1.35},
        "STOCH_CROSS":   {70: 1.60, 50: 1.45, 0: 1.35},
        "EMA_MOMENTUM":  {70: 1.80, 50: 1.60, 0: 1.40},
    }

    target_aggressive = {
        "PULLBACK":      {70: 2.00, 50: 1.70, 0: 1.50},
        "TREND_RIDE":    {70: 2.50, 50: 2.00, 0: 1.70},
        "CPR_BREAKOUT":  {70: 2.00, 50: 1.70, 0: 1.50},
        "GAP_TRADE":     {70: 2.00, 50: 1.70, 0: 1.50},
        "SUPERTREND":    {70: 2.00, 50: 1.70, 0: 1.50},
        "STOCH_CROSS":   {70: 2.00, 50: 1.70, 0: 1.50},
        "EMA_MOMENTUM":  {70: 2.50, 50: 2.00, 0: 1.70},
    }

    # Iteration 1: Baseline
    configs.append({"name": "BASELINE", "disabled": base_disabled | {"EMA_MOMENTUM", "SUPERTREND", "STOCH_CROSS"}})

    # Iteration 2: Re-enable SUPERTREND (80% WR in 100d backtest)
    configs.append({"name": "RE-ENABLE_SUPERTREND", "disabled": base_disabled | {"EMA_MOMENTUM", "STOCH_CROSS"}})

    # Iteration 3: Re-enable STOCH_CROSS (85% WR in 100d backtest)
    configs.append({"name": "RE-ENABLE_STOCH+ST", "disabled": base_disabled | {"EMA_MOMENTUM"}})

    # Iteration 4: Re-enable all (EMA_MOMENTUM was 65% WR in 100d backtest)
    configs.append({"name": "ALL_ENABLED", "disabled": base_disabled})

    # Iteration 5: Wider targets (1.5-2.0x)
    configs.append({"name": "WIDE_TARGETS", "disabled": base_disabled,
                    "target_mult": target_wide})

    # Iteration 6: Aggressive targets
    configs.append({"name": "AGG_TARGETS", "disabled": base_disabled,
                    "target_mult": target_aggressive})

    # Iteration 7: Tighter SL (8% for all)
    configs.append({"name": "TIGHT_SL_8PCT", "disabled": base_disabled,
                    "sl_pct": {s: 8.0 for s in STRATEGY_SL_PCT}})

    # Iteration 8: Wider trail (20% trigger, 12% trail)
    configs.append({"name": "WIDE_TRAIL_20_12", "disabled": base_disabled,
                    "settings": {"TRAIL_TRIGGER_PCT": 20.0, "TRAIL_PCT": 12.0}})

    # Iteration 9: Very wide trail (25% trigger, 15% trail)
    configs.append({"name": "WIDE_TRAIL_25_15", "disabled": base_disabled,
                    "settings": {"TRAIL_TRIGGER_PCT": 25.0, "TRAIL_PCT": 15.0}})

    # Iteration 10: Higher delta (0.80) for better premium tracking
    configs.append({"name": "HIGH_DELTA_0.80", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80}})

    # Iteration 11: Higher delta + wide targets
    configs.append({"name": "DELTA80+WIDE_TGT", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80},
                    "target_mult": target_wide})

    # Iteration 12: Lower theta (0.15) -- less decay pressure
    configs.append({"name": "LOW_THETA_0.15", "disabled": base_disabled,
                    "settings": {"THETA_BASE": 0.15, "PREMIUM_THETA_PER_CANDLE": 0.15}})

    # Iteration 13: Combo -- high delta + low theta + wide targets
    configs.append({"name": "D80+T15+WIDE_TGT", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15},
                    "target_mult": target_wide})

    # Iteration 14: Combo + tighter SL (8%)
    configs.append({"name": "COMBO+SL8", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15},
                    "target_mult": target_wide,
                    "sl_pct": {s: 8.0 for s in STRATEGY_SL_PCT}})

    # Iteration 15: Combo + re-enable high-WR strategies
    configs.append({"name": "COMBO+STRATS", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15},
                    "target_mult": target_wide})

    # Iteration 16: Raise max lots to 50 for better compounding
    configs.append({"name": "COMBO+50LOTS", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15, "MAX_LOTS_CAP": 50},
                    "target_mult": target_wide})

    # Iteration 17: Raise max lots to 100
    configs.append({"name": "COMBO+100LOTS", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15, "MAX_LOTS_CAP": 100},
                    "target_mult": target_wide})

    # Iteration 18: Combo + more trades per day (12)
    configs.append({"name": "COMBO+12TRADES", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15, "MAX_LOTS_CAP": 50},
                    "target_mult": target_wide,
                    "max_total_per_day": 12})

    # Iteration 19: Combo + aggressive trail (20/12)
    configs.append({"name": "COMBO+WIDE_TRAIL", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15, "MAX_LOTS_CAP": 50,
                                 "TRAIL_TRIGGER_PCT": 20.0, "TRAIL_PCT": 12.0},
                    "target_mult": target_wide})

    # Iteration 20: Lower min ADX (8) for more signals
    configs.append({"name": "COMBO+LOW_ADX8", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15, "MAX_LOTS_CAP": 50},
                    "target_mult": target_wide,
                    "min_adx": 8})

    # Iteration 21: Higher max ADX (45) for trending days
    configs.append({"name": "COMBO+HIGH_ADX45", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15, "MAX_LOTS_CAP": 50},
                    "target_mult": target_wide,
                    "max_adx": 45})

    # Iteration 22: Aggressive targets + wide trail + lots 50
    configs.append({"name": "AGG_TGT+TRAIL+50", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15, "MAX_LOTS_CAP": 50,
                                 "TRAIL_TRIGGER_PCT": 20.0, "TRAIL_PCT": 12.0},
                    "target_mult": target_aggressive})

    # Iteration 23: No adaptive mode (raw strategy performance)
    configs.append({"name": "NO_ADAPTIVE", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15, "MAX_LOTS_CAP": 50},
                    "target_mult": target_wide,
                    "use_adaptive": False})

    # Iteration 24: Shorter cooldown (2 bars)
    configs.append({"name": "COMBO+CD2", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15, "MAX_LOTS_CAP": 50},
                    "target_mult": target_wide,
                    "cooldown_bars": 2, "sl_cooldown_bars": 4})

    # Iteration 25: Capital per lot = 3000 (more aggressive sizing)
    configs.append({"name": "COMBO+CPL3K", "disabled": base_disabled,
                    "settings": {"PREMIUM_DELTA": 0.80, "THETA_BASE": 0.15,
                                 "MAX_LOTS_CAP": 50, "CAPITAL_PER_LOT": 3000},
                    "target_mult": target_wide})

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--days", type=int, default=100)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    print(f"\nLoading {args.days} trading days of Nifty data...")
    df = load_real_data(days=args.days)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days")

    configs = generate_param_grid()[:args.iterations]
    print(f"\nRunning {len(configs)} parameter configurations...")
    print("=" * 120)
    print(f"{'#':>3} {'Name':<25} {'Trades':>6} {'WR%':>6} {'PF':>6} "
          f"{'Final Cap':>12} {'Return%':>8} {'DD%':>6} {'GeoDaily%':>9} {'AvgWin':>8} {'AvgLoss':>8}")
    print("-" * 120)

    all_results = []
    best_geo = -999
    best_name = ""

    for i, params in enumerate(configs):
        name = params.get("name", f"config_{i+1}")
        try:
            r = run_single(df, params, starting_capital=args.capital,
                           lot_size=settings.NIFTY_LOT_SIZE)

            geo = r["geo_daily_return_pct"]
            marker = " **BEST**" if geo > best_geo else ""
            if geo > best_geo:
                best_geo = geo
                best_name = name

            print(f"{i+1:>3} {name:<25} {r['total_trades']:>6} {r['win_rate']:>5.1f}% "
                  f"{r['profit_factor']:>5.2f} Rs {r['final_capital']:>10,.0f} "
                  f"{r['return_pct']:>+7.1f}% {r['max_drawdown_pct']:>5.1f}% "
                  f"{geo:>+8.2f}% {r['avg_win']:>+8,.0f} {r['avg_loss']:>+8,.0f}{marker}")

            all_results.append({"name": name, **r})
        except Exception as e:
            print(f"{i+1:>3} {name:<25} FAILED: {e}")

    print("=" * 120)
    print(f"\nBEST CONFIG: {best_name} ({best_geo:+.2f}% geometric daily return)")

    if all_results:
        best = max(all_results, key=lambda x: x["geo_daily_return_pct"])
        print(f"\n  Final Capital: Rs {best['final_capital']:,.0f}")
        print(f"  Return: {best['return_pct']}%")
        print(f"  Win Rate: {best['win_rate']}%")
        print(f"  Profit Factor: {best['profit_factor']}")
        print(f"  Max Drawdown: {best['max_drawdown_pct']}%")
        print(f"  Avg Daily Return: {best['avg_daily_return_pct']}%")
        print(f"  Geo Daily Return: {best['geo_daily_return_pct']}%")

        summary = []
        for r in sorted(all_results, key=lambda x: -x["geo_daily_return_pct"]):
            summary.append({
                "name": r["name"],
                "trades": r["total_trades"],
                "win_rate": r["win_rate"],
                "pf": r["profit_factor"],
                "final_capital": r["final_capital"],
                "return_pct": r["return_pct"],
                "max_dd": r["max_drawdown_pct"],
                "geo_daily": r["geo_daily_return_pct"],
            })
        with open(settings.DATA_DIR / "optimizer_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\n  Saved: data/optimizer_results.json")

    return all_results


if __name__ == "__main__":
    main()
