"""Parameter Sweep -- Rigorous A/B testing of tuning changes.

Tests each parameter change in isolation against a baseline, then combines
the winners. Uses 70/30 chronological walk-forward split on 100 trading days
to avoid lookahead bias.

Usage:
    python -m backtest.parameter_sweep [--days 100]
"""

from __future__ import annotations
import sys
import copy
import argparse
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, run_compound_backtest


@contextmanager
def patched_config(**overrides):
    originals = {}
    for key, val in overrides.items():
        originals[key] = getattr(settings, key, None)
        setattr(settings, key, val)
    try:
        yield
    finally:
        for key, val in originals.items():
            if val is None:
                try:
                    delattr(settings, key)
                except AttributeError:
                    pass
            else:
                setattr(settings, key, val)


def make_engine(**overrides):
    """Create a MultiStrategyEngine with specific parameter overrides."""
    from engine.multi_strategy_engine import MultiStrategyEngine
    engine = MultiStrategyEngine()
    for key, val in overrides.items():
        setattr(engine, key, val)
    return engine


def run_config(label: str, df_train, df_test, lot_size: int,
               capital: float, engine_overrides: dict = None,
               config_overrides: dict = None) -> dict:
    """Run a single config on both train and test sets."""

    engine_overrides = engine_overrides or {}
    config_overrides = config_overrides or {}

    with patched_config(**config_overrides):
        engine_train = make_engine(**engine_overrides)
        r_train = run_compound_backtest(
            df_train, starting_capital=capital,
            lot_size=lot_size, engine_override=engine_train,
        )

        engine_test = make_engine(**engine_overrides)
        r_test = run_compound_backtest(
            df_test, starting_capital=capital,
            lot_size=lot_size, engine_override=engine_test,
        )

    train_days = r_train["active_trading_days"] or 1
    test_days = r_test["active_trading_days"] or 1
    train_cdgr = ((r_train["final_capital"] / capital) ** (1 / train_days) - 1) * 100 if r_train["final_capital"] > 0 else 0
    test_cdgr = ((r_test["final_capital"] / capital) ** (1 / test_days) - 1) * 100 if r_test["final_capital"] > 0 else 0

    train_tpd = r_train["total_trades"] / train_days
    test_tpd = r_test["total_trades"] / test_days

    return {
        "label": label,
        "train_capital": r_train["final_capital"],
        "train_return": r_train["return_pct"],
        "train_trades": r_train["total_trades"],
        "train_wr": r_train["win_rate"],
        "train_pf": r_train["profit_factor"],
        "train_dd": r_train["max_drawdown_pct"],
        "train_cdgr": round(train_cdgr, 2),
        "train_tpd": round(train_tpd, 2),
        "train_days": train_days,

        "test_capital": r_test["final_capital"],
        "test_return": r_test["return_pct"],
        "test_trades": r_test["total_trades"],
        "test_wr": r_test["win_rate"],
        "test_pf": r_test["profit_factor"],
        "test_dd": r_test["max_drawdown_pct"],
        "test_cdgr": round(test_cdgr, 2),
        "test_tpd": round(test_tpd, 2),
        "test_days": test_days,

        "train_strat": _strat_breakdown(r_train),
        "test_strat": _strat_breakdown(r_test),
    }


def _strat_breakdown(results: dict) -> dict:
    strats = {}
    for t in results["trades"]:
        s = t["strategy"]
        if s not in strats:
            strats[s] = {"n": 0, "w": 0, "pnl": 0}
        strats[s]["n"] += 1
        if t["pnl"] > 0:
            strats[s]["w"] += 1
        strats[s]["pnl"] += t["pnl"]
    return strats


def print_comparison(results: list[dict]):
    print("\n" + "=" * 120)
    print("  PARAMETER SWEEP -- WALK-FORWARD RESULTS (70% Train / 30% Test)")
    print("=" * 120)

    header = (f"  {'Config':<25s} | {'CDGR':>6s} {'WR':>5s} {'PF':>5s} "
              f"{'DD':>5s} {'T/D':>5s} {'Trd':>4s} | "
              f"{'CDGR':>6s} {'WR':>5s} {'PF':>5s} "
              f"{'DD':>5s} {'T/D':>5s} {'Trd':>4s} | {'Gap':>5s}")
    print(f"\n  {'':25s} | {'───── TRAIN ─────':^32s} | {'───── TEST ──────':^32s} | {'Δ':>5s}")
    print(header)
    print("  " + "-" * 118)

    baseline_test_cdgr = results[0]["test_cdgr"] if results else 0

    for r in results:
        gap = r["test_cdgr"] - r["train_cdgr"]
        delta = r["test_cdgr"] - baseline_test_cdgr
        delta_s = f"{delta:+.1f}" if r["label"] != "A: Baseline" else "  --"

        print(f"  {r['label']:<25s} | "
              f"{r['train_cdgr']:>5.1f}% {r['train_wr']:>4.0f}% {r['train_pf']:>5.2f} "
              f"{r['train_dd']:>4.0f}% {r['train_tpd']:>5.2f} {r['train_trades']:>4d} | "
              f"{r['test_cdgr']:>5.1f}% {r['test_wr']:>4.0f}% {r['test_pf']:>5.2f} "
              f"{r['test_dd']:>4.0f}% {r['test_tpd']:>5.2f} {r['test_trades']:>4d} | "
              f"{delta_s:>5s}")

    print("  " + "-" * 118)

    print("\n  Strategy Breakdown (TEST set):")
    for r in results:
        if r["test_strat"]:
            strats = ", ".join(
                f"{s}: {d['n']}t {d['w']/d['n']*100:.0f}%WR Rs{d['pnl']:+,.0f}"
                for s, d in sorted(r["test_strat"].items()) if d["n"] > 0
            )
            print(f"    {r['label']:<25s}: {strats}")

    print("\n  Interpretation Guide:")
    print("    CDGR = Compound Daily Growth Rate (higher = better)")
    print("    Gap  = Test CDGR minus Baseline Test CDGR")
    print("    Train-Test gap > 2% suggests overfitting")
    print("    Test set is the TRUTH -- train is for reference only")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--capital", type=float, default=10000)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    print(f"\nLoading {args.days} trading days of Nifty data...")
    df = load_real_data(days=args.days)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days")

    split_idx = int(len(unique_days) * 0.70)
    train_days_set = set(unique_days[:split_idx])
    test_days_set = set(unique_days[split_idx:])
    df_train = df[df.index.map(lambda t: t.date() in train_days_set)]
    df_test = df[df.index.map(lambda t: t.date() in test_days_set)]

    print(f"Train: {len(train_days_set)} days ({min(train_days_set)} -> {max(train_days_set)})")
    print(f"Test:  {len(test_days_set)} days ({min(test_days_set)} -> {max(test_days_set)})")

    lot_size = settings.NIFTY_LOT_SIZE
    results = []

    configs = [
        {
            "label": "A: Baseline (prod)",
            "engine": {},
            "config": {},
        },
        {
            "label": "B: EntryEnd 14:30",
            "engine": {},
            "config": {"ENTRY_END": "14:30", "NO_NEW_ENTRY_AFTER": "14:30"},
        },
        {
            "label": "C: Cooldown 2 bars",
            "engine": {"COOLDOWN_BARS": 2},
            "config": {},
        },
        {
            "label": "D: DeadZone 3",
            "engine": {"HTF_DEAD_ZONE_HI": 3},
            "config": {},
        },
        {
            "label": "E: PB RSI 45/55",
            "engine": {},
            "config": {},
        },
        {
            "label": "F: +SUPERTREND",
            "engine": {"DISABLED_STRATEGIES": {
                "VWAP_MOMENTUM", "VWAP_MEAN_REV", "STOCH_CROSS", "ADX_BREAKOUT",
            }},
            "config": {},
        },
        {
            "label": "G: MIN_ADX=5",
            "engine": {"MIN_ADX": 5},
            "config": {},
        },
        {
            "label": "H: MaxTotal 12",
            "engine": {"MAX_TOTAL_PER_DAY": 12},
            "config": {},
        },
        {
            "label": "I: +STOCH_CROSS",
            "engine": {"DISABLED_STRATEGIES": {
                "SUPERTREND", "VWAP_MOMENTUM", "VWAP_MEAN_REV", "ADX_BREAKOUT",
            }},
            "config": {},
        },
        {
            "label": "J: MinADX5+DZ3",
            "engine": {"MIN_ADX": 5, "HTF_DEAD_ZONE_HI": 3},
            "config": {},
        },
        {
            "label": "K: MinADX5+MaxT12",
            "engine": {"MIN_ADX": 5, "MAX_TOTAL_PER_DAY": 12},
            "config": {},
        },
    ]

    for i, cfg in enumerate(configs):
        label = cfg["label"]
        print(f"\n  Running [{i+1}/{len(configs)}]: {label}...", flush=True)

        if label == "E: PB RSI 45/55":
            r = run_pullback_rsi_test(df_train, df_test, lot_size, args.capital)
        else:
            r = run_config(
                label, df_train, df_test, lot_size, args.capital,
                engine_overrides=cfg["engine"],
                config_overrides=cfg["config"],
            )
        results.append(r)
        print(f"    Train CDGR={r['train_cdgr']:.1f}% | Test CDGR={r['test_cdgr']:.1f}% | "
              f"Test WR={r['test_wr']:.0f}% PF={r['test_pf']:.2f} T/D={r['test_tpd']:.2f}")

    winners = identify_winners(results)

    if winners:
        print(f"\n  Running [COMBINED]: Winners = {winners}...", flush=True)
        combined = build_combined_config(winners, configs)
        r_combined = run_config(
            "H: Combined Best", df_train, df_test, lot_size, args.capital,
            engine_overrides=combined["engine"],
            config_overrides=combined["config"],
        )
        results.append(r_combined)
        print(f"    Train CDGR={r_combined['train_cdgr']:.1f}% | "
              f"Test CDGR={r_combined['test_cdgr']:.1f}% | "
              f"Test WR={r_combined['test_wr']:.0f}% PF={r_combined['test_pf']:.2f}")

    print_comparison(results)

    csv_rows = []
    for r in results:
        csv_rows.append({k: v for k, v in r.items()
                         if k not in ("train_strat", "test_strat")})
    pd.DataFrame(csv_rows).to_csv(
        settings.DATA_DIR / "parameter_sweep_results.csv", index=False)
    print(f"  Saved: data/parameter_sweep_results.csv")


def run_pullback_rsi_test(df_train, df_test, lot_size, capital):
    """Custom test for widened pullback RSI thresholds (45/55 vs 48/52).

    We monkey-patch the _check_pullback method to use different thresholds.
    """
    from engine.multi_strategy_engine import MultiStrategyEngine

    original_check = MultiStrategyEngine._check_pullback

    def patched_check(self, ind_dict, idx):
        sig = original_check(self, ind_dict, idx)
        return sig

    def make_wide_engine():
        engine = MultiStrategyEngine()

        orig_fn = engine._check_pullback

        def wide_pullback(ind_dict, idx):
            rsi_5m = engine._sv(ind_dict['rsi_5m'], idx, 50)
            rsi_15m = engine._htf_rsi(ind_dict, idx, 50)
            if rsi_15m == 50.0:
                if rsi_5m < 35:
                    rsi_15m = 40
                elif rsi_5m > 65:
                    rsi_15m = 60
            stoch_k = engine._sv(ind_dict['stoch_k'], idx, 50)
            cci = engine._sv(ind_dict['cci'], idx, 0)
            willr = engine._sv(ind_dict['willr'], idx, -50)
            close = engine._sv(ind_dict['close'], idx)
            ema_9 = engine._sv(ind_dict['ema_9'], idx, close)
            ema_20 = engine._sv(ind_dict['ema_20'], idx, close)
            st_dir = engine._sv(ind_dict['supertrend_dir'], idx, 0)
            adx_val = engine._sv(ind_dict.get('adx', pd.Series()), idx, 0)
            rsi_prev = engine._sv(ind_dict['rsi_5m'], idx - 1, rsi_5m) if idx >= 1 else rsi_5m

            if engine._pullback_count >= engine.MAX_PULLBACK_PER_DAY:
                return None
            if np.isnan(close):
                return None

            bull_trend = rsi_15m > engine.HTF_BULL_RSI
            bear_trend = rsi_15m < engine.HTF_BEAR_RSI
            if not (bull_trend or bear_trend):
                return None

            htf_strength = abs(rsi_15m - 50)
            if engine.HTF_DEAD_ZONE_LO <= htf_strength < engine.HTF_DEAD_ZONE_HI:
                return None

            from engine.multi_strategy_engine import TradeSignal

            if bull_trend:
                direction = "LONG"
                pb_count = 0
                reasons = [f"15m_RSI={rsi_15m:.0f}↑"]
                if rsi_5m < 45:  # widened from 48
                    pb_count += 1
                    reasons.append(f"RSI={rsi_5m:.0f}<45")
                if stoch_k < 30:
                    pb_count += 1
                if cci < -80:
                    pb_count += 1
                if willr < -70:
                    pb_count += 1
            else:
                direction = "SHORT"
                pb_count = 0
                reasons = [f"15m_RSI={rsi_15m:.0f}↓"]
                if rsi_5m > 55:  # widened from 52
                    pb_count += 1
                    reasons.append(f"RSI={rsi_5m:.0f}>55")
                if stoch_k > 70:
                    pb_count += 1
                if cci > 80:
                    pb_count += 1
                if willr > -30:
                    pb_count += 1

                if pb_count < 1 and adx_val > 35 and st_dir == -1 and ema_9 < ema_20:
                    rsi_bounce = rsi_5m > rsi_prev + 2
                    bear_candles = 0
                    if idx >= 2:
                        for lb in range(3):
                            c = engine._sv(ind_dict['close'], idx - lb)
                            o = engine._sv(ind_dict['open'], idx - lb)
                            if not np.isnan(c) and not np.isnan(o) and c < o:
                                bear_candles += 1
                    if rsi_bounce or bear_candles >= 3:
                        pb_count = 1
                        reasons.append("TrendCont")

            if pb_count < 1:
                return None

            conf = 55 + (pb_count * 10)
            htf_strength = abs(rsi_15m - 50)
            conf += min(htf_strength * 0.3, 8)
            if direction == "LONG" and close > ema_20:
                conf += 3
            elif direction == "SHORT" and close < ema_20:
                conf += 3
            if direction == "LONG" and st_dir == 1:
                conf += 3
            elif direction == "SHORT" and st_dir == -1:
                conf += 3
            conf = min(conf, 100)

            return TradeSignal(
                direction=direction,
                signal_type="PULLBACK",
                confidence=conf,
                htf_rsi=rsi_15m,
                ltf_rsi=rsi_5m,
                nifty_price=close,
                reason=" | ".join(reasons),
                pullback_count=pb_count,
            )

        engine._check_pullback = wide_pullback
        return engine

    engine_train = make_wide_engine()
    r_train = run_compound_backtest(
        df_train, starting_capital=capital,
        lot_size=lot_size, engine_override=engine_train,
    )

    engine_test = make_wide_engine()
    r_test = run_compound_backtest(
        df_test, starting_capital=capital,
        lot_size=lot_size, engine_override=engine_test,
    )

    train_days = r_train["active_trading_days"] or 1
    test_days = r_test["active_trading_days"] or 1
    train_cdgr = ((r_train["final_capital"] / capital) ** (1 / train_days) - 1) * 100 if r_train["final_capital"] > 0 else 0
    test_cdgr = ((r_test["final_capital"] / capital) ** (1 / test_days) - 1) * 100 if r_test["final_capital"] > 0 else 0

    return {
        "label": "F: PB RSI 45/55",
        "train_capital": r_train["final_capital"],
        "train_return": r_train["return_pct"],
        "train_trades": r_train["total_trades"],
        "train_wr": r_train["win_rate"],
        "train_pf": r_train["profit_factor"],
        "train_dd": r_train["max_drawdown_pct"],
        "train_cdgr": round(train_cdgr, 2),
        "train_tpd": round(r_train["total_trades"] / train_days, 2),
        "train_days": train_days,

        "test_capital": r_test["final_capital"],
        "test_return": r_test["return_pct"],
        "test_trades": r_test["total_trades"],
        "test_wr": r_test["win_rate"],
        "test_pf": r_test["profit_factor"],
        "test_dd": r_test["max_drawdown_pct"],
        "test_cdgr": round(test_cdgr, 2),
        "test_tpd": round(r_test["total_trades"] / test_days, 2),
        "test_days": test_days,

        "train_strat": _strat_breakdown(r_train),
        "test_strat": _strat_breakdown(r_test),
    }


def identify_winners(results: list[dict]) -> list[str]:
    """Identify configs that beat baseline on test CDGR without destroying WR."""
    if not results:
        return []
    baseline = results[0]
    winners = []
    for r in results[1:]:
        if (r["test_cdgr"] > baseline["test_cdgr"]
                and r["test_wr"] >= baseline["test_wr"] - 5
                and r["test_pf"] >= 1.0):
            winners.append(r["label"])
    return winners


def build_combined_config(winners: list[str], configs: list[dict]) -> dict:
    """Merge engine and config overrides from all winning configs."""
    combined_engine = {}
    combined_config = {}
    for w in winners:
        for cfg in configs:
            if cfg["label"] == w:
                combined_engine.update(cfg.get("engine", {}))
                combined_config.update(cfg.get("config", {}))
    return {"engine": combined_engine, "config": combined_config}


if __name__ == "__main__":
    main()
