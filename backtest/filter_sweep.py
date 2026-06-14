"""Filter variant sweep for low-trade strategies -- 100 days, realistic backtest.

Tests TIGHT / MEDIUM / LOOSE filter configs for ORB_BREAKOUT, BB_SQUEEZE,
VWAP_BOUNCE, and RSI_DIVERGENCE in isolation, then runs BEST COMBO.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data
from backtest.realistic_backtest import run_realistic_backtest
from engine.multi_strategy_engine import MultiStrategyEngine, TradeSignal

ALL_STRATEGIES = frozenset({
    "STOCH_CROSS", "PULLBACK", "EMA_MOMENTUM", "SUPERTREND", "RSI_REVERSION",
    "VWAP_MOMENTUM", "VWAP_MEAN_REV", "CPR_RANGE", "GAP_TRADE", "CPR_BREAKOUT",
    "ADX_BREAKOUT", "TREND_RIDE", "ORB_BREAKOUT", "BB_SQUEEZE", "VWAP_BOUNCE",
    "RSI_DIVERGENCE",
})

COMBO_BASE = frozenset({
    "PULLBACK", "TREND_RIDE", "STOCH_CROSS", "CPR_BREAKOUT",
})


@dataclass(frozen=True)
class OrbFilters:
    vol_min: float
    adx_min: float
    time_start: str
    time_end: str


@dataclass(frozen=True)
class BbSqueezeFilters:
    width_mult: float
    vol_min: float | None
    cpr_lo: float | None
    cpr_hi: float | None
    adx_lo: float
    adx_hi: float


@dataclass(frozen=True)
class VwapBounceFilters:
    dist_atr: float
    was_atr: float
    adx_block: float | None


@dataclass(frozen=True)
class RsiDivFilters:
    stoch_long: float
    stoch_short: float
    willr_long: float
    willr_short: float
    adx_max: float
    time_start: str
    time_end: str


ORB_VARIANTS = {
    "TIGHT": OrbFilters(vol_min=1.3, adx_min=15, time_start="09:45", time_end="11:00"),
    "MEDIUM": OrbFilters(vol_min=1.1, adx_min=12, time_start="09:45", time_end="12:00"),
    "LOOSE": OrbFilters(vol_min=1.0, adx_min=10, time_start="09:45", time_end="13:00"),
}

BB_VARIANTS = {
    "TIGHT": BbSqueezeFilters(1.3, 1.2, 20, 45, 12, 30),
    "MEDIUM": BbSqueezeFilters(1.5, 1.0, 15, 50, 10, 35),
    "LOOSE": BbSqueezeFilters(1.8, None, None, None, 8, 40),
}

VWAP_VARIANTS = {
    "TIGHT": VwapBounceFilters(0.25, 0.3, 40),
    "MEDIUM": VwapBounceFilters(0.4, 0.2, 45),
    "LOOSE": VwapBounceFilters(0.5, 0.15, None),
}

RSI_DIV_VARIANTS = {
    "TIGHT": RsiDivFilters(35, 65, -80, -20, 30, "11:00", "14:00"),
    "MEDIUM": RsiDivFilters(40, 60, -70, -30, 35, "10:30", "14:30"),
    "LOOSE": RsiDivFilters(45, 55, -60, -40, 40, "10:00", "15:00"),
}

SWEEP_STRATEGIES = ("ORB_BREAKOUT", "BB_SQUEEZE", "VWAP_BOUNCE", "RSI_DIVERGENCE")
VARIANT_ORDER = ("TIGHT", "MEDIUM", "LOOSE")


class FilterSweepEngine(MultiStrategyEngine):
    """Engine with parameterized filters for one target strategy."""

    def __init__(
        self,
        enabled: frozenset,
        orb: OrbFilters | None = None,
        bb: BbSqueezeFilters | None = None,
        vwap: VwapBounceFilters | None = None,
        rsi_div: RsiDivFilters | None = None,
    ):
        super().__init__()
        self.DISABLED_STRATEGIES = ALL_STRATEGIES - enabled
        self._orb_f = orb or ORB_VARIANTS["TIGHT"]
        self._bb_f = bb or BB_VARIANTS["TIGHT"]
        self._vwap_f = vwap or VWAP_VARIANTS["TIGHT"]
        self._rsi_div_f = rsi_div or RSI_DIV_VARIANTS["TIGHT"]

    def _check_orb_breakout(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        cfg = self._orb_f
        if self._orb_count >= self.MAX_ORB_PER_DAY:
            return None
        if np.isnan(self._orb_high) or np.isnan(self._orb_low):
            return None

        time_str = self._current_time
        if time_str < cfg.time_start or time_str >= cfg.time_end:
            return None

        close = self._sv(ind_dict["close"], idx)
        close_prev = self._sv(ind_dict["close"], idx - 1) if idx >= 1 else close
        if np.isnan(close) or np.isnan(close_prev):
            return None

        adx_val = self._sv(ind_dict.get("adx", pd.Series()), idx, 0)
        if adx_val < cfg.adx_min:
            return None

        rsi_5m = self._sv(ind_dict["rsi_5m"], idx, 50)
        plus_di = self._sv(ind_dict.get("plus_di", pd.Series()), idx, 0)
        minus_di = self._sv(ind_dict.get("minus_di", pd.Series()), idx, 0)
        vol = self._sv(ind_dict["volume"], idx, 0)
        vol_avg = self._sv(ind_dict["vol_avg"], idx, 1)
        vol_ratio = vol / vol_avg if vol_avg > 0 else 0
        if vol > 0 and vol_ratio < cfg.vol_min:
            return None

        direction = None
        reasons = []

        if (close > self._orb_high and close_prev <= self._orb_high
                and rsi_5m > 50 and plus_di > minus_di):
            direction = "LONG"
            reasons = [
                f"ORB break>{self._orb_high:.0f}",
                f"Vol={vol_ratio:.1f}x",
                f"RSI={rsi_5m:.0f}",
                f"+DI>{minus_di:.0f}",
            ]
        elif (close < self._orb_low and close_prev >= self._orb_low
              and rsi_5m < 50 and minus_di > plus_di):
            direction = "SHORT"
            reasons = [
                f"ORB break<{self._orb_low:.0f}",
                f"Vol={vol_ratio:.1f}x",
                f"RSI={rsi_5m:.0f}",
                f"-DI>{plus_di:.0f}",
            ]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        conf = 70
        if vol_ratio > 1.6:
            conf += 4
        if adx_val > 20:
            conf += 3
        di_spread = abs(plus_di - minus_di)
        if di_spread > 8:
            conf += min(int(di_spread / 3), 5)
        conf = min(conf, 100)

        return TradeSignal(
            direction=direction,
            signal_type="ORB_BREAKOUT",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
        )

    def _check_bb_squeeze(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        cfg = self._bb_f
        if self._squeeze_count >= self.MAX_SQUEEZE_PER_DAY:
            return None
        if idx < 50:
            return None

        bb_width = self._sv(ind_dict.get("bb_width", pd.Series()), idx - 1, np.nan)
        bb_min50 = self._sv(ind_dict.get("bb_width_min50", pd.Series()), idx - 1, np.nan)
        if np.isnan(bb_width) or np.isnan(bb_min50) or bb_min50 <= 0:
            return None
        if bb_width > bb_min50 * cfg.width_mult:
            return None

        cpr_w = ind_dict.get("cpr_width", 0)
        if cfg.cpr_lo is not None and cfg.cpr_hi is not None:
            if cpr_w < cfg.cpr_lo or cpr_w > cfg.cpr_hi:
                return None

        adx_val = self._sv(ind_dict.get("adx", pd.Series()), idx, 0)
        if adx_val < cfg.adx_lo or adx_val > cfg.adx_hi:
            return None

        close = self._sv(ind_dict["close"], idx)
        high_prev = self._sv(ind_dict["high"], idx - 1) if idx >= 1 else close
        low_prev = self._sv(ind_dict["low"], idx - 1) if idx >= 1 else close
        bb_pctb = self._sv(ind_dict.get("bb_pctb", pd.Series()), idx, 0.5)
        ema_9 = self._sv(ind_dict["ema_9"], idx, 0)
        ema_20 = self._sv(ind_dict["ema_20"], idx, close)
        vol = self._sv(ind_dict["volume"], idx, 0)
        vol_avg = self._sv(ind_dict["vol_avg"], idx, 1)
        vol_ratio = vol / vol_avg if vol_avg > 0 else 0

        if np.isnan(close) or ema_9 == 0:
            return None
        if cfg.vol_min is not None and vol > 0 and vol_ratio < cfg.vol_min:
            return None

        direction = None
        reasons = []

        if close > high_prev and bb_pctb > 0.85 and ema_9 > ema_20:
            direction = "LONG"
            reasons = [
                "BB squeeze break↑",
                f"BB%B={bb_pctb:.2f}",
                f"Vol={vol_ratio:.1f}x",
                f"CPR_W={cpr_w:.0f}",
            ]
        elif close < low_prev and bb_pctb < 0.15 and ema_9 < ema_20:
            direction = "SHORT"
            reasons = [
                "BB squeeze break↓",
                f"BB%B={bb_pctb:.2f}",
                f"Vol={vol_ratio:.1f}x",
                f"CPR_W={cpr_w:.0f}",
            ]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        squeeze_tightness = bb_min50 / bb_width if bb_width > 0 else 1.0
        conf = 68
        if vol_ratio > 1.5:
            conf += 4
        if squeeze_tightness < 0.85:
            conf += 4
        if 18 <= adx_val <= 25:
            conf += 3
        conf = min(conf, 100)

        return TradeSignal(
            direction=direction,
            signal_type="BB_SQUEEZE",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=self._sv(ind_dict["rsi_5m"], idx, 50),
            nifty_price=close,
            reason=" | ".join(reasons),
        )

    def _check_vwap_bounce(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        cfg = self._vwap_f
        if self._vwap_bounce_count >= self.MAX_VWAP_BOUNCE_PER_DAY:
            return None
        if idx < 5:
            return None

        close = self._sv(ind_dict["close"], idx)
        vwap_val = self._sv(ind_dict.get("vwap", pd.Series()), idx, np.nan)
        atr_val = self._sv(ind_dict["atr"], idx, 30)
        ema_9 = self._sv(ind_dict["ema_9"], idx, 0)
        ema_20 = self._sv(ind_dict["ema_20"], idx, close)
        adx_val = self._sv(ind_dict.get("adx", pd.Series()), idx, 0)
        cpr_w = ind_dict.get("cpr_width", 0)

        if np.isnan(close) or np.isnan(vwap_val) or atr_val <= 0:
            return None
        if cfg.adx_block is not None and adx_val > cfg.adx_block:
            return None
        if cpr_w <= 25:
            return None

        rsi_5m = self._sv(ind_dict["rsi_5m"], idx, 50)
        rsi_5m_prev = self._sv(ind_dict["rsi_5m"], idx - 1, rsi_5m)
        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        stoch_k = self._sv(ind_dict["stoch_k"], idx, 50)
        stoch_k_prev = self._sv(ind_dict["stoch_k"], idx - 1, stoch_k)
        vol = self._sv(ind_dict["volume"], idx, 0)
        vol_avg = self._sv(ind_dict["vol_avg"], idx, 1)
        vol_ratio = vol / vol_avg if vol_avg > 0 else 1.0

        vwap_dist = abs(close - vwap_val)
        at_vwap = vwap_dist <= cfg.dist_atr * atr_val

        direction = None
        reasons = []

        if ema_9 > ema_20 and rsi_15m > 52 and at_vwap:
            was_extended = False
            for lb in range(1, 6):
                c_lb = self._sv(ind_dict["close"], idx - lb)
                v_lb = self._sv(ind_dict.get("vwap", pd.Series()), idx - lb, np.nan)
                a_lb = self._sv(ind_dict["atr"], idx - lb, atr_val)
                if not np.isnan(c_lb) and not np.isnan(v_lb) and a_lb > 0:
                    if c_lb > v_lb + cfg.was_atr * a_lb:
                        was_extended = True
                        break
            rsi_cross_up = rsi_5m_prev <= 45 < rsi_5m
            stoch_rising = stoch_k > stoch_k_prev
            if was_extended and rsi_cross_up and stoch_rising:
                direction = "LONG"
                reasons = [
                    "VWAP bounce↑",
                    "RSI5 cross 45",
                    f"RSI15={rsi_15m:.0f}",
                    f"VWAP dist={vwap_dist:.1f}",
                ]

        elif ema_9 < ema_20 and rsi_15m < 48 and at_vwap:
            was_extended = False
            for lb in range(1, 6):
                c_lb = self._sv(ind_dict["close"], idx - lb)
                v_lb = self._sv(ind_dict.get("vwap", pd.Series()), idx - lb, np.nan)
                a_lb = self._sv(ind_dict["atr"], idx - lb, atr_val)
                if not np.isnan(c_lb) and not np.isnan(v_lb) and a_lb > 0:
                    if c_lb < v_lb - cfg.was_atr * a_lb:
                        was_extended = True
                        break
            rsi_cross_down = rsi_5m_prev >= 55 > rsi_5m
            if was_extended and rsi_cross_down:
                direction = "SHORT"
                reasons = [
                    "VWAP bounce↓",
                    "RSI5 cross 55",
                    f"RSI15={rsi_15m:.0f}",
                    f"VWAP dist={vwap_dist:.1f}",
                ]

        if not direction:
            return None

        conf = 68
        touch_precision = 1.0 - min(vwap_dist / (cfg.dist_atr * atr_val), 1.0)
        conf += int(touch_precision * 5)
        htf_strength = abs(rsi_15m - 50)
        if htf_strength > 8:
            conf += min(int(htf_strength * 0.3), 5)
        if vol_ratio > 1.2:
            conf += 3
        conf = min(conf, 100)

        return TradeSignal(
            direction=direction,
            signal_type="VWAP_BOUNCE",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
        )

    def _check_rsi_divergence(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        cfg = self._rsi_div_f
        if self._divergence_count >= self.MAX_DIVERGENCE_PER_DAY:
            return None
        if idx < 10:
            return None

        time_str = self._current_time
        if time_str < cfg.time_start or time_str >= cfg.time_end:
            return None

        rsi_div = self._sv(ind_dict.get("rsi_div", pd.Series()), idx, 0)
        if rsi_div == 0:
            return None

        adx_val = self._sv(ind_dict.get("adx", pd.Series()), idx, 0)
        if adx_val >= cfg.adx_max:
            return None

        rsi_5m = self._sv(ind_dict["rsi_5m"], idx, 50)
        rsi_5m_prev = self._sv(ind_dict["rsi_5m"], idx - 1, rsi_5m)
        stoch_k = self._sv(ind_dict["stoch_k"], idx, 50)
        stoch_k_prev = self._sv(ind_dict["stoch_k"], idx - 1, stoch_k)
        willr = self._sv(ind_dict["willr"], idx, -50)
        close = self._sv(ind_dict["close"], idx)
        vwap_val = self._sv(ind_dict.get("vwap", pd.Series()), idx, np.nan)
        atr_val = self._sv(ind_dict["atr"], idx, 30)

        if np.isnan(close):
            return None

        direction = None
        reasons = []

        if (rsi_div == 1 and rsi_5m > rsi_5m_prev
                and stoch_k < cfg.stoch_long and stoch_k > stoch_k_prev
                and willr < cfg.willr_long):
            direction = "LONG"
            reasons = [
                "Bullish RSI div",
                f"RSI={rsi_5m:.0f}↑",
                f"Stoch={stoch_k:.0f}",
                f"WR={willr:.0f}",
            ]
        elif (rsi_div == -1 and rsi_5m < rsi_5m_prev
              and stoch_k > cfg.stoch_short and stoch_k < stoch_k_prev
              and willr > cfg.willr_short):
            direction = "SHORT"
            reasons = [
                "Bearish RSI div",
                f"RSI={rsi_5m:.0f}↓",
                f"Stoch={stoch_k:.0f}",
                f"WR={willr:.0f}",
            ]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)

        lookback = 10
        close_now = close
        close_lb = self._sv(ind_dict["close"], idx - lookback, close)
        rsi_lb = self._sv(ind_dict["rsi_5m"], idx - lookback, rsi_5m)
        price_move = abs(close_now - close_lb)
        rsi_move = abs(rsi_5m - rsi_lb)
        div_magnitude = rsi_move / max(price_move / max(atr_val, 1), 0.01)

        conf = 66
        if div_magnitude > 2:
            conf += min(int(div_magnitude), 6)
        if not np.isnan(vwap_val) and atr_val > 0:
            vwap_prox = abs(close - vwap_val) / atr_val
            if vwap_prox < 0.5:
                conf += 4
        if direction == "LONG" and willr < cfg.willr_long - 5:
            conf += 3
        elif direction == "SHORT" and willr > cfg.willr_short + 5:
            conf += 3
        conf = min(conf, 100)

        return TradeSignal(
            direction=direction,
            signal_type="RSI_DIVERGENCE",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
        )


def _run_isolated(
    strategy: str,
    df: pd.DataFrame,
    capital: int,
    lot_size: int,
    orb: OrbFilters | None = None,
    bb: BbSqueezeFilters | None = None,
    vwap: VwapBounceFilters | None = None,
    rsi_div: RsiDivFilters | None = None,
) -> dict:
    engine = FilterSweepEngine(
        enabled=frozenset({strategy}),
        orb=orb,
        bb=bb,
        vwap=vwap,
        rsi_div=rsi_div,
    )
    return run_realistic_backtest(
        df, starting_capital=capital, lot_size=lot_size, engine_override=engine,
    )


def _run_combo(
    df: pd.DataFrame,
    capital: int,
    lot_size: int,
    best_orb: OrbFilters,
    best_bb: BbSqueezeFilters,
    best_vwap: VwapBounceFilters,
    best_rsi: RsiDivFilters,
) -> dict:
    enabled = COMBO_BASE | {
        "ORB_BREAKOUT", "BB_SQUEEZE", "VWAP_BOUNCE", "RSI_DIVERGENCE",
    }
    engine = FilterSweepEngine(
        enabled=enabled,
        orb=best_orb,
        bb=best_bb,
        vwap=best_vwap,
        rsi_div=best_rsi,
    )
    return run_realistic_backtest(
        df, starting_capital=capital, lot_size=lot_size, engine_override=engine,
    )


def _format_row(r: dict) -> str:
    pnl = r["total_pnl"]
    return (
        f"{r['total_trades']:>3} trades | {r['win_rate']:>4.1f}% WR | "
        f"{r['profit_factor']:>4.2f} PF | Rs {pnl:>7,.0f} | "
        f"{r['max_drawdown_pct']:>4.1f}% DD"
    )


def _pick_best(variant_results: dict[str, dict]) -> tuple[str, dict]:
    """Pick best variant by profit factor, then total PnL."""

    def score(name: str) -> tuple:
        r = variant_results[name]
        pf = r["profit_factor"] if r["total_trades"] >= 1 else -1.0
        return (pf, r["total_pnl"], r["total_trades"])

    best_name = max(variant_results, key=score)
    return best_name, variant_results[best_name]


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    days = 100
    capital = 10_000
    lot_size = settings.NIFTY_LOT_SIZE

    print(f"\nLoading {days} trading days of Nifty data...")
    df = load_real_data(days=days)
    unique_days = sorted(set(df.index.date))
    print(f"Loaded {len(df)} candles across {len(unique_days)} days\n")

    all_results: dict[str, dict[str, dict]] = {}
    best_filters: dict[str, object] = {}

    variant_maps = {
        "ORB_BREAKOUT": ORB_VARIANTS,
        "BB_SQUEEZE": BB_VARIANTS,
        "VWAP_BOUNCE": VWAP_VARIANTS,
        "RSI_DIVERGENCE": RSI_DIV_VARIANTS,
    }

    for strategy in SWEEP_STRATEGIES:
        all_results[strategy] = {}
        variants = variant_maps[strategy]
        for variant_name in VARIANT_ORDER:
            print(f"  Running {strategy} {variant_name}...", flush=True)
            kwargs = {}
            if strategy == "ORB_BREAKOUT":
                kwargs["orb"] = variants[variant_name]
            elif strategy == "BB_SQUEEZE":
                kwargs["bb"] = variants[variant_name]
            elif strategy == "VWAP_BOUNCE":
                kwargs["vwap"] = variants[variant_name]
            else:
                kwargs["rsi_div"] = variants[variant_name]

            result = _run_isolated(strategy, df, capital, lot_size, **kwargs)
            all_results[strategy][variant_name] = result

        best_name, _ = _pick_best(all_results[strategy])
        best_filters[strategy] = variants[best_name]

    print("\n  Running BEST COMBO...", flush=True)
    combo_result = _run_combo(
        df, capital, lot_size,
        best_orb=best_filters["ORB_BREAKOUT"],
        best_bb=best_filters["BB_SQUEEZE"],
        best_vwap=best_filters["VWAP_BOUNCE"],
        best_rsi=best_filters["RSI_DIVERGENCE"],
    )

    print()
    print("STRATEGY FILTER SWEEP RESULTS (100 days, realistic backtest)")
    print("=" * 60)
    print()

    for strategy in SWEEP_STRATEGIES:
        print(f"{strategy}:")
        for variant_name in VARIANT_ORDER:
            r = all_results[strategy][variant_name]
            print(f"  {variant_name:7s}: {_format_row(r)}")
        best_name, _ = _pick_best(all_results[strategy])
        print(f"  >> best: {best_name}")
        print()

    print("BEST COMBO (all winners + existing):")
    print(f"  {_format_row(combo_result)}")
    print()

    return all_results, combo_result


if __name__ == "__main__":
    main()
