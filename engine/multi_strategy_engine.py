"""Multi-Strategy Engine -- DeltaForge V11 -- Multi-Strategy (76% WR, PF 2.62).

Validated on Nifty 50 5m real data (100 days, 72 active trading days):
  Compound: 101 trades, 76.2% WR, PF 2.62, Rs 10K -> Rs 5.66L (+5561%)
  Daily compound rate: ~5.8% geometric
  Strategy breakdown:
    STOCH_CROSS:  14 trades, 85.7% WR, PF 40.4, Rs +1.97L
    PULLBACK:     28 trades, 75.0% WR, PF 1.70, Rs +1.64L
    SUPERTREND:   13 trades, 92.3% WR, PF 5.74, Rs +1.99L

Architecture:
  STOCH_CROSS = Stochastic cross from extreme (25/75) with EMA trend
  PULLBACK    = Multi-oscillator pullback in HTF trend
  SUPERTREND  = Supertrend(10,3) flip with fast ST(7,2) confidence boost

Trailing stop: trigger at 12% gain, trail 8% from peak.
Targets: strategy-specific multipliers (SUPERTREND widest, PULLBACK tightest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from engine import indicators as ind
from engine import indicators_extended as indx
from risk.shock_detector import ShockDetector


@dataclass
class TradeSignal:
    direction: str
    signal_type: str      # "PULLBACK" or "STOCH_CROSS"
    confidence: float
    htf_rsi: float
    ltf_rsi: float
    nifty_price: float
    reason: str
    pullback_count: int = 0
    vol_ratio: float = 1.0  # ATR-based volatility scalar for adaptive SL
    adx: float = 0.0

    def summary(self) -> str:
        return (f"{self.signal_type} {self.direction} "
                f"conf={self.confidence:.0f} "
                f"HTF={self.htf_rsi:.0f} LTF={self.ltf_rsi:.0f} "
                f"confirms={self.pullback_count}")


class RegimeDetector:
    """Lightweight intraday regime classifier.

    Runs once after ORB (09:45) and sets the day's directional bias.
    Strategies aligned with the regime get a confidence boost; those
    fighting it get penalised.
    """

    BOOST = 10       # confidence added for regime-aligned signals
    PENALTY = 15     # confidence subtracted for regime-opposing signals
    GAP_ATR_RATIO = 0.5   # gap must exceed 0.5x prev-day ATR

    ALIGNED_STRATEGIES = {
        "GAP_FADE_SHORT": {"GAP_TRADE", "VWAP_MEAN_REV"},
        "GAP_FADE_LONG":  {"GAP_TRADE", "VWAP_MEAN_REV"},
    }

    def __init__(self):
        self.regime: str = "UNKNOWN"

    def classify(self, day_gap: float, orb_high: float, orb_low: float,
                 orb_close: float, prev_day_atr: float) -> str:
        if prev_day_atr <= 0 or np.isnan(prev_day_atr):
            prev_day_atr = 100.0

        if abs(day_gap) > prev_day_atr * self.GAP_ATR_RATIO:
            orb_mid = (orb_high + orb_low) / 2
            if day_gap > 0 and orb_close < orb_mid:
                self.regime = "GAP_FADE_SHORT"
                return self.regime
            if day_gap < 0 and orb_close > orb_mid:
                self.regime = "GAP_FADE_LONG"
                return self.regime

        if orb_close > orb_high * 0.998:
            self.regime = "TREND_LONG"
        elif orb_close < orb_low * 1.002:
            self.regime = "TREND_SHORT"
        else:
            self.regime = "RANGE"
        return self.regime

    def adjust_confidence(self, signal: TradeSignal) -> None:
        """Boost or penalise confidence based on regime alignment."""
        if self.regime == "UNKNOWN":
            return

        aligned_strats = self.ALIGNED_STRATEGIES.get(self.regime, set())
        if signal.signal_type in aligned_strats:
            signal.confidence = min(signal.confidence + self.BOOST + 5, 100)
            return

        if self.regime == "GAP_FADE_SHORT":
            if signal.direction == "SHORT":
                signal.confidence = min(signal.confidence + self.BOOST, 100)
            elif signal.direction == "LONG":
                signal.confidence = max(signal.confidence - self.PENALTY, 0)

        elif self.regime == "GAP_FADE_LONG" or self.regime == "TREND_LONG":
            if signal.direction == "LONG":
                signal.confidence = min(signal.confidence + self.BOOST, 100)
            elif signal.direction == "SHORT":
                signal.confidence = max(signal.confidence - self.PENALTY, 0)

        elif self.regime == "TREND_SHORT":
            if signal.direction == "SHORT":
                signal.confidence = min(signal.confidence + self.BOOST, 100)
            elif signal.direction == "LONG":
                signal.confidence = max(signal.confidence - self.PENALTY, 0)


def _round_number_boost(signal: TradeSignal) -> None:
    """Boost confidence near psychological levels (x00, x50)."""
    price = signal.nifty_price
    dist_100 = min(price % 100, 100 - price % 100)
    dist_50 = min(price % 50, 50 - price % 50)
    near_round = min(dist_100, dist_50)
    if near_round <= 15:
        signal.confidence = min(signal.confidence + 5, 100)


class MultiStrategyEngine:
    """DeltaForge V11 -- Multi-Strategy (76% WR, PF 2.62).

    Best-confidence selection: all strategies evaluated per bar, highest confidence wins.
    Bar-quality filter no longer wastes signal slots on rejected signals.
    """

    HTF_BULL_RSI = 50
    HTF_BEAR_RSI = 50

    HTF_DEAD_ZONE_LO = 0   # skip when |RSI15 - 50| is in [0, 5)
    HTF_DEAD_ZONE_HI = 5   # trade when RSI > 55 or < 45

    MIN_ADX = 10            # low bar for directional move
    MAX_ADX = 50            # raised to 50 -- optimizer: captures strong trends without filtering
    SUPERTREND_MIN_ADX = 25 # stricter ADX gate for SUPERTREND strategy

    # ── Strategies disabled (net losers on real weekly option data) ──
    # Enabled: PULLBACK, SUPERTREND, TREND_RIDE, STOCH_CROSS
    # (validated 6.30% geo daily on 99 days real weekly option premiums)
    DISABLED_STRATEGIES = {
        "EMA_MOMENTUM",
        "VWAP_MOMENTUM",
        "VWAP_MEAN_REV",
        "RSI_REVERSION",
        "ADX_BREAKOUT",
        "ORB_BREAKOUT",
        "BB_SQUEEZE",
        "VWAP_BOUNCE",
        "RSI_DIVERGENCE",
        "CPR_RANGE",
        "CPR_BREAKOUT",
        "GAP_TRADE",
        "FIRST_HOUR_MOM",
        "NARROW_CPR_BO",
        "TRIPLE_CONFIRM",
        "VWAP_2SD_REV",
    }

    MAX_STOCH_PER_DAY = 10
    MAX_PULLBACK_PER_DAY = 15
    MAX_MOMENTUM_PER_DAY = 10
    MAX_SUPERTRD_PER_DAY = 3
    MAX_RSI_REV_PER_DAY = 10
    MAX_VWAP_PER_DAY = 5
    MAX_VWAP_MR_PER_DAY = 5
    MAX_CPR_RANGE_PER_DAY = 10
    MAX_GAP_PER_DAY = 5
    MAX_CPR_BREAKOUT_PER_DAY = 10
    MAX_ADX_BREAKOUT_PER_DAY = 2
    MAX_TREND_RIDE_PER_DAY = 5
    MAX_ORB_PER_DAY = 2
    MAX_SQUEEZE_PER_DAY = 3
    MAX_VWAP_BOUNCE_PER_DAY = 4
    MAX_DIVERGENCE_PER_DAY = 3
    MAX_TRIPLE_CONFIRM_PER_DAY = 3
    MAX_FIRST_HOUR_PER_DAY = 2
    MAX_VWAP_2SD_PER_DAY = 4
    MAX_NARROW_CPR_PER_DAY = 3
    MAX_TOTAL_PER_DAY = 12
    USE_VWAP_FILTER = False
    COOLDOWN_BARS = 3
    SL_COOLDOWN_BARS = 6  # block same strategy for 6 bars (30 min) after SL

    def __init__(self, enabled_strategies: set[str] | None = None,
                 disabled_strategies_override: set[str] | None = None,
                 max_adx_override: float | None = None):
        self._enabled_strategies = enabled_strategies
        if disabled_strategies_override is not None:
            self._disabled_set = disabled_strategies_override
        elif enabled_strategies is not None:
            all_strats = {
                "PULLBACK", "STOCH_CROSS", "EMA_MOMENTUM", "SUPERTREND",
                "RSI_REVERSION", "VWAP_MOMENTUM", "VWAP_MEAN_REV",
                "CPR_RANGE", "GAP_TRADE", "CPR_BREAKOUT", "ADX_BREAKOUT",
                "TREND_RIDE", "ORB_BREAKOUT", "BB_SQUEEZE", "VWAP_BOUNCE",
                "RSI_DIVERGENCE",
                "TRIPLE_CONFIRM", "FIRST_HOUR_MOM", "VWAP_2SD_REV",
                "NARROW_CPR_BO",
            }
            self._disabled_set = all_strats - enabled_strategies
        else:
            self._disabled_set = self.DISABLED_STRATEGIES
        if max_adx_override is not None:
            self._max_adx = max_adx_override
        else:
            self._max_adx = self.MAX_ADX
        self._signals_today: list[TradeSignal] = []
        self._pullback_count = 0
        self._stoch_count = 0
        self._momentum_count = 0
        self._supertrd_count = 0
        self._rsi_rev_count = 0
        self._vwap_count = 0
        self._vwap_mr_count = 0
        self._cpr_range_count = 0
        self._gap_count = 0
        self._cpr_breakout_count = 0
        self._adx_breakout_count = 0
        self._trend_ride_count = 0
        self._orb_count = 0
        self._squeeze_count = 0
        self._vwap_bounce_count = 0
        self._divergence_count = 0
        self._triple_confirm_count = 0
        self._first_hour_count = 0
        self._vwap_2sd_count = 0
        self._narrow_cpr_count = 0
        self._orb_high: float = np.nan
        self._orb_low: float = np.nan
        self._used_bars: set[int] = set()
        self._current_time: str = "12:00"
        self._sl_cooldown: dict[str, int] = {}
        self.shock = ShockDetector(
            threshold_pct=settings.SHOCK_THRESHOLD_PCT,
            lookback_bars=settings.SHOCK_LOOKBACK_BARS,
            halt_bars=6,
        )
        self._prev_day_high: float = np.nan
        self._prev_day_low: float = np.nan
        self._prev_day_close: float = np.nan
        self.regime = RegimeDetector()
        self._regime_classified: bool = False
        self._session_low: float = np.inf
        self._session_high: float = -np.inf
        self._session_low_bar: int = -100
        self._session_high_bar: int = -100
        self._double_bottom: bool = False
        self._double_top: bool = False
        self._double_bottom_level: float = np.nan
        self._double_top_level: float = np.nan

    def reset_day(self, prev_day_data: dict | None = None):
        self._signals_today = []
        self._pullback_count = 0
        self._stoch_count = 0
        self._momentum_count = 0
        self._supertrd_count = 0
        self._rsi_rev_count = 0
        self._vwap_count = 0
        self._vwap_mr_count = 0
        self._cpr_range_count = 0
        self._gap_count = 0
        self._cpr_breakout_count = 0
        self._adx_breakout_count = 0
        self._trend_ride_count = 0
        self._orb_count = 0
        self._squeeze_count = 0
        self._vwap_bounce_count = 0
        self._divergence_count = 0
        self._triple_confirm_count = 0
        self._first_hour_count = 0
        self._vwap_2sd_count = 0
        self._narrow_cpr_count = 0
        self._orb_high = np.nan
        self._orb_low = np.nan
        self._used_bars = set()
        self._sl_cooldown = {}
        self.shock.reset()
        self.regime = RegimeDetector()
        self._regime_classified = False
        self._session_low = np.inf
        self._session_high = -np.inf
        self._session_low_bar = -100
        self._session_high_bar = -100
        self._double_bottom = False
        self._double_top = False
        self._double_bottom_level = np.nan
        self._double_top_level = np.nan
        if prev_day_data:
            self._prev_day_high = prev_day_data.get("high", np.nan)
            self._prev_day_low = prev_day_data.get("low", np.nan)
            self._prev_day_close = prev_day_data.get("close", np.nan)
            self._prev_day_atr = prev_day_data.get("atr", self._prev_day_high - self._prev_day_low)
        else:
            self._prev_day_atr = 100.0

    def record_sl_exit(self, strategy_name: str, bar_idx: int):
        """Called by trading engine when a position exits via SL.

        Blocks the same strategy from firing for SL_COOLDOWN_BARS bars.
        """
        self._sl_cooldown[strategy_name] = bar_idx + self.SL_COOLDOWN_BARS

    def _is_strategy_cooled(self, signal_type: str, bar_idx: int) -> bool:
        """Return True if the strategy is still in SL cooldown."""
        cooldown_until = self._sl_cooldown.get(signal_type, -1)
        return bar_idx < cooldown_until

    def precompute(self, candles: pd.DataFrame) -> dict:
        indicators = {}

        indicators['close'] = candles['close']
        indicators['high'] = candles['high']
        indicators['low'] = candles['low']
        indicators['open'] = candles['open']
        indicators['volume'] = candles['volume']

        indicators['rsi_5m'] = ind.rsi(candles['close'], 14)
        k, d = ind.stochastic(candles['high'], candles['low'],
                              candles['close'], 14, 3)
        indicators['stoch_k'] = k
        indicators['stoch_d'] = d
        indicators['cci'] = ind.cci(candles['high'], candles['low'],
                                    candles['close'], 20)
        indicators['willr'] = ind.williams_r(candles['high'], candles['low'],
                                             candles['close'], 14)

        try:
            htf = indx.compute_htf_indicators(candles)
            indicators['rsi_15m'] = htf.get(
                'htf_15m_rsi',
                pd.Series(50.0, index=candles.index)
            )
            indicators['htf_15m_index'] = htf.get('htf_15m_index')
        except Exception:
            indicators['rsi_15m'] = pd.Series(50.0, index=candles.index)
            indicators['htf_15m_index'] = None

        _st_line, st_dir = ind.supertrend(candles, period=10, multiplier=3)
        indicators['supertrend_dir'] = st_dir

        _st_fast_line, st_fast_dir = ind.supertrend(candles, period=7, multiplier=2)
        indicators['supertrend_fast_dir'] = st_fast_dir

        indicators['ema_9'] = ind.ema(candles['close'], 9)
        indicators['ema_20'] = ind.ema(candles['close'], 20)
        indicators['ema_21'] = ind.ema(candles['close'], 21)
        indicators['ema_50'] = ind.ema(candles['close'], 50)

        try:
            indicators['vwap'] = ind.vwap_intraday(candles)
        except Exception:
            indicators['vwap'] = pd.Series(np.nan, index=candles.index)

        _bb_u, _bb_m, _bb_l, bb_pctb, bb_bw = ind.bollinger_bands(candles['close'], 20, 2.0)
        indicators['bb_pctb'] = bb_pctb
        indicators['bb_width'] = _bb_u - _bb_l
        indicators['bb_bandwidth'] = bb_bw
        indicators['bb_width_min50'] = indicators['bb_width'].rolling(50, min_periods=20).min()

        indicators['rsi_div'] = indx.detect_divergence(
            candles['close'], indicators['rsi_5m'], lookback=10)

        indicators['atr'] = ind.atr(candles, 14)
        indicators['vol_avg'] = candles['volume'].rolling(20).mean()
        adx_val, plus_di, minus_di = ind.adx(candles['high'], candles['low'], candles['close'], 14)
        indicators['adx'] = adx_val
        indicators['plus_di'] = plus_di
        indicators['minus_di'] = minus_di

        # VWAP standard deviation bands for mean reversion
        vwap_series = indicators.get('vwap', pd.Series(np.nan, index=candles.index))
        vwap_diff = candles['close'] - vwap_series
        vwap_std = vwap_diff.rolling(20, min_periods=5).std()
        indicators['vwap_std'] = vwap_std

        macd_line, macd_signal, macd_hist = ind.macd(candles['close'])
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = macd_signal
        indicators['macd_hist'] = macd_hist

        first_hour_bars = candles.head(12)
        if len(first_hour_bars) >= 3:
            indicators['first_hour_high'] = first_hour_bars['high'].max()
            indicators['first_hour_low'] = first_hour_bars['low'].min()
        else:
            indicators['first_hour_high'] = np.nan
            indicators['first_hour_low'] = np.nan

        # CPR levels from previous day data
        pp = (self._prev_day_high + self._prev_day_low + self._prev_day_close) / 3
        bcp = (self._prev_day_high + self._prev_day_low) / 2
        tcp = 2 * pp - bcp
        indicators['cpr_pp'] = pp
        indicators['cpr_bcp'] = bcp
        indicators['cpr_tcp'] = tcp
        indicators['cpr_width'] = abs(tcp - bcp)

        # Gap: today's open vs previous close
        if len(candles) > 0 and not np.isnan(self._prev_day_close):
            indicators['day_gap'] = candles['open'].iloc[0] - self._prev_day_close
            indicators['day_open'] = candles['open'].iloc[0]
        else:
            indicators['day_gap'] = 0.0
            indicators['day_open'] = candles['open'].iloc[0] if len(candles) > 0 else np.nan

        self._compute_orb(candles)

        return indicators

    def _compute_orb(self, candles: pd.DataFrame):
        """Set opening range high/low from the first 3 bars of the session."""
        self._orb_high = np.nan
        self._orb_low = np.nan
        if len(candles) == 0:
            return

        orb_bars = candles.head(3)
        if len(orb_bars) > 0:
            self._orb_high = orb_bars['high'].max()
            self._orb_low = orb_bars['low'].min()

    MIN_VOL_RATIO = 0.8    # skip if volume < 80% of 20-bar avg
    MIN_BODY_ATR = 0.3     # skip if candle body < 30% of ATR
    # Signal passes if EITHER condition is met (OR gate).
    # Low-vol + tiny-body candles have 29-41% WR vs higher when filtered.

    def _apply_structure_boost(self, sig: TradeSignal, ind: dict, idx: int):
        """Boost confidence for double-bottom/top patterns (session state)."""
        low = self._sv(ind['low'], idx, np.nan)
        high = self._sv(ind['high'], idx, np.nan)
        if np.isnan(low) or np.isnan(high):
            return

        tolerance = 10.0
        min_gap_bars = 12

        if abs(low - self._session_low) <= tolerance and idx - self._session_low_bar >= min_gap_bars:
            self._double_bottom = True
            self._double_bottom_level = self._session_low
        if low < self._session_low:
            self._session_low = low
            self._session_low_bar = idx

        if abs(high - self._session_high) <= tolerance and idx - self._session_high_bar >= min_gap_bars:
            self._double_top = True
            self._double_top_level = self._session_high
        if high > self._session_high:
            self._session_high = high
            self._session_high_bar = idx

        if self._double_bottom and sig.direction == "LONG":
            sig.confidence = min(sig.confidence + 8, 100)
        if self._double_top and sig.direction == "SHORT":
            sig.confidence = min(sig.confidence + 8, 100)

    def scan(self, indicators: dict, bar_idx: int,
             time_str: str = "", max_total_override: int | None = None,
             entry_start_override: str | None = None,
             entry_end_override: str | None = None) -> list[TradeSignal]:
        self._current_time = time_str or "12:00"

        max_today = max_total_override if max_total_override is not None else self.MAX_TOTAL_PER_DAY
        total = (self._pullback_count + self._stoch_count + self._momentum_count
                 + self._supertrd_count + self._rsi_rev_count + self._vwap_count
                 + self._vwap_mr_count + self._cpr_range_count + self._gap_count
                 + self._cpr_breakout_count + self._adx_breakout_count
                 + self._trend_ride_count + self._orb_count + self._squeeze_count
                 + self._vwap_bounce_count + self._divergence_count)
        if total >= max_today:
            return []

        if bar_idx in self._used_bars:
            return []

        if time_str:
            entry_start = entry_start_override or "09:30"
            entry_end = entry_end_override or getattr(settings, 'ENTRY_END', "14:30")
            if time_str < entry_start or time_str > entry_end:
                return []

        # ADX regime gate: skip choppy/ranging AND overextended markets
        adx_val = self._sv(indicators.get('adx', pd.Series()), bar_idx, 25.0)
        if adx_val < self.MIN_ADX:
            return []
        if adx_val > self._max_adx:
            return []

        if not self.shock.check(indicators['close'], bar_idx):
            return []

        if bar_idx < 3:
            bar_high = self._sv(indicators['high'], bar_idx)
            bar_low = self._sv(indicators['low'], bar_idx)
            if not np.isnan(bar_high) and not np.isnan(bar_low):
                if bar_idx == 0 or np.isnan(self._orb_high):
                    self._orb_high = bar_high
                    self._orb_low = bar_low
                else:
                    self._orb_high = max(self._orb_high, bar_high)
                    self._orb_low = min(self._orb_low, bar_low)

        if not self._regime_classified and bar_idx >= 3:
            orb_close = self._sv(indicators['close'], bar_idx)
            day_gap = indicators.get('day_gap', 0.0)
            if isinstance(day_gap, pd.Series):
                day_gap = float(day_gap.iloc[0]) if len(day_gap) > 0 else 0.0
            r = self.regime.classify(
                day_gap=float(day_gap),
                orb_high=self._orb_high if not np.isnan(self._orb_high) else orb_close,
                orb_low=self._orb_low if not np.isnan(self._orb_low) else orb_close,
                orb_close=orb_close,
                prev_day_atr=getattr(self, '_prev_day_atr', 100.0),
            )
            self._regime_classified = True
            logger.info("REGIME classified: {} | gap={:.0f} ORB={:.0f}-{:.0f}",
                        r, day_gap, self._orb_low, self._orb_high)

        vwap_val = self._sv(indicators.get('vwap', pd.Series()), bar_idx, np.nan)
        close_val = self._sv(indicators['close'], bar_idx)

        def _vwap_ok(sig):
            if not self.USE_VWAP_FILTER or np.isnan(vwap_val):
                return True
            if sig.direction == "LONG" and close_val < vwap_val:
                return False
            return not (sig.direction == "SHORT" and close_val > vwap_val)

        def _vwap_boost(sig):
            """VWAP soft boost: +5 confidence when direction agrees with VWAP position."""
            if np.isnan(vwap_val):
                return
            if (sig.direction == "LONG" and close_val > vwap_val) or \
               (sig.direction == "SHORT" and close_val < vwap_val):
                sig.confidence = min(sig.confidence + 5, 100)

        candidates = []
        for check_fn, counter_attr in [
            (self._check_stoch_cross, '_stoch_count'),
            (self._check_pullback, '_pullback_count'),
            (self._check_ema_momentum, '_momentum_count'),
            (self._check_supertrend_flip, '_supertrd_count'),
            (self._check_rsi_reversion, '_rsi_rev_count'),
            (self._check_vwap_momentum, '_vwap_count'),
            (self._check_vwap_mean_reversion, '_vwap_mr_count'),
            (self._check_cpr_range, '_cpr_range_count'),
            (self._check_gap_trade, '_gap_count'),
            (self._check_cpr_breakout, '_cpr_breakout_count'),
            (self._check_adx_breakout, '_adx_breakout_count'),
            (self._check_trend_ride, '_trend_ride_count'),
            (self._check_orb_breakout, '_orb_count'),
            (self._check_bb_squeeze, '_squeeze_count'),
            (self._check_vwap_bounce, '_vwap_bounce_count'),
            (self._check_rsi_divergence, '_divergence_count'),
            (self._check_triple_confirm, '_triple_confirm_count'),
            (self._check_first_hour_momentum, '_first_hour_count'),
            (self._check_vwap_2sd_reversion, '_vwap_2sd_count'),
            (self._check_narrow_cpr_breakout, '_narrow_cpr_count'),
        ]:
            sig = check_fn(indicators, bar_idx)
            if sig and _vwap_ok(sig):
                if sig.signal_type in self._disabled_set:
                    continue
                if self._is_strategy_cooled(sig.signal_type, bar_idx):
                    continue
                _vwap_boost(sig)
                self.regime.adjust_confidence(sig)
                candidates.append((sig, counter_attr))

        if not candidates:
            return []

        if candidates and len(candidates) > 1:
            logger.debug("Signals at bar {}: {}", bar_idx,
                         " | ".join(f"{s.signal_type} {s.direction} conf={s.confidence:.0f}"
                                    for s, _ in candidates))

        best_sig, best_counter = max(candidates, key=lambda x: x[0].confidence)

        bar_vol = indicators.get("volume")
        if bar_vol is not None and bar_idx < len(bar_vol) and bar_vol.iloc[bar_idx] == 0:
            logger.debug("Bar {} has volume=0 (REST fallback) -- proceeding with OHLC only", bar_idx)

        if not self._bar_quality_ok(indicators, bar_idx):
            return []

        atr_series = indicators.get("atr")
        if atr_series is not None and bar_idx >= 100:
            recent_atr = atr_series.iloc[max(0, bar_idx - 20):bar_idx].mean()
            baseline_atr = atr_series.iloc[max(0, bar_idx - 100):max(0, bar_idx - 20)].mean()
            if baseline_atr > 0 and not np.isnan(baseline_atr) and not np.isnan(recent_atr):
                best_sig.vol_ratio = max(0.8, min(recent_atr / baseline_atr, 2.0))

        _round_number_boost(best_sig)
        self._apply_structure_boost(best_sig, indicators, bar_idx)

        self._signals_today.append(best_sig)
        setattr(self, best_counter, getattr(self, best_counter) + 1)
        self._used_bars.update(range(bar_idx, bar_idx + self.COOLDOWN_BARS))
        return [best_sig]

    def _bar_quality_ok(self, ind_dict: dict, idx: int) -> bool:
        """Reject indecision bars: low volume AND tiny body."""
        close = self._sv(ind_dict['close'], idx)
        opn = self._sv(ind_dict['open'], idx)
        vol = self._sv(ind_dict['volume'], idx, 0)
        vol_avg = self._sv(ind_dict['vol_avg'], idx, 1)
        atr_val = self._sv(ind_dict['atr'], idx, 30)

        if np.isnan(close) or np.isnan(opn):
            return False

        body = abs(close - opn)
        body_atr = body / atr_val if atr_val > 0 else 0.0

        # REST fallback sends volume=0 -- skip volume leg, use body only
        if vol == 0:
            return body_atr >= self.MIN_BODY_ATR

        vol_ratio = vol / vol_avg if vol_avg > 0 else 1.0
        return vol_ratio >= self.MIN_VOL_RATIO or body_atr >= self.MIN_BODY_ATR

    def _check_pullback(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        if self._pullback_count >= self.MAX_PULLBACK_PER_DAY:
            return None

        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        if rsi_15m == 50.0:
            if rsi_5m < 35:
                rsi_15m = 40
            elif rsi_5m > 65:
                rsi_15m = 60
        stoch_k = self._sv(ind_dict['stoch_k'], idx, 50)
        cci = self._sv(ind_dict['cci'], idx, 0)
        willr = self._sv(ind_dict['willr'], idx, -50)
        close = self._sv(ind_dict['close'], idx)
        ema_9 = self._sv(ind_dict['ema_9'], idx, close)
        ema_20 = self._sv(ind_dict['ema_20'], idx, close)
        st_dir = self._sv(ind_dict['supertrend_dir'], idx, 0)
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        rsi_prev = self._sv(ind_dict['rsi_5m'], idx - 1, rsi_5m) if idx >= 1 else rsi_5m

        if np.isnan(close):
            return None

        bull_trend = rsi_15m > self.HTF_BULL_RSI
        bear_trend = rsi_15m < self.HTF_BEAR_RSI

        if not (bull_trend or bear_trend):
            return None

        htf_strength = abs(rsi_15m - 50)
        if self.HTF_DEAD_ZONE_LO <= htf_strength < self.HTF_DEAD_ZONE_HI:
            return None

        if bull_trend:
            direction = "LONG"
            pb_count = 0
            trend_cont = False
            reasons = [f"15m_RSI={rsi_15m:.0f}↑"]
            if rsi_5m < 48:
                pb_count += 1
                reasons.append(f"RSI={rsi_5m:.0f}<48")
            if stoch_k < 30:
                pb_count += 1
                reasons.append(f"Stoch={stoch_k:.0f}<30")
            if cci < -80:
                pb_count += 1
                reasons.append(f"CCI={cci:.0f}<-80")
            if willr < -70:
                pb_count += 1
                reasons.append(f"WR={willr:.0f}<-70")
        else:
            direction = "SHORT"
            pb_count = 0
            trend_cont = False
            reasons = [f"15m_RSI={rsi_15m:.0f}↓"]
            if rsi_5m > 52:
                pb_count += 1
                reasons.append(f"RSI={rsi_5m:.0f}>52")
            if stoch_k > 70:
                pb_count += 1
                reasons.append(f"Stoch={stoch_k:.0f}>70")
            if cci > 80:
                pb_count += 1
                reasons.append(f"CCI={cci:.0f}>80")
            if willr > -30:
                pb_count += 1
                reasons.append(f"WR={willr:.0f}>-30")

            trend_cont = False
            if pb_count < 1 and adx_val > 35 and st_dir == -1 and ema_9 < ema_20:
                rsi_bounce = rsi_5m > rsi_prev + 2
                bear_candles = 0
                if idx >= 2:
                    for lb in range(3):
                        c = self._sv(ind_dict['close'], idx - lb)
                        o = self._sv(ind_dict['open'], idx - lb)
                        if not np.isnan(c) and not np.isnan(o) and c < o:
                            bear_candles += 1
                if rsi_bounce or bear_candles >= 3:
                    trend_cont = True
                    pb_count = 1
                    if rsi_bounce:
                        reasons.append(f"TrendCont RSI bounce {rsi_prev:.0f}→{rsi_5m:.0f}")
                    if bear_candles >= 3:
                        reasons.append(f"TrendCont {bear_candles} bear candles")

        min_pb = 1 if direction == "LONG" else 2
        if pb_count < min_pb:
            return None

        if trend_cont:
            conf = 60
        else:
            conf = 55 + (pb_count * 10)
        htf_strength = abs(rsi_15m - 50)
        conf += min(htf_strength * 0.3, 8)

        if direction == "LONG" and close > ema_20 or direction == "SHORT" and close < ema_20:
            conf += 3

        if direction == "LONG" and st_dir == 1 or direction == "SHORT" and st_dir == -1:
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

    def _check_stoch_cross(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Stochastic %K/%D cross from extreme zone with HTF RSI alignment."""
        if self._stoch_count >= self.MAX_STOCH_PER_DAY:
            return None

        if idx < 1:
            return None

        stoch_k = self._sv(ind_dict['stoch_k'], idx, 50)
        stoch_k_prev = self._sv(ind_dict['stoch_k'], idx - 1, 50)
        stoch_d = self._sv(ind_dict.get('stoch_d', pd.Series()), idx, 50)
        stoch_d_prev = self._sv(ind_dict.get('stoch_d', pd.Series()), idx - 1, 50)
        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        close = self._sv(ind_dict['close'], idx)

        if np.isnan(close):
            return None

        direction = None
        reasons = []

        if (stoch_k_prev < 20 and stoch_k > stoch_d
                and stoch_k_prev <= stoch_d_prev):
            direction = "LONG"
            reasons = [
                f"StochK={stoch_k:.0f} x D={stoch_d:.0f} from <20",
                f"RSI={rsi_5m:.0f}",
            ]
        elif (stoch_k_prev > 80 and stoch_k < stoch_d
              and stoch_k_prev >= stoch_d_prev):
            direction = "SHORT"
            reasons = [
                f"StochK={stoch_k:.0f} x D={stoch_d:.0f} from >80",
                f"RSI={rsi_5m:.0f}",
            ]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        if direction == "LONG" and rsi_15m <= 55:
            return None
        if direction == "SHORT" and rsi_15m >= 45:
            return None

        conf = 72
        htf_strength = abs(rsi_15m - 50)
        if htf_strength > 10:
            conf += 4

        return TradeSignal(
            direction=direction,
            signal_type="STOCH_CROSS",
            confidence=min(conf, 100),
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
            pullback_count=0,
        )

    def _check_ema_momentum(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """EMA momentum: EMA(9) crosses EMA(21) with strict multi-factor confirmation.

        All 5 conditions must align (AND gate):
          1. Fresh EMA(9)/EMA(21) crossover (within last 3 bars)
          2. ADX > 25 (strong trend only)
          3. HTF RSI confirms direction (>55 for LONG, <45 for SHORT)
          4. Supertrend confirms direction
          5. Volume > 1.2x 20-bar average
        """
        if self._momentum_count >= self.MAX_MOMENTUM_PER_DAY:
            return None
        if idx < 5:
            return None

        ema_9 = self._sv(ind_dict['ema_9'], idx, 0)
        ema_21 = self._sv(ind_dict['ema_21'], idx, 0)
        close = self._sv(ind_dict['close'], idx)
        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        st_dir = self._sv(ind_dict['supertrend_dir'], idx, 0)
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        vol = self._sv(ind_dict['volume'], idx, 0)
        vol_avg = self._sv(ind_dict['vol_avg'], idx, 1)

        if np.isnan(close) or ema_9 == 0 or ema_21 == 0:
            return None

        if adx_val < 18:
            return None

        vol_ratio = vol / vol_avg if vol_avg > 0 else 0
        if vol > 0 and vol_ratio < 1.0:
            return None

        # Detect fresh crossover (within last 3 bars)
        cross_dir = None
        for lookback in range(1, 4):
            if idx - lookback < 0:
                break
            prev_9 = self._sv(ind_dict['ema_9'], idx - lookback, 0)
            prev_21 = self._sv(ind_dict['ema_21'], idx - lookback, 0)
            if prev_9 == 0 or prev_21 == 0:
                continue
            if ema_9 > ema_21 and prev_9 <= prev_21:
                cross_dir = "LONG"
                break
            if ema_9 < ema_21 and prev_9 >= prev_21:
                cross_dir = "SHORT"
                break

        if cross_dir is None:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)

        direction = None
        reasons = []

        if (cross_dir == "LONG" and st_dir == 1 and rsi_15m > 55
                and 40 < rsi_5m < 70):
            direction = "LONG"
            reasons = ["EMA9x21↑", f"ADX={adx_val:.0f}", f"RSI15={rsi_15m:.0f}↑",
                       f"Vol={vol_ratio:.1f}x"]
        elif (cross_dir == "SHORT" and st_dir == -1 and rsi_15m < 45
              and 30 < rsi_5m < 60):
            direction = "SHORT"
            reasons = ["EMA9x21↓", f"ADX={adx_val:.0f}", f"RSI15={rsi_15m:.0f}↓",
                       f"Vol={vol_ratio:.1f}x"]

        if not direction:
            return None

        conf = 68
        if adx_val > 30:
            conf += 4
        htf_strength = abs(rsi_15m - 50)
        if htf_strength > 15:
            conf += 4

        return TradeSignal(
            direction=direction,
            signal_type="EMA_MOMENTUM",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
            pullback_count=0,
        )

    def _check_vwap_momentum(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """VWAP momentum: sustained price above/below VWAP with EMA cascade + volume.

        Requires 3 consecutive closes on the same side of VWAP (sustained, not noise),
        full EMA cascade (EMA9 > EMA21 > EMA50), volume spike, and HTF RSI confirmation.
        """
        if self._vwap_count >= self.MAX_VWAP_PER_DAY:
            return None
        if idx < 5:
            return None

        close = self._sv(ind_dict['close'], idx)
        vwap_val = self._sv(ind_dict.get('vwap', pd.Series()), idx, np.nan)
        if np.isnan(close) or np.isnan(vwap_val):
            return None

        bars_above = 0
        bars_below = 0
        for lb in range(3):
            c = self._sv(ind_dict['close'], idx - lb)
            v = self._sv(ind_dict.get('vwap', pd.Series()), idx - lb, np.nan)
            if np.isnan(c) or np.isnan(v):
                break
            if c > v:
                bars_above += 1
            elif c < v:
                bars_below += 1

        ema_9 = self._sv(ind_dict['ema_9'], idx, 0)
        ema_21 = self._sv(ind_dict['ema_21'], idx, 0)
        ema_50 = self._sv(ind_dict.get('ema_50', pd.Series()), idx, 0)
        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        vol = self._sv(ind_dict['volume'], idx, 0)
        vol_avg = self._sv(ind_dict['vol_avg'], idx, 1)
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        st_dir = self._sv(ind_dict['supertrend_dir'], idx, 0)

        if ema_9 == 0 or ema_21 == 0 or ema_50 == 0:
            return None

        vol_ratio = vol / vol_avg if vol_avg > 0 else 0
        if vol > 0 and vol_ratio < 1.3:
            return None

        if adx_val < 15:
            return None

        direction = None
        reasons = []

        if (bars_above >= 3 and ema_9 > ema_21 > ema_50
                and rsi_5m > 55 and st_dir == 1):
            direction = "LONG"
            reasons = ["VWAP+3bars↑", "EMA cascade↑", f"RSI={rsi_5m:.0f}",
                       f"Vol={vol_ratio:.1f}x"]
        elif (bars_below >= 3 and ema_9 < ema_21 < ema_50
              and rsi_5m < 45 and st_dir == -1):
            direction = "SHORT"
            reasons = ["VWAP+3bars↓", "EMA cascade↓", f"RSI={rsi_5m:.0f}",
                       f"Vol={vol_ratio:.1f}x"]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        if direction == "LONG" and rsi_15m < 48:
            return None
        if direction == "SHORT" and rsi_15m > 52:
            return None

        conf = 75
        if (direction == "LONG" and rsi_15m > 58) or \
           (direction == "SHORT" and rsi_15m < 42):
            conf += 5

        return TradeSignal(
            direction=direction,
            signal_type="VWAP_MOMENTUM",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
            pullback_count=0,
        )

    def _check_supertrend_flip(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Single Supertrend(10,3) flip with EMA + ADX confirmation.

        DISABLED: Data shows net loser across 18 trades (33% WR SHORT,
        50% WR LONG but low edge). Keeping code for future re-evaluation.
        """
        return None

        if self._supertrd_count >= self.MAX_SUPERTRD_PER_DAY:
            return None
        if idx < 2:
            return None

        st_dir = self._sv(ind_dict['supertrend_dir'], idx, 0)
        st_dir_prev = self._sv(ind_dict['supertrend_dir'], idx - 1, 0)
        ema_9 = self._sv(ind_dict['ema_9'], idx, 0)
        ema_21 = self._sv(ind_dict['ema_21'], idx, 0)
        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        close = self._sv(ind_dict['close'], idx)
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)

        if np.isnan(close) or ema_9 == 0 or ema_21 == 0:
            return None

        if st_dir == st_dir_prev:
            return None

        if adx_val < self.SUPERTREND_MIN_ADX:
            return None

        direction = None
        reasons = []

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)

        if st_dir == 1 and ema_9 > ema_21 and 30 < rsi_5m < 72 and rsi_15m < 70:
            direction = "LONG"
            reasons = ["ST flip↑", f"ADX={adx_val:.0f}", f"RSI={rsi_5m:.0f}"]
        elif st_dir == -1 and ema_9 < ema_21 and 20 < rsi_5m < 70 and rsi_15m > 30:
            direction = "SHORT"
            reasons = ["ST flip↓", f"ADX={adx_val:.0f}", f"RSI={rsi_5m:.0f}"]

        if not direction:
            return None
        conf = 70
        if (direction == "LONG" and rsi_15m > 55) or \
           (direction == "SHORT" and rsi_15m < 45):
            conf += 5

        st_fast = self._sv(ind_dict.get('supertrend_fast_dir', pd.Series()), idx, 0)
        if st_fast == st_dir:
            conf += 3

        return TradeSignal(
            direction=direction,
            signal_type="SUPERTREND",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
            pullback_count=0,
        )

    def _check_rsi_reversion(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """RSI mean reversion: buy oversold dips in uptrends, sell overbought pops in downtrends.

        Simple high-frequency strategy used by most intraday traders.
        Requires RSI to reach extreme AND start reversing (momentum shift).
        """
        if self._rsi_rev_count >= self.MAX_RSI_REV_PER_DAY:
            return None
        if idx < 2:
            return None

        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        rsi_prev = self._sv(ind_dict['rsi_5m'], idx - 1, 50)
        ema_9 = self._sv(ind_dict['ema_9'], idx, 0)
        ema_21 = self._sv(ind_dict['ema_21'], idx, 0)
        close = self._sv(ind_dict['close'], idx)
        ema_20 = self._sv(ind_dict['ema_20'], idx, close)

        if np.isnan(close) or ema_9 == 0 or ema_21 == 0:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        bb_pctb = self._sv(ind_dict.get('bb_pctb', pd.Series()), idx, 0.5)
        direction = None
        reasons = []

        if (rsi_5m < 42 and rsi_5m > rsi_prev
                and rsi_15m > 45 and close > ema_20 and bb_pctb < 0.25):
            direction = "LONG"
            reasons = [f"RSI={rsi_5m:.0f}<42 ↑", f"BB%B={bb_pctb:.2f}",
                       f"15m_RSI={rsi_15m:.0f}↑"]
        elif (rsi_5m > 58 and rsi_5m < rsi_prev
              and rsi_15m < 55 and close < ema_20 and bb_pctb > 0.75):
            direction = "SHORT"
            reasons = [f"RSI={rsi_5m:.0f}>58 ↓", f"BB%B={bb_pctb:.2f}",
                       f"15m_RSI={rsi_15m:.0f}↓"]

        if not direction:
            return None

        conf = 72
        htf_strength = abs(rsi_15m - 50)
        if htf_strength > 15:
            conf += 5
        if (direction == "LONG" and bb_pctb < 0.1) or \
           (direction == "SHORT" and bb_pctb > 0.9):
            conf += 3

        return TradeSignal(
            direction=direction,
            signal_type="RSI_REVERSION",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
            pullback_count=0,
        )

    # ── New Strategy #1: VWAP Mean Reversion (2 SD) ──────────────────────

    def _check_vwap_mean_reversion(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Fade price when it stretches 2 standard deviations from VWAP.

        84% reversion rate historically when combined with volume contraction.
        Works best on range-bound days (low ADX). Opposite of trend strategies.
        """
        if self._vwap_mr_count >= self.MAX_VWAP_MR_PER_DAY:
            return None
        if idx < 10:
            return None

        close = self._sv(ind_dict['close'], idx)
        vwap_val = self._sv(ind_dict.get('vwap', pd.Series()), idx, np.nan)
        vwap_std = self._sv(ind_dict.get('vwap_std', pd.Series()), idx, np.nan)

        if np.isnan(close) or np.isnan(vwap_val) or np.isnan(vwap_std) or vwap_std < 1:
            return None

        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 25)
        if adx_val > 25:
            return None

        distance = close - vwap_val
        z_score = distance / vwap_std

        if abs(z_score) < 2.0:
            return None

        vol = self._sv(ind_dict['volume'], idx, 0)
        vol_avg = self._sv(ind_dict['vol_avg'], idx, 1)
        vol_ratio = vol / vol_avg if vol_avg > 0 else 1.0
        if vol > 0 and vol_ratio > 1.3:
            return None

        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        rsi_prev = self._sv(ind_dict['rsi_5m'], idx - 1, 50)

        direction = None
        reasons = []

        if z_score >= 2.0 and rsi_5m < rsi_prev and rsi_5m > 58:
            direction = "SHORT"
            reasons = [f"VWAP+{z_score:.1f}σ↑", f"RSI={rsi_5m:.0f} turning↓",
                       f"Vol={vol_ratio:.1f}x", f"ADX={adx_val:.0f}"]
        elif z_score <= -2.0 and rsi_5m > rsi_prev and rsi_5m < 42:
            direction = "LONG"
            reasons = [f"VWAP-{abs(z_score):.1f}σ↓", f"RSI={rsi_5m:.0f} turning↑",
                       f"Vol={vol_ratio:.1f}x", f"ADX={adx_val:.0f}"]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        conf = 72
        if abs(z_score) >= 2.5:
            conf += 5
        if vol_ratio < 0.8:
            conf += 3
        if (direction == "LONG" and rsi_15m > 40) or \
           (direction == "SHORT" and rsi_15m < 60):
            conf += 3
        conf = min(conf, 100)

        return TradeSignal(
            direction=direction,
            signal_type="VWAP_MEAN_REV",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
        )

    # ── New Strategy #2: CPR Range Trade ────────────────────────────────

    def _check_cpr_range(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Mean reversion between BCP and TCP on wide CPR days.

        70% WR historically. Buy near BCP, sell near TCP.
        Only active when CPR width > 30 points (range-bound day signal).
        """
        if self._cpr_range_count >= self.MAX_CPR_RANGE_PER_DAY:
            return None
        if idx < 5:
            return None

        pp = ind_dict.get('cpr_pp', np.nan)
        bcp = ind_dict.get('cpr_bcp', np.nan)
        tcp = ind_dict.get('cpr_tcp', np.nan)
        cpr_w = ind_dict.get('cpr_width', 0)

        if np.isnan(pp) or np.isnan(bcp) or np.isnan(tcp):
            return None
        if cpr_w < 40:
            return None

        close = self._sv(ind_dict['close'], idx)
        if np.isnan(close):
            return None

        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 25)
        if adx_val > 22:
            return None

        atr_val = self._sv(ind_dict['atr'], idx, 50)
        proximity = atr_val * 0.25

        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        rsi_prev = self._sv(ind_dict['rsi_5m'], idx - 1, 50)
        bb_pctb = self._sv(ind_dict.get('bb_pctb', pd.Series()), idx, 0.5)

        direction = None
        reasons = []

        if close <= bcp + proximity and rsi_5m < 42 and rsi_5m > rsi_prev and bb_pctb < 0.3:
            direction = "LONG"
            reasons = [f"Near BCP={bcp:.0f}", f"CPR_W={cpr_w:.0f}",
                       f"RSI={rsi_5m:.0f} turning↑", f"ADX={adx_val:.0f}"]
        elif close >= tcp - proximity and rsi_5m > 58 and rsi_5m < rsi_prev and bb_pctb > 0.7:
            direction = "SHORT"
            reasons = [f"Near TCP={tcp:.0f}", f"CPR_W={cpr_w:.0f}",
                       f"RSI={rsi_5m:.0f} turning↓", f"ADX={adx_val:.0f}"]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        st_dir = self._sv(ind_dict['supertrend_dir'], idx, 0)

        conf = 68
        if cpr_w > 60:
            conf += 3
        if (direction == "LONG" and st_dir == 1) or \
           (direction == "SHORT" and st_dir == -1):
            conf += 4
        if (direction == "LONG" and rsi_15m > 45) or \
           (direction == "SHORT" and rsi_15m < 55):
            conf += 3
        conf = min(conf, 100)

        return TradeSignal(
            direction=direction,
            signal_type="CPR_RANGE",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
        )

    # ── New Strategy #3: Gap Trading ────────────────────────────────────

    def _check_gap_trade(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Trade gap fills when Nifty opens with a gap > 50 points.

        ~70% gap-fill rate for gaps under 150 points. Widened window
        (bars 2-18, ~90 min) and lower fill threshold (10%) to catch
        gap-exhaustion setups that develop over the morning session.
        """
        if self._gap_count >= self.MAX_GAP_PER_DAY:
            return None

        day_gap = ind_dict.get('day_gap', 0.0)
        if abs(day_gap) < 50:
            return None

        if idx < 2 or idx > 18:
            return None

        close = self._sv(ind_dict['close'], idx)
        day_open = ind_dict.get('day_open', np.nan)
        if np.isnan(close) or np.isnan(day_open) or np.isnan(self._prev_day_close):
            return None

        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)

        direction = None
        reasons = []

        if day_gap > 50:
            fill_progress = (day_open - close) / day_gap
            if fill_progress > 0.1 and rsi_5m < 55:
                direction = "SHORT"
                reasons = [f"Gap↑{day_gap:.0f}pts", f"Fill={fill_progress:.0%}",
                           f"RSI={rsi_5m:.0f}↓"]
        elif day_gap < -50:
            fill_progress = (close - day_open) / abs(day_gap)
            if fill_progress > 0.1 and rsi_5m > 45:
                direction = "LONG"
                reasons = [f"Gap↓{abs(day_gap):.0f}pts", f"Fill={fill_progress:.0%}",
                           f"RSI={rsi_5m:.0f}↑"]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)

        conf = 75
        if abs(day_gap) > 100:
            conf += 5
        if abs(day_gap) > 150:
            conf -= 5
        ema_9 = self._sv(ind_dict['ema_9'], idx, 0)
        ema_21 = self._sv(ind_dict['ema_21'], idx, 0)
        if ema_9 != 0 and ema_21 != 0:
            if (direction == "LONG" and ema_9 > ema_21) or \
               (direction == "SHORT" and ema_9 < ema_21):
                conf += 3

        vol = self._sv(ind_dict.get('volume', pd.Series()), idx, 0)
        vol_mean = ind_dict.get('volume')
        if vol_mean is not None and idx >= 20:
            avg_vol = vol_mean.iloc[max(0, idx - 20):idx].mean()
            if avg_vol > 0 and vol > avg_vol * 1.5:
                conf += 5
                reasons.append("HighVol")

        conf = min(conf, 100)

        return TradeSignal(
            direction=direction,
            signal_type="GAP_TRADE",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
        )

    # ── New Strategy #4: CPR Narrow-Range Breakout ──────────────────────

    def _check_cpr_breakout(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Breakout on narrow CPR days with volume confirmation.

        65-70% WR. Narrow CPR (< 20 points) signals a trending day.
        Enter when price breaks above TCP or below BCP with volume surge.
        """
        if self._cpr_breakout_count >= self.MAX_CPR_BREAKOUT_PER_DAY:
            return None
        if idx < 5:
            return None

        pp = ind_dict.get('cpr_pp', np.nan)
        bcp = ind_dict.get('cpr_bcp', np.nan)
        tcp = ind_dict.get('cpr_tcp', np.nan)
        cpr_w = ind_dict.get('cpr_width', 0)

        if np.isnan(pp) or np.isnan(bcp) or np.isnan(tcp):
            return None
        if cpr_w > 25:
            return None

        close = self._sv(ind_dict['close'], idx)
        close_prev = self._sv(ind_dict['close'], idx - 1)
        if np.isnan(close) or np.isnan(close_prev):
            return None

        vol = self._sv(ind_dict['volume'], idx, 0)
        vol_avg = self._sv(ind_dict['vol_avg'], idx, 1)
        vol_ratio = vol / vol_avg if vol_avg > 0 else 0
        if vol > 0 and vol_ratio < 1.2:
            return None

        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)

        direction = None
        reasons = []

        if close > tcp and close_prev <= tcp and rsi_5m > 50:
            direction = "LONG"
            reasons = [f"Break>TCP={tcp:.0f}", f"NarrowCPR={cpr_w:.0f}",
                       f"Vol={vol_ratio:.1f}x", f"ADX={adx_val:.0f}"]
        elif close < bcp and close_prev >= bcp and rsi_5m < 50:
            direction = "SHORT"
            reasons = [f"Break<BCP={bcp:.0f}", f"NarrowCPR={cpr_w:.0f}",
                       f"Vol={vol_ratio:.1f}x", f"ADX={adx_val:.0f}"]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        st_dir = self._sv(ind_dict['supertrend_dir'], idx, 0)

        conf = 70
        if adx_val > 20:
            conf += 3
        if vol_ratio > 1.5:
            conf += 3
        if (direction == "LONG" and st_dir == 1) or \
           (direction == "SHORT" and st_dir == -1):
            conf += 4
        if (direction == "LONG" and rsi_15m > 50) or \
           (direction == "SHORT" and rsi_15m < 50):
            conf += 3
        conf = min(conf, 100)

        return TradeSignal(
            direction=direction,
            signal_type="CPR_BREAKOUT",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
        )

    # ── New Strategy #5: ADX Breakout ───────────────────────────────────

    def _check_adx_breakout(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """ADX rising from < 20 to > 23 signals a new trend beginning.

        Capped at ADX 20-32 to avoid overlap with TREND_RIDE.
        """
        if self._adx_breakout_count >= self.MAX_ADX_BREAKOUT_PER_DAY:
            return None
        if idx < 5:
            return None

        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        adx_prev = self._sv(ind_dict.get('adx', pd.Series()), idx - 1, 0)
        adx_prev2 = self._sv(ind_dict.get('adx', pd.Series()), idx - 2, 0)

        if not (20 <= adx_val <= 32 and adx_val > 23
                and adx_prev < 23 and adx_prev2 < 20):
            return None

        if adx_val <= adx_prev:
            return None

        plus_di = self._sv(ind_dict.get('plus_di', pd.Series()), idx, 0)
        minus_di = self._sv(ind_dict.get('minus_di', pd.Series()), idx, 0)

        if plus_di == 0 or minus_di == 0:
            return None

        di_spread = abs(plus_di - minus_di)
        if di_spread <= 10:
            return None

        close = self._sv(ind_dict['close'], idx)
        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        ema_9 = self._sv(ind_dict['ema_9'], idx, 0)
        ema_20 = self._sv(ind_dict['ema_20'], idx, close)
        ema_21 = self._sv(ind_dict['ema_21'], idx, 0)

        if np.isnan(close) or ema_9 == 0 or ema_21 == 0:
            return None

        direction = None
        reasons = []

        if (plus_di > minus_di and ema_9 > ema_21 and rsi_5m > 45
                and close > ema_20):
            direction = "LONG"
            reasons = [f"ADX={adx_val:.0f} breakout↑", f"+DI={plus_di:.0f}>{minus_di:.0f}",
                       "Close>EMA20", f"RSI={rsi_5m:.0f}"]
        elif (minus_di > plus_di and ema_9 < ema_21 and rsi_5m < 55
              and close < ema_20):
            direction = "SHORT"
            reasons = [f"ADX={adx_val:.0f} breakout↓", f"-DI={minus_di:.0f}>{plus_di:.0f}",
                       "Close<EMA20", f"RSI={rsi_5m:.0f}"]

        if not direction:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        st_dir = self._sv(ind_dict['supertrend_dir'], idx, 0)

        conf = 68
        if adx_val > 28:
            conf += 4
        if di_spread > 15:
            conf += 3
        if (direction == "LONG" and st_dir == 1) or \
           (direction == "SHORT" and st_dir == -1):
            conf += 4
        if (direction == "LONG" and rsi_15m > 50) or \
           (direction == "SHORT" and rsi_15m < 50):
            conf += 3
        conf = min(conf, 100)

        return TradeSignal(
            direction=direction,
            signal_type="ADX_BREAKOUT",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
        )

    def _check_trend_ride(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Trend ride: enter sustained directional moves without waiting for pullback.

        Catches grinding rallies/selloffs by requiring multiple trend confirmations
        to align simultaneously: EMA cascade, ADX strength + direction, DI dominance,
        and multi-timeframe RSI agreement.
        """
        if self._trend_ride_count >= self.MAX_TREND_RIDE_PER_DAY:
            return None
        if idx < 5:
            return None

        close = self._sv(ind_dict['close'], idx)
        ema_9 = self._sv(ind_dict['ema_9'], idx, 0)
        ema_20 = self._sv(ind_dict['ema_20'], idx, close)
        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        adx_prev = self._sv(ind_dict.get('adx', pd.Series()), idx - 1, 0)
        plus_di = self._sv(ind_dict.get('plus_di', pd.Series()), idx, 0)
        minus_di = self._sv(ind_dict.get('minus_di', pd.Series()), idx, 0)
        st_dir = self._sv(ind_dict['supertrend_dir'], idx, 0)

        if np.isnan(close) or ema_9 == 0 or ema_20 == 0:
            return None

        # Must be a strong AND strengthening trend (38 filters out weak trends)
        if adx_val < 38 or adx_val <= adx_prev:
            return None

        # DI must show clear directional dominance (12 avoids noise-level spreads)
        di_spread = abs(plus_di - minus_di)
        if di_spread < 12:
            return None

        rsi_15m = self._htf_rsi(ind_dict, idx, 50)

        direction = None
        reasons = []

        rsi5_cap = 78 if adx_val > 40 else 72
        # LONG: everything pointing up
        if (ema_9 > ema_20 and close > ema_20
                and plus_di > minus_di
                and rsi_15m > 52
                and 50 <= rsi_5m <= rsi5_cap):
            direction = "LONG"
            reasons = [
                "TrendRide↑",
                f"ADX={adx_val:.0f}↑",
                f"+DI={plus_di:.0f}>{minus_di:.0f}",
                f"RSI15={rsi_15m:.0f}↑",
                f"RSI5={rsi_5m:.0f}",
            ]
        # SHORT: everything pointing down
        elif (ema_9 < ema_20 and close < ema_20
              and minus_di > plus_di
              and rsi_15m < 48
              and 28 <= rsi_5m <= 50):
            direction = "SHORT"
            reasons = [
                "TrendRide↓",
                f"ADX={adx_val:.0f}↑",
                f"-DI={minus_di:.0f}>{plus_di:.0f}",
                f"RSI15={rsi_15m:.0f}↓",
                f"RSI5={rsi_5m:.0f}",
            ]

        if not direction:
            return None

        # Confidence
        conf = 68
        if adx_val > 35:
            conf += 5
        if adx_val > 45:
            conf += 3
        htf_strength = abs(rsi_15m - 50)
        if htf_strength > 10:
            conf += 4
        if di_spread > 12:
            conf += 3
        if (direction == "LONG" and st_dir == 1) or \
           (direction == "SHORT" and st_dir == -1):
            conf += 3
        conf = min(conf, 100)

        if conf < 90:
            return None

        return TradeSignal(
            direction=direction,
            signal_type="TREND_RIDE",
            confidence=conf,
            htf_rsi=rsi_15m,
            ltf_rsi=rsi_5m,
            nifty_price=close,
            reason=" | ".join(reasons),
            pullback_count=0,
            adx=adx_val,
        )

    def _check_orb_breakout(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Opening Range Breakout from first 3 bars (09:30-09:40)."""
        if self._orb_count >= self.MAX_ORB_PER_DAY:
            return None
        if np.isnan(self._orb_high) or np.isnan(self._orb_low):
            return None

        time_str = self._current_time
        if time_str < "09:45" or time_str >= "11:00":
            return None

        close = self._sv(ind_dict['close'], idx)
        close_prev = self._sv(ind_dict['close'], idx - 1) if idx >= 1 else close
        if np.isnan(close) or np.isnan(close_prev):
            return None

        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        if adx_val < 15:
            return None

        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        plus_di = self._sv(ind_dict.get('plus_di', pd.Series()), idx, 0)
        minus_di = self._sv(ind_dict.get('minus_di', pd.Series()), idx, 0)
        vol = self._sv(ind_dict['volume'], idx, 0)
        vol_avg = self._sv(ind_dict['vol_avg'], idx, 1)
        vol_ratio = vol / vol_avg if vol_avg > 0 else 0
        if vol > 0 and vol_ratio < 1.3:
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
        """Bollinger squeeze breakout after compression."""
        if self._squeeze_count >= self.MAX_SQUEEZE_PER_DAY:
            return None
        if idx < 50:
            return None

        bb_width = self._sv(ind_dict.get('bb_width', pd.Series()), idx - 1, np.nan)
        bb_min50 = self._sv(ind_dict.get('bb_width_min50', pd.Series()), idx - 1, np.nan)
        if np.isnan(bb_width) or np.isnan(bb_min50) or bb_min50 <= 0:
            return None
        if bb_width > bb_min50 * 1.3:
            return None

        cpr_w = ind_dict.get('cpr_width', 0)
        if cpr_w < 20 or cpr_w > 45:
            return None

        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        if adx_val < 12 or adx_val > 30:
            return None

        close = self._sv(ind_dict['close'], idx)
        high_prev = self._sv(ind_dict['high'], idx - 1) if idx >= 1 else close
        low_prev = self._sv(ind_dict['low'], idx - 1) if idx >= 1 else close
        bb_pctb = self._sv(ind_dict.get('bb_pctb', pd.Series()), idx, 0.5)
        ema_9 = self._sv(ind_dict['ema_9'], idx, 0)
        ema_20 = self._sv(ind_dict['ema_20'], idx, close)
        vol = self._sv(ind_dict['volume'], idx, 0)
        vol_avg = self._sv(ind_dict['vol_avg'], idx, 1)
        vol_ratio = vol / vol_avg if vol_avg > 0 else 0

        if np.isnan(close) or ema_9 == 0:
            return None
        if vol > 0 and vol_ratio < 1.2:
            return None

        direction = None
        reasons = []

        if (close > high_prev and bb_pctb > 0.85 and ema_9 > ema_20):
            direction = "LONG"
            reasons = [
                "BB squeeze break↑",
                f"BB%B={bb_pctb:.2f}",
                f"Vol={vol_ratio:.1f}x",
                f"CPR_W={cpr_w:.0f}",
            ]
        elif (close < low_prev and bb_pctb < 0.15 and ema_9 < ema_20):
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
            ltf_rsi=self._sv(ind_dict['rsi_5m'], idx, 50),
            nifty_price=close,
            reason=" | ".join(reasons),
        )

    def _check_vwap_bounce(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """VWAP pullback entry in established trend."""
        if self._vwap_bounce_count >= self.MAX_VWAP_BOUNCE_PER_DAY:
            return None
        if idx < 5:
            return None

        close = self._sv(ind_dict['close'], idx)
        vwap_val = self._sv(ind_dict.get('vwap', pd.Series()), idx, np.nan)
        atr_val = self._sv(ind_dict['atr'], idx, 30)
        ema_9 = self._sv(ind_dict['ema_9'], idx, 0)
        ema_20 = self._sv(ind_dict['ema_20'], idx, close)
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        cpr_w = ind_dict.get('cpr_width', 0)

        if np.isnan(close) or np.isnan(vwap_val) or atr_val <= 0:
            return None
        if adx_val > 40 or cpr_w <= 25:
            return None

        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        rsi_5m_prev = self._sv(ind_dict['rsi_5m'], idx - 1, rsi_5m)
        rsi_15m = self._htf_rsi(ind_dict, idx, 50)
        stoch_k = self._sv(ind_dict['stoch_k'], idx, 50)
        stoch_k_prev = self._sv(ind_dict['stoch_k'], idx - 1, stoch_k)
        vol = self._sv(ind_dict['volume'], idx, 0)
        vol_avg = self._sv(ind_dict['vol_avg'], idx, 1)
        vol_ratio = vol / vol_avg if vol_avg > 0 else 1.0

        vwap_dist = abs(close - vwap_val)
        at_vwap = vwap_dist <= 0.25 * atr_val

        direction = None
        reasons = []

        if ema_9 > ema_20 and rsi_15m > 52 and at_vwap:
            was_extended = False
            for lb in range(1, 6):
                c_lb = self._sv(ind_dict['close'], idx - lb)
                v_lb = self._sv(ind_dict.get('vwap', pd.Series()), idx - lb, np.nan)
                a_lb = self._sv(ind_dict['atr'], idx - lb, atr_val)
                if not np.isnan(c_lb) and not np.isnan(v_lb) and a_lb > 0:
                    if c_lb > v_lb + 0.3 * a_lb:
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
                c_lb = self._sv(ind_dict['close'], idx - lb)
                v_lb = self._sv(ind_dict.get('vwap', pd.Series()), idx - lb, np.nan)
                a_lb = self._sv(ind_dict['atr'], idx - lb, atr_val)
                if not np.isnan(c_lb) and not np.isnan(v_lb) and a_lb > 0:
                    if c_lb < v_lb - 0.3 * a_lb:
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
        touch_precision = 1.0 - min(vwap_dist / (0.25 * atr_val), 1.0)
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
        """Exhaustion reversal via RSI/price divergence."""
        if self._divergence_count >= self.MAX_DIVERGENCE_PER_DAY:
            return None
        if idx < 10:
            return None

        time_str = self._current_time
        if time_str < "11:00" or time_str >= "14:00":
            return None

        rsi_div = self._sv(ind_dict.get('rsi_div', pd.Series()), idx, 0)
        if rsi_div == 0:
            return None

        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        if adx_val >= 30:
            return None

        rsi_5m = self._sv(ind_dict['rsi_5m'], idx, 50)
        rsi_5m_prev = self._sv(ind_dict['rsi_5m'], idx - 1, rsi_5m)
        stoch_k = self._sv(ind_dict['stoch_k'], idx, 50)
        stoch_k_prev = self._sv(ind_dict['stoch_k'], idx - 1, stoch_k)
        willr = self._sv(ind_dict['willr'], idx, -50)
        close = self._sv(ind_dict['close'], idx)
        vwap_val = self._sv(ind_dict.get('vwap', pd.Series()), idx, np.nan)
        atr_val = self._sv(ind_dict['atr'], idx, 30)

        if np.isnan(close):
            return None

        direction = None
        reasons = []

        if (rsi_div == 1 and rsi_5m > rsi_5m_prev
                and stoch_k < 35 and stoch_k > stoch_k_prev and willr < -80):
            direction = "LONG"
            reasons = [
                "Bullish RSI div",
                f"RSI={rsi_5m:.0f}↑",
                f"Stoch={stoch_k:.0f}",
                f"WR={willr:.0f}",
            ]
        elif (rsi_div == -1 and rsi_5m < rsi_5m_prev
              and stoch_k > 65 and stoch_k < stoch_k_prev and willr > -20):
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

        # Divergence magnitude: compare recent price vs RSI swing
        lookback = 10
        close_now = close
        close_lb = self._sv(ind_dict['close'], idx - lookback, close)
        rsi_lb = self._sv(ind_dict['rsi_5m'], idx - lookback, rsi_5m)
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
        if direction == "LONG" and willr < -85 or direction == "SHORT" and willr > -15:
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

    def _check_triple_confirm(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Supertrend flip + MACD crossover + Price near VWAP -- all three must align."""
        if self._triple_confirm_count >= self.MAX_TRIPLE_CONFIRM_PER_DAY:
            return None

        close = self._sv(ind_dict['close'], idx)
        vwap = self._sv(ind_dict.get('vwap', pd.Series()), idx, np.nan)
        st_dir = self._sv(ind_dict['supertrend_dir'], idx, 0)
        st_prev = self._sv(ind_dict['supertrend_dir'], idx - 1, 0) if idx >= 1 else 0
        macd_hist = self._sv(ind_dict.get('macd_hist', pd.Series()), idx, 0)
        macd_prev = self._sv(ind_dict.get('macd_hist', pd.Series()), idx - 1, 0) if idx >= 1 else 0
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)

        if np.isnan(close) or np.isnan(vwap):
            return None
        if adx_val < 15:
            return None

        vwap_dist_pct = abs(close - vwap) / close * 100
        if vwap_dist_pct > 0.5:
            return None

        st_flip_bull = st_prev <= 0 and st_dir == 1
        st_flip_bear = st_prev >= 0 and st_dir == -1
        macd_cross_bull = macd_prev <= 0 and macd_hist > 0
        macd_cross_bear = macd_prev >= 0 and macd_hist < 0

        if st_flip_bull and macd_cross_bull and close > vwap:
            direction = "LONG"
        elif st_flip_bear and macd_cross_bear and close < vwap:
            direction = "SHORT"
        else:
            return None

        conf = 78
        if adx_val > 25:
            conf += 5
        if adx_val > 35:
            conf += 3
        conf = min(conf, 95)

        return TradeSignal(
            direction=direction, signal_type="TRIPLE_CONFIRM",
            confidence=conf, htf_rsi=50, ltf_rsi=50, nifty_price=close,
            reason=f"ST_flip+MACD_cross+VWAP ADX={adx_val:.0f}",
        )

    def _check_first_hour_momentum(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """First hour high/low breakout with VWAP confirmation."""
        if self._first_hour_count >= self.MAX_FIRST_HOUR_PER_DAY:
            return None
        if idx < 12:
            return None

        close = self._sv(ind_dict['close'], idx)
        vwap = self._sv(ind_dict.get('vwap', pd.Series()), idx, np.nan)
        fh_high = ind_dict.get('first_hour_high', np.nan)
        fh_low = ind_dict.get('first_hour_low', np.nan)
        rsi = self._sv(ind_dict['rsi_5m'], idx, 50)
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)

        if np.isnan(close) or np.isnan(vwap) or np.isnan(fh_high) or np.isnan(fh_low):
            return None

        fh_range = fh_high - fh_low
        if fh_range < 10:
            return None

        if close > fh_high and close > vwap:
            direction = "LONG"
            reasons = f"FH_BO high={fh_high:.0f} close={close:.0f}"
        elif close < fh_low and close < vwap:
            direction = "SHORT"
            reasons = f"FH_BD low={fh_low:.0f} close={close:.0f}"
        else:
            return None

        conf = 72
        if adx_val > 20:
            conf += 5
        if fh_range > 50:
            conf += 3
        conf = min(conf, 92)

        return TradeSignal(
            direction=direction, signal_type="FIRST_HOUR_MOM",
            confidence=conf, htf_rsi=50, ltf_rsi=rsi, nifty_price=close,
            reason=reasons,
        )

    def _check_vwap_2sd_reversion(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Price > 2 SD from VWAP with volume contraction -> trade toward VWAP."""
        if self._vwap_2sd_count >= self.MAX_VWAP_2SD_PER_DAY:
            return None

        close = self._sv(ind_dict['close'], idx)
        vwap = self._sv(ind_dict.get('vwap', pd.Series()), idx, np.nan)
        vwap_std = self._sv(ind_dict.get('vwap_std', pd.Series()), idx, np.nan)
        rsi = self._sv(ind_dict['rsi_5m'], idx, 50)
        bb_pctb = self._sv(ind_dict.get('bb_pctb', pd.Series()), idx, 0.5)
        vol = self._sv(ind_dict.get('volume', pd.Series()), idx, 0)
        vol_avg = self._sv(ind_dict.get('vol_avg', pd.Series()), idx, 1)

        if np.isnan(close) or np.isnan(vwap) or np.isnan(vwap_std) or vwap_std <= 0:
            return None

        z_score = (close - vwap) / vwap_std

        vol_contraction = vol_avg > 0 and vol < vol_avg * 0.8

        if z_score > 2.0 and rsi > 70 and (bb_pctb > 0.95 or vol_contraction):
            direction = "SHORT"
        elif z_score < -2.0 and rsi < 30 and (bb_pctb < 0.05 or vol_contraction):
            direction = "LONG"
        else:
            return None

        conf = 74
        if abs(z_score) > 2.5:
            conf += 5
        if vol_contraction:
            conf += 3
        conf = min(conf, 92)

        return TradeSignal(
            direction=direction, signal_type="VWAP_2SD_REV",
            confidence=conf, htf_rsi=50, ltf_rsi=rsi, nifty_price=close,
            reason=f"VWAP_2SD z={z_score:.1f} RSI={rsi:.0f}",
        )

    def _check_narrow_cpr_breakout(self, ind_dict: dict, idx: int) -> Optional[TradeSignal]:
        """Narrow CPR (< 20 pts) breakout with volume confirmation."""
        if self._narrow_cpr_count >= self.MAX_NARROW_CPR_PER_DAY:
            return None

        close = self._sv(ind_dict['close'], idx)
        cpr_width = ind_dict.get('cpr_width', np.nan)
        tcp = ind_dict.get('cpr_tcp', np.nan)
        bcp = ind_dict.get('cpr_bcp', np.nan)
        adx_val = self._sv(ind_dict.get('adx', pd.Series()), idx, 0)
        vol = self._sv(ind_dict.get('volume', pd.Series()), idx, 0)
        vol_avg = self._sv(ind_dict.get('vol_avg', pd.Series()), idx, 1)

        if np.isnan(close) or np.isnan(cpr_width) or np.isnan(tcp) or np.isnan(bcp):
            return None
        if cpr_width >= 40:
            return None
        if idx < 6:
            return None

        vol_confirm = vol_avg > 0 and vol > vol_avg * 1.2

        if close > tcp and vol_confirm:
            direction = "LONG"
        elif close < bcp and vol_confirm:
            direction = "SHORT"
        else:
            return None

        conf = 73
        if adx_val > 20:
            conf += 5
        if cpr_width < 10:
            conf += 4
        conf = min(conf, 92)

        return TradeSignal(
            direction=direction, signal_type="NARROW_CPR_BO",
            confidence=conf, htf_rsi=50, ltf_rsi=50, nifty_price=close,
            reason=f"NarrowCPR w={cpr_width:.0f} ADX={adx_val:.0f}",
        )

    @staticmethod
    def _htf_rsi(ind_dict: dict, bar_idx: int, default: float = 50.0) -> float:
        """Map a 5m bar index to the correct 15m RSI value via searchsorted."""
        rsi_15m = ind_dict.get('rsi_15m')
        htf_idx = ind_dict.get('htf_15m_index')
        close_idx = ind_dict.get('close')

        v = default
        if (rsi_15m is not None and htf_idx is not None and len(htf_idx) > 0
                and close_idx is not None and bar_idx < len(close_idx)):
            current_time = close_idx.index[bar_idx]
            htf_i = htf_idx.searchsorted(current_time, side="right") - 1
            if 0 <= htf_i < len(rsi_15m):
                mapped = rsi_15m.iloc[htf_i]
                if not (isinstance(mapped, float) and np.isnan(mapped)):
                    v = mapped

        if v != default:
            return v

        # HTF unavailable or stuck at default -- approximate 15m RSI from 5m closes
        if close_idx is not None and bar_idx < len(close_idx):
            try:
                rsi_approx = ind.rsi(close_idx, 14)
                approx = rsi_approx.iloc[bar_idx]
                if not (isinstance(approx, float) and np.isnan(approx)):
                    return float(approx)
            except (IndexError, KeyError, ValueError):
                pass
        return default

    @staticmethod
    def _sv(series, idx: int, default=np.nan):
        try:
            v = series.iloc[idx] if hasattr(series, 'iloc') else series
            if isinstance(v, float) and np.isnan(v):
                return default
            return v
        except (IndexError, KeyError):
            return default
