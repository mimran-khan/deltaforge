"""VWAP + EMA crossover momentum strategy for options."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from config import settings
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class VWAPMomentumStrategy(BaseStrategy):
    """VWAP pullback + EMA crossover momentum.

    Rules:
    - 9 EMA crosses above 20 EMA (bullish) or below (bearish)
    - Price above VWAP for long / below VWAP for short
    - RSI > 50 for long / < 50 for short
    - Volume on crossover candle > 1.5x 20-candle average
    - Trade window: 9:30-14:00
    - Max 2 signals per day
    """

    name = "VWAP_MOM"

    def __init__(self):
        self.signals_today = 0
        self.max_signals = 2
        self._prev_ema_state: Optional[str] = None  # "above" or "below"

    def reset(self):
        self.signals_today = 0
        self._prev_ema_state = None

    def on_candle(self, df: pd.DataFrame, current_time: pd.Timestamp) -> Optional[Signal]:
        if len(df) < 25:
            return None

        if self.signals_today >= self.max_signals:
            return None

        time_str = current_time.strftime("%H:%M")
        if time_str < settings.VWAP_ENTRY_START or time_str > settings.VWAP_ENTRY_END:
            return None

        required = {"ema_fast", "ema_slow", "vwap", "rsi", "atr", "volume_sma"}
        if not required.issubset(df.columns):
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        ema_f = latest["ema_fast"]
        ema_s = latest["ema_slow"]
        prev_ema_f = prev["ema_fast"]
        prev_ema_s = prev["ema_slow"]

        current_state = "above" if ema_f > ema_s else "below"

        crossover_up = prev_ema_f <= prev_ema_s and ema_f > ema_s
        crossover_down = prev_ema_f >= prev_ema_s and ema_f < ema_s

        vol_ok = (latest["volume_sma"] > 0 and
                  latest["volume"] >= settings.VWAP_VOLUME_MULTIPLIER * latest["volume_sma"])

        self._prev_ema_state = current_state

        # ── Bullish crossover ───────────────────────────────────────
        if crossover_up:
            vwap_ok = latest["close"] > latest["vwap"]
            rsi_ok = latest["rsi"] > settings.VWAP_RSI_LONG_THRESHOLD

            if vwap_ok and rsi_ok and vol_ok:
                sl = latest["close"] - 1.5 * latest["atr"]
                target = latest["close"] + 2.0 * latest["atr"]
                self.signals_today += 1
                logger.info("[VWAP_MOM] LONG crossover at {:.2f}, RSI={:.1f}",
                            latest["close"], latest["rsi"])
                return Signal(
                    signal_type=SignalType.LONG,
                    strategy_name=self.name,
                    entry_price=latest["close"],
                    stop_loss_index=sl,
                    target_index=target,
                    option_type="CE",
                    confidence=0.8,
                    reason=f"EMA crossover up + VWAP confirm, RSI={latest['rsi']:.0f}",
                    timestamp=current_time,
                )

        # ── Bearish crossover ───────────────────────────────────────
        if crossover_down:
            vwap_ok = latest["close"] < latest["vwap"]
            rsi_ok = latest["rsi"] < settings.VWAP_RSI_SHORT_THRESHOLD

            if vwap_ok and rsi_ok and vol_ok:
                sl = latest["close"] + 1.5 * latest["atr"]
                target = latest["close"] - 2.0 * latest["atr"]
                self.signals_today += 1
                logger.info("[VWAP_MOM] SHORT crossover at {:.2f}, RSI={:.1f}",
                            latest["close"], latest["rsi"])
                return Signal(
                    signal_type=SignalType.SHORT,
                    strategy_name=self.name,
                    entry_price=latest["close"],
                    stop_loss_index=sl,
                    target_index=target,
                    option_type="PE",
                    confidence=0.8,
                    reason=f"EMA crossover down + VWAP confirm, RSI={latest['rsi']:.0f}",
                    timestamp=current_time,
                )

        return None
