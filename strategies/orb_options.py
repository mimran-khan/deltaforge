"""Opening Range Breakout (ORB) strategy for Nifty/BankNifty options."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from config import settings
from strategies.base_strategy import BaseStrategy, Signal, SignalType


class ORBStrategy(BaseStrategy):
    """15-minute Opening Range Breakout applied to index -> options execution.

    Rules:
    - Capture ORB high/low from 9:15-9:30 (first 15 minutes)
    - Long: candle CLOSES above ORB high + VWAP confirmation + volume surge
    - Short: candle CLOSES below ORB low + below VWAP + volume surge
    - Skip if ORB range > MAX_RANGE_POINTS (poor R:R)
    - Only enter between 9:30-11:00 AM
    - One signal per day (first valid breakout only)
    """

    name = "ORB"

    def __init__(self):
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self.orb_range: float = 0
        self.orb_captured = False
        self.signal_given = False

    def reset(self):
        self.orb_high = None
        self.orb_low = None
        self.orb_range = 0
        self.orb_captured = False
        self.signal_given = False

    def on_candle(self, df: pd.DataFrame, current_time: pd.Timestamp) -> Optional[Signal]:
        if len(df) < 3:
            return None

        time_str = current_time.strftime("%H:%M")

        if not self.orb_captured:
            return self._capture_orb(df, time_str, current_time)

        if self.signal_given:
            return None

        return self._check_breakout(df, time_str, current_time)

    def _capture_orb(self, df: pd.DataFrame, time_str: str,
                     current_time: pd.Timestamp) -> None:
        """Accumulate candles in the 9:15-9:30 window to set ORB range."""
        orb_candles = df[
            (df.index.hour == 9) &
            (df.index.minute >= 15) &
            (df.index.minute < 30)
        ]

        if len(orb_candles) == 0:
            return None

        if time_str >= settings.ORB_ENTRY_START:
            self.orb_high = orb_candles["high"].max()
            self.orb_low = orb_candles["low"].min()
            self.orb_range = self.orb_high - self.orb_low
            self.orb_captured = True

            if self.orb_range > settings.ORB_MAX_RANGE_POINTS:
                logger.info("[ORB] Range {:.0f} pts > max {} -- skipping today",
                            self.orb_range, settings.ORB_MAX_RANGE_POINTS)
                self.signal_given = True
                return None

            logger.info("[ORB] Captured: high={:.2f} low={:.2f} range={:.0f}",
                        self.orb_high, self.orb_low, self.orb_range)

        return None

    def _check_breakout(self, df: pd.DataFrame, time_str: str,
                        current_time: pd.Timestamp) -> Optional[Signal]:
        if time_str < settings.ORB_ENTRY_START or time_str > settings.ORB_ENTRY_END:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        has_vwap = "vwap" in df.columns
        has_volume = "volume_sma" in df.columns and "volume" in df.columns

        vol_ok = True
        if has_volume and latest["volume_sma"] > 0:
            vol_ok = latest["volume"] >= settings.ORB_VOLUME_MULTIPLIER * latest["volume_sma"]

        # ── Bullish breakout ────────────────────────────────────────
        if latest["close"] > self.orb_high and prev["close"] <= self.orb_high:
            vwap_ok = latest["close"] > latest.get("vwap", 0) if has_vwap else True
            if vwap_ok and vol_ok:
                sl = self.orb_low
                target = latest["close"] + self.orb_range * settings.ORB_RR_RATIO
                self.signal_given = True
                logger.info("[ORB] LONG breakout at {:.2f}, SL={:.2f}, T={:.2f}",
                            latest["close"], sl, target)
                return Signal(
                    signal_type=SignalType.LONG,
                    strategy_name=self.name,
                    entry_price=latest["close"],
                    stop_loss_index=sl,
                    target_index=target,
                    option_type="CE",
                    reason=f"ORB breakout above {self.orb_high:.0f}",
                    timestamp=current_time,
                )

        # ── Bearish breakout ────────────────────────────────────────
        if latest["close"] < self.orb_low and prev["close"] >= self.orb_low:
            vwap_ok = latest["close"] < latest.get("vwap", float("inf")) if has_vwap else True
            if vwap_ok and vol_ok:
                sl = self.orb_high
                target = latest["close"] - self.orb_range * settings.ORB_RR_RATIO
                self.signal_given = True
                logger.info("[ORB] SHORT breakout at {:.2f}, SL={:.2f}, T={:.2f}",
                            latest["close"], sl, target)
                return Signal(
                    signal_type=SignalType.SHORT,
                    strategy_name=self.name,
                    entry_price=latest["close"],
                    stop_loss_index=sl,
                    target_index=target,
                    option_type="PE",
                    reason=f"ORB breakdown below {self.orb_low:.0f}",
                    timestamp=current_time,
                )

        return None
