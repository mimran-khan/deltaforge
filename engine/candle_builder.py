"""Builds OHLCV candles from tick data or historical API data."""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import settings

_CANDLE_CSV = Path(settings.DATA_DIR) / "candles_live.csv"


class CandleBuilder:
    """Aggregates ticks into OHLCV candles at a given interval."""

    def __init__(self, interval_minutes: int = 5):
        self.interval = interval_minutes
        self.candles: pd.DataFrame = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        self._current_bucket: Optional[datetime] = None
        self._o = self._h = self._l = self._c = 0.0
        self._v = 0

    def _bucket_start(self, ts: datetime) -> datetime:
        minutes = (ts.hour * 60 + ts.minute)
        bucket_min = (minutes // self.interval) * self.interval
        return ts.replace(hour=bucket_min // 60, minute=bucket_min % 60,
                          second=0, microsecond=0)

    def on_tick(self, price: float, volume: int, timestamp: datetime) -> Optional[pd.Series]:
        """Feed a tick. Returns completed candle row or None."""
        bucket = self._bucket_start(timestamp)

        if self._current_bucket is None:
            self._current_bucket = bucket
            self._o = self._h = self._l = self._c = price
            self._v = volume
            return None

        if bucket == self._current_bucket:
            self._h = max(self._h, price)
            self._l = min(self._l, price)
            self._c = price
            self._v += volume
            return None

        completed = pd.Series({
            "open": self._o, "high": self._h,
            "low": self._l, "close": self._c, "volume": self._v,
        }, name=self._current_bucket)

        self.candles.loc[self._current_bucket] = completed
        self._persist_candles()

        self._current_bucket = bucket
        self._o = self._h = self._l = self._c = price
        self._v = volume
        return completed

    def get_candles(self) -> pd.DataFrame:
        """Return all completed candles + current in-progress candle."""
        if self._current_bucket is not None:
            current = pd.DataFrame(
                [{"open": self._o, "high": self._h,
                  "low": self._l, "close": self._c, "volume": self._v}],
                index=[self._current_bucket],
            )
            if self.candles.empty:
                return current
            return pd.concat([self.candles, current])
        return self.candles.copy()

    @staticmethod
    def from_historical(raw_data: list, interval_minutes: int = 5) -> pd.DataFrame:
        """Convert Angel One historical API response to OHLCV DataFrame.

        Raw data format: [[timestamp, open, high, low, close, volume], ...]
        """
        if not raw_data:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(raw_data,
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(int)
        return df

    def seed(self, historical_df: pd.DataFrame):
        """Pre-fill with historical candles so indicators warm up immediately.

        The last bar becomes the in-progress candle so live ticks in the
        same bucket merge naturally and the first boundary crossing creates
        a genuinely new completed bar.
        """
        if historical_df.empty:
            return
        self.candles = historical_df.iloc[:-1].copy()
        last = historical_df.iloc[-1]
        self._current_bucket = self._bucket_start(historical_df.index[-1])
        self._o = float(last["open"])
        self._h = float(last["high"])
        self._l = float(last["low"])
        self._c = float(last["close"])
        self._v = int(last["volume"])
        logger.info("CandleBuilder seeded with {} completed + 1 in-progress bar",
                     len(self.candles))

    def reset(self):
        self.candles = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        self._current_bucket = None

    def _persist_candles(self):
        """Save completed candles to CSV for crash recovery."""
        try:
            if self.candles.empty:
                return
            self.candles.to_csv(_CANDLE_CSV, index_label="timestamp")
        except Exception as e:
            logger.warning("Candle persistence failed: {}", e)

    def load_from_disk(self) -> int:
        """Load recent candles from CSV for indicator warmup.

        Loads up to 100 most recent COMPLETED bars across all days.
        15m RSI needs ~45 bars of 5m data to compute correctly.
        """
        if not _CANDLE_CSV.exists():
            return 0
        try:
            df = pd.read_csv(_CANDLE_CSV, index_col="timestamp", parse_dates=True)
            if df.empty:
                return 0

            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            df["volume"] = df["volume"].astype(int)

            recent = df.tail(100)
            if recent.empty:
                return 0

            today = datetime.now().date()
            last_bar_date = recent.index[-1].date()

            if last_bar_date < today:
                self.candles = recent.copy()
                self._current_bucket = None
                logger.info(
                    "Loaded {} historical candles from disk (all completed, prior day)",
                    len(recent),
                )
            else:
                self.seed(recent)
                logger.info("Loaded {} candles from disk cache (multi-day)", len(recent))

            return len(recent)
        except Exception as e:
            logger.warning("Candle load from disk failed: {}", e)
            return 0
