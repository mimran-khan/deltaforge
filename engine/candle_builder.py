"""Builds OHLCV candles from tick data or historical API data."""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


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

        self._current_bucket = bucket
        self._o = self._h = self._l = self._c = price
        self._v = volume
        return completed

    def get_candles(self) -> pd.DataFrame:
        """Return all completed candles + current in-progress candle."""
        if self._current_bucket is not None:
            current = pd.Series({
                "open": self._o, "high": self._h,
                "low": self._l, "close": self._c, "volume": self._v,
            }, name=self._current_bucket)
            return pd.concat([self.candles, current.to_frame().T])
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

    def reset(self):
        self.candles = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        self._current_bucket = None
