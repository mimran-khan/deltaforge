"""Separate capital tracker for multi-asset futures trading.

Uses its own file (multi_asset_capital.json) -- completely isolated from the
Nifty options capital pool in capital.json.  No shared state, no interference.

This is a simplified version of CapitalTracker focused on futures:
- Per-instrument capital pools (allocated by percentage)
- Simpler lot sizing (margin-based, not premium-based)
- Separate drawdown tracking
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz
from loguru import logger

from config import settings
from config.instruments import Instrument, get_futures_instruments

IST = pytz.timezone("Asia/Kolkata")

FUTURES_CAPITAL_FILE = settings.DATA_DIR / "multi_asset_capital.json"


class FuturesCapitalTracker:
    """Manages capital for all futures instruments (Gold, Crude, USDINR)."""

    def __init__(self, total_allocation: float = 0):
        self.total_capital = total_allocation or getattr(settings, 'FUTURES_STARTING_CAPITAL', 50000)
        self.pools: dict[str, InstrumentPool] = {}
        self._trade_log: list = []

        for inst in get_futures_instruments():
            alloc = self.total_capital * (inst.capital_alloc_pct / 100)
            self.pools[inst.name] = InstrumentPool(
                instrument_name=inst.name,
                allocated=alloc,
                current=alloc,
            )

        self._load()

    def _load(self):
        if not FUTURES_CAPITAL_FILE.exists():
            return
        try:
            with open(FUTURES_CAPITAL_FILE) as f:
                data = json.load(f)
            for name, pool_data in data.get("pools", {}).items():
                if name in self.pools:
                    self.pools[name].current = pool_data.get("current", self.pools[name].allocated)
                    self.pools[name].daily_pnl = pool_data.get("daily_pnl", 0)
                    self.pools[name].peak = pool_data.get("peak", self.pools[name].current)
                    self.pools[name].trades_today = pool_data.get("trades_today", 0)
            logger.info("Loaded futures capital: {} pools", len(self.pools))
        except Exception as e:
            logger.warning("Failed to load futures capital: {}", e)

    def save(self):
        data = {
            "total_capital": round(self.total_capital, 2),
            "pools": {
                name: pool.to_dict() for name, pool in self.pools.items()
            },
            "last_updated": datetime.now(IST).isoformat(),
        }
        try:
            tmp = FUTURES_CAPITAL_FILE.with_suffix('.tmp')
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(str(tmp), str(FUTURES_CAPITAL_FILE))
        except Exception as e:
            logger.error("Futures capital save error: {}", e)

    def start_day(self):
        for pool in self.pools.values():
            pool.daily_pnl = 0
            pool.trades_today = 0
        self.save()

    def record_trade(self, instrument_name: str, pnl: float):
        if instrument_name not in self.pools:
            logger.warning("No pool for instrument {}", instrument_name)
            return
        pool = self.pools[instrument_name]
        pool.current += pnl
        pool.daily_pnl += pnl
        pool.trades_today += 1
        if pool.current > pool.peak:
            pool.peak = pool.current
        self.save()

    def get_lots(self, instrument: Instrument, entry_price: float) -> int:
        pool = self.pools.get(instrument.name)
        if not pool:
            return 0
        margin_per_lot = entry_price * instrument.lot_size * instrument.margin_pct / 100
        if margin_per_lot <= 0:
            return 1
        lots = int(pool.current / margin_per_lot)
        if lots <= 0:
            return 0
        return min(lots, 5)

    def get_summary(self) -> dict:
        return {
            "total": round(sum(p.current for p in self.pools.values()), 2),
            "pools": {name: pool.to_dict() for name, pool in self.pools.items()},
        }


class InstrumentPool:
    def __init__(self, instrument_name: str, allocated: float, current: float):
        self.instrument_name = instrument_name
        self.allocated = allocated
        self.current = current
        self.daily_pnl: float = 0
        self.peak: float = current
        self.trades_today: int = 0

    @property
    def drawdown_pct(self) -> float:
        if self.peak <= 0:
            return 0
        return (self.peak - self.current) / self.peak * 100

    def to_dict(self) -> dict:
        return {
            "allocated": round(self.allocated, 2),
            "current": round(self.current, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "peak": round(self.peak, 2),
            "drawdown_pct": round(self.drawdown_pct, 2),
            "trades_today": self.trades_today,
        }
