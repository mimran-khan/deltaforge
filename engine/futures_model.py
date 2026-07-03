"""Futures position model for direct price-based P&L.

Unlike PremiumState (which simulates option premium with delta/gamma/theta),
this model handles futures where P&L = (exit - entry) * lot_size * lots.

No theta decay, no delta, no gamma.  Used for MCX Gold, MCX Crude, NSE USDINR.

This module is purely additive -- the existing PremiumState in premium_model.py
is unchanged and continues to handle Nifty options.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config.instruments import Instrument

logger = logging.getLogger(__name__)


@dataclass
class FuturesPosition:
    """Tracks a futures position through its lifecycle."""
    instrument_name: str
    direction: str             # "LONG" or "SHORT"
    entry_price: float
    lot_size: int
    tick_size: float
    tick_value: float
    lots: int
    qty: int                   # lot_size * lots

    sl_price: float
    target_price: float

    peak_price: float = 0.0    # best price seen (for trailing)
    trail_active: bool = False

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price

    def current_pnl(self, current_price: float) -> float:
        """Unrealized P&L at current market price.

        Uses tick-based calculation: (price_diff / tick_size) * tick_value * lots.
        This gives correct results regardless of whether price_divisor was applied,
        as long as tick_size and prices are in the same unit system.
        """
        if self.direction == "LONG":
            price_diff = current_price - self.entry_price
        else:
            price_diff = self.entry_price - current_price

        if self.tick_size > 0:
            ticks = price_diff / self.tick_size
            return ticks * self.tick_value * self.lots

        return price_diff * self.lot_size * self.lots

    def current_price_for_direction(self, market_price: float) -> float:
        """Effective price considering direction (for SL/target comparison)."""
        return market_price

    def update_peak(self, current_price: float):
        """Track the best price seen for trailing stop."""
        if self.direction == "LONG":
            if current_price > self.peak_price:
                self.peak_price = current_price
        else:
            if self.peak_price == self.entry_price or current_price < self.peak_price:
                self.peak_price = current_price

    def update_trail(self, current_price: float,
                     trigger_pct: float, trail_pct: float) -> Optional[float]:
        """Update trailing stop. Returns trail floor/ceiling if active, else None.

        For LONG: trail activates when price rises trigger_pct% above entry,
                  floor = peak_price * (1 - trail_pct/100)
        For SHORT: trail activates when price drops trigger_pct% below entry,
                   ceiling = peak_price * (1 + trail_pct/100)
        """
        self.update_peak(current_price)

        if self.direction == "LONG":
            gain_pct = (self.peak_price - self.entry_price) / self.entry_price * 100
            if gain_pct >= trigger_pct:
                self.trail_active = True
                return self.peak_price * (1 - trail_pct / 100)
        else:
            gain_pct = (self.entry_price - self.peak_price) / self.entry_price * 100
            if gain_pct >= trigger_pct:
                self.trail_active = True
                return self.peak_price * (1 + trail_pct / 100)
        return None

    def check_exit(self, current_price: float,
                   trail_level: Optional[float]) -> Optional[str]:
        """Check if any exit condition is met. Returns reason or None."""
        if self.direction == "LONG":
            if current_price <= self.sl_price:
                return "SL"
            if current_price >= self.target_price:
                return "TGT"
            if trail_level is not None and current_price <= trail_level:
                return "TRAIL"
        else:
            if current_price >= self.sl_price:
                return "SL"
            if current_price <= self.target_price:
                return "TGT"
            if trail_level is not None and current_price >= trail_level:
                return "TRAIL"
        return None


def create_futures_position(
    instrument: Instrument,
    direction: str,
    entry_price: float,
    lots: int,
    sl_pct: float,
    target_mult: float = 1.5,
) -> FuturesPosition:
    """Create a futures position with SL and target based on instrument specs.

    Args:
        instrument: Instrument definition from registry
        direction: "LONG" or "SHORT"
        entry_price: fill price (must already be divided by price_divisor)
        lots: number of lots
        sl_pct: stop loss as % of entry price (e.g., 2.0 = 2%)
        target_mult: risk-reward multiplier (e.g., 1.5 = 1.5x risk)
    """
    if instrument.price_divisor > 1.0 and entry_price > 1000:
        logger.error(
            "PRICE FORMAT ERROR: %s entry_price=%.2f looks undivided "
            "(price_divisor=%.0f). Expected ~%.2f. Dividing now.",
            instrument.name, entry_price, instrument.price_divisor,
            entry_price / instrument.price_divisor,
        )
        entry_price = entry_price / instrument.price_divisor

    risk_amount = entry_price * sl_pct / 100

    if direction == "LONG":
        sl_price = entry_price - risk_amount
        target_price = entry_price + risk_amount * target_mult
    else:
        sl_price = entry_price + risk_amount
        target_price = entry_price - risk_amount * target_mult

    return FuturesPosition(
        instrument_name=instrument.name,
        direction=direction,
        entry_price=entry_price,
        lot_size=instrument.lot_size,
        tick_size=instrument.tick_size,
        tick_value=instrument.tick_value,
        lots=lots,
        qty=instrument.lot_size * lots,
        sl_price=round(sl_price, 4),
        target_price=round(target_price, 4),
        peak_price=entry_price,
    )
