"""Deterministic option premium model for backtest/production parity.

Uses the Black-Scholes delta approximation to model how option
premiums move with the underlying. Same code runs in backtest
and live trading -- no random noise.

The live system will override `get_live_premium()` with actual
broker LTP data, but the underlying logic is identical.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PremiumState:
    """Tracks the premium of an options position deterministically."""
    entry_premium: float
    entry_index_price: float
    delta: float
    theta_per_candle: float
    direction: str  # "LONG" or "SHORT"
    sl_premium: float
    target_premium: float

    # Trailing stop state
    peak_premium: float = 0.0
    trail_active: bool = False

    def __post_init__(self):
        self.peak_premium = self.entry_premium

    def current_premium(self, current_index_price: float,
                         candles_elapsed: int) -> float:
        """Calculate current premium deterministically from index move + theta."""
        if self.direction == "LONG":
            index_move = current_index_price - self.entry_index_price
        else:
            index_move = self.entry_index_price - current_index_price

        premium_from_delta = index_move * self.delta
        time_decay = candles_elapsed * self.theta_per_candle

        current = self.entry_premium + premium_from_delta - time_decay
        return max(current, 0.5)  # premium can't go below 0.5

    def update_trail(self, current_prem: float,
                      trigger_pct: float, trail_pct: float) -> float | None:
        """Update trailing stop. Returns trail floor if triggered, else None."""
        if current_prem > self.peak_premium:
            self.peak_premium = current_prem

        gain_pct = (self.peak_premium - self.entry_premium) / self.entry_premium * 100
        if gain_pct >= trigger_pct:
            self.trail_active = True
            return self.peak_premium * (1 - trail_pct / 100)
        return None

    def check_exit(self, current_prem: float,
                    trail_floor: float | None) -> str | None:
        """Check if any exit condition is hit. Returns reason or None."""
        if current_prem <= self.sl_premium:
            return "SL"
        if current_prem >= self.target_premium:
            return "TGT"
        if trail_floor is not None and current_prem <= trail_floor:
            return "TRAIL"
        return None


def create_premium_state(
    entry_index_price: float,
    direction: str,
    base_premium: float = 95.0,
    delta: float = 0.45,
    theta_per_candle: float = 0.15,
    sl_pct: float = 35.0,
    confluence_score: float = 50.0,
) -> PremiumState:
    """Create a premium state for a new trade, deterministically.

    Premium is based on confluence score strength:
    - Higher confluence = slightly higher entry premium (more ATM strike)
    - This is deterministic, no random noise.
    """
    abs_conf = abs(confluence_score)
    premium_adj = (abs_conf - 40) / 100 * 8  # +0 to +4.8 based on confluence
    entry_premium = base_premium + max(0, premium_adj)

    # Dynamic target based on confluence strength
    if abs_conf >= 70:
        target_prem = entry_premium * 1.35
    elif abs_conf >= 50:
        target_prem = entry_premium * 1.25
    else:
        target_prem = entry_premium * 1.18

    sl_prem = entry_premium * (1 - sl_pct / 100)

    return PremiumState(
        entry_premium=round(entry_premium, 2),
        entry_index_price=entry_index_price,
        delta=delta,
        theta_per_candle=theta_per_candle,
        direction=direction,
        sl_premium=round(sl_prem, 2),
        target_premium=round(target_prem, 2),
    )
