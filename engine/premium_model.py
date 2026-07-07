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
    dte: float = 3.0  # days to expiry at entry

    # Trailing stop state
    peak_premium: float = 0.0
    trail_active: bool = False

    def __post_init__(self):
        self.peak_premium = self.entry_premium

    def current_premium(self, current_index_price: float,
                         candles_elapsed: int) -> float:
        """Calculate current premium with DTE-aware gamma and theta.

        Near expiry (DTE <= 1): delta is amplified (gamma effect) and
        theta decay accelerates (sqrt model).
        """
        if self.direction == "LONG":
            index_move = current_index_price - self.entry_index_price
        else:
            index_move = self.entry_index_price - current_index_price

        effective_delta = self.delta
        dte_now = max(0.1, self.dte - candles_elapsed / 75.0)  # ~75 bars/day
        if dte_now <= 1.0:
            effective_delta = min(self.delta * 1.3, 0.95)
        elif dte_now <= 2.0:
            effective_delta = min(self.delta * 1.15, 0.90)

        premium_from_delta = index_move * effective_delta

        gamma_base = 0.002
        if dte_now <= 1.0:
            gamma_base = 0.005
        elif dte_now <= 2.0:
            gamma_base = 0.003
        gamma_pnl = 0.5 * gamma_base * (index_move ** 2)

        if dte_now <= 2.0:
            theta_mult = 1.0 / max(0.3, dte_now ** 0.5)
        else:
            theta_mult = 1.0
        time_decay = candles_elapsed * self.theta_per_candle * theta_mult

        current = self.entry_premium + premium_from_delta + gamma_pnl - time_decay
        return max(current, 0.5)

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


STRATEGY_TARGET_MULT = {
    "SUPERTREND":    {70: 1.50, 50: 1.50, 0: 1.50},
    "STOCH_CROSS":   {70: 1.50, 50: 1.50, 0: 1.50},
    "PULLBACK":      {70: 1.50, 50: 1.50, 0: 1.50},
    "TREND_RIDE":    {70: 1.50, 50: 1.50, 0: 1.50},
    "RSI_REVERSION": {70: 1.60, 50: 1.45, 0: 1.35},
    "VWAP_MOMENTUM": {70: 1.60, 50: 1.45, 0: 1.35},
    "EMA_MOMENTUM":  {70: 2.50, 50: 2.00, 0: 1.70},
    "VWAP_MEAN_REV": {70: 1.40, 50: 1.30, 0: 1.20},
    "CPR_RANGE":     {70: 1.40, 50: 1.30, 0: 1.20},
    "GAP_TRADE":     {70: 2.00, 50: 1.70, 0: 1.50},
    "CPR_BREAKOUT":  {70: 2.00, 50: 1.70, 0: 1.50},
    "ADX_BREAKOUT":  {70: 1.60, 50: 1.45, 0: 1.35},
    "ORB_BREAKOUT":  {70: 2.00, 50: 1.70, 0: 1.50},
    "BB_SQUEEZE":    {70: 2.00, 50: 1.70, 0: 1.50},
    "VWAP_BOUNCE":   {70: 1.60, 50: 1.45, 0: 1.35},
    "RSI_DIVERGENCE": {70: 1.60, 50: 1.45, 0: 1.35},
    "TRIPLE_CONFIRM": {70: 2.50, 50: 2.00, 0: 1.70},
    "FIRST_HOUR_MOM": {70: 2.00, 50: 1.70, 0: 1.50},
    "VWAP_2SD_REV":   {70: 1.40, 50: 1.30, 0: 1.20},
    "NARROW_CPR_BO":  {70: 2.00, 50: 1.70, 0: 1.50},
}

STRATEGY_SL_PCT = {
    "SUPERTREND":    8.0,
    "STOCH_CROSS":   8.0,
    "PULLBACK":      8.0,
    "TREND_RIDE":    8.0,
    "RSI_REVERSION": 10.0,
    "VWAP_MOMENTUM": 10.0,
    "EMA_MOMENTUM":  10.0,
    "VWAP_MEAN_REV": 10.0,
    "CPR_RANGE":     10.0,
    "GAP_TRADE":     10.0,
    "CPR_BREAKOUT":  10.0,
    "ADX_BREAKOUT":  10.0,
    "ORB_BREAKOUT":  10.0,
    "BB_SQUEEZE":    10.0,
    "VWAP_BOUNCE":   8.0,
    "RSI_DIVERGENCE": 8.0,
    "TRIPLE_CONFIRM": 15.0,
    "FIRST_HOUR_MOM": 12.0,
    "VWAP_2SD_REV":   10.0,
    "NARROW_CPR_BO":  12.0,
}

STRATEGY_HOLD_BARS = {
    "PULLBACK":    12,
    "SUPERTREND":  20,
    "TREND_RIDE":  15,
    "STOCH_CROSS": 12,
}

STRATEGY_TRAIL = {
    "PULLBACK":    {"trigger": 5.0, "pullback": 2.0},
    "SUPERTREND":  {"trigger": 10.0, "pullback": 4.0},
    "TREND_RIDE":  {"trigger": 5.0, "pullback": 2.0},
    "STOCH_CROSS": {"trigger": 5.0, "pullback": 2.0},
}


def create_premium_state(
    entry_index_price: float,
    direction: str,
    base_premium: float = 95.0,
    delta: float = 0.45,
    theta_per_candle: float = 0.15,
    sl_pct: float = 35.0,
    confluence_score: float = 50.0,
    signal_type: str = "",
    dte: float = 3.0,
) -> PremiumState:
    """Create a premium state for a new trade, deterministically.

    Uses strategy-specific target multipliers: SUPERTREND (80% WR) gets
    wider targets, PULLBACK gets tighter targets for faster exits.
    """
    abs_conf = abs(confluence_score)
    premium_adj = (abs_conf - 40) / 100 * 8
    entry_premium = base_premium + max(0, premium_adj)

    tiers = STRATEGY_TARGET_MULT.get(
        signal_type, {70: 1.45, 50: 1.35, 0: 1.25})
    if abs_conf >= 70:
        mult = tiers[70]
    elif abs_conf >= 50:
        mult = tiers[50]
    else:
        mult = tiers[0]
    target_prem = entry_premium * mult

    effective_sl = STRATEGY_SL_PCT.get(signal_type, sl_pct)
    sl_prem = entry_premium * (1 - effective_sl / 100)

    return PremiumState(
        entry_premium=round(entry_premium, 2),
        entry_index_price=entry_index_price,
        delta=delta,
        theta_per_candle=theta_per_candle,
        direction=direction,
        sl_premium=round(sl_prem, 2),
        target_premium=round(target_prem, 2),
        dte=dte,
    )
