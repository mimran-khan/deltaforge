"""Risk engine for multi-asset futures trading.

Separate from the existing RiskEngine (which handles Nifty options).
Applies per-instrument risk rules without touching the Nifty risk flow.
"""

from __future__ import annotations

from datetime import datetime

import pytz
from loguru import logger

from config.instruments import Instrument
from risk.futures_capital_tracker import FuturesCapitalTracker

IST = pytz.timezone("Asia/Kolkata")


class FuturesRiskEngine:
    """Per-instrument risk checks for futures trading."""

    def __init__(self, capital_tracker: FuturesCapitalTracker):
        self.capital = capital_tracker
        self._halted_instruments: set[str] = set()

    def can_trade(self, instrument: Instrument) -> tuple[bool, str]:
        """Check if a new trade is allowed for the given instrument.

        Returns (allowed, reason).
        """
        if instrument.name in self._halted_instruments:
            return False, f"{instrument.name} halted for the day"

        pool = self.capital.pools.get(instrument.name)
        if not pool:
            return False, f"No capital pool for {instrument.name}"

        daily_loss_limit = pool.allocated * 0.05
        if abs(pool.daily_pnl) > daily_loss_limit and pool.daily_pnl < 0:
            self._halted_instruments.add(instrument.name)
            logger.warning("RISK: {} halted -- daily loss Rs {:.0f} > limit Rs {:.0f}",
                           instrument.display_name, abs(pool.daily_pnl), daily_loss_limit)
            return False, f"Daily loss limit hit (Rs {abs(pool.daily_pnl):.0f})"

        if pool.drawdown_pct > 20:
            return False, f"Drawdown {pool.drawdown_pct:.1f}% > 20% threshold"

        overrides = instrument.strategy
        max_trades = overrides.max_total_per_day or 8
        if pool.trades_today >= max_trades:
            return False, f"Max trades per day reached ({max_trades})"

        now = datetime.now(IST)
        t = now.strftime("%H:%M")
        if t < instrument.hours.entry_start or t > instrument.hours.entry_end:
            return False, f"Outside entry window ({instrument.hours.entry_start}-{instrument.hours.entry_end})"

        return True, "OK"

    def reset_day(self):
        self._halted_instruments.clear()
