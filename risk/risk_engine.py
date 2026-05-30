"""Independent risk gatekeeper -- approves or rejects every trade signal."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from config import settings
from risk.capital_tracker import CapitalTracker
from strategies.base_strategy import Signal


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    lots: int = 0
    premium_sl_pct: float = 0.0


class RiskEngine:
    """Pre-trade risk gatekeeper. Strategy proposes, risk engine disposes."""

    def __init__(self, capital_tracker: CapitalTracker):
        self.capital = capital_tracker
        self._halted = False
        self._halt_reason = ""

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def halt(self, reason: str):
        self._halted = True
        self._halt_reason = reason
        logger.warning("RISK HALT: {}", reason)

    def resume(self):
        self._halted = False
        self._halt_reason = ""

    def evaluate(self, signal: Signal) -> RiskDecision:
        """Evaluate a signal against all risk rules. Returns approval or rejection."""

        # Rule 0: System halted
        if self._halted:
            return RiskDecision(False, f"System halted: {self._halt_reason}")

        # Rule 1: Daily loss limit
        daily_limit = self.capital.get_daily_loss_limit()
        if abs(self.capital.daily_pnl) >= daily_limit and self.capital.daily_pnl < 0:
            self.halt(f"Daily loss limit hit: Rs {self.capital.daily_pnl:.0f}")
            return RiskDecision(False, f"Daily loss limit Rs {daily_limit:.0f} breached")

        # Rule 2: Weekly loss limit
        weekly_limit = self.capital.get_weekly_loss_limit()
        if abs(self.capital.weekly_pnl) >= weekly_limit and self.capital.weekly_pnl < 0:
            self.halt(f"Weekly loss limit hit: Rs {self.capital.weekly_pnl:.0f}")
            return RiskDecision(False, f"Weekly loss limit Rs {weekly_limit:.0f} breached")

        # Rule 3: Max trades per day
        if self.capital.trades_today >= settings.MAX_TRADES_PER_DAY:
            return RiskDecision(False, f"Max trades/day ({settings.MAX_TRADES_PER_DAY}) reached")

        # Rule 4: Consecutive loss breaker
        if self.capital.consecutive_losses >= settings.MAX_CONSECUTIVE_LOSSES:
            self.halt(f"{self.capital.consecutive_losses} consecutive losses")
            return RiskDecision(
                False,
                f"Consecutive loss limit ({settings.MAX_CONSECUTIVE_LOSSES}) hit"
            )

        # Rule 5: Capital too low
        lot_size = settings.NIFTY_LOT_SIZE
        min_premium_cost = 50 * lot_size  # assume minimum Rs 50 premium
        if self.capital.current_capital < min_premium_cost:
            self.halt(f"Capital Rs {self.capital.current_capital:.0f} below minimum")
            return RiskDecision(False, "Insufficient capital")

        # Rule 6: Compute lot size
        max_lots = self.capital.get_max_lots()

        # Approved
        logger.info("Risk APPROVED: {} lots={} capital=Rs {:.0f}",
                     signal.strategy_name, max_lots, self.capital.current_capital)

        return RiskDecision(
            approved=True,
            reason="All risk checks passed",
            lots=max_lots,
            premium_sl_pct=settings.PREMIUM_SL_PCT,
        )

    def check_daily_limits(self) -> bool:
        """Continuous check -- call on every P&L update."""
        daily_limit = self.capital.get_daily_loss_limit()
        if self.capital.daily_pnl < 0 and abs(self.capital.daily_pnl) >= daily_limit:
            self.halt(f"Daily loss Rs {self.capital.daily_pnl:.0f} exceeds limit Rs {daily_limit:.0f}")
            return False
        return True

    def start_day(self):
        self.capital.start_day()
        self._halted = False
        self._halt_reason = ""
