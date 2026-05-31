"""Three-layer risk management (FIA standard).

Layer 1: Pre-trade gates (before order placement)
Layer 2: Real-time monitoring (during session)
Layer 3: Post-trade reconciliation (end of day)

All checks operate independently of external systems.
If Telegram fails, risk still halts. If broker fails, risk still halts.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

import pytz
from loguru import logger

from config import settings
from risk.capital_tracker import CapitalTracker

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    lots: int = 0
    premium_sl_pct: float = 0.0


class RiskEngine:
    """Pre-trade gatekeeper + real-time monitor.

    Every signal must pass ALL gates. Any single failure = rejection.
    """

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

    def evaluate(self, confluence_score: float = 0,
                 direction: str = "NEUTRAL",
                 strength: str = "NONE",
                 signal_obj=None) -> RiskDecision:
        """Evaluate a trade against all pre-trade risk rules.

        Accepts either raw confluence data or a Signal object for
        backward compatibility.
        """

        # Gate 0: System halted
        if self._halted:
            return RiskDecision(False, f"Halted: {self._halt_reason}")

        # Gate 1: Minimum capital
        if self.capital.current_capital < settings.MIN_CAPITAL_TO_TRADE:
            self.halt(f"Capital Rs {self.capital.current_capital:.0f} < minimum Rs {settings.MIN_CAPITAL_TO_TRADE}")
            return RiskDecision(False, "Capital below minimum")

        # Gate 2: Daily loss limit
        daily_limit = self.capital.get_daily_loss_limit()
        if self.capital.daily_pnl < 0 and abs(self.capital.daily_pnl) >= daily_limit:
            self.halt(f"Daily loss Rs {self.capital.daily_pnl:.0f} >= limit Rs {daily_limit:.0f}")
            return RiskDecision(False, "Daily loss limit breached")

        # Gate 3: Weekly loss limit
        weekly_limit = self.capital.get_weekly_loss_limit()
        if self.capital.weekly_pnl < 0 and abs(self.capital.weekly_pnl) >= weekly_limit:
            self.halt(f"Weekly loss Rs {self.capital.weekly_pnl:.0f} >= limit Rs {weekly_limit:.0f}")
            return RiskDecision(False, "Weekly loss limit breached")

        # Gate 4: Max trades per day
        if self.capital.trades_today >= settings.MAX_TRADES_PER_DAY:
            return RiskDecision(False, f"Max {settings.MAX_TRADES_PER_DAY} trades/day reached")

        # Gate 5: Consecutive losses
        if self.capital.consecutive_losses >= settings.MAX_CONSECUTIVE_LOSSES:
            self.halt(f"{self.capital.consecutive_losses} consecutive losses")
            return RiskDecision(False, "Consecutive loss limit hit")

        # Gate 6: Drawdown tier check
        dd = self.capital.drawdown_pct
        if dd >= settings.DRAWDOWN_HALT_PCT:
            self.halt(f"Drawdown {dd:.1f}% >= halt threshold {settings.DRAWDOWN_HALT_PCT}%")
            return RiskDecision(False, "Max drawdown reached")

        # Gate 7: Expiry day ban
        if settings.SKIP_EXPIRY_DAY:
            today_weekday = datetime.now(IST).weekday()
            if today_weekday == settings.NIFTY_EXPIRY_DAY:
                return RiskDecision(False, "No trading on expiry day")

        # Gate 8: Time window
        now_str = datetime.now(IST).strftime("%H:%M")
        if now_str < settings.ENTRY_START:
            return RiskDecision(False, f"Before entry window ({settings.ENTRY_START})")
        if now_str > settings.ENTRY_END:
            return RiskDecision(False, f"After entry window ({settings.ENTRY_END})")

        # Gate 9: Confluence threshold
        if abs(confluence_score) < settings.CONFLUENCE_THRESHOLD:
            return RiskDecision(False, f"Confluence {confluence_score:.0f} < threshold {settings.CONFLUENCE_THRESHOLD}")

        strength_order = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3, "EXTREME": 4}
        if strength_order.get(strength, 0) < strength_order.get(settings.MIN_STRENGTH, 3):
            return RiskDecision(False, f"Strength {strength} < min {settings.MIN_STRENGTH}")

        # Compute lot size with drawdown-aware sizing
        lots = self.capital.get_max_lots(
            premium_per_unit=settings.PREMIUM_BASE)
        if lots <= 0:
            return RiskDecision(False, "Lot sizing returned 0 (drawdown halt)")

        logger.info("RISK APPROVED: {}x lots, DD={:.1f}%, Cap=Rs {:.0f}",
                     lots, dd, self.capital.current_capital)

        return RiskDecision(
            approved=True,
            reason="All gates passed",
            lots=lots,
            premium_sl_pct=settings.PREMIUM_SL_PCT,
        )

    def check_realtime(self) -> bool:
        """Real-time monitoring -- call on every tick/candle.

        Returns False if trading should stop immediately.
        """
        # Check 1: Daily loss
        daily_limit = self.capital.get_daily_loss_limit()
        if self.capital.daily_pnl < 0 and abs(self.capital.daily_pnl) >= daily_limit:
            self.halt(f"RT: Daily loss Rs {self.capital.daily_pnl:.0f}")
            return False

        # Check 2: Drawdown
        dd = self.capital.drawdown_pct
        if dd >= settings.DRAWDOWN_HALT_PCT:
            self.halt(f"RT: Drawdown {dd:.1f}%")
            return False

        # Check 3: Capital minimum
        if self.capital.current_capital < settings.MIN_CAPITAL_TO_TRADE:
            self.halt(f"RT: Capital Rs {self.capital.current_capital:.0f}")
            return False

        return True

    # Backward compat alias
    def check_daily_limits(self) -> bool:
        return self.check_realtime()

    def start_day(self):
        self.capital.start_day()
        self._halted = False
        self._halt_reason = ""
