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
from typing import Optional

import pytz
from loguru import logger

from config import settings
from risk.capital_tracker import CapitalTracker
from risk.kill_switch import set_halt as _set_halt_flag, clear_halt as _clear_halt_flag

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
        self._vix_cache: Optional[float] = None
        self._vix_fetched_at: Optional[datetime] = None

    def _fetch_vix(self) -> Optional[float]:
        """Fetch India VIX with 5-minute cache. Returns None if unavailable."""
        now = datetime.now(IST)
        if (self._vix_fetched_at and self._vix_cache is not None
                and (now - self._vix_fetched_at).total_seconds() < 300):
            return self._vix_cache

        vix_value = None
        try:
            import urllib.request
            import json as _json
            req = urllib.request.Request(
                "https://www.nseindia.com/api/allIndices",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read().decode())
            for item in data.get("data", []):
                if "VIX" in item.get("index", "").upper():
                    vix_value = float(item.get("last", 0))
                    break
        except Exception as e:
            logger.warning("VIX fetch unavailable: {}", e)

        self._vix_fetched_at = now
        self._vix_cache = vix_value
        return vix_value

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def halt(self, reason: str):
        self._halted = True
        self._halt_reason = reason
        _set_halt_flag(reason)
        logger.warning("RISK HALT: {}", reason)

    def resume(self):
        self._halted = False
        self._halt_reason = ""
        _clear_halt_flag()

    def evaluate(self, confluence_score: float = 0,
                 direction: str = "NEUTRAL",
                 strength: str = "NONE",
                 signal_obj=None,
                 lot_multiplier: float = 1.0,
                 min_confidence_override: int | None = None) -> RiskDecision:
        """Evaluate a trade against all pre-trade risk rules.

        Gates 0-9 must ALL pass. Any single failure = rejection.
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

        # Gate 2.5: Daily profit target -- secure the bag
        if settings.DAILY_PROFIT_TARGET_PCT > 0 and self.capital.day_start_capital > 0:
            profit_target = self.capital.day_start_capital * settings.DAILY_PROFIT_TARGET_PCT / 100
            if self.capital.daily_pnl >= profit_target:
                logger.info("Daily profit target hit: Rs {:.0f} >= {:.0f} ({}% of day start Rs {:.0f}). No new entries.",
                           self.capital.daily_pnl, profit_target, settings.DAILY_PROFIT_TARGET_PCT,
                           self.capital.day_start_capital)
                return RiskDecision(False, f"Daily profit target {settings.DAILY_PROFIT_TARGET_PCT}% reached")

        # Gate 3: Weekly loss limit
        weekly_limit = self.capital.get_weekly_loss_limit()
        if self.capital.weekly_pnl < 0 and abs(self.capital.weekly_pnl) >= weekly_limit:
            self.halt(f"Weekly loss Rs {self.capital.weekly_pnl:.0f} >= limit Rs {weekly_limit:.0f}")
            return RiskDecision(False, "Weekly loss limit breached")

        # Gate 4: Consecutive losses
        if self.capital.consecutive_losses >= settings.MAX_CONSECUTIVE_LOSSES:
            self.halt(f"{self.capital.consecutive_losses} consecutive losses")
            return RiskDecision(False, "Consecutive loss limit hit")

        # Gate 6: Drawdown tier check
        dd = self.capital.drawdown_pct
        if dd >= settings.DRAWDOWN_HALT_PCT:
            self.halt(f"Drawdown {dd:.1f}% >= halt threshold {settings.DRAWDOWN_HALT_PCT}%")
            return RiskDecision(False, "Max drawdown reached")

        # Gate 5.5: VIX circuit breaker
        vix = self._fetch_vix()
        if vix is None:
            logger.warning("VIX unavailable -- skipping VIX gate (proceeding with caution)")
        elif vix > settings.MAX_VIX_THRESHOLD:
            return RiskDecision(
                False,
                f"VIX {vix:.1f} > threshold {settings.MAX_VIX_THRESHOLD}",
            )

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

        # Gate 9: Signal confidence threshold
        min_conf = min_confidence_override if min_confidence_override is not None else getattr(settings, 'PULLBACK_MIN_CONFIDENCE', 50)
        if confluence_score < min_conf:
            return RiskDecision(False, f"Confidence {confluence_score:.0f} < min {min_conf}")

        # Compute lot size with drawdown-aware sizing
        lots = self.capital.get_max_lots(
            premium_per_unit=settings.PREMIUM_BASE)
        if lot_multiplier != 1.0:
            lots = max(1, int(lots * lot_multiplier))
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

        # Check: Daily profit target (soft - just log, don't halt)
        if settings.DAILY_PROFIT_TARGET_PCT > 0 and self.capital.day_start_capital > 0:
            profit_target = self.capital.day_start_capital * settings.DAILY_PROFIT_TARGET_PCT / 100
            if self.capital.daily_pnl >= profit_target:
                logger.info("Profit target reached: Rs {:.0f} / {:.0f} ({}%)",
                           self.capital.daily_pnl, profit_target, settings.DAILY_PROFIT_TARGET_PCT)

        # Check 2: Weekly loss
        weekly_limit = self.capital.get_weekly_loss_limit()
        if self.capital.weekly_pnl < 0 and abs(self.capital.weekly_pnl) >= weekly_limit:
            self.halt(f"RT: Weekly loss Rs {self.capital.weekly_pnl:.0f}")
            return False

        # Check 3: Consecutive losses
        if self.capital.consecutive_losses >= settings.MAX_CONSECUTIVE_LOSSES:
            self.halt(f"RT: {self.capital.consecutive_losses} consecutive losses")
            return False

        # Check 4: Drawdown
        dd = self.capital.drawdown_pct
        if dd >= settings.DRAWDOWN_HALT_PCT:
            self.halt(f"RT: Drawdown {dd:.1f}%")
            return False

        # Check 5: Capital minimum
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
