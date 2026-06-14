"""Persistent equity tracker with compound lot sizing.

Saves ALL state to disk so kill-switch watchdog can read it.
Implements drawdown tiers for progressive risk reduction.
"""

from __future__ import annotations
import fcntl
import json
import os
import shutil
from datetime import datetime
from typing import Optional

import pytz
from loguru import logger

from config import settings

IST = pytz.timezone("Asia/Kolkata")


class CapitalTracker:
    def __init__(self):
        self.starting_capital: float = settings.STARTING_CAPITAL
        self.current_capital: float = settings.STARTING_CAPITAL
        self.day_start_capital: float = settings.STARTING_CAPITAL
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.trades_today: int = 0
        self.wins_today: int = 0
        self.losses_today: int = 0
        self.consecutive_losses: int = 0
        self.peak_capital: float = settings.STARTING_CAPITAL
        self.max_drawdown: float = 0.0
        self._trade_log: list = []
        self._last_start_date: str = ""
        self._last_weekly_reset: str = ""
        self.week_start_capital: float = settings.STARTING_CAPITAL
        self._load()

    def _load(self):
        if settings.CAPITAL_FILE.exists():
            try:
                with open(settings.CAPITAL_FILE) as f:
                    data = json.load(f)
                self.current_capital = data.get("current_capital", self.starting_capital)
                self.total_pnl = data.get("total_pnl", 0)
                self.peak_capital = data.get("peak_capital", self.current_capital)
                self.weekly_pnl = data.get("weekly_pnl", 0)
                self.max_drawdown = data.get("max_drawdown", 0)
                self.daily_pnl = data.get("daily_pnl", 0)
                self.trades_today = data.get("trades_today", 0)
                self.consecutive_losses = data.get("consecutive_losses", 0)
                self.day_start_capital = data.get("day_start_capital", self.current_capital)
                self.wins_today = data.get("wins_today", 0)
                self.losses_today = data.get("losses_today", 0)
                self._last_start_date = data.get("last_start_date", "")
                self._last_weekly_reset = data.get("last_weekly_reset", "")
                self.week_start_capital = data.get("week_start_capital", self.current_capital)
                logger.info("Loaded capital: Rs {:.0f} (peak: Rs {:.0f})",
                           self.current_capital, self.peak_capital)
            except Exception as e:
                logger.error("Capital load error: {}", e)

    def save(self):
        if self.trades_today != self.wins_today + self.losses_today:
            logger.critical(
                "Counter invariant broken: trades_today={} != wins={} + losses={}",
                self.trades_today, self.wins_today, self.losses_today,
            )
            self.trades_today = self.wins_today + self.losses_today

        data = {
            "current_capital": round(self.current_capital, 2),
            "total_pnl": round(self.total_pnl, 2),
            "peak_capital": round(self.peak_capital, 2),
            "weekly_pnl": round(self.weekly_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "day_start_capital": round(self.day_start_capital, 2),
            "wins_today": self.wins_today,
            "losses_today": self.losses_today,
            "last_updated": datetime.now(IST).isoformat(),
            "last_start_date": getattr(self, '_last_start_date', ''),
            "last_weekly_reset": getattr(self, '_last_weekly_reset', ''),
            "week_start_capital": round(getattr(self, 'week_start_capital', self.current_capital), 2),
        }
        try:
            tmp = settings.CAPITAL_FILE.with_suffix('.tmp')
            lock_path = settings.CAPITAL_FILE.with_suffix('.lock')
            with open(lock_path, 'w') as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    with open(tmp, "w") as f:
                        json.dump(data, f, indent=2)
                    os.replace(str(tmp), str(settings.CAPITAL_FILE))
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception as e:
            logger.error("Capital save error: {}", e)

    def start_day(self):
        today = datetime.now(IST).date().isoformat()

        if getattr(self, '_last_start_date', '') == today:
            logger.info("Day already started ({}), skipping reset. Cap=Rs {:.0f}", today, self.current_capital)
            return

        # Backup capital file before resetting daily state
        if settings.CAPITAL_FILE.exists():
            bak = settings.CAPITAL_FILE.with_suffix('.bak')
            try:
                shutil.copy2(str(settings.CAPITAL_FILE), str(bak))
            except Exception as e:
                logger.warning("Capital backup failed: {}", e)

        # Auto-reset weekly PnL on Monday (once per week)
        if datetime.now(IST).weekday() == 0:
            if getattr(self, '_last_weekly_reset', '') != today:
                self.reset_weekly()
                self._last_weekly_reset = today

        self._last_start_date = today
        self.day_start_capital = self.current_capital
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.consecutive_losses = 0
        self._trade_log = []
        self.save()
        logger.info("Day started: Rs {:.0f}", self.current_capital)

    def record_trade(self, pnl: float, strategy: str, symbol: str,
                     entry_price: float, exit_price: float,
                     quantity: int, reason: str = ""):
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.weekly_pnl += pnl
        self.current_capital += pnl
        self.trades_today += 1

        if pnl >= 0:
            self.wins_today += 1
            self.consecutive_losses = 0
        else:
            self.losses_today += 1
            self.consecutive_losses += 1

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        dd = (self.peak_capital - self.current_capital) / self.peak_capital * 100 \
            if self.peak_capital > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        self._trade_log.append({
            "timestamp": datetime.now(IST).isoformat(),
            "strategy": strategy, "symbol": symbol,
            "entry": entry_price, "exit": exit_price,
            "quantity": quantity, "pnl": round(pnl, 2),
            "reason": reason, "capital_after": round(self.current_capital, 2),
        })
        self.save()
        logger.info("Trade: {} {} PnL=Rs {:.0f} Cap=Rs {:.0f}",
                     strategy, "W" if pnl >= 0 else "L", pnl, self.current_capital)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_capital <= 0:
            return 0
        return (self.peak_capital - self.current_capital) / self.peak_capital * 100

    def get_sizing_multiplier(self) -> float:
        """Drawdown-based position sizing. Returns 0.0-1.0."""
        dd = self.drawdown_pct
        if dd >= settings.DRAWDOWN_HALT_PCT:
            return 0.0
        if dd >= settings.DRAWDOWN_HALFSIZE_PCT:
            return 0.5
        return 1.0

    def get_max_lots(self, underlying: str = "NIFTY",
                     premium_per_unit: float = 100.0) -> int:
        """Compute lots from capital -- this is how compounding works.

        1 lot per CAPITAL_PER_LOT of current capital, capped at MAX_LOTS_CAP.
        Drawdown multiplier can reduce this to 0 (halt).
        """
        sizing_mult = self.get_sizing_multiplier()
        if sizing_mult <= 0:
            return 0

        deployable = self.current_capital * (settings.CAPITAL_DEPLOY_PCT / 100)
        per_lot = getattr(settings, 'CAPITAL_PER_LOT', 10_000)
        cap = getattr(settings, 'MAX_LOTS_CAP', 10)
        lots = max(1, int(deployable / per_lot))
        lots = int(lots * sizing_mult)
        return max(1, min(lots, cap))

    def get_daily_loss_limit(self) -> float:
        return self.day_start_capital * (settings.DAILY_LOSS_LIMIT_PCT / 100)

    def get_weekly_loss_limit(self) -> float:
        base = getattr(self, 'week_start_capital', self.current_capital)
        return base * (settings.WEEKLY_LOSS_LIMIT_PCT / 100)

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_start_capital <= 0:
            return 0
        return (self.daily_pnl / self.day_start_capital) * 100

    def get_summary(self) -> dict:
        wr = self.wins_today / self.trades_today * 100 if self.trades_today > 0 else 0
        return {
            "capital": self.current_capital,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "trades": self.trades_today,
            "wins": self.wins_today,
            "losses": self.losses_today,
            "win_rate": wr,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown,
            "drawdown_current": self.drawdown_pct,
            "consecutive_losses": self.consecutive_losses,
            "peak_capital": self.peak_capital,
            "trade_log": self._trade_log,
        }

    def reset_weekly(self):
        self.weekly_pnl = 0.0
        self.week_start_capital = self.current_capital
        self.save()
