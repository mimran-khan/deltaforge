"""Track capital, P&L, and compute lot sizing based on equity tiers."""

from __future__ import annotations
import json
from datetime import date, datetime
from typing import Optional

from loguru import logger

from config import settings


class CapitalTracker:
    """Persistent equity tracker with compound lot sizing."""

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
                logger.info("Loaded capital: Rs {:.0f}", self.current_capital)
            except Exception as e:
                logger.error("Capital load error: {}", e)

    def save(self):
        data = {
            "current_capital": self.current_capital,
            "total_pnl": self.total_pnl,
            "peak_capital": self.peak_capital,
            "weekly_pnl": self.weekly_pnl,
            "last_updated": datetime.now().isoformat(),
        }
        try:
            with open(settings.CAPITAL_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Capital save error: {}", e)

    def start_day(self):
        self.day_start_capital = self.current_capital
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.consecutive_losses = 0
        self._trade_log = []
        logger.info("Day started with capital: Rs {:.0f}", self.current_capital)

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

        dd = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "symbol": symbol,
            "entry": entry_price,
            "exit": exit_price,
            "quantity": quantity,
            "pnl": pnl,
            "reason": reason,
            "capital_after": self.current_capital,
        }
        self._trade_log.append(trade_record)
        self.save()

        logger.info("Trade recorded: {} {} PnL=Rs {:.0f} Capital=Rs {:.0f}",
                     strategy, "WIN" if pnl >= 0 else "LOSS", pnl, self.current_capital)

    def get_max_lots(self, underlying: str = "NIFTY") -> int:
        """Determine lot count based on current capital tier."""
        for min_cap, max_lots, inst in reversed(settings.LOT_TIERS):
            if self.current_capital >= min_cap and inst == underlying:
                return max_lots
        return 1

    def get_daily_loss_limit(self) -> float:
        return self.day_start_capital * (settings.DAILY_LOSS_LIMIT_PCT / 100)

    def get_weekly_loss_limit(self) -> float:
        return self.starting_capital * (settings.WEEKLY_LOSS_LIMIT_PCT / 100)

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_start_capital <= 0:
            return 0
        return (self.daily_pnl / self.day_start_capital) * 100

    def get_summary(self) -> dict:
        win_rate = (self.wins_today / self.trades_today * 100
                    if self.trades_today > 0 else 0)
        return {
            "capital": self.current_capital,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "trades": self.trades_today,
            "wins": self.wins_today,
            "losses": self.losses_today,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown,
            "consecutive_losses": self.consecutive_losses,
            "trade_log": self._trade_log,
        }

    def reset_weekly(self):
        self.weekly_pnl = 0.0
        self.save()
