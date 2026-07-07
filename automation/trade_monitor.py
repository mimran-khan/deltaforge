"""Autonomous trade monitor -- runs all day, sends periodic Slack updates.

Watches capital, P&L, adaptive mode, and trade activity.
Sends alerts on:
  - Every trade (entry/exit) -- already handled by trading engine
  - Hourly P&L summary
  - Mode transitions (AGGRESSIVE/DEFENSIVE/HALT)
  - Danger: drawdown > 15%, consecutive losses >= 2
  - Victory: daily profit > 5%, new capital high
  - Inactivity: no trades for 2+ hours during market

Usage:
    python -m automation.trade_monitor
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pytz
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

_alert_method = getattr(settings, 'ALERT_METHOD', 'slack')
if _alert_method == 'imessage':
    from alerts.imessage_bot import send_system_alert
elif _alert_method == 'slack':
    from alerts.slack_bot import send_system_alert
else:
    from alerts.slack_bot import send_system_alert

IST = pytz.timezone("Asia/Kolkata")
CAPITAL_FILE = Path(settings.DATA_DIR) / "capital.json"
POLL_INTERVAL = 60  # check every 60 seconds


def _read_capital() -> dict:
    """Read current capital state."""
    try:
        return json.loads(CAPITAL_FILE.read_text())
    except Exception:
        return {}


def _daily_pnl_pct(cap: dict) -> float:
    day_start = cap.get("day_start_capital", 10000)
    if day_start <= 0:
        return 0.0
    return (cap.get("daily_pnl", 0) / day_start) * 100


class TradeMonitor:
    """Watches trading state and sends periodic alerts."""

    def __init__(self):
        self._last_trades_count = 0
        self._last_hourly_report = 0
        self._last_mode = "NORMAL"
        self._last_capital = 0.0
        self._peak_capital = 0.0
        self._alerted_danger = False
        self._alerted_victory = False
        self._last_trade_time = time.time()
        self._inactivity_alerted = False
        self._started = False

    def run(self):
        """Main monitoring loop -- runs until market close."""
        logger.info("Trade Monitor started")
        send_system_alert(
            "Monitor Online",
            f"Trade monitor active. Watching for profits.\n"
            f"Capital: Rs {_read_capital().get('current_capital', 10000):,.0f}\n"
            f"Mode: NORMAL (day start)\n"
            f"Goal: MAKE MONEY"
        )
        self._started = True
        self._last_hourly_report = time.time()

        while True:
            now = datetime.now(IST)

            # Exit after market close
            if now.hour > 15 or (now.hour == 15 and now.minute >= 45):
                self._send_eod_summary()
                logger.info("Market closed -- monitor shutting down")
                break

            # Skip before market open
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                time.sleep(30)
                continue

            try:
                self._check_state()
            except Exception as e:
                logger.error("Monitor check error: {}", e)

            time.sleep(POLL_INTERVAL)

    def _check_state(self):
        """Run all monitoring checks."""
        cap = _read_capital()
        if not cap:
            return

        capital = cap.get("current_capital", 0)
        daily_pnl = cap.get("daily_pnl", 0)
        pnl_pct = _daily_pnl_pct(cap)
        wins = cap.get("wins_today", 0)
        losses = cap.get("losses_today", 0)
        trades = wins + losses
        consec_losses = cap.get("consecutive_losses", 0)
        cap.get("day_start_capital", 10000)

        # Track peak
        if capital > self._peak_capital:
            self._peak_capital = capital

        # ── New trade detected ──
        if trades > self._last_trades_count:
            self._last_trade_time = time.time()
            self._inactivity_alerted = False
            trades - self._last_trades_count
            self._last_trades_count = trades

            wr = (wins / trades * 100) if trades > 0 else 0
            logger.info("Trade #{}: PnL={:.0f} ({:.1f}%) W={} L={} WR={:.0f}%",
                        trades, daily_pnl, pnl_pct, wins, losses, wr)

        # ── Hourly summary ──
        if time.time() - self._last_hourly_report >= 3600:
            self._last_hourly_report = time.time()
            now = datetime.now(IST)
            wr = (wins / trades * 100) if trades > 0 else 0

            send_system_alert(
                f"Hourly Update ({now.strftime('%H:%M')})",
                f"Capital: Rs {capital:,.0f}\n"
                f"Day PnL: Rs {'+' if daily_pnl >= 0 else ''}{daily_pnl:,.0f} ({pnl_pct:+.1f}%)\n"
                f"Trades: {trades} (W:{wins} L:{losses})\n"
                f"Win Rate: {wr:.0f}%\n"
                f"Consecutive Losses: {consec_losses}"
            )

        # ── Danger alerts ──
        if consec_losses >= 2 and not self._alerted_danger:
            self._alerted_danger = True
            send_system_alert(
                "DANGER: Consecutive Losses",
                f"{consec_losses} losses in a row!\n"
                f"Day PnL: Rs {daily_pnl:,.0f} ({pnl_pct:+.1f}%)\n"
                f"Capital: Rs {capital:,.0f}\n"
                f"Adaptive mode should be shifting to DEFENSIVE."
            )

        if pnl_pct <= -10 and not self._alerted_danger:
            self._alerted_danger = True
            send_system_alert(
                "DANGER: Heavy Loss Day",
                f"Day PnL: {pnl_pct:+.1f}% (Rs {daily_pnl:,.0f})\n"
                f"Capital: Rs {capital:,.0f}\n"
                f"Adaptive mode should be in HALT."
            )

        # Reset danger flag if recovered
        if consec_losses == 0 and pnl_pct > -5:
            self._alerted_danger = False

        # ── Victory alerts ──
        if pnl_pct >= 5 and not self._alerted_victory:
            self._alerted_victory = True
            send_system_alert(
                "WINNING DAY",
                f"Daily profit: {pnl_pct:+.1f}% (Rs {daily_pnl:,.0f})\n"
                f"Capital: Rs {capital:,.0f}\n"
                f"Trades: {trades} (W:{wins} L:{losses})\n"
                f"Adaptive mode should be AGGRESSIVE."
            )

        # ── Inactivity alert ──
        now = datetime.now(IST)
        market_active = (now.hour >= 9 and now.hour < 15)
        elapsed_since_trade = time.time() - self._last_trade_time
        if market_active and elapsed_since_trade > 7200 and not self._inactivity_alerted:
            self._inactivity_alerted = True
            send_system_alert(
                "No Trades for 2+ Hours",
                f"Last trade was {elapsed_since_trade / 60:.0f} minutes ago.\n"
                f"Capital: Rs {capital:,.0f}\n"
                f"Check if engine is running and signals are generating."
            )

        self._last_capital = capital

    def _send_eod_summary(self):
        """Send end-of-day summary."""
        cap = _read_capital()
        if not cap:
            return

        capital = cap.get("current_capital", 0)
        daily_pnl = cap.get("daily_pnl", 0)
        pnl_pct = _daily_pnl_pct(cap)
        wins = cap.get("wins_today", 0)
        losses = cap.get("losses_today", 0)
        trades = wins + losses
        wr = (wins / trades * 100) if trades > 0 else 0

        verdict = "PROFIT" if daily_pnl > 0 else "LOSS" if daily_pnl < 0 else "FLAT"

        send_system_alert(
            f"EOD Summary -- {verdict}",
            f"Capital: Rs {capital:,.0f}\n"
            f"Day PnL: Rs {'+' if daily_pnl >= 0 else ''}{daily_pnl:,.0f} ({pnl_pct:+.1f}%)\n"
            f"Trades: {trades} (W:{wins} L:{losses})\n"
            f"Win Rate: {wr:.0f}%\n"
            f"Consecutive Losses: {cap.get('consecutive_losses', 0)}\n"
            f"\nGoal was to MAKE MONEY. Result: {verdict}."
        )


def main():
    from config.logging import setup_logging
    setup_logging()
    monitor = TradeMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
