"""Daily autonomous scheduler -- runs the full trading day lifecycle."""

from __future__ import annotations
import sys
import time
import subprocess
from datetime import datetime, date
from pathlib import Path

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.broker import BrokerConnection
from engine.trading_engine import TradingEngine
from risk.kill_switch import clear_halt, is_halted
from alerts.telegram_bot import send_system_alert

IST = pytz.timezone("Asia/Kolkata")

LOG_FILE = settings.LOG_DIR / f"trading_{date.today().isoformat()}.log"
logger.add(LOG_FILE, rotation="1 day", retention="30 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def is_trading_day() -> bool:
    """Check if today is a weekday (basic check -- holidays not included)."""
    today = datetime.now(IST)
    return today.weekday() < 5  # Mon=0 ... Fri=4


class DailyScheduler:
    """Orchestrates the full autonomous trading day."""

    def __init__(self):
        self.broker = BrokerConnection()
        self.engine: TradingEngine | None = None
        self.scheduler = BackgroundScheduler(timezone=IST)
        self._watchdog_proc = None

    def run(self):
        """Entry point -- sets up scheduled jobs and blocks."""
        if not is_trading_day():
            logger.info("Not a trading day. Exiting.")
            return

        logger.info("=" * 60)
        logger.info("DAILY SCHEDULER STARTED")
        logger.info("Date: {}", datetime.now(IST).strftime("%Y-%m-%d %A"))
        logger.info("Mode: {}", settings.TRADING_MODE.upper())
        logger.info("=" * 60)

        clear_halt()

        self._schedule_jobs()
        self.scheduler.start()

        self._login_and_prepare()

        try:
            self._wait_for_market_and_trade()
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted")
        finally:
            self._cleanup()

    def _schedule_jobs(self):
        h, m = settings.SQUARE_OFF_TIME.split(":")
        self.scheduler.add_job(
            self._square_off,
            "cron", hour=int(h), minute=int(m),
            id="square_off", replace_existing=True,
        )

        h, m = settings.EOD_REPORT_TIME.split(":")
        self.scheduler.add_job(
            self._eod_report,
            "cron", hour=int(h), minute=int(m),
            id="eod_report", replace_existing=True,
        )

        h, m = settings.SESSION_LOGOUT_TIME.split(":")
        self.scheduler.add_job(
            self._logout,
            "cron", hour=int(h), minute=int(m),
            id="logout", replace_existing=True,
        )

    def _login_and_prepare(self):
        logger.info("Logging into Angel One...")
        max_retries = 3
        for attempt in range(max_retries):
            if self.broker.login():
                logger.info("Login successful (attempt {})", attempt + 1)
                break
            logger.warning("Login attempt {} failed, retrying in 30s...", attempt + 1)
            time.sleep(30)
        else:
            logger.critical("All login attempts failed. Cannot trade today.")
            send_system_alert("LOGIN FAILED", "All login attempts failed. Manual intervention needed.")
            return

        logger.info("Downloading instruments...")
        instruments = self.broker.download_instruments()
        if not instruments:
            logger.error("Instrument download failed")

        self.engine = TradingEngine()
        self.engine.broker = self.broker
        self.engine.start_day()

        self._start_watchdog()

    def _wait_for_market_and_trade(self):
        """Wait until market open, then run the trading loop."""
        now = datetime.now(IST)
        open_parts = settings.MARKET_OPEN.split(":")
        open_hour, open_min = int(open_parts[0]), int(open_parts[1])

        if now.hour < open_hour or (now.hour == open_hour and now.minute < open_min):
            wait_until = now.replace(hour=open_hour, minute=open_min, second=0)
            wait_secs = (wait_until - now).total_seconds()
            logger.info("Waiting {:.0f}s for market open at {}", wait_secs, settings.MARKET_OPEN)
            time.sleep(max(0, wait_secs))

        if self.engine and self.broker.is_active:
            logger.info("Market open -- starting trading loop")
            self.engine.run_loop(poll_interval=5)
        else:
            logger.error("Engine or broker not ready")

    def _square_off(self):
        if self.engine and self.engine.position_mgr.has_open_positions:
            logger.info("Scheduled square-off triggered")
            self.engine.position_mgr.square_off_all("SCHEDULED_SQUARE_OFF")

    def _eod_report(self):
        if self.engine:
            self.engine.end_day()

    def _logout(self):
        logger.info("Scheduled session logout")
        self.broker.logout()
        send_system_alert("Session Ended", "Daily session logged out (SEBI compliance)")

    def _start_watchdog(self):
        """Start kill switch as a separate process."""
        try:
            venv_python = PROJECT_ROOT / "venv" / "bin" / "python3"
            self._watchdog_proc = subprocess.Popen(
                [str(venv_python), "-m", "risk.kill_switch"],
                cwd=str(PROJECT_ROOT),
            )
            logger.info("Watchdog started (PID {})", self._watchdog_proc.pid)
        except Exception as e:
            logger.error("Watchdog start failed: {}", e)

    def _cleanup(self):
        if self._watchdog_proc:
            self._watchdog_proc.terminate()
            logger.info("Watchdog terminated")
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler shutdown complete")


def main():
    scheduler = DailyScheduler()
    scheduler.run()


if __name__ == "__main__":
    main()
