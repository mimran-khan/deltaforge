"""Daily autonomous scheduler -- runs the full trading day lifecycle."""

from __future__ import annotations
import atexit
import json
import random
import signal as _signal
import sys
import threading
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
from alerts.command_listener import CommandPoller
from risk.kill_switch import (
    set_halt,
    acquire_session_lock, release_session_lock, is_session_running,
)

_alert_method = getattr(settings, 'ALERT_METHOD', 'slack')
if _alert_method == 'imessage':
    from alerts.imessage_bot import send_system_alert
elif _alert_method == 'slack':
    from alerts.slack_bot import send_system_alert
else:
    from alerts.telegram_bot import send_system_alert

IST = pytz.timezone("Asia/Kolkata")


def _load_holidays() -> set[str]:
    """Load NSE holiday dates from config/holidays.json."""
    holidays_file = settings.BASE_DIR / "config" / "holidays.json"
    if not holidays_file.exists():
        return set()
    try:
        with open(holidays_file) as f:
            data = json.load(f)
        return {h["date"] for h in data.get("holidays", [])}
    except Exception:
        return set()


_NSE_HOLIDAYS = _load_holidays()


def is_trading_day() -> bool:
    """Check if today is a weekday and not an NSE holiday."""
    today = datetime.now(IST)
    if today.weekday() >= 5:
        return False
    today_str = today.strftime("%Y-%m-%d")
    if today_str in _NSE_HOLIDAYS:
        logger.info("NSE holiday: {} -- skipping Nifty", today_str)
        return False
    return True


def is_mcx_evening_day() -> bool:
    """Check if MCX evening session is available today.

    On NSE holidays that fall on weekdays, MCX morning is closed
    but the evening session (17:00-23:30) still runs.
    """
    today = datetime.now(IST)
    if today.weekday() >= 5:
        return False
    today_str = today.strftime("%Y-%m-%d")
    if today_str in _NSE_HOLIDAYS:
        return True
    return False


class DailyScheduler:
    """Orchestrates the full autonomous trading day."""

    def __init__(self, enable_commands: bool = True):
        self.broker = BrokerConnection()
        self.engine: TradingEngine | None = None
        self._multi_engine = None
        self._multi_thread = None
        self.scheduler = BackgroundScheduler(timezone=IST)
        self._watchdog_proc = None
        self._dashboard_proc = None
        self._command_poller: CommandPoller | None = None
        self._enable_commands = enable_commands

    def run(self):
        """Entry point -- sets up scheduled jobs and blocks."""
        self._mcx_only_mode = False

        if not is_trading_day():
            if is_mcx_evening_day() and settings.MULTI_ASSET_ENABLED:
                logger.info("NSE holiday but MCX evening session available -- MCX-only mode")
                self._mcx_only_mode = True
            else:
                logger.info("Not a trading day. Exiting.")
                return

        running, pid = is_session_running()
        if running:
            logger.critical("Another trading session is already running (PID {}). Exiting.", pid)
            send_system_alert("SESSION BLOCKED",
                              f"Cannot start: session PID {pid} already running.")
            return

        if not acquire_session_lock():
            logger.critical("Failed to acquire session lock. Exiting.")
            return

        atexit.register(release_session_lock)

        def _sig_handler(signum, frame):
            logger.info("Signal {} received -- shutting down", signum)
            self._cleanup()
            sys.exit(0)

        _signal.signal(_signal.SIGTERM, _sig_handler)
        _signal.signal(_signal.SIGINT, _sig_handler)

        logger.info("=" * 60)
        logger.info("DeltaForge -- DAILY SCHEDULER STARTED")
        logger.info("Date: {}", datetime.now(IST).strftime("%Y-%m-%d %A"))
        logger.info("Mode: {} {}", settings.TRADING_MODE.upper(),
                     "(MCX-ONLY)" if self._mcx_only_mode else "")
        logger.info("=" * 60)

        # Clear previous-day HALT unconditionally — risk engine will
        # re-halt during the session if drawdown / limits still breached.
        halt_file = settings.DATA_DIR / "HALT"
        if halt_file.exists():
            logger.info("New trading day — clearing previous HALT flag")
            halt_file.unlink(missing_ok=True)

        self._schedule_jobs()
        self.scheduler.start()

        if self._mcx_only_mode:
            self._login_and_prepare_mcx_only()
        else:
            self._login_and_prepare()

        try:
            if not self._mcx_only_mode:
                self._wait_for_market_and_trade()
            self._idle_until_logout()
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

        # Refresh broker session every 4 hours (tokens expire after ~6 hrs)
        self.scheduler.add_job(
            self._refresh_session,
            "interval", hours=4,
            id="session_refresh", replace_existing=True,
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

        logger.info("Waiting 15s after instrument download to avoid API rate-limit...")
        time.sleep(15)
        time.sleep(random.uniform(0, 5))

        # Crash recovery: check for orphaned positions from previous session
        self._check_orphaned_positions()

        self.engine = TradingEngine()
        self.engine.broker = self.broker
        self.engine.start_day()

        if self.engine._skip_today:
            logger.info("Engine skipped today -- exiting scheduler")
            self.engine = None
            return

        seeded = self.engine.seed_historical_candles()
        if seeded > 0:
            logger.info("HTF indicators pre-warmed with {} bars", seeded)
        else:
            logger.warning("No historical candles seeded -- HTF RSI will need warmup time")

        caught = self.engine.catch_up_missed_bars()
        if caught > 0:
            if getattr(settings, 'LATE_START_CATCHUP_DRY_RUN', True):
                logger.info("Late-start catch-up scanned {} morning bars (dry-run)", caught)
            else:
                logger.info("Late-start catch-up replayed {} morning bars", caught)

        if self._enable_commands:
            self._command_poller = CommandPoller(engine=self.engine)
            self._command_poller.start()

        self._start_watchdog()
        self._ensure_dashboard()
        self._start_multi_asset_engine()

    def _login_and_prepare_mcx_only(self):
        """Lightweight startup for MCX evening-only sessions (NSE holidays).

        Skips Nifty engine entirely. Logs in, downloads instruments,
        waits until 16:45, then starts the multi-asset engine.
        """
        logger.info("MCX-only mode: logging in...")
        max_retries = 3
        for attempt in range(max_retries):
            if self.broker.login():
                logger.info("Login successful (attempt {})", attempt + 1)
                break
            logger.warning("Login attempt {} failed, retrying in 30s...", attempt + 1)
            time.sleep(30)
        else:
            logger.critical("All login attempts failed. Cannot trade today.")
            send_system_alert("LOGIN FAILED",
                              "All login attempts failed (MCX-only day).")
            return

        logger.info("Downloading instruments...")
        self.broker.download_instruments()

        now = datetime.now(IST)
        mcx_pre_open = now.replace(hour=16, minute=45, second=0, microsecond=0)
        if now < mcx_pre_open:
            wait_secs = (mcx_pre_open - now).total_seconds()
            logger.info("MCX evening pre-open at 16:45 -- waiting {:.0f} minutes",
                        wait_secs / 60)
            send_system_alert("MCX-ONLY DAY",
                              f"NSE holiday. MCX evening session starts at 17:00. "
                              f"Engine will activate at 16:45 ({wait_secs/60:.0f}m from now).")
            time.sleep(wait_secs)

        logger.info("Starting multi-asset engine for MCX evening session")
        self._start_multi_asset_engine()

        if self._enable_commands:
            self._command_poller = CommandPoller(engine=None)
            self._command_poller.start()

        self._start_watchdog()
        self._ensure_dashboard()

    def _wait_for_market_and_trade(self):
        """Wait until market open, then run the trading loop."""
        if not self.engine:
            logger.info("No engine -- nothing to trade today")
            return

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

    def _start_multi_asset_engine(self):
        """Start the multi-asset futures engine in a background thread.

        Only activates when MULTI_ASSET_ENABLED=true.
        Runs completely independently of the Nifty TradingEngine.
        Thread handles its own startup delay to avoid blocking the main thread.
        """

        if not settings.MULTI_ASSET_ENABLED:
            return

        try:
            from engine.multi_asset_engine import MultiAssetEngine
        except ImportError as e:
            logger.warning("Multi-asset engine not available: {}", e)
            return

        logger.info("Multi-Asset Engine will start in background (75s delay for rate-limit avoidance)")

        def _multi_asset_thread_target():
            try:
                time.sleep(75)
                logger.info("Starting Multi-Asset Engine (MULTI_ASSET_ENABLED=true)")
                self._multi_engine = MultiAssetEngine(broker=self.broker)
                self._multi_engine.start_day()
                instruments = list(self._multi_engine.instrument_names)
                send_system_alert("MULTI-ASSET STARTED",
                                  f"Engine running: {', '.join(instruments)}")
                self._multi_engine.run_loop(poll_interval=5)
                logger.info("Multi-Asset Engine run_loop exited normally")
            except Exception as e:
                logger.critical("Multi-Asset Engine CRASHED: {}", e)
                send_system_alert("MULTI-ASSET CRASHED",
                                  f"Engine thread died: {e}")

        self._multi_thread = threading.Thread(
            target=_multi_asset_thread_target,
            daemon=False,
            name="multi-asset-engine",
        )
        self._multi_thread.start()

    def _idle_until_logout(self):
        """Keep process alive so APScheduler EOD/logout jobs can fire.

        When MULTI_ASSET_ENABLED, extends idle period to cover MCX close (23:35).
        """
        logout_time = settings.SESSION_LOGOUT_TIME
        if settings.MULTI_ASSET_ENABLED:
            from config.instruments import get_latest_eod_time
            latest_eod = get_latest_eod_time()
            if latest_eod > logout_time:
                logout_time = latest_eod
                logger.info("Extended idle until {} for multi-asset EOD", logout_time)

        logout_parts = logout_time.split(":")
        logout_h, logout_m = int(logout_parts[0]), int(logout_parts[1])
        while True:
            now = datetime.now(IST)
            if now.hour > logout_h or (now.hour == logout_h and now.minute >= logout_m):
                logger.info("Reached logout time ({}) -- shutting down", logout_time)
                break
            time.sleep(60)

    def _square_off(self):
        if not self.engine:
            return
        if self.engine.position_mgr and self.engine.position_mgr.has_open_positions:
            logger.info("Scheduled square-off triggered (live)")
            self.engine.position_mgr.square_off_all("SCHEDULED_SQUARE_OFF")
        if self.engine._paper_positions:
            logger.info("Scheduled square-off triggered (paper)")
            self.engine._square_off_all("SCHEDULED_SQUARE_OFF")

    def _eod_report(self):
        if self.engine:
            self._archive_candles()
            self.engine.end_day()
        self._generate_static_report()

    def _generate_static_report(self):
        """Regenerate the static HTML trade report in docs/index.html."""
        try:
            from reports.generate import generate
            path = generate()
            logger.info("Static report updated: {}", path)
        except Exception as e:
            logger.warning("Static report generation failed (non-fatal): {}", e)

    def _archive_candles(self):
        """Append today's candles to nifty_5m_combined.csv for future backtesting."""
        try:
            live_csv = settings.DATA_DIR / "candles_live.csv"
            combined_csv = settings.DATA_DIR / "nifty_5m_combined.csv"

            if not live_csv.exists():
                logger.info("No candles_live.csv to archive")
                return

            import pandas as pd
            live_df = pd.read_csv(live_csv, index_col="timestamp", parse_dates=True)
            if live_df.empty:
                return

            today = datetime.now(IST).date()
            today_candles = live_df[live_df.index.date == today]
            if today_candles.empty:
                logger.info("No today's candles to archive")
                return

            if combined_csv.exists():
                combined = pd.read_csv(combined_csv, index_col=0, parse_dates=True)
                combined = pd.concat([combined, today_candles])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
            else:
                combined = today_candles

            combined.to_csv(combined_csv, index_label="timestamp")
            logger.info("Archived {} candles to nifty_5m_combined.csv (total: {})",
                        len(today_candles), len(combined))
        except Exception as e:
            logger.warning("Candle archiving failed: {}", e)

    def _refresh_session(self):
        """Refresh broker session to prevent token expiry mid-day.

        Also restarts the WebSocket feed so it uses the new auth/feed tokens
        instead of the stale ones from the original login.
        """
        logger.info("Refreshing broker session...")
        try:
            self.broker.logout()
            if self.broker.login():
                logger.info("Session refreshed successfully")
                if self.engine:
                    self.engine.broker = self.broker
                    self.engine.restart_market_feed()
            else:
                logger.critical("Session refresh FAILED -- halting trading")
                set_halt("Broker session refresh failed")
                send_system_alert("SESSION REFRESH FAILED", "Trading halted. Manual re-login required.")
        except Exception as e:
            logger.error("Session refresh error: {}", e)

    def _logout(self):
        logger.info("Scheduled session logout")
        if self._multi_engine:
            self._multi_engine.stop()
            if self._multi_thread and self._multi_thread.is_alive():
                self._multi_thread.join(timeout=15)
        if self.engine:
            self.engine.stop()
        self.broker.logout()
        send_system_alert("Session Ended", "Daily session logged out.")

    def _check_orphaned_positions(self):
        """Check for open positions from a previous crashed session."""
        try:
            positions = self.broker.get_positions()
            open_nfo = [
                p for p in positions
                if p.get("exchange") == "NFO"
                and int(p.get("netqty", 0)) != 0
            ]
            if open_nfo:
                symbols = ", ".join(p.get("tradingsymbol", "?") for p in open_nfo)
                msg = (f"{len(open_nfo)} open NFO position(s) from previous session.\n"
                       f"Symbols: {symbols}\n"
                       f"Manual check required.")
                logger.critical("ORPHANED POSITIONS: {}", msg)
                send_system_alert("ORPHANED POSITION DETECTED", msg)
        except Exception as e:
            logger.debug("Orphaned position check skipped: {}", e)

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

    def _ensure_dashboard(self):
        """Start dashboard API/UI on :8900 if not already listening."""
        import socket
        import urllib.error
        import urllib.request

        def _dashboard_project_root() -> str | None:
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8900/api/status", timeout=2
                ) as resp:
                    payload = json.loads(resp.read().decode())
                return payload.get("project_root")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                return None

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if sock.connect_ex(("127.0.0.1", 8900)) == 0:
                expected = str(PROJECT_ROOT.resolve())
                actual = _dashboard_project_root()
                if actual and actual != expected:
                    logger.warning(
                        "Dashboard on :8900 belongs to {} -- restarting for {}",
                        actual, expected,
                    )
                    try:
                        subprocess.run(
                            "lsof -ti :8900 | xargs kill",
                            shell=True,
                            check=False,
                        )
                    except (subprocess.SubprocessError, FileNotFoundError):
                        pass
                    time.sleep(1)
                else:
                    logger.info("Dashboard already running on http://localhost:8900")
                    return
        finally:
            sock.close()

        try:
            venv_python = PROJECT_ROOT / "venv" / "bin" / "python3"
            log_file = PROJECT_ROOT / "logs" / "dashboard.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as log:
                self._dashboard_proc = subprocess.Popen(
                    [str(venv_python), "-m", "dashboard.server"],
                    cwd=str(PROJECT_ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            logger.info(
                "Dashboard started (PID {}) at http://localhost:8900",
                self._dashboard_proc.pid,
            )
        except Exception as e:
            logger.error("Dashboard start failed: {}", e)

    def _cleanup(self):
        if self._multi_engine:
            logger.info("Stopping multi-asset engine...")
            self._multi_engine.stop()
        if self._multi_thread and self._multi_thread.is_alive():
            self._multi_thread.join(timeout=15)
            if self._multi_thread.is_alive():
                logger.warning("Multi-asset thread did not exit within 15s -- proceeding with shutdown")
            else:
                logger.info("Multi-asset engine stopped")
        elif self._multi_engine:
            logger.info("Multi-asset engine stopped")
        if self._command_poller:
            self._command_poller.stop()
        if self._watchdog_proc:
            self._watchdog_proc.terminate()
            logger.info("Watchdog terminated")
        try:
            if self.broker and self.broker.is_active:
                self.broker.logout()
                logger.info("Broker session logged out")
        except Exception as e:
            logger.warning("Broker logout error: {}", e)
        self.scheduler.shutdown(wait=False)
        release_session_lock()
        logger.info("Scheduler shutdown complete")


def main():
    from config.logging import setup_logging
    setup_logging()
    scheduler = DailyScheduler()
    scheduler.run()


if __name__ == "__main__":
    main()
