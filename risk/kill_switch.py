"""Independent watchdog process -- monitors P&L and kills trading if limits breached."""

from __future__ import annotations
import json
import os
import time
import signal as sig
import sys
from datetime import datetime
from pathlib import Path

import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from risk.capital_tracker import CapitalTracker


HALT_FLAG = settings.DATA_DIR / "HALT"
SESSION_LOCK = settings.DATA_DIR / "session.lock"
POLLER_LOCK = settings.DATA_DIR / "poller.lock"


def is_halted() -> bool:
    return HALT_FLAG.exists()


def set_halt(reason: str):
    HALT_FLAG.write_text(json.dumps({
        "halted_at": datetime.now(IST).isoformat(),
        "reason": reason,
    }))
    logger.critical("HALT FLAG SET: {}", reason)


def clear_halt():
    if HALT_FLAG.exists():
        HALT_FLAG.unlink()
        logger.info("Halt flag cleared")


# ── Session lock (prevents duplicate trading sessions) ──────────


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def is_session_running() -> tuple[bool, int]:
    """Check if a trading session is already running.

    Returns (running, pid). Cleans stale locks when:
    - PID is dead/zombie
    - Lock is older than 6 hours (hung process)
    """
    if not SESSION_LOCK.exists():
        return False, 0
    try:
        data = json.loads(SESSION_LOCK.read_text())
        pid = data.get("pid", 0)
        started = data.get("started_at", "")

        if started:
            try:
                lock_time = datetime.fromisoformat(started)
                age_hours = (datetime.now(IST) - lock_time).total_seconds() / 3600
                if age_hours > 6:
                    logger.warning(
                        "Session lock is {:.1f}h old (PID {}). Force-clearing stale lock.",
                        age_hours, pid,
                    )
                    SESSION_LOCK.unlink(missing_ok=True)
                    return False, 0
            except ValueError:
                pass

        if pid and _pid_alive(pid):
            return True, pid

        logger.info("Cleaned stale session lock (PID {} dead)", pid)
        SESSION_LOCK.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Session lock read error ({}), clearing", e)
        SESSION_LOCK.unlink(missing_ok=True)
    return False, 0


def acquire_session_lock() -> bool:
    """Acquire the session lock. Returns False if another session owns it."""
    running, pid = is_session_running()
    if running:
        logger.warning("Session already running (PID {})", pid)
        return False
    try:
        SESSION_LOCK.write_text(json.dumps({
            "pid": os.getpid(),
            "started_at": datetime.now(IST).isoformat(),
        }))
        return True
    except Exception as e:
        logger.error("Failed to acquire session lock: {}", e)
        return False


def release_session_lock():
    """Release the session lock if we own it."""
    if not SESSION_LOCK.exists():
        return
    try:
        data = json.loads(SESSION_LOCK.read_text())
        if data.get("pid") == os.getpid():
            SESSION_LOCK.unlink(missing_ok=True)
            logger.info("Session lock released")
    except Exception:
        SESSION_LOCK.unlink(missing_ok=True)


# ── Poller lock (prevents duplicate command listeners) ────────


def is_poller_running() -> tuple[bool, int]:
    """Check if a command listener is already running."""
    if not POLLER_LOCK.exists():
        return False, 0
    try:
        data = json.loads(POLLER_LOCK.read_text())
        pid = data.get("pid", 0)
        if pid and _pid_alive(pid):
            return True, pid
        POLLER_LOCK.unlink(missing_ok=True)
        logger.info("Cleaned stale poller lock (PID {} dead)", pid)
    except Exception:
        POLLER_LOCK.unlink(missing_ok=True)
    return False, 0


def acquire_poller_lock() -> bool:
    """Acquire the poller lock. Returns False if another poller owns it."""
    running, pid = is_poller_running()
    if running:
        logger.warning("Command listener already running (PID {})", pid)
        return False
    try:
        POLLER_LOCK.write_text(json.dumps({
            "pid": os.getpid(),
            "started_at": datetime.now(IST).isoformat(),
        }))
        return True
    except Exception as e:
        logger.error("Failed to acquire poller lock: {}", e)
        return False


def release_poller_lock():
    """Release the poller lock if we own it."""
    if not POLLER_LOCK.exists():
        return
    try:
        data = json.loads(POLLER_LOCK.read_text())
        if data.get("pid") == os.getpid():
            POLLER_LOCK.unlink(missing_ok=True)
            logger.info("Poller lock released")
    except Exception:
        POLLER_LOCK.unlink(missing_ok=True)


def send_kill_alert(reason: str):
    """Send alert for kill switch activation via configured channel."""
    try:
        _method = getattr(settings, 'ALERT_METHOD', 'slack')
        if _method == 'imessage':
            from alerts.imessage_bot import send_alert
        elif _method == 'slack':
            from alerts.slack_bot import send_alert
        else:
            from alerts.telegram_bot import send_alert
        send_alert(f"KILL SWITCH ACTIVATED\n\nReason: {reason}\n\nAll trading halted.")
    except Exception as e:
        logger.error("Kill alert failed: {}", e)


def watchdog_loop(check_interval: int = 30):
    """Run as separate process -- monitors capital and halts if needed.

    Usage: python -m risk.kill_switch
    """
    logger.info("Kill switch watchdog started (check every {}s)", check_interval)

    tracker = CapitalTracker()
    _halt_sent = False

    def shutdown(signum, frame):
        logger.info("Watchdog shutting down")
        sys.exit(0)

    sig.signal(sig.SIGTERM, shutdown)
    sig.signal(sig.SIGINT, shutdown)

    while True:
        try:
            tracker._load()

            breached = False
            reason = ""

            daily_limit = tracker.day_start_capital * (settings.DAILY_LOSS_LIMIT_PCT / 100)
            if tracker.daily_pnl < 0 and abs(tracker.daily_pnl) >= daily_limit:
                breached = True
                reason = f"Daily loss Rs {tracker.daily_pnl:.0f} exceeds limit Rs {daily_limit:.0f}"

            weekly_limit = tracker.get_weekly_loss_limit()
            if tracker.weekly_pnl < 0 and abs(tracker.weekly_pnl) >= weekly_limit:
                breached = True
                reason = f"Weekly loss Rs {tracker.weekly_pnl:.0f} exceeds limit Rs {weekly_limit:.0f}"

            if tracker.consecutive_losses >= settings.MAX_CONSECUTIVE_LOSSES:
                breached = True
                reason = f"{tracker.consecutive_losses} consecutive losses (max {settings.MAX_CONSECUTIVE_LOSSES})"

            dd = tracker.drawdown_pct
            if dd >= settings.DRAWDOWN_HALT_PCT:
                breached = True
                reason = f"Drawdown {dd:.1f}% >= halt threshold {settings.DRAWDOWN_HALT_PCT}%"

            if tracker.current_capital < settings.MIN_CAPITAL_TO_TRADE:
                breached = True
                reason = f"Capital critically low: Rs {tracker.current_capital:.0f}"

            if breached and not _halt_sent:
                set_halt(reason)
                send_kill_alert(reason)
                logger.critical(reason)
                _halt_sent = True
            elif not breached and _halt_sent:
                clear_halt()
                logger.info("Breach resolved -- HALT flag cleared automatically")
                _halt_sent = False

            now = datetime.now(IST)
            if now.hour == 15 and now.minute >= 15:
                logger.info("Market closed. Watchdog idle until tomorrow.")
                time.sleep(600)

        except Exception as e:
            logger.error("Watchdog error: {}", e)

        time.sleep(check_interval)


if __name__ == "__main__":
    watchdog_loop()
