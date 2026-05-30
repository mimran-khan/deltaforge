"""Independent watchdog process -- monitors P&L and kills trading if limits breached."""

from __future__ import annotations
import json
import time
import signal as sig
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from risk.capital_tracker import CapitalTracker


HALT_FLAG = settings.DATA_DIR / "HALT"


def is_halted() -> bool:
    return HALT_FLAG.exists()


def set_halt(reason: str):
    HALT_FLAG.write_text(json.dumps({
        "halted_at": datetime.now().isoformat(),
        "reason": reason,
    }))
    logger.critical("HALT FLAG SET: {}", reason)


def clear_halt():
    if HALT_FLAG.exists():
        HALT_FLAG.unlink()
        logger.info("Halt flag cleared")


def send_kill_alert(reason: str):
    """Send Telegram alert for kill switch activation."""
    try:
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

    def shutdown(signum, frame):
        logger.info("Watchdog shutting down")
        sys.exit(0)

    sig.signal(sig.SIGTERM, shutdown)
    sig.signal(sig.SIGINT, shutdown)

    while True:
        try:
            tracker._load()

            daily_limit = tracker.day_start_capital * (settings.DAILY_LOSS_LIMIT_PCT / 100)

            if tracker.daily_pnl < 0 and abs(tracker.daily_pnl) >= daily_limit:
                reason = (f"Daily loss Rs {tracker.daily_pnl:.0f} "
                          f"exceeds limit Rs {daily_limit:.0f}")
                set_halt(reason)
                send_kill_alert(reason)
                logger.critical(reason)

            if tracker.current_capital < 3000:
                reason = f"Capital critically low: Rs {tracker.current_capital:.0f}"
                set_halt(reason)
                send_kill_alert(reason)
                logger.critical(reason)

            now = datetime.now()
            if now.hour == 15 and now.minute >= 15:
                logger.info("Market closed. Watchdog idle until tomorrow.")
                time.sleep(600)

        except Exception as e:
            logger.error("Watchdog error: {}", e)

        time.sleep(check_interval)


if __name__ == "__main__":
    watchdog_loop()
