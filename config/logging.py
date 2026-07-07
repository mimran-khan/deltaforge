"""Centralized logging configuration.

Three sinks:
  1. Console  -- colored, human-friendly, INFO+
  2. File     -- daily rotation, 30-day retention, DEBUG+
  3. JSON     -- machine-parseable, for dashboards / alerting, DEBUG+

Usage:
    from config.logging import setup_logging
    setup_logging()            # default: INFO console, DEBUG files
    setup_logging(verbose=True) # DEBUG everywhere
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime

import pytz
from loguru import logger

from config import settings

IST = pytz.timezone("Asia/Kolkata")


def _today_ist() -> date:
    return datetime.now(IST).date()


def _json_serializer(message):
    """Loguru custom sink -- writes one JSON object per line."""
    record = message.record
    ist_time = record["time"].astimezone(IST)
    entry = {
        "ts": ist_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
        "level": record["level"].name,
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    if record["exception"]:
        entry["exception"] = str(record["exception"])
    return json.dumps(entry, default=str)


def _json_sink(message):
    json_dir = settings.LOG_DIR / "json"
    json_dir.mkdir(exist_ok=True)
    path = json_dir / f"trading_{_today_ist().isoformat()}.jsonl"
    with open(path, "a") as f:
        f.write(_json_serializer(message) + "\n")


def _ist_formatter(record):
    """Convert loguru record time to IST before formatting."""
    record["extra"]["ist_time"] = record["time"].astimezone(IST)


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure all logging sinks. Call once at process start.

    All timestamps in console, file, and JSON logs are in IST (UTC+05:30).
    """
    settings.ensure_dirs()

    import os
    os.environ["TZ"] = "Asia/Kolkata"
    try:
        import time as _time
        _time.tzset()
    except AttributeError:
        pass

    logger.remove()

    console_level = "DEBUG" if verbose else ("WARNING" if quiet else "INFO")
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<7}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan> | "
            "{message}"
        ),
        level=console_level,
        colorize=True,
    )

    log_file = settings.LOG_DIR / f"trading_{_today_ist().isoformat()}.log"
    logger.add(
        str(log_file),
        rotation="1 day",
        retention="30 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {module}:{function}:{line} | {message}",
        level="DEBUG",
        enqueue=True,
    )

    logger.add(
        _json_sink,
        level="DEBUG",
        enqueue=True,
    )

    logger.info("Logging initialized (console={}, file=DEBUG, json=DEBUG)", console_level)
