#!/usr/bin/env python3
"""TradingAgent -- Autonomous Nifty/BankNifty Options Scalping System.

Usage:
    python main.py                  # Run daily trading session
    python main.py --backtest       # Run backtest on synthetic data
    python main.py --paper          # Force paper trading mode
    python main.py --live           # Force live trading mode
"""

import sys
import argparse
from pathlib import Path
from datetime import date

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        level="INFO",
    )
    log_file = settings.LOG_DIR / f"trading_{date.today().isoformat()}.log"
    logger.add(
        log_file,
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        level="DEBUG",
    )


def run_trading():
    from automation.daily_scheduler import DailyScheduler
    scheduler = DailyScheduler()
    scheduler.run()


def run_backtest(days: int = 180, capital: float = 10000):
    from backtest.run_backtest import main as bt_main
    sys.argv = ["backtest", "--days", str(days), "--capital", str(capital)]
    bt_main()


def main():
    parser = argparse.ArgumentParser(description="TradingAgent")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--bt-days", type=int, default=180, help="Backtest days")
    parser.add_argument("--bt-capital", type=float, default=10000, help="Backtest capital")
    parser.add_argument("--paper", action="store_true", help="Force paper mode")
    parser.add_argument("--live", action="store_true", help="Force live mode")
    args = parser.parse_args()

    setup_logging()

    if args.paper:
        settings.TRADING_MODE = "paper"
    elif args.live:
        settings.TRADING_MODE = "live"

    logger.info("TradingAgent v1.0")
    logger.info("Mode: {}", settings.TRADING_MODE.upper())
    logger.info("Capital: Rs {:,.0f}", settings.STARTING_CAPITAL)

    if args.backtest:
        run_backtest(days=args.bt_days, capital=args.bt_capital)
    else:
        run_trading()


if __name__ == "__main__":
    main()
