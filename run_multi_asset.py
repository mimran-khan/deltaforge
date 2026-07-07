#!/usr/bin/env python3
"""Standalone launcher for the Multi-Asset Engine.

Runs Gold Petal, Crude Oil Mini, and USDINR futures strategies
completely independently of the Nifty options engine.

Usage:
    python run_multi_asset.py
"""

import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from utils.logging import setup_logging

from engine.broker import BrokerConnection
from engine.multi_asset_engine import MultiAssetEngine

setup_logging()

engine = None


def _shutdown(signum, frame):
    logger.info("Multi-Asset: shutdown signal received ({})", signum)
    if engine:
        engine.stop()


def main():
    global engine

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info("=" * 60)
    logger.info("Multi-Asset Standalone Launcher")
    logger.info("=" * 60)

    broker = BrokerConnection()
    if not broker.is_active:
        logger.error("Broker login failed -- cannot start multi-asset engine")
        sys.exit(1)

    engine = MultiAssetEngine(broker=broker)
    engine.start_day()
    engine.run_loop(poll_interval=5)

    logger.info("Multi-Asset engine finished for today")


if __name__ == "__main__":
    main()
