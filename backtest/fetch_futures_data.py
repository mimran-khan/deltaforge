"""Fetch historical 5-min candle data for MCX/CDS futures from Angel One.

Downloads data for Gold Petal, Crude Oil Mini, and USDINR futures.
Saves as <instrument>_5m_combined.csv in the data/ directory for backtesting.

Usage:
    python -m backtest.fetch_futures_data --days 60
    python -m backtest.fetch_futures_data --instrument GOLD_PETAL --days 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.instruments import Instrument, get_futures_instruments
from engine.broker import BrokerConnection

IST = pytz.timezone("Asia/Kolkata")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_DELAY = 2.0


def resolve_token(broker: BrokerConnection, instrument: Instrument) -> tuple[str, str]:
    """Resolve the current futures token for an instrument from scrip master."""
    instruments_path = DATA_DIR / "instruments.json"
    if not instruments_path.exists():
        logger.info("Downloading instruments from broker...")
        broker.download_instruments()

    if not instruments_path.exists():
        raise RuntimeError(f"instruments.json not available -- cannot resolve {instrument.name}")

    with open(instruments_path) as f:
        all_scrips = json.load(f)

    def _parse_expiry(expiry_str: str) -> datetime:
        try:
            return IST.localize(datetime.strptime(expiry_str, "%d%b%Y"))
        except (ValueError, TypeError):
            return IST.localize(datetime(2099, 12, 31))

    matches = [
        s for s in all_scrips
        if s.get("exch_seg") == instrument.exchange
        and s.get("name", "").startswith(instrument.symbol_prefix)
        and s.get("instrumenttype") in ("FUTCOM", "FUTCUR", "FUTSTK")
    ]
    if not matches:
        raise RuntimeError(f"No scrip match for {instrument.symbol_prefix} on {instrument.exchange}")

    now = datetime.now(IST)
    future_matches = [s for s in matches if _parse_expiry(s.get("expiry", "")) >= now]
    pool = future_matches if future_matches else matches
    pool.sort(key=lambda s: _parse_expiry(s.get("expiry", "")))
    nearest = pool[0]

    token = str(nearest.get("token", ""))
    symbol = nearest.get("symbol", instrument.symbol_prefix)
    logger.info("Resolved {}: token={} symbol={} expiry={}",
                instrument.name, token, symbol, nearest.get("expiry", ""))
    return token, symbol


def fetch_5m_candles(broker: BrokerConnection, exchange: str,
                     token: str, from_date: date, to_date: date) -> pd.DataFrame:
    """Fetch 5-minute candles for a date range, paginated by day."""
    all_rows = []
    current = from_date

    while current <= to_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        from_str = f"{current.strftime('%Y-%m-%d')} 09:00"
        to_str = f"{current.strftime('%Y-%m-%d')} 23:45"

        logger.info("  Fetching {} ...", current.strftime("%Y-%m-%d"))
        candles = broker.get_historical(exchange, token, "FIVE_MINUTE", from_str, to_str)

        if candles:
            for c in candles:
                all_rows.append({
                    "timestamp": c[0],
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": int(c[5]) if len(c) > 5 else 0,
                })
            logger.info("    Got {} candles", len(candles))
        else:
            logger.info("    No data (holiday or no session)")

        time.sleep(API_DELAY)
        current += timedelta(days=1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def download_instrument(broker: BrokerConnection, instrument: Instrument,
                        days: int) -> Optional[Path]:
    """Download historical data for a single instrument."""
    logger.info("=" * 50)
    logger.info("Downloading {} ({}) -- last {} days", instrument.display_name, instrument.exchange, days)

    try:
        token, symbol = resolve_token(broker, instrument)
    except RuntimeError as e:
        logger.error("Token resolution failed: {}", e)
        return None

    to_date = date.today()
    from_date = to_date - timedelta(days=days)

    df = fetch_5m_candles(broker, instrument.exchange, token, from_date, to_date)
    if df.empty:
        logger.warning("No data retrieved for {}", instrument.name)
        return None

    out_file = DATA_DIR / f"{instrument.name.lower()}_5m_combined.csv"

    if out_file.exists():
        existing = pd.read_csv(out_file, index_col="timestamp", parse_dates=True)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

    df.to_csv(out_file, index_label="timestamp")
    unique_days = len(set(df.index.date))
    logger.info("Saved {} candles ({} trading days) to {}", len(df), unique_days, out_file.name)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Fetch MCX/CDS futures historical data")
    parser.add_argument("--days", type=int, default=60, help="Number of calendar days to fetch")
    parser.add_argument("--instrument", type=str, default=None,
                        help="Specific instrument name (e.g. GOLD_PETAL, CRUDEOILM)")
    args = parser.parse_args()

    broker = BrokerConnection()
    if not broker.login():
        logger.error("Broker login failed")
        sys.exit(1)

    instruments = get_futures_instruments()
    if args.instrument:
        instruments = [i for i in instruments if i.name.upper() == args.instrument.upper()]
        if not instruments:
            logger.error("Unknown instrument: {}", args.instrument)
            sys.exit(1)

    for inst in instruments:
        download_instrument(broker, inst, args.days)
        time.sleep(5)

    broker.logout()
    logger.info("Done. Run backtests with: python -m backtest.run_backtest --instrument <name> --days <N>")


if __name__ == "__main__":
    main()
