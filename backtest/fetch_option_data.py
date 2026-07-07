"""Fetch real historical Nifty option 5-min candle data from Angel One.

Downloads ATM CE and PE premiums for each trading day, aligned with
our 5-min Nifty index candles. This gives us REAL option prices for
backtesting instead of the synthetic premium model.

Usage:
    python -m backtest.fetch_option_data --days 30
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backtest.run_backtest import load_real_data
from engine.broker import BrokerConnection

IST = pytz.timezone("Asia/Kolkata")
OPTION_DATA_DIR = BASE_DIR / "data" / "option_candles"
OPTION_DATA_DIR.mkdir(parents=True, exist_ok=True)

_EXPIRY_FORMATS = ("%d%b%Y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y")


def _parse_expiry(exp_str: str) -> Optional[date]:
    if not exp_str:
        return None
    cleaned = exp_str.strip().upper()
    for fmt in _EXPIRY_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _find_atm_option(instruments: list, spot_price: float,
                     option_type: str, trade_date: date) -> Optional[dict]:
    """Find the ATM option for a given date (nearest expiry >= trade_date)."""
    step = 50
    atm_strike = round(spot_price / step) * step
    atm_strike_raw = atm_strike * 100

    nifty_opts = [
        i for i in instruments
        if i.get("name") == "NIFTY"
        and i.get("instrumenttype") == "OPTIDX"
        and i.get("exch_seg") == "NFO"
        and i.get("symbol", "").endswith(option_type)
    ]

    candidates = []
    for o in nifty_opts:
        strike_raw = float(o.get("strike", 0))
        if abs(strike_raw - atm_strike_raw) > step * 100:
            continue

        exp = _parse_expiry(o.get("expiry", ""))
        if exp is None or exp < trade_date:
            continue
        candidates.append((o, abs(strike_raw - atm_strike_raw), exp))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[2], x[1]))
    return candidates[0][0]


def fetch_option_candles(broker: BrokerConnection, token: str,
                         trade_date: date) -> pd.DataFrame:
    """Fetch 5-min candle data for a single option on a single day."""
    from_dt = datetime.combine(trade_date, datetime.min.time()).replace(
        hour=9, minute=0)
    to_dt = datetime.combine(trade_date, datetime.min.time()).replace(
        hour=15, minute=35)

    from_str = from_dt.strftime("%Y-%m-%d %H:%M")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M")

    data = broker.get_historical(
        exchange="NFO",
        token=token,
        interval="FIVE_MINUTE",
        from_date=from_str,
        to_date=to_str,
    )

    if not data:
        return pd.DataFrame()

    rows = []
    for candle in data:
        ts = pd.Timestamp(candle[0])
        rows.append({
            "timestamp": ts,
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": int(candle[5]),
        })

    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    return df


def run_fetch(days: int = 30):
    """Main entry: fetch option data for the last N trading days."""
    logger.info("Loading Nifty index data...")
    nifty_df = load_real_data(days=days + 10)
    unique_days = sorted(set(nifty_df.index.date))[-days:]
    logger.info("Will fetch option data for {} trading days", len(unique_days))

    logger.info("Connecting to Angel One...")
    broker = BrokerConnection()
    if not broker.login():
        logger.error("Broker login failed. Check credentials in .env")
        return

    logger.info("Downloading instrument master...")
    instruments = broker.download_instruments()
    if not instruments:
        logger.error("No instruments loaded")
        return

    nifty_options = [
        i for i in instruments
        if i.get("name") == "NIFTY"
        and i.get("instrumenttype") == "OPTIDX"
        and i.get("exch_seg") == "NFO"
    ]
    logger.info("Found {} NIFTY options in instrument master", len(nifty_options))

    fetched = 0
    skipped = 0

    for trade_date in unique_days:
        cache_file = OPTION_DATA_DIR / f"{trade_date}.csv"
        if cache_file.exists():
            logger.info("  {} -- cached", trade_date)
            skipped += 1
            continue

        day_nifty = nifty_df[nifty_df.index.date == trade_date]
        if day_nifty.empty:
            continue

        spot_open = day_nifty["open"].iloc[0]

        results = {}
        for opt_type in ["CE", "PE"]:
            option = _find_atm_option(instruments, spot_open, opt_type, trade_date)
            if not option:
                logger.warning("  {} -- no {} option found for spot={:.0f}",
                               trade_date, opt_type, spot_open)
                continue

            symbol = option["symbol"]
            token = option["token"]
            strike = float(option.get("strike", 0)) / 100
            expiry = option.get("expiry", "?")

            logger.info("  {} {} -- {} (token={}, strike={}, exp={})",
                        trade_date, opt_type, symbol, token, strike, expiry)

            candles = fetch_option_candles(broker, token, trade_date)
            if candles.empty:
                logger.warning("    No candle data returned for {}", symbol)
                continue

            candles["option_type"] = opt_type
            candles["symbol"] = symbol
            candles["strike"] = strike
            candles["expiry"] = expiry
            results[opt_type] = candles

            time.sleep(0.5)

        if results:
            combined = pd.concat(results.values())
            combined.to_csv(cache_file)
            fetched += 1
            logger.info("  {} -- saved {} candles", trade_date, len(combined))
        else:
            logger.warning("  {} -- no option data fetched", trade_date)

        time.sleep(1)

    logger.info("Done: fetched={}, cached={}, total={}", fetched, skipped, len(unique_days))

    summary_file = OPTION_DATA_DIR / "fetch_summary.json"
    summary = {
        "last_run": datetime.now(IST).isoformat(),
        "days_requested": days,
        "days_available": len(unique_days),
        "days_fetched": fetched,
        "days_cached": skipped,
        "date_range": f"{unique_days[0]} to {unique_days[-1]}",
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    run_fetch(days=args.days)
