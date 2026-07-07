"""Download real Nifty/BankNifty historical data from Yahoo Finance."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import settings


def download_nifty_intraday(interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    """Download Nifty 50 intraday data. Yahoo provides ~60 days of 5m data."""
    logger.info("Downloading Nifty 50 {} data (period={})...", interval, period)
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        logger.error("No intraday data received")
        return df
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df["volume"] > 0]
    if not df.empty:
        logger.info("Downloaded {} candles from {} to {}",
                    len(df), df.index[0], df.index[-1])
    else:
        logger.warning("Intraday download returned empty dataframe")
    return df


def download_nifty_daily(years: int = 2) -> pd.DataFrame:
    """Download Nifty 50 daily data for longer-term analysis."""
    logger.info("Downloading Nifty 50 daily data ({} years)...", years)
    end = datetime.now(IST)
    start = end - timedelta(days=years * 365)
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(start=start, end=end, interval="1d")
    if df.empty:
        logger.error("No daily data received")
        return df
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]]
    logger.info("Downloaded {} daily candles from {} to {}",
                len(df), df.index[0], df.index[-1])
    return df


def download_banknifty_intraday(interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    logger.info("Downloading BankNifty {} data...", interval)
    ticker = yf.Ticker("^NSEBANK")
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df["volume"] > 0]
    logger.info("Downloaded {} BankNifty candles", len(df))
    return df


def save_data(df: pd.DataFrame, name: str):
    path = settings.DATA_DIR / f"{name}.csv"
    df.to_csv(path)
    logger.info("Saved {} rows to {}", len(df), path)
    return path


def load_data(name: str) -> pd.DataFrame:
    path = settings.DATA_DIR / f"{name}.csv"
    if not path.exists():
        logger.error("Data file not found: {}", path)
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def download_all():
    """Download all data and save to data/ directory."""
    nifty_5m = download_nifty_intraday("5m", "60d")
    if not nifty_5m.empty:
        save_data(nifty_5m, "nifty_5m")

    nifty_15m = download_nifty_intraday("15m", "60d")
    if not nifty_15m.empty:
        save_data(nifty_15m, "nifty_15m")

    nifty_daily = download_nifty_daily(2)
    if not nifty_daily.empty:
        save_data(nifty_daily, "nifty_daily")

    banknifty_5m = download_banknifty_intraday("5m", "60d")
    if not banknifty_5m.empty:
        save_data(banknifty_5m, "banknifty_5m")

    return {
        "nifty_5m": nifty_5m,
        "nifty_15m": nifty_15m,
        "nifty_daily": nifty_daily,
        "banknifty_5m": banknifty_5m,
    }


if __name__ == "__main__":
    download_all()
