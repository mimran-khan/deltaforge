"""SQLite performance database -- records every trade for analysis.

Data collection from day 1. No auto-disable logic.
Manual review weekly via strategy_stats() and daily_summary().
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pytz
from loguru import logger

from config import settings

IST = pytz.timezone("Asia/Kolkata")


class PerformanceDB:

    def __init__(self, db_path: Optional[str] = None):
        path = db_path or str(settings.DB_PATH)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL,
                htf_rsi REAL,
                adx REAL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                pnl REAL NOT NULL,
                hold_bars INTEGER,
                exit_reason TEXT,
                lots INTEGER,
                capital_after REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        self._migrate_add_instrument_columns()

    def _migrate_add_instrument_columns(self):
        """Add instrument and exchange columns if they don't exist.

        Backward compatible: existing rows default to NIFTY/NFO.
        Safe to call repeatedly -- checks column existence first.
        """
        try:
            cols = {
                row[1]
                for row in self.conn.execute("PRAGMA table_info(trades)").fetchall()
            }
            if "instrument" not in cols:
                self.conn.execute(
                    "ALTER TABLE trades ADD COLUMN instrument TEXT DEFAULT 'NIFTY'"
                )
                logger.info("DB migration: added 'instrument' column to trades")
            if "exchange" not in cols:
                self.conn.execute(
                    "ALTER TABLE trades ADD COLUMN exchange TEXT DEFAULT 'NFO'"
                )
                logger.info("DB migration: added 'exchange' column to trades")
            self.conn.commit()
        except Exception as e:
            logger.warning("DB migration check failed (non-fatal): {}", e)

    def record_trade(self, date: str, time: str, strategy: str,
                     direction: str, entry_price: float, exit_price: float,
                     pnl: float, confidence: float = 0, htf_rsi: float = 0,
                     adx: float = 0, hold_bars: int = 0,
                     exit_reason: str = "", lots: int = 1,
                     capital_after: float = 0,
                     instrument: str = "NIFTY",
                     exchange: str = "NFO"):
        with self._lock:
            try:
                self.conn.execute("""
                    INSERT INTO trades
                        (date, time, strategy, direction, confidence, htf_rsi,
                         adx, entry_price, exit_price, pnl, hold_bars,
                         exit_reason, lots, capital_after,
                         instrument, exchange)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (date, time, strategy, direction, confidence, htf_rsi,
                      adx, entry_price, exit_price, round(pnl, 2), hold_bars,
                      exit_reason, lots, round(capital_after, 2),
                      instrument, exchange))
                self.conn.commit()
            except Exception as e:
                logger.error("PerfDB record error: {}", e)

    def daily_summary(self, target_date: Optional[str] = None) -> dict:
        d = target_date or datetime.now(IST).date().isoformat()
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE date = ?", (d,)
            ).fetchall()

        if not rows:
            return {"date": d, "trades": 0, "wins": 0, "losses": 0,
                    "total_pnl": 0, "wr": 0}

        wins = sum(1 for r in rows if r["pnl"] > 0)
        losses = len(rows) - wins
        total_pnl = sum(r["pnl"] for r in rows)
        wr = wins / len(rows) * 100

        return {
            "date": d,
            "trades": len(rows),
            "wins": wins,
            "losses": losses,
            "total_pnl": round(total_pnl, 2),
            "wr": round(wr, 1),
        }

    def strategy_stats(self, strategy: Optional[str] = None,
                       min_trades: int = 5) -> list[dict]:
        with self._lock:
            if strategy:
                rows = self.conn.execute(
                    "SELECT strategy, COUNT(*) as n, "
                    "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
                    "SUM(pnl) as total_pnl, AVG(pnl) as avg_pnl "
                    "FROM trades WHERE strategy = ? GROUP BY strategy",
                    (strategy,)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT strategy, COUNT(*) as n, "
                    "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
                    "SUM(pnl) as total_pnl, AVG(pnl) as avg_pnl "
                    "FROM trades GROUP BY strategy HAVING n >= ?",
                    (min_trades,)
                ).fetchall()

        return [
            {
                "strategy": r["strategy"],
                "trades": r["n"],
                "wins": r["wins"],
                "wr": round(r["wins"] / r["n"] * 100, 1) if r["n"] > 0 else 0,
                "total_pnl": round(r["total_pnl"], 2),
                "avg_pnl": round(r["avg_pnl"], 2),
            }
            for r in rows
        ]

    def close(self):
        self.conn.close()
