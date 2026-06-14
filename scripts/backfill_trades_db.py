#!/usr/bin/env python3
"""Backfill trades.db by pairing PAPER_ENTRY and PAPER_EXIT events."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, time
from pathlib import Path

IST_MARKET_OPEN = time(9, 15)
IST_MARKET_CLOSE = time(15, 30)
LOT_SIZE = 65
STARTING_CAPITAL = 10_000.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [
    PROJECT_ROOT / "data",
]

CREATE_TRADES_SQL = """
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
"""

INSERT_TRADE_SQL = """
INSERT INTO trades
    (date, time, strategy, direction, confidence, htf_rsi, adx,
     entry_price, exit_price, pnl, hold_bars, exit_reason, lots, capital_after)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

RSI_PATTERN = re.compile(r"(?:RSI15|RSI)=(\d+(?:\.\d+)?)")
ADX_PATTERN = re.compile(r"ADX=(\d+(?:\.\d+)?)")


def event_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("events_*.jsonl"))
    rolling = data_dir / "events.jsonl"
    if rolling.exists():
        files.append(rolling)
    return files


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def is_market_hours(ts: str) -> bool:
    try:
        t = parse_ts(ts).time()
    except ValueError:
        return False
    return IST_MARKET_OPEN <= t <= IST_MARKET_CLOSE


def is_test_event(event: dict) -> bool:
    return "test" in str(event.get("reason", "")).lower()


def is_orphan_exit(event: dict) -> bool:
    """Exit events with placeholder entry_premium=100 have no real entry."""
    return event.get("entry_premium") == 100


def load_paper_events(data_dir: Path) -> list[dict]:
    seen_lines: set[str] = set()
    events: list[dict] = []

    for path in event_files(data_dir):
        try:
            with path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line in seen_lines:
                        continue
                    seen_lines.add(line)
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") not in ("PAPER_ENTRY", "PAPER_EXIT"):
                        continue
                    if is_test_event(event) or not is_market_hours(event.get("ts", "")):
                        continue
                    events.append(event)
        except OSError as exc:
            print(f"  warning: could not read {path}: {exc}", file=sys.stderr)

    events.sort(key=lambda e: e["ts"])
    return events


def extract_strategy(reason: str) -> str:
    text = reason or ""
    upper = text.upper()
    if "15M_RSI" in upper or "15m_RSI" in text:
        return "PULLBACK"
    if "TrendRide" in text:
        return "TREND_RIDE"
    if "EMA9x21" in text:
        return "EMA_MOMENTUM"
    if "ST flip" in text or "ST FLIP" in upper:
        return "SUPERTREND"
    if "Stoch=" in text and "RSI" not in text:
        return "STOCH_CROSS"
    if "Stoch=" in text:
        return "STOCH_CROSS"
    signal = text.split("|")[0].strip()
    return signal or "UNKNOWN"


def extract_htf_rsi(reason: str) -> float | None:
    match = RSI_PATTERN.search(reason or "")
    if not match:
        return None
    return float(match.group(1))


def extract_adx(reason: str) -> float | None:
    match = ADX_PATTERN.search(reason or "")
    if not match:
        return None
    return float(match.group(1))


def entry_price(entry: dict) -> float:
    for key in ("entry_premium", "premium"):
        value = entry.get(key)
        if value is not None:
            return float(value)
    return 100.0


def compute_exit_price(entry_px: float, pnl: float, direction: str) -> float:
    delta = pnl / LOT_SIZE
    if direction == "SHORT":
        return round(entry_px - delta, 4)
    return round(entry_px + delta, 4)


def compute_hold_bars(entry_ts: str, exit_ts: str) -> int:
    minutes = (parse_ts(exit_ts) - parse_ts(entry_ts)).total_seconds() / 60
    return max(0, int(round(minutes / 5)))


def pair_trades(events: list[dict]) -> tuple[list[dict], int]:
    pending: dict[str, dict | None] = {}
    trades: list[dict] = []
    unmatched_exits = 0
    replaced_entries = 0

    for event in events:
        direction = event.get("direction", "")
        if event["event"] == "PAPER_ENTRY":
            if pending.get(direction) is not None:
                replaced_entries += 1
            pending[direction] = event
            continue

        if is_orphan_exit(event):
            unmatched_exits += 1
            continue

        entry = pending.get(direction)
        if entry is None:
            unmatched_exits += 1
            continue

        pending[direction] = None
        reason = entry.get("reason", "")
        pnl = float(event.get("pnl", 0))
        exit_reason = str(event.get("reason", ""))
        entry_px = entry_price(entry)
        direction = event.get("direction", entry.get("direction", ""))
        exit_px = compute_exit_price(entry_px, pnl, direction)
        exit_dt = parse_ts(event["ts"])

        trades.append({
            "date": exit_dt.date().isoformat(),
            "time": exit_dt.strftime("%H:%M:%S"),
            "strategy": extract_strategy(reason),
            "direction": direction,
            "confidence": entry.get("confidence"),
            "htf_rsi": extract_htf_rsi(reason),
            "adx": extract_adx(reason),
            "entry_price": round(entry_px, 4),
            "exit_price": exit_px,
            "pnl": round(pnl, 2),
            "hold_bars": compute_hold_bars(entry["ts"], event["ts"]),
            "exit_reason": exit_reason,
            "lots": entry.get("lots", 1) or 1,
        })

    return trades, unmatched_exits, replaced_entries


def attach_capital_after(trades: list[dict]) -> None:
    capital = STARTING_CAPITAL
    for trade in trades:
        capital += trade["pnl"]
        trade["capital_after"] = round(capital, 2)


def write_database(db_path: Path, trades: list[dict]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    for suffix in ("-shm", "-wal"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(CREATE_TRADES_SQL)
        for trade in trades:
            conn.execute(
                INSERT_TRADE_SQL,
                (
                    trade["date"],
                    trade["time"],
                    trade["strategy"],
                    trade["direction"],
                    trade["confidence"],
                    trade["htf_rsi"],
                    trade["adx"],
                    trade["entry_price"],
                    trade["exit_price"],
                    trade["pnl"],
                    trade["hold_bars"],
                    trade["exit_reason"],
                    trade["lots"],
                    trade["capital_after"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def resolve_data_dir() -> Path:
    for directory in DATA_DIRS:
        if directory.exists() and event_files(directory):
            return directory
    return DATA_DIRS[0]


def main() -> int:
    data_dir = resolve_data_dir()
    print(f"Loading events from {data_dir}")

    events = load_paper_events(data_dir)
    entries = sum(1 for e in events if e["event"] == "PAPER_ENTRY")
    exits = sum(1 for e in events if e["event"] == "PAPER_EXIT")
    print(f"  Found {entries} PAPER_ENTRY and {exits} PAPER_EXIT events")

    trades, unmatched_exits, replaced_entries = pair_trades(events)
    attach_capital_after(trades)

    print(f"  Paired {len(trades)} trades")
    if unmatched_exits:
        print(f"  Skipped {unmatched_exits} unmatched/orphan PAPER_EXIT events")
    if replaced_entries:
        print(f"  Replaced {replaced_entries} stale open entries (single-position model)")

    if trades:
        print("\nTrades:")
        for trade in trades:
            print(
                f"  {trade['date']} {trade['time']} {trade['strategy']:12} "
                f"{trade['direction']:5} pnl={trade['pnl']:>8.2f} "
                f"exit={trade['exit_reason']} cap={trade['capital_after']:,.2f}"
            )
        print(f"\n  Final capital: Rs {trades[-1]['capital_after']:,.2f}")
    else:
        print("  No trades to insert")

    errors: list[str] = []
    for directory in DATA_DIRS:
        db_path = directory / "trades.db"
        try:
            write_database(db_path, trades)
            print(f"Wrote {db_path} ({len(trades)} rows)")
        except OSError as exc:
            msg = f"Failed to write {db_path}: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
