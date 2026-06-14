#!/usr/bin/env python3
"""Rebuild capital.json by replaying PAPER_EXIT events from event logs."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, time
from pathlib import Path

import pytz

IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

STARTING_CAPITAL = float(os.environ.get("STARTING_CAPITAL", "10000"))
TARGET_CAPITAL = float(os.environ.get("TARGET_CAPITAL", "0"))
WEEKLY_RESET_DATE = os.environ.get("WEEKLY_RESET_DATE", "")
LAST_START_DATE = os.environ.get("LAST_START_DATE", "")
LAST_WEEKLY_RESET = os.environ.get("LAST_WEEKLY_RESET", "")
WEEK_START_CAPITAL = float(os.environ.get("WEEK_START_CAPITAL", str(STARTING_CAPITAL)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [
    PROJECT_ROOT / "data",
]


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
    return MARKET_OPEN <= t <= MARKET_CLOSE


def is_test_event(event: dict) -> bool:
    return "test" in str(event.get("reason", "")).lower()


def is_orphan_exit(event: dict) -> bool:
    """Exit events with placeholder entry_premium=100 have no real entry."""
    return event.get("entry_premium") == 100


def load_paper_exits(data_dir: Path) -> list[dict]:
    seen_lines: set[str] = set()
    exits: list[dict] = []

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
                    if event.get("event") != "PAPER_EXIT":
                        continue
                    if is_test_event(event) or not is_market_hours(event.get("ts", "")):
                        continue
                    if is_orphan_exit(event):
                        continue
                    exits.append(event)
        except OSError as exc:
            print(f"  warning: could not read {path}: {exc}", file=sys.stderr)

    exits.sort(key=lambda e: e["ts"])
    return exits


def replay_capital(exits: list[dict]) -> dict:
    capital = STARTING_CAPITAL
    peak = STARTING_CAPITAL
    max_drawdown = 0.0
    weekly_pnl = 0.0
    total_pnl = 0.0

    day_start_capital = STARTING_CAPITAL
    daily_pnl = 0.0
    trades_today = 0
    wins_today = 0
    losses_today = 0
    consecutive_losses = 0
    current_day = ""

    trade_log: list[dict] = []
    processed = 0
    skipped_before_reset = 0
    reset_applied = False

    for exit_event in exits:
        trade_date = exit_event["ts"][:10]
        if trade_date < WEEKLY_RESET_DATE:
            skipped_before_reset += 1
            continue

        if not reset_applied:
            capital = WEEK_START_CAPITAL
            peak = max(peak, capital)
            weekly_pnl = 0.0
            reset_applied = True

        if trade_date != current_day:
            current_day = trade_date
            day_start_capital = capital
            daily_pnl = 0.0
            trades_today = 0
            wins_today = 0
            losses_today = 0
            consecutive_losses = 0

        pnl = float(exit_event.get("pnl", 0))
        exit_reason = str(exit_event.get("reason", ""))
        capital += pnl
        daily_pnl += pnl
        weekly_pnl += pnl
        total_pnl += pnl
        trades_today += 1

        if pnl >= 0:
            wins_today += 1
            consecutive_losses = 0
        else:
            losses_today += 1
            consecutive_losses += 1

        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak * 100 if peak > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        trade_log.append({
            "timestamp": exit_event["ts"],
            "direction": exit_event.get("direction"),
            "signal_type": exit_event.get("signal_type"),
            "pnl": round(pnl, 2),
            "exit_reason": exit_reason,
            "capital_after": round(capital, 2),
        })
        processed += 1

    return {
        "current_capital": round(capital, 2),
        "total_pnl": round(total_pnl, 2),
        "peak_capital": round(peak, 2),
        "weekly_pnl": round(weekly_pnl, 2),
        "max_drawdown": round(max_drawdown, 2),
        "daily_pnl": round(daily_pnl, 2),
        "trades_today": trades_today,
        "consecutive_losses": consecutive_losses,
        "day_start_capital": round(day_start_capital, 2),
        "wins_today": wins_today,
        "losses_today": losses_today,
        "last_updated": datetime.now(IST).isoformat(),
        "last_start_date": LAST_START_DATE,
        "last_weekly_reset": LAST_WEEKLY_RESET,
        "week_start_capital": WEEK_START_CAPITAL,
        "_processed_exits": processed,
        "_skipped_before_reset": skipped_before_reset,
        "_reset_applied": reset_applied,
        "_trade_log": trade_log,
    }


def write_capital(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {k: v for k, v in payload.items() if not k.startswith("_")}
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def resolve_data_dir() -> Path:
    for directory in DATA_DIRS:
        if directory.exists() and event_files(directory):
            return directory
    return DATA_DIRS[0]


def main() -> int:
    data_dir = resolve_data_dir()
    print(f"Loading events from {data_dir}")

    exits = load_paper_exits(data_dir)
    print(f"  Found {len(exits)} valid PAPER_EXIT events (deduped, market hours only)")

    result = replay_capital(exits)
    processed = result.pop("_processed_exits")
    skipped = result.pop("_skipped_before_reset")
    reset_applied = result.pop("_reset_applied")
    trade_log = result.pop("_trade_log")

    print(f"  Replayed {processed} PAPER_EXIT events since weekly reset {WEEKLY_RESET_DATE}")
    print(f"  Skipped {skipped} pre-reset exits; reset capital to Rs {WEEK_START_CAPITAL:,.0f} on {WEEKLY_RESET_DATE}")
    if not reset_applied:
        print("  WARNING: weekly reset date had no qualifying exits", file=sys.stderr)
    print(f"  Final capital: Rs {result['current_capital']:,.2f} (target Rs {TARGET_CAPITAL:,.2f})")

    if TARGET_CAPITAL > 0 and abs(result["current_capital"] - TARGET_CAPITAL) > 0.10:
        print(
            f"  WARNING: capital differs from target by "
            f"Rs {result['current_capital'] - TARGET_CAPITAL:,.2f}",
            file=sys.stderr,
        )

    print("\nTrade replay:")
    for trade in trade_log:
        print(
            f"  {trade['timestamp'][:19]} {trade['signal_type']:12} "
            f"{trade['exit_reason']:10} pnl={trade['pnl']:>8.2f} "
            f"cap={trade['capital_after']:,.2f}"
        )

    errors: list[str] = []
    for directory in DATA_DIRS:
        out_path = directory / "capital.json"
        try:
            write_capital(out_path, result)
            print(f"\nWrote {out_path}")
        except OSError as exc:
            msg = f"Failed to write {out_path}: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
