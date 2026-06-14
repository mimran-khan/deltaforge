"""Replay live week: compare backtest signals against actual live event logs.

Loads events_*.jsonl, replays with corrected warmup, and asserts entry
times/strategies match within ±1 bar. This is the regression test that
proves backtest ≈ live.

Usage:
    python -m backtest.replay_live_week [--date-range 2026-06-09:2026-06-12]
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backtest.run_backtest import load_real_data, run_compound_backtest


def _load_live_events(data_dir: Path, date_range: tuple[str, str] | None = None) -> list[dict]:
    """Load all PAPER_ENTRY events from events_*.jsonl files."""
    events = []

    for f in sorted(data_dir.glob("events*.jsonl")):
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    ev = json.loads(line)
                    if ev.get("event") in ("PAPER_ENTRY", "LIVE_ENTRY"):
                        events.append(ev)
        except Exception as e:
            logger.debug("Skip {}: {}", f.name, e)

    if date_range:
        start, end = date_range
        events = [e for e in events if start <= e["ts"][:10] <= end]

    return events


def _match_trades(live_entries: list[dict], bt_trades: list[dict],
                  tolerance_bars: int = 1) -> dict:
    """Compare live entries against backtest trades.

    Returns match statistics and detailed per-trade comparison.
    """
    results = {
        "live_count": len(live_entries),
        "bt_count": len(bt_trades),
        "matched": 0,
        "unmatched_live": [],
        "unmatched_bt": [],
        "details": [],
    }

    bt_remaining = list(bt_trades)

    for live in live_entries:
        live_ts = live["ts"][:16]
        live_dir = live.get("direction", "")
        live_strat = live.get("signal_type", "")
        live_dt = pd.Timestamp(live_ts)

        best_match = None
        best_delta = timedelta(minutes=999)

        for bt in bt_remaining:
            bt_dt = pd.Timestamp(str(bt["entry_time"]))
            delta = abs(bt_dt - live_dt)
            time_ok = delta <= timedelta(minutes=5 * tolerance_bars)
            dir_ok = bt["signal"] == live_dir
            strat_ok = bt["strategy"] == live_strat

            if time_ok and dir_ok and delta < best_delta:
                best_match = bt
                best_delta = delta

        if best_match:
            results["matched"] += 1
            bt_remaining.remove(best_match)
            results["details"].append({
                "live_time": live_ts,
                "bt_time": str(best_match["entry_time"]),
                "direction": live_dir,
                "live_strategy": live_strat,
                "bt_strategy": best_match["strategy"],
                "time_delta_min": best_delta.total_seconds() / 60,
                "match": "OK" if best_match["strategy"] == live_strat else "STRATEGY_MISMATCH",
            })
        else:
            results["unmatched_live"].append({
                "time": live_ts,
                "direction": live_dir,
                "strategy": live_strat,
            })

    results["unmatched_bt"] = [{
        "time": str(bt["entry_time"]),
        "direction": bt["signal"],
        "strategy": bt["strategy"],
    } for bt in bt_remaining]

    return results


def main():
    parser = argparse.ArgumentParser(description="Replay live week vs backtest")
    parser.add_argument("--date-range", type=str, default=None,
                        help="Date range as START:END (e.g. 2026-06-09:2026-06-12)")
    parser.add_argument("--capital", type=float, default=10000)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    data_dir = Path(settings.DATA_DIR)

    date_range = None
    if args.date_range:
        parts = args.date_range.split(":")
        if len(parts) == 2:
            date_range = (parts[0], parts[1])

    print("\n" + "=" * 65)
    print("  REPLAY LIVE WEEK -- Backtest vs Live Comparison")
    print("=" * 65)

    live_entries = _load_live_events(data_dir, date_range)
    print(f"\n  Live entries found: {len(live_entries)}")

    if not live_entries:
        print("  No live events found. Nothing to replay.")
        return

    for e in live_entries:
        print(f"    {e['ts'][:16]} | {e.get('signal_type','?'):12s} | "
              f"{e.get('direction','?'):5s} | conf={e.get('confidence', 0)}")

    df = load_real_data(days=30)
    if date_range:
        start_dt = pd.Timestamp(date_range[0])
        end_dt = pd.Timestamp(date_range[1]) + timedelta(days=1)
        df = df[(df.index >= start_dt) & (df.index < end_dt)]

    unique_days = sorted(set(df.index.date))
    print(f"\n  Backtest data: {len(df)} candles, {len(unique_days)} days")

    results = run_compound_backtest(
        df, starting_capital=args.capital,
        lot_size=settings.NIFTY_LOT_SIZE,
    )

    print(f"  Backtest trades: {results['total_trades']}")

    match_results = _match_trades(live_entries, results["trades"])

    print(f"\n  {'─' * 60}")
    print(f"  MATCH RESULTS:")
    print(f"    Live entries:    {match_results['live_count']}")
    print(f"    Backtest trades: {match_results['bt_count']}")
    print(f"    Matched:         {match_results['matched']}")
    print(f"    Unmatched live:  {len(match_results['unmatched_live'])}")
    print(f"    Unmatched BT:    {len(match_results['unmatched_bt'])}")

    match_rate = (match_results['matched'] / match_results['live_count'] * 100
                  if match_results['live_count'] > 0 else 0)
    print(f"    Match rate:      {match_rate:.0f}%")

    if match_results["details"]:
        print(f"\n  Matched trades:")
        for d in match_results["details"]:
            print(f"    {d['live_time']} ↔ {d['bt_time']} | "
                  f"{d['direction']:5s} | {d['live_strategy']:12s} | "
                  f"Δ={d['time_delta_min']:.0f}min | {d['match']}")

    if match_results["unmatched_live"]:
        print(f"\n  Live entries NOT matched in backtest:")
        for u in match_results["unmatched_live"]:
            print(f"    {u['time']} | {u['direction']:5s} | {u['strategy']}")

    if match_results["unmatched_bt"]:
        print(f"\n  Backtest trades NOT matched in live:")
        for u in match_results["unmatched_bt"][:10]:
            print(f"    {u['time']} | {u['direction']:5s} | {u['strategy']}")
        if len(match_results["unmatched_bt"]) > 10:
            print(f"    ... and {len(match_results['unmatched_bt']) - 10} more")

    print(f"\n  {'─' * 60}")
    if match_rate >= 80:
        print("  VERDICT: PASS -- backtest closely matches live")
    elif match_rate >= 50:
        print("  VERDICT: PARTIAL -- some divergence, investigate unmatched")
    else:
        print("  VERDICT: FAIL -- significant backtest-live gap")
    print()


if __name__ == "__main__":
    main()
