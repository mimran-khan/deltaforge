#!/usr/bin/env python3
"""Counterfactual: what if today's guardrails had been active? (read-only sim)"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings

DATA = ROOT / "data"

# ── load candles from latest engine_state export ─────────────────
state = json.loads((DATA / "engine_state.json").read_text())
candles = state["candles"]

# ── actual trades today ──────────────────────────────────────────
TRADES = [
    {
        "id": 1,
        "strategy": "PULLBACK",
        "signal_bar": "11:40",
        "entry_time": "11:45",
        "entry_index": 23955.05,
        "entry_premium": 104.22,
        "sl_premium": 93.798,
        "exit_premium": 93.498,
        "pnl": -1452.79,
        "exit_reason": "SL",
        "exit_time": "12:10",
        "htf_rsi": 80,
        "ltf_rsi": 62,
        "direction": "LONG",
        "lots": 2,
        "qty": 130,
    },
    {
        "id": 2,
        "strategy": "STOCH_CROSS",
        "signal_bar": "12:15",
        "entry_time": "12:20",
        "entry_index": 23908.65,
        "entry_premium": 103.18,
        "sl_premium": 92.862,
        "exit_premium": None,  # simulate
        "pnl": None,
        "exit_reason": None,
        "exit_time": None,
        "htf_rsi": 74,
        "ltf_rsi": 42,
        "direction": "LONG",
        "lots": 2,
        "qty": 130,
    },
]

DELTA = settings.PREMIUM_DELTA
THETA = settings.PREMIUM_THETA_PER_CANDLE
SLIPPAGE = settings.SLIPPAGE_POINTS


def twap_vwap_proxy(rows: list[dict]) -> list[float]:
    """Volume is 0 today — use cumulative typical-price average as VWAP proxy."""
    out, cum = [], 0.0
    for i, c in enumerate(rows):
        tp = (c["h"] + c["l"] + c["c"]) / 3
        cum += tp
        out.append(cum / (i + 1))
    return out


def session_high_so_far(rows: list[dict]) -> list[float]:
    hi = 0.0
    out = []
    for c in rows:
        hi = max(hi, c["h"])
        out.append(hi)
    return out


def lower_highs_since_peak(rows: list[dict]) -> list[bool]:
    """True once we've made a lower swing-high after the session peak."""
    peak = 0.0
    last_swing_high = 0.0
    flagged = False
    out = []
    for c in rows:
        peak = max(peak, c["h"])
        # simple: after peak, if this bar's high < prior bar high while below peak zone
        if c["h"] >= peak - 0.01:
            last_swing_high = c["h"]
        elif last_swing_high > 0 and c["h"] < last_swing_high:
            flagged = True
        out.append(flagged)
    return out


def idx_for_time(t: str) -> int:
    for i, c in enumerate(candles):
        if c["t"] == t:
            return i
    raise KeyError(t)


def premium_at(entry_index: float, entry_prem: float, index_price: float, bars: int) -> float:
    move = index_price - entry_index
    return max(0.5, entry_prem + move * DELTA - bars * THETA)


def simulate_trade2_exit(rows: list[dict], start_i: int) -> dict:
    t = TRADES[1]
    entry_i = start_i + 1  # entry on next bar open after 12:15 signal
    bars_held = 0
    for j in range(entry_i, len(rows)):
        bars_held += 1
        px = rows[j]["c"]
        cur = premium_at(t["entry_index"], t["entry_premium"], px, bars_held)
        if cur <= t["sl_premium"]:
            exit_prem = t["sl_premium"] - SLIPPAGE
            raw = (exit_prem - t["entry_premium"]) * t["qty"]
            costs = 58.0  # ~same as trade 1
            return {
                "exit_time": rows[j]["t"],
                "exit_premium": round(exit_prem, 3),
                "pnl": round(raw - costs, 2),
                "exit_reason": "SL",
                "bars_held": bars_held,
            }
    last = rows[-1]
    cur = premium_at(t["entry_index"], t["entry_premium"], last["c"], len(rows) - entry_i)
    exit_prem = cur - SLIPPAGE
    raw = (exit_prem - t["entry_premium"]) * t["qty"]
    return {
        "exit_time": last["t"] + " (EOD sim)",
        "exit_premium": round(exit_prem, 3),
        "pnl": round(raw - 58.0, 2),
        "exit_reason": "OPEN/EOD",
        "bars_held": len(rows) - entry_i,
    }


@dataclass
class FilterResult:
    name: str
    blocked: list[int]
    reasons: dict[int, str]


def apply_filters(rows: list[dict], twap: list[float], sess_hi: list[float], lh: list[bool]):
    results = []

    # 1) Below VWAP/TWAP blocks LONG
    b1, r1 = [], {}
    for tr in TRADES:
        i = idx_for_time(tr["signal_bar"])
        if tr["direction"] == "LONG" and rows[i]["c"] < twap[i]:
            b1.append(tr["id"])
            r1[tr["id"]] = f"close {rows[i]['c']:.1f} < TWAP {twap[i]:.1f}"
    results.append(FilterResult("① Below VWAP (TWAP proxy)", b1, r1))

    # 2) HTF/LTF divergence — block LONG if HTF>=58 and LTF<=HTF-18
    b2, r2 = [], {}
    for tr in TRADES:
        if tr["direction"] == "LONG" and tr["htf_rsi"] >= 58 and tr["ltf_rsi"] <= tr["htf_rsi"] - 18:
            b2.append(tr["id"])
            r2[tr["id"]] = f"HTF={tr['htf_rsi']} LTF={tr['ltf_rsi']} gap={tr['htf_rsi']-tr['ltf_rsi']}"
    results.append(FilterResult("② HTF/LTF alignment (gap≥18)", b2, r2))

    # 3) Global 30m cooldown after any SL
    b3, r3 = [], {}
    sl_time = TRADES[0]["exit_time"]
    for tr in TRADES[1:]:
        # signal within 6 bars (30m) after SL
        if tr["signal_bar"] < "12:40" and tr["signal_bar"] >= sl_time:
            b3.append(tr["id"])
            r3[tr["id"]] = f"signal {tr['signal_bar']} within 30m of SL @ {sl_time}"
    results.append(FilterResult("③ Global SL cooldown (30m)", b3, r3))

    # 4) Regime — block PULLBACK/STOCH LONG if ≥40 pts below session high
    b4, r4 = [], {}
    for tr in TRADES:
        if tr["strategy"] in ("PULLBACK", "STOCH_CROSS") and tr["direction"] == "LONG":
            i = idx_for_time(tr["signal_bar"])
            dist = sess_hi[i] - rows[i]["c"]
            if dist >= 40:
                b4.append(tr["id"])
                r4[tr["id"]] = f"{dist:.0f} pts below session high {sess_hi[i]:.1f}"
    results.append(FilterResult("④ ≥40 pts below session high", b4, r4))

    # 5) Lower-highs intraday structure
    b5, r5 = [], {}
    for tr in TRADES:
        i = idx_for_time(tr["signal_bar"])
        if tr["direction"] == "LONG" and lh[i]:
            b5.append(tr["id"])
            r5[tr["id"]] = "lower-high structure after morning peak"
    results.append(FilterResult("⑤ Lower highs since peak", b5, r5))

    return results


def main():
    twap = twap_vwap_proxy(candles)
    sess_hi = session_high_so_far(candles)
    lh = lower_highs_since_peak(candles)

    t2_sim = simulate_trade2_exit(candles, idx_for_time("12:15"))
    TRADES[1].update(t2_sim)
    if t2_sim["exit_reason"] == "SL":
        TRADES[1]["pnl"] = t2_sim["pnl"]

    actual_pnl = TRADES[0]["pnl"] + (TRADES[1]["pnl"] or 0)

    filters = apply_filters(candles, twap, sess_hi, lh)

    print("=" * 72)
    print("COUNTERFACTUAL — 2026-06-15 (paper, started 11:43)")
    print("=" * 72)
    print(f"Session high (from available candles): {max(sess_hi):.1f}")
    print(f"Candle window: {candles[0]['t']} → {candles[-1]['t']} ({len(candles)} bars)")
    print()
    print("ACTUAL TRADES")
    for t in TRADES:
        pnl = t["pnl"]
        print(
            f"  #{t['id']} {t['strategy']:12} {t['signal_bar']} bar | "
            f"entry {t['entry_index']:.1f} | PnL Rs {pnl:,.0f} | {t.get('exit_reason','?')}"
            + (f" @ {t.get('exit_time')}" if t.get("exit_time") else "")
        )
    print(f"\n  Actual total PnL: Rs {actual_pnl:,.0f}")
    print(f"  Capital: Rs 21,501 → Rs {21501.11 + actual_pnl:,.0f}")

    print("\n" + "-" * 72)
    print("FILTER-BY-FILTER IMPACT (each tested alone)")
    print("-" * 72)

    for fr in filters:
        saved = sum(TRADES[tid - 1]["pnl"] for tid in fr.blocked if TRADES[tid - 1]["pnl"])
        remaining = [t for t in TRADES if t["id"] not in fr.blocked]
        rem_pnl = sum(t["pnl"] for t in remaining)
        print(f"\n{fr.name}")
        if fr.blocked:
            for tid in fr.blocked:
                print(f"  BLOCK trade #{tid}: {fr.reasons[tid]}")
            print(f"  → Avoided loss: Rs {abs(saved):,.0f} | Day PnL would be: Rs {rem_pnl:,.0f}")
        else:
            print("  No trades blocked")

    # Combined (union of all filters)
    blocked_any = set()
    for fr in filters:
        blocked_any.update(fr.blocked)
    comb_pnl = sum(t["pnl"] for t in TRADES if t["id"] not in blocked_any)
    comb_saved = sum(t["pnl"] for t in TRADES if t["id"] in blocked_any)

    print("\n" + "=" * 72)
    print("ALL FIVE FILTERS COMBINED")
    print("=" * 72)
    if blocked_any:
        print(f"  Blocked trades: {sorted(blocked_any)}")
        print(f"  Avoided loss:   Rs {abs(comb_saved):,.0f}")
        print(f"  Day PnL:        Rs {comb_pnl:,.0f}  (vs actual Rs {actual_pnl:,.0f})")
        print(f"  Capital EOD:    Rs {21501.11 + comb_pnl:,.0f}  (vs actual Rs {21501.11 + actual_pnl:,.0f})")
    else:
        print("  No trades blocked")

    # Context at each signal
    print("\n" + "-" * 72)
    print("MARKET CONTEXT AT SIGNAL BARS")
    print("-" * 72)
    for tr in TRADES:
        i = idx_for_time(tr["signal_bar"])
        print(
            f"  #{tr['id']} @ {tr['signal_bar']}: close={candles[i]['c']:.1f} | "
            f"TWAP={twap[i]:.1f} ({'below' if candles[i]['c']<twap[i] else 'above'}) | "
            f"vs high -{sess_hi[i]-candles[i]['c']:.0f} pts | "
            f"HTF/LTF {tr['htf_rsi']}/{tr['ltf_rsi']}"
        )


if __name__ == "__main__":
    main()
