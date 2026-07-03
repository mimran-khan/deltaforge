"""Generate a single self-contained HTML trade report from trades.db + events.jsonl.

Outputs docs/index.html -- premium interactive dashboard with TradingView charts,
glassmorphism design, and animated gradients.

Usage:
    python -m reports.generate          # from project root
    python reports/generate.py          # direct
"""
from __future__ import annotations

import base64
import calendar
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, date
from html import escape
from pathlib import Path
from typing import Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")

STRATEGY_DISPLAY = {
    "TREND_RIDE_0": "Trend Ride",
    "TREND_RIDE": "Trend Ride v2",
    "PULLBACK": "Pullback",
    "PULLBACK_1": "Pullback Aggressive",
    "PULLBACK_2": "Pullback Conservative",
    "PULLBACK_3": "Pullback Momentum",
    "PULLBACK_4": "Pullback Reversal",
    "SUPERTREND": "SuperTrend",
    "ADX_BREAKOUT_0": "ADX Breakout",
    "STOCH_CROSS_0": "Stochastic Cross",
    "EMA_MOMENTUM": "EMA Momentum",
}


def strategy_name(raw: str) -> str:
    return STRATEGY_DISPLAY.get(raw, raw.replace("_", " ").title())


def _bg_b64() -> str:
    if BG_IMAGE.exists():
        return base64.b64encode(BG_IMAGE.read_bytes()).decode()
    return ""


BASE_DIR = Path(__file__).resolve().parent.parent
PROD_DATA_DIR = Path(os.environ.get("DELTAFORGE_DATA_DIR", str(Path.home() / "TradingAgent" / "data")))
DATA_DIR = PROD_DATA_DIR if PROD_DATA_DIR.exists() else BASE_DIR / "data"
DB_PATH = DATA_DIR / "trades.db"
EVENTS_FILE = DATA_DIR / "events.jsonl"
CAPITAL_FILE = DATA_DIR / "capital.json"
OUTPUT_FILE = BASE_DIR / "docs" / "index.html"
BG_IMAGE = BASE_DIR / "docs" / "banner.jpg"
HOLIDAYS_FILE = BASE_DIR / "config" / "holidays.json"


def _load_holiday_dates() -> list[str]:
    if HOLIDAYS_FILE.exists():
        data = json.loads(HOLIDAYS_FILE.read_text())
        return [h["date"] for h in data.get("holidays", [])]
    return []


# ── Data Loading ─────────────────────────────────────────────

def load_trades() -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM trades ORDER BY date ASC, time ASC, id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_events() -> list[dict]:
    events = []
    event_files = sorted(DATA_DIR.glob("events*.jsonl"))
    for ef in event_files:
        if ef.stat().st_size == 0:
            continue
        for line in ef.read_text().strip().splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return events


def load_capital() -> dict:
    if not CAPITAL_FILE.exists():
        return {}
    try:
        return json.loads(CAPITAL_FILE.read_text())
    except Exception:
        return {}


def match_events_to_trades(
    trades: list[dict], events: list[dict]
) -> dict[int, dict]:
    entries = [e for e in events if e.get("event") == "PAPER_ENTRY"]
    exits = [e for e in events if e.get("event") == "PAPER_EXIT"]
    used_entries: set[int] = set()
    used_exits: set[int] = set()
    trade_events: dict[int, dict] = {}

    for trade in trades:
        tid = trade["id"]
        t_date = trade["date"]
        t_time = trade.get("time", "")[:5]
        t_strategy = trade.get("strategy", "")
        t_direction = trade.get("direction", "")

        entry_ev = None
        for idx, ev in enumerate(entries):
            if idx in used_entries:
                continue
            ev_ts = ev.get("ts", "")
            if not ev_ts.startswith(t_date):
                continue
            if ev.get("signal_type", "") in t_strategy and ev.get("direction", "") == t_direction:
                entry_ev = ev
                used_entries.add(idx)
                break

        exit_ev = None
        for idx, ev in enumerate(exits):
            if idx in used_exits:
                continue
            ev_ts = ev.get("ts", "")
            if not ev_ts.startswith(t_date):
                continue
            if ev.get("signal_type", "") in t_strategy and ev.get("direction", "") == t_direction:
                exit_ev = ev
                used_exits.add(idx)
                break

        trade_events[tid] = {"entry": entry_ev or {}, "exit": exit_ev or {}}
    return trade_events


def load_day_events(events: list[dict]) -> dict[str, dict]:
    days: dict[str, dict] = {}
    for ev in events:
        ts = ev.get("ts", "")
        d = ts[:10] if len(ts) >= 10 else ""
        if not d:
            continue
        if ev.get("event") == "DAY_START":
            days.setdefault(d, {})["start"] = ev
        elif ev.get("event") == "DAY_END":
            days.setdefault(d, {})["end"] = ev
    return days


# ── Stats Computation ────────────────────────────────────────

def compute_overall(trades: list[dict], capital: dict) -> dict:
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "wr": 0,
            "total_pnl": 0, "best": 0, "worst": 0, "avg_pnl": 0,
            "profit_factor": 0, "current_capital": 0, "peak_capital": 0,
            "max_dd": 0, "trading_days": 0, "strategies": [],
        }

    wins = [t for t in trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl") or 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    days = sorted(set(t["date"] for t in trades))

    strats: dict[str, dict] = {}
    for t in trades:
        s = t.get("strategy", "UNKNOWN")
        b = strats.setdefault(s, {"n": 0, "w": 0, "pnl": 0.0})
        b["n"] += 1
        b["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            b["w"] += 1

    strategy_list = [
        {"name": s, "trades": d["n"], "wins": d["w"],
         "wr": round(d["w"] / d["n"] * 100, 1) if d["n"] else 0,
         "pnl": round(d["pnl"], 2)}
        for s, d in sorted(strats.items(), key=lambda x: x[1]["pnl"], reverse=True)
    ]

    return {
        "total_trades": len(trades), "wins": len(wins), "losses": len(losses),
        "wr": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl": round(total_pnl, 2),
        "best": round(max((t.get("pnl", 0) for t in trades), default=0), 2),
        "worst": round(min((t.get("pnl", 0) for t in trades), default=0), 2),
        "avg_pnl": round(total_pnl / len(trades), 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "current_capital": capital.get("current_capital", 0),
        "peak_capital": capital.get("peak_capital", 0),
        "max_dd": capital.get("max_drawdown", 0),
        "trading_days": len(days), "strategies": strategy_list,
    }


def _pnl_class(pnl: float) -> str:
    return "pos" if pnl > 0 else ("neg" if pnl < 0 else "")


def _pnl_sign(pnl: float) -> str:
    return f"+{pnl:,.2f}" if pnl > 0 else f"{pnl:,.2f}"


def _build_strat_cards(strategies: list[dict]) -> str:
    cards = ""
    for s in strategies[:5]:
        pnl = s["pnl"]
        cls = "pos" if pnl > 0 else "neg"
        arrow = "&#9650;" if pnl > 0 else "&#9660;"
        cards += (
            f'      <div class="glass sc-card">'
            f'<div class="sc-name">{strategy_name(s["name"])}</div>'
            f'<div class="sc-pnl {cls}">{_pnl_sign(pnl)} <span class="kpi-arrow {"up" if pnl > 0 else "down"}">{arrow}</span></div>'
            f'<div class="sc-meta">{s["trades"]} trades &middot; {s["wr"]}% WR</div>'
            f'</div>\n'
        )
    return cards


def _last_day_label(trades: list[dict]) -> str:
    if not trades:
        return "N/A"
    last_date = max(t["date"] for t in trades)
    try:
        d = datetime.strptime(last_date, "%Y-%m-%d")
        return d.strftime("%A, %b %d")
    except (ValueError, TypeError):
        return last_date


def _last_day_stats(trades: list[dict]) -> str:
    if not trades:
        return "<span>No trades</span>"
    last_date = max(t["date"] for t in trades)
    day_trades = [t for t in trades if t["date"] == last_date]
    pnl = sum(t.get("pnl", 0) for t in day_trades)
    wins = sum(1 for t in day_trades if t.get("pnl", 0) > 0)
    losses = len(day_trades) - wins
    cls = "pos" if pnl > 0 else "neg"
    arrow = "&#9650;" if pnl >= 0 else "&#9660;"
    return (
        f'<span class="ld-pnl {cls}">{_pnl_sign(pnl)} <span class="kpi-arrow {"up" if pnl >= 0 else "down"}">{arrow}</span></span>'
        f'<span class="ld-detail">{len(day_trades)} trades &middot; {wins}W / {losses}L</span>'
    )


def _recent_trades_html(trades: list[dict]) -> str:
    rows = ""
    for t in trades[-5:][::-1]:
        pnl = t.get("pnl", 0)
        cls = "pos" if pnl > 0 else "neg"
        arrow = "&#9650;" if pnl > 0 else "&#9660;"
        time_str = ""
        try:
            dt = datetime.fromisoformat(t.get("entry_time", ""))
            time_str = dt.strftime("%b %d, %H:%M")
        except (ValueError, TypeError):
            time_str = t.get("date", "")
        rows += (
            f'<div class="rt-row">'
            f'<span class="rt-time">{time_str}</span>'
            f'<span class="rt-strat">{strategy_name(t.get("strategy", ""))}</span>'
            f'<span class="rt-dir">{t.get("direction", "")}</span>'
            f'<span class="rt-pnl {cls}">{_pnl_sign(pnl)} <span class="kpi-arrow {"up" if pnl > 0 else "down"}">{arrow}</span></span>'
            f'</div>\n'
        )
    return rows


# ── HTML Generation ──────────────────────────────────────────

def generate_html(trades: list[dict], events: list[dict], capital: dict) -> str:
    trade_events = match_events_to_trades(trades, events)
    day_events = load_day_events(events)
    overall = compute_overall(trades, capital)
    starting_cap = capital.get("initial_capital") or capital.get("peak_capital") or 10000
    now = datetime.now(IST)

    # Build equity curve data — one point per day (end-of-day value)
    # Account for capital injections
    injections: dict[str, float] = {}
    for inj in capital.get("capital_injections", []):
        injections[inj["date"]] = injections.get(inj["date"], 0) + inj["amount"]

    running = starting_cap
    daily_equity: dict[str, float] = {}
    injection_applied: set = set()
    for t in sorted(trades, key=lambda x: x["date"]):
        d = t["date"]
        if d in injections and d not in injection_applied:
            running += injections[d]
            injection_applied.add(d)
        running += t.get("pnl", 0)
        daily_equity[d] = round(running, 2)
    equity_data = [{"time": d, "value": v} for d, v in sorted(daily_equity.items())]

    # Daily P&L for bar chart
    daily_pnl: dict[str, float] = defaultdict(float)
    daily_n: dict[str, int] = defaultdict(int)
    for t in trades:
        d = t.get("date", "")
        if d:
            daily_pnl[d] += t.get("pnl", 0)
            daily_n[d] += 1
    daily_bars = [{"time": d, "value": round(v, 2), "color": "rgba(16,185,129,0.8)" if v >= 0 else "rgba(239,68,68,0.8)"} for d, v in sorted(daily_pnl.items())]

    # Trade log data
    by_date: dict[str, list[dict]] = {}
    for t in trades:
        by_date.setdefault(t["date"], []).append(t)
    sorted_dates = sorted(by_date.keys(), reverse=True)

    trade_log_html = ""
    for i, d in enumerate(sorted_dates):
        day_trades = by_date[d]
        day_pnl_val = sum(t.get("pnl", 0) for t in day_trades)
        wins = sum(1 for t in day_trades if t.get("pnl", 0) > 0)
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            day_label = dt.strftime("%a, %b %d")
        except ValueError:
            day_label = d
        open_cls = " open" if i == 0 else ""
        rows = ""
        for t in day_trades:
            tp = t.get("pnl", 0)
            time_str = (t.get("time", "") or "")[:5]
            strat = t.get("strategy", "")
            direction = t.get("direction", "")
            dir_cls = "long" if direction == "LONG" else "short"
            exit_r = (t.get("exit_reason", "") or "")[:6]
            entry_p = t.get("entry_price", 0)
            exit_p = t.get("exit_price", 0)
            lots = t.get("lots", 1) or 1
            hold = t.get("hold_bars", "")
            hold_str = f"{hold}b" if hold else ""
            pnl_cls = _pnl_class(tp)
            rows += f'<div class="trow"><span class="t-time">{time_str}</span><span class="t-strat">{strategy_name(strat)}</span><span class="t-dir {dir_cls}">{direction}</span><span class="t-price">{entry_p:.1f} &rarr; {exit_p:.1f}</span><span class="t-exit">{escape(exit_r)}</span><span class="t-lots">{lots}L</span><span class="t-pnl {pnl_cls}">{_pnl_sign(tp)}</span></div>'

        trade_log_html += f'''<div class="day-group{open_cls}" data-date="{d}">
<div class="day-header">
  <span class="dh-date">{day_label}</span>
  <span class="dh-meta">{len(day_trades)} trades &middot; {wins}W {len(day_trades)-wins}L</span>
  <span class="dh-pnl {_pnl_class(day_pnl_val)}">{_pnl_sign(day_pnl_val)}</span>
</div>
<div class="day-body">{rows}</div>
</div>'''

    # Calendar — slider-based, one month at a time with navigation
    cal_months_data = []
    if daily_pnl:
        all_dates = sorted(daily_pnl.keys())
        first = datetime.strptime(all_dates[0], "%Y-%m-%d")
        last = datetime.strptime(all_dates[-1], "%Y-%m-%d")
        abs_max = max(abs(v) for v in daily_pnl.values()) or 1
        months = []
        cur = first.replace(day=1)
        while cur <= last:
            months.append((cur.year, cur.month))
            cur = cur.replace(year=cur.year + (1 if cur.month == 12 else 0), month=(cur.month % 12) + 1)

        for idx, (year, month) in enumerate(months):
            month_name = calendar.month_name[month]
            cal_weeks = calendar.monthcalendar(year, month)
            m_pnl = sum(v for dd, v in daily_pnl.items() if dd.startswith(f"{year}-{month:02d}"))
            m_trades = sum(1 for t in trades if t.get("date", "").startswith(f"{year}-{month:02d}"))
            active = " active" if idx == len(months) - 1 else ""
            cells = ""
            for hdr in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
                cells += f'<div class="cal-hdr">{hdr}</div>'
            for week in cal_weeks:
                for dow, day_num in enumerate(week):
                    if day_num == 0:
                        cells += '<div class="cal-cell"></div>'
                        continue
                    date_str = f"{year}-{month:02d}-{day_num:02d}"
                    if dow >= 5:
                        cells += f'<div class="cal-cell off"><span class="cn">{day_num}</span></div>'
                    elif date_str in daily_pnl:
                        pnl = daily_pnl[date_str]
                        intensity = min(abs(pnl) / abs_max, 1)
                        if pnl > 0:
                            bg = f"rgba(16,185,129,{0.15 + intensity * 0.4})"
                        else:
                            bg = f"rgba(239,68,68,{0.15 + intensity * 0.4})"
                        pnl_str = f"+{pnl/1000:.1f}k" if pnl > 0 else f"{pnl/1000:.1f}k"
                        cells += f'<div class="cal-cell traded" style="background:{bg}" title="{date_str}"><span class="cn">{day_num}</span><span class="cv {_pnl_class(pnl)}">{pnl_str}</span></div>'
                    else:
                        cells += f'<div class="cal-cell"><span class="cn">{day_num}</span></div>'
            cal_months_data.append({
                "html": cells,
                "name": f"{month_name} {year}",
                "pnl": m_pnl,
                "trades": m_trades,
                "active": active,
            })

    cal_html = '<div class="cal-slider">'
    cal_html += '<div class="cal-nav"><button class="cal-arrow" id="cal-prev">&larr;</button><div class="cal-nav-title" id="cal-title"></div><button class="cal-arrow" id="cal-next">&rarr;</button></div>'
    for i, m in enumerate(cal_months_data):
        cal_html += f'<div class="cal-slide{m["active"]}" data-idx="{i}" data-name="{m["name"]}" data-pnl="{m["pnl"]:.0f}" data-trades="{m["trades"]}"><div class="cal-grid">{m["html"]}</div></div>'
    cal_html += '</div>'

    # Strategy table
    strat_rows = ""
    for s in overall["strategies"]:
        avg = s["pnl"] / s["trades"] if s["trades"] else 0
        wr_color = "var(--green)" if s["wr"] >= 50 else "var(--red)"
        strat_rows += f'<tr><td class="s-name">{strategy_name(s["name"])}<div class="strat-bar"><div class="strat-bar-fill" style="width:{s["wr"]}%;background:{wr_color}"></div></div></td><td>{s["trades"]}</td><td>{s["wins"]}</td><td><span style="color:{wr_color}">{s["wr"]}%</span></td><td class="{_pnl_class(s["pnl"])}">{_pnl_sign(s["pnl"])}</td><td class="{_pnl_class(avg)}">{_pnl_sign(avg)}</td></tr>'

    # Filter options
    filter_opts = "".join(f'<option value="{strategy_name(s["name"])}">{strategy_name(s["name"])}</option>' for s in overall["strategies"])

    # Max streaks
    streak_w = streak_l = cur_w = cur_l = 0
    for d in sorted(daily_pnl.keys()):
        if daily_pnl[d] > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        streak_w = max(streak_w, cur_w)
        streak_l = max(streak_l, cur_l)

    # Avg hold
    bars_list = [t.get("hold_bars", 0) for t in trades if t.get("hold_bars")]
    avg_hold = f"{sum(bars_list)/len(bars_list):.1f}" if bars_list else "—"

    # Ticker — last 15 trades scrolling
    ticker_items = ""
    for t in trades[-15:]:
        tp = t.get("pnl", 0)
        cls = "pos" if tp > 0 else "neg"
        ticker_items += f'<div class="ticker-item"><span class="ti-name">{strategy_name(t.get("strategy", ""))}</span><span class="ti-pnl {cls}">{_pnl_sign(tp)}</span></div>'
    ticker_html = ticker_items + ticker_items

    # Drawdown data
    peak_eq = starting_cap
    dd_data = []
    running_eq = starting_cap
    for d in sorted(daily_equity.keys()):
        v = daily_equity[d]
        if v > peak_eq:
            peak_eq = v
        dd_pct = ((v - peak_eq) / peak_eq) * 100 if peak_eq > 0 else 0
        dd_data.append({"time": d, "value": round(dd_pct, 2)})

    # Win rate over time (rolling 5-trade window)
    wr_data = []
    wins_running = 0
    for i, t in enumerate(sorted(trades, key=lambda x: x["date"])):
        if t.get("pnl", 0) > 0:
            wins_running += 1
        wr = round((wins_running / (i + 1)) * 100, 1)
        wr_data.append({"time": t["date"], "value": wr})
    # Deduplicate to last per day
    wr_daily: dict[str, float] = {}
    for w in wr_data:
        wr_daily[w["time"]] = w["value"]
    wr_data_clean = [{"time": d, "value": v} for d, v in sorted(wr_daily.items())]

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DeltaForge — Performance Dashboard</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23060910'/%3E%3Cpath d='M6 22 L12 16 L18 19 L26 10' stroke='%238b5cf6' stroke-width='2.5' fill='none' stroke-linecap='round'/%3E%3Ccircle cx='26' cy='10' r='2.5' fill='%2306b6d4'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --bg:#060910;--surface:rgba(15,20,30,0.7);--glass:rgba(255,255,255,0.03);
  --border:rgba(255,255,255,0.06);--border-h:rgba(255,255,255,0.12);
  --text:#f1f5f9;--text2:#94a3b8;--text3:#64748b;--text4:#334155;
  --green:#10b981;--green-l:#6ee7b7;--red:#ef4444;--red-l:#fca5a5;
  --accent:#8b5cf6;--accent2:#06b6d4;
  --mono:'JetBrains Mono',monospace;--sans:'Inter',system-ui,sans-serif;
}}
html{{font-size:14px;scroll-behavior:smooth}}
body{{font-family:var(--sans);background:var(--bg);color:var(--text);overflow-x:hidden;min-height:100vh}}

/* Sticky header zone (topbar + ticker) */
.header-fixed{{position:sticky;top:0;z-index:10;background:var(--bg)}}
/* Hero banner image — overview only */
.hero-banner{{position:relative;overflow:hidden;border-radius:16px;border:1px solid var(--border);box-shadow:0 4px 24px rgba(0,0,0,0.3);margin-bottom:24px}}
.hero-banner img{{width:100%;height:180px;object-fit:cover;object-position:center 35%;display:block}}
.hero-banner::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:70px;background:linear-gradient(to top,var(--bg) 0%,transparent 100%)}}

/* Animated background particles */
.bg-canvas{{position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none}}

/* Animated glow orbs */
.bg-glow{{position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;overflow:hidden}}
.orb{{position:absolute;border-radius:50%;filter:blur(100px);opacity:0.3;animation:float 25s ease-in-out infinite}}
.orb-1{{width:700px;height:700px;background:radial-gradient(circle,rgba(139,92,246,0.3),transparent 70%);top:-250px;left:-150px;animation-delay:0s}}
.orb-2{{width:600px;height:600px;background:radial-gradient(circle,rgba(6,182,212,0.25),transparent 70%);bottom:-200px;right:-150px;animation-delay:-8s}}
.orb-3{{width:500px;height:500px;background:radial-gradient(circle,rgba(16,185,129,0.18),transparent 70%);top:35%;left:55%;animation-delay:-16s}}
.orb-4{{width:350px;height:350px;background:radial-gradient(circle,rgba(244,63,94,0.15),transparent 70%);top:70%;left:10%;animation-delay:-12s}}
@keyframes float{{0%,100%{{transform:translate(0,0) scale(1)}}25%{{transform:translate(20px,-40px) scale(1.05)}}50%{{transform:translate(-15px,25px) scale(0.95)}}75%{{transform:translate(30px,10px) scale(1.02)}}}}

/* Glass morphism */
.glass{{background:rgba(12,16,28,0.6);backdrop-filter:blur(20px) saturate(1.2);-webkit-backdrop-filter:blur(20px) saturate(1.2);border:1px solid var(--border);border-radius:16px;transition:border-color 0.25s,box-shadow 0.25s,transform 0.25s}}
.glass:hover{{border-color:var(--border-h);box-shadow:0 8px 40px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.04)}}

/* Layout */
.app{{position:relative;z-index:1;max-width:1280px;margin:0 auto;padding:24px 32px 48px}}
.topbar{{display:flex;justify-content:space-between;align-items:center;padding:16px 24px;margin-bottom:0;background:rgba(6,9,16,0.95);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);border-radius:0}}
.refresh-counter{{font-size:0.72rem;color:var(--muted);background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:8px;padding:6px 12px;display:flex;align-items:center;gap:6px;white-space:nowrap}}
.refresh-counter .rc-dot{{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}}
.refresh-counter .rc-label{{opacity:0.7}}
.refresh-counter .rc-time{{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}}

/* Ticker */
.ticker{{overflow:hidden;margin-bottom:0;padding:8px 0;background:rgba(6,9,16,0.9);border-bottom:1px solid var(--border);position:relative}}
.ticker::before,.ticker::after{{content:'';position:absolute;top:0;bottom:0;width:60px;z-index:2}}
.ticker::before{{left:0;background:linear-gradient(90deg,var(--bg),transparent)}}
.ticker::after{{right:0;background:linear-gradient(270deg,var(--bg),transparent)}}
.ticker-track{{display:flex;gap:32px;animation:scroll 30s linear infinite;width:max-content}}
.ticker-item{{display:flex;align-items:center;gap:8px;font-size:0.72rem;font-family:var(--mono);white-space:nowrap;color:var(--text3)}}
.ticker-item .ti-name{{color:var(--text2);font-weight:500}}
.ticker-item .ti-pnl{{font-weight:700}}
@keyframes scroll{{0%{{transform:translateX(0)}}100%{{transform:translateX(-50%)}}}}
.logo{{font-size:1.6rem;font-weight:800;letter-spacing:-0.03em;
  background:linear-gradient(135deg,#fff 0%,#8b5cf6 50%,#06b6d4 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.meta{{font-size:0.78rem;color:var(--text2);font-family:var(--mono);display:flex;align-items:center;gap:12px}}
.live-dot{{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:0.4;transform:scale(0.85)}}}}

/* Navigation */
.nav{{display:flex;gap:4px;padding:4px;background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:12px;margin-bottom:24px}}
.nav-btn{{flex:1;padding:10px 16px;border:none;background:transparent;color:var(--text3);font-size:0.8rem;font-weight:600;cursor:pointer;border-radius:8px;font-family:var(--sans);transition:all 0.2s}}
.nav-btn:hover{{color:var(--text);background:rgba(255,255,255,0.04)}}
.nav-btn.active{{color:#fff;background:linear-gradient(135deg,rgba(139,92,246,0.35),rgba(6,182,212,0.25));border:1px solid rgba(139,92,246,0.4);box-shadow:0 4px 20px rgba(139,92,246,0.2),inset 0 1px 0 rgba(255,255,255,0.1)}}

/* KPI Cards */
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px}}
.kpi{{padding:20px 22px;position:relative;overflow:hidden}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),var(--accent2),transparent);opacity:0.5}}
.kpi::after{{content:'';position:absolute;top:0;right:0;width:60px;height:60px;background:radial-gradient(circle,rgba(139,92,246,0.06),transparent 70%);pointer-events:none}}
.kpi-label{{font-size:0.65rem;color:var(--text2);text-transform:uppercase;letter-spacing:0.08em;font-weight:500;margin-bottom:8px}}
.kpi-val{{font-size:1.5rem;font-weight:700;font-family:var(--mono);line-height:1.2;color:var(--text)}}
.kpi-sub{{font-size:0.7rem;color:var(--text3);font-family:var(--mono);margin-top:6px}}
.kpi.hero{{grid-column:span 2}}
.kpi.hero .kpi-val{{font-size:2.4rem;background:linear-gradient(135deg,var(--green-l),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.kpi.hero-neg{{grid-column:span 2}}
.kpi.hero-neg .kpi-val{{font-size:2.4rem;background:linear-gradient(135deg,var(--red-l),var(--red));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}

/* Panels */
.panel{{display:none}}.panel.active{{display:block;animation:fadeUp 0.25s ease}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}

/* Strategy cards row */
.strat-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}}
.sc-card{{padding:16px;border-radius:12px;display:flex;flex-direction:column;gap:4px}}
.sc-name{{font-size:0.72rem;color:var(--text2);font-weight:600;text-transform:uppercase;letter-spacing:0.04em}}
.sc-pnl{{font-size:1.1rem;font-weight:700;font-family:var(--mono)}}
.sc-meta{{font-size:0.65rem;color:var(--text4);font-family:var(--mono)}}

/* Last day summary */
.last-day{{padding:16px 24px;border-radius:12px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between}}
.ld-head{{font-size:0.78rem;font-weight:600;color:var(--text2)}}
.ld-stats{{display:flex;align-items:center;gap:16px}}
.ld-pnl{{font-size:1.1rem;font-weight:700;font-family:var(--mono)}}
.ld-detail{{font-size:0.7rem;color:var(--text4);font-family:var(--mono)}}

/* Recent trades strip */
.recent-trades{{padding:20px;border-radius:14px;margin-top:20px}}
.rt-list{{display:flex;flex-direction:column;gap:8px;margin-top:12px}}
.rt-row{{display:grid;grid-template-columns:120px 1fr 60px 100px;align-items:center;padding:10px 14px;background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:8px;font-size:0.75rem;font-family:var(--mono);transition:background 0.15s}}
.rt-row:hover{{background:rgba(139,92,246,0.04)}}
.rt-time{{color:var(--text3)}}
.rt-strat{{color:var(--text);font-weight:500}}
.rt-dir{{color:var(--text2);font-size:0.65rem;text-transform:uppercase}}
.rt-pnl{{text-align:right;font-weight:600}}


/* Chart containers */
.chart-wrap{{margin-bottom:20px;padding:20px;overflow:hidden}}
.chart-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}}
.chart-title{{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text3);font-weight:600}}
.chart-sub{{font-size:0.68rem;color:var(--text4);font-family:var(--mono)}}
.chart-container{{width:100%;height:280px;border-radius:8px;overflow:hidden}}
.chart-container-sm{{height:180px}}

/* Trade Log */
.trades-summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}}
.trades-summary .glass{{padding:16px 20px;text-align:center}}
.trades-summary .ts-val{{font-size:1.3rem;font-weight:700;font-family:var(--mono)}}
.trades-summary .ts-label{{font-size:0.65rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.06em;margin-top:4px}}
.day-group{{margin-bottom:10px}}
.day-header{{display:grid;grid-template-columns:140px 1fr 130px;align-items:center;padding:16px 22px;cursor:pointer;border-radius:14px;background:var(--glass);backdrop-filter:blur(8px);border:1px solid var(--border);transition:all 0.2s}}
.day-header:hover{{border-color:var(--border-h);background:rgba(255,255,255,0.06);transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,0,0,0.3)}}
.dh-date{{font-weight:600;font-size:0.9rem}}
.dh-meta{{font-size:0.75rem;color:var(--text3);display:flex;gap:12px;align-items:center}}
.dh-pnl{{font-family:var(--mono);font-weight:700;text-align:right;font-size:1rem}}
.day-body{{display:none;padding:14px 22px 18px;margin-top:-6px;background:rgba(0,0,0,0.4);backdrop-filter:blur(4px);border:1px solid var(--border);border-top:none;border-radius:0 0 14px 14px}}
.day-group.open .day-body{{display:block;animation:fadeUp 0.2s ease}}
.day-group.open .day-header{{border-radius:14px 14px 0 0;border-bottom-color:transparent;background:rgba(255,255,255,0.05)}}
.trow{{display:grid;grid-template-columns:55px 120px 60px 1fr 55px 40px 100px;gap:8px;align-items:center;padding:10px 4px;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.78rem;font-family:var(--mono);color:var(--text2);border-radius:6px;transition:background 0.1s}}
.trow:hover{{background:rgba(255,255,255,0.03)}}
.trow:last-child{{border-bottom:none}}
.t-time{{color:var(--text3);font-size:0.72rem}}
.t-strat{{font-family:var(--sans);font-weight:600;color:var(--text);font-size:0.76rem}}
.t-dir{{font-size:0.65rem;font-weight:700;padding:3px 8px;border-radius:5px;text-align:center;letter-spacing:0.02em}}
.t-dir.long{{color:var(--green-l);background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.25)}}
.t-dir.short{{color:var(--red-l);background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.25)}}
.t-price{{color:var(--text3);font-size:0.72rem}}
.t-exit{{color:var(--text4);font-size:0.68rem;padding:2px 6px;background:rgba(255,255,255,0.04);border-radius:4px}}
.t-lots{{color:var(--text4);font-size:0.72rem}}
.t-pnl{{font-weight:700;text-align:right;font-size:0.85rem}}

/* Calendar */
.cal-summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
.cal-summary .glass{{padding:16px 20px;text-align:center}}
.cal-summary .cs-val{{font-size:1.2rem;font-weight:700;font-family:var(--mono)}}
.cal-summary .cs-label{{font-size:0.62rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.06em;margin-top:4px}}
/* Calendar slider */
.cal-slider{{position:relative}}
.cal-nav{{display:flex;align-items:center;justify-content:center;gap:20px;margin-bottom:24px;padding:16px;background:var(--glass);backdrop-filter:blur(8px);border:1px solid var(--border);border-radius:12px}}
.cal-arrow{{background:rgba(255,255,255,0.06);border:1px solid var(--border);color:var(--text);width:36px;height:36px;border-radius:8px;font-size:1.1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.15s}}
.cal-arrow:hover{{background:rgba(139,92,246,0.15);border-color:var(--accent);color:#fff}}
.cal-arrow:disabled{{opacity:0.3;cursor:default}}
.cal-nav-title{{text-align:center;min-width:200px}}
.cal-nav-title .cnt{{font-size:1.1rem;font-weight:700}}
.cal-nav-title .cnsub{{font-size:0.72rem;color:var(--text3);font-family:var(--mono);margin-top:3px}}
.cal-slide{{display:none;padding:28px 20px;animation:fadeUp 0.25s ease}}
.cal-slide.active{{display:block}}
.cal-grid{{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;max-width:700px;margin:0 auto}}
.cal-hdr{{font-size:0.72rem;color:var(--text3);text-align:center;padding:10px 0;font-weight:600}}
.cal-cell{{aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:12px;font-size:0.78rem;color:var(--text4);min-height:72px;border:1px solid transparent;transition:all 0.2s}}
.cal-cell.traded{{cursor:pointer;border:1px solid rgba(255,255,255,0.1);box-shadow:inset 0 0 12px rgba(0,0,0,0.15)}}.cal-cell.traded:hover{{transform:scale(1.06);box-shadow:0 8px 28px rgba(0,0,0,0.5);z-index:2;border-color:rgba(255,255,255,0.25)}}
.cal-cell.off{{opacity:0.15}}
.cal-cell .cn{{font-size:0.78rem;line-height:1;font-weight:600}}
.cal-cell .cv{{font-family:var(--mono);font-size:0.68rem;font-weight:700;margin-top:4px}}

/* Stats */
.stats-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px}}
.stat{{padding:20px;position:relative;overflow:hidden}}
.stat::after{{content:'';position:absolute;bottom:0;left:20px;right:20px;height:1px;background:linear-gradient(90deg,transparent,var(--border-h),transparent)}}
.stat-label{{font-size:0.62rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:8px}}
.stat-val{{font-size:1.3rem;font-weight:700;font-family:var(--mono)}}

/* Strategy table */
.strat-wrap{{padding:24px;border-radius:16px;overflow:hidden}}
.strat-wrap .chart-title{{margin-bottom:18px;font-size:0.78rem}}
.strat-table{{width:100%;border-collapse:separate;border-spacing:0}}
.strat-table th{{text-align:left;padding:12px 16px;font-size:0.65rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.08em;font-weight:600;border-bottom:1px solid var(--border);background:rgba(255,255,255,0.02)}}
.strat-table th:first-child{{border-radius:8px 0 0 0}}
.strat-table th:last-child{{border-radius:0 8px 0 0}}
.strat-table td{{padding:14px 16px;font-family:var(--mono);font-size:0.82rem;color:var(--text2);border-bottom:1px solid rgba(255,255,255,0.04)}}
.strat-table tr{{transition:background 0.1s}}
.strat-table tr:hover td{{background:rgba(139,92,246,0.04)}}
.strat-table tr:last-child td{{border-bottom:none}}
.s-name{{font-family:var(--sans)!important;font-weight:600;color:var(--text)!important;font-size:0.84rem!important}}
.strat-bar{{height:4px;border-radius:2px;margin-top:6px;background:var(--border)}}
.strat-bar-fill{{height:100%;border-radius:2px;transition:width 0.4s ease}}

/* Filters */
.filter-bar{{display:flex;gap:10px;align-items:center;margin-bottom:20px;padding:14px 20px;background:var(--glass);backdrop-filter:blur(8px);border:1px solid var(--border);border-radius:12px;flex-wrap:wrap}}
.filter-bar select{{background:rgba(255,255,255,0.06);border:1px solid var(--border);color:var(--text);padding:9px 14px;border-radius:8px;font-size:0.78rem;font-family:var(--sans);cursor:pointer;transition:all 0.15s;-webkit-appearance:none;appearance:none;padding-right:28px;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center}}
.filter-bar select:hover{{border-color:var(--border-h);background-color:rgba(255,255,255,0.08)}}
.filter-bar select:focus{{outline:none;border-color:var(--accent);box-shadow:0 0 0 2px rgba(139,92,246,0.15)}}
.filter-count{{margin-left:auto;font-size:0.75rem;color:var(--text2);font-family:var(--mono);padding:6px 12px;background:rgba(139,92,246,0.1);border-radius:6px;border:1px solid rgba(139,92,246,0.2)}}

/* Colors */
.pos{{color:var(--green-l)}}.neg{{color:var(--red-l)}}
.kpi-arrow{{display:inline-block;font-size:0.6em;margin-left:6px;vertical-align:middle}}
.kpi-arrow.up{{color:var(--green-l)}}.kpi-arrow.down{{color:var(--red-l)}}
.kpi-info{{position:absolute;top:10px;right:10px;width:20px;height:20px;border-radius:50%;border:1.5px solid rgba(139,92,246,0.5);display:flex;align-items:center;justify-content:center;font-size:0.6rem;color:rgba(139,92,246,0.8);cursor:help;transition:all 0.15s;font-family:var(--sans);font-style:italic;font-weight:700;background:rgba(139,92,246,0.06)}}
.kpi-info:hover{{border-color:var(--accent);color:#fff;background:rgba(139,92,246,0.25)}}
.kpi-info:hover .kpi-tip{{opacity:1;transform:translateX(-50%) translateY(0);pointer-events:auto}}
.kpi-tip{{position:absolute;top:calc(100% + 8px);left:50%;transform:translateX(-50%) translateY(-4px);background:rgba(16,20,32,0.95);backdrop-filter:blur(12px);color:var(--text2);padding:8px 12px;border:1px solid var(--border);border-radius:8px;font-size:0.68rem;font-family:var(--mono);font-style:normal;font-weight:400;white-space:nowrap;pointer-events:none;opacity:0;transition:all 0.2s;z-index:100;box-shadow:0 8px 24px rgba(0,0,0,0.4)}}

/* Footer */
.footer{{position:relative;z-index:1;max-width:1280px;margin:0 auto;padding:40px 32px 32px;border-top:1px solid var(--border)}}
.footer-inner{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px}}
.footer-brand{{display:flex;align-items:center;gap:12px}}
.footer-brand .logo-sm{{font-size:1rem;font-weight:700;background:linear-gradient(135deg,#fff 0%,#8b5cf6 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.footer-brand .tagline{{font-size:0.72rem;color:var(--text4)}}
.footer-links{{display:flex;gap:16px;align-items:center}}
.footer-links a{{color:var(--text3);text-decoration:none;font-size:0.75rem;font-family:var(--mono);display:flex;align-items:center;gap:6px;padding:6px 12px;border-radius:6px;border:1px solid var(--border);transition:all 0.15s}}
.footer-links a:hover{{color:var(--text);border-color:var(--accent);background:rgba(139,92,246,0.08)}}
.footer-links svg{{width:16px;height:16px}}
.footer-copy{{width:100%;text-align:center;font-size:0.65rem;color:var(--text4);margin-top:20px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.03)}}

/* Loading screen — auto-hides via CSS animation */
.loader{{position:fixed;top:0;left:0;width:100%;height:100%;background:var(--bg);z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;animation:loaderHide 0.5s ease 0.8s forwards}}
.loader-logo{{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#fff,#8b5cf6,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:16px}}
.loader-bar{{width:120px;height:3px;background:var(--border);border-radius:3px;overflow:hidden}}
.loader-bar::after{{content:'';display:block;width:40%;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:3px;animation:load 0.8s ease-in-out infinite}}
@keyframes load{{0%{{transform:translateX(-100%)}}100%{{transform:translateX(350%)}}}}
@keyframes loaderHide{{to{{opacity:0;visibility:hidden;pointer-events:none}}}}

/* Page enter animations */
.animate-in{{opacity:0;transform:translateY(16px);transition:opacity 0.5s ease,transform 0.5s ease}}
.animate-in.visible{{opacity:1;transform:translateY(0)}}
.animate-in:nth-child(1){{transition-delay:0.05s}}
.animate-in:nth-child(2){{transition-delay:0.1s}}
.animate-in:nth-child(3){{transition-delay:0.15s}}
.animate-in:nth-child(4){{transition-delay:0.2s}}
.animate-in:nth-child(5){{transition-delay:0.25s}}
.animate-in:nth-child(6){{transition-delay:0.3s}}
.animate-in:nth-child(7){{transition-delay:0.35s}}

/* Counter animation */
.counter{{display:inline-block}}

/* Tooltips */

/* Search */
.search-wrap{{position:relative}}
.search-wrap input{{width:100%;background:rgba(255,255,255,0.04);border:1px solid var(--border);color:var(--text);padding:10px 14px 10px 36px;border-radius:8px;font-size:0.78rem;font-family:var(--sans);transition:all 0.15s}}
.search-wrap input:focus{{outline:none;border-color:var(--accent);background:rgba(255,255,255,0.06);box-shadow:0 0 0 2px rgba(139,92,246,0.1)}}
.search-wrap input::placeholder{{color:var(--text4)}}
.search-wrap svg{{position:absolute;left:12px;top:50%;transform:translateY(-50%);width:14px;height:14px;color:var(--text4)}}

/* Extra chart containers */
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}}
.chart-row .glass{{padding:20px;overflow:hidden}}

/* KPI interactive */
.kpi{{cursor:default}}
.kpi.clickable{{cursor:pointer}}
.kpi.clickable:hover{{transform:translateY(-2px)}}

/* Scrollbar */
::-webkit-scrollbar{{width:6px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:rgba(255,255,255,0.1);border-radius:3px}}
::-webkit-scrollbar-thumb:hover{{background:rgba(255,255,255,0.2)}}

/* Selection */
::selection{{background:rgba(139,92,246,0.3);color:#fff}}

/* Responsive */
@media(max-width:768px){{
  .app{{padding:16px 12px}}
  .topbar{{padding:14px 16px;flex-direction:column;gap:8px;text-align:center}}
  .kpi-grid{{grid-template-columns:1fr 1fr;gap:10px}}
  .kpi.hero,.kpi.hero-neg{{grid-column:span 2}}
  .kpi.hero .kpi-val{{font-size:1.6rem}}
  .kpi{{padding:14px 16px}}
  .trow{{grid-template-columns:45px 80px 45px 1fr 80px;font-size:0.68rem}}
  .t-exit,.t-lots,.t-price{{display:none}}
  .chart-container{{height:180px}}
  .chart-row{{grid-template-columns:1fr}}
  .nav{{overflow-x:auto;flex-wrap:nowrap;-webkit-overflow-scrolling:touch}}
  .nav-btn{{padding:8px 12px;font-size:0.72rem;flex:none;white-space:nowrap}}
  .ticker{{display:none}}
  .stats-row{{grid-template-columns:1fr 1fr}}
  .cal-summary{{grid-template-columns:1fr 1fr}}
  .cal-grid{{gap:4px}}
  .cal-cell{{min-height:48px}}
  .filter-bar{{flex-direction:column;align-items:stretch}}
  .filter-bar select{{width:100%}}
  .filter-count{{margin-left:0;text-align:center}}
  .footer-inner{{flex-direction:column;text-align:center}}
  .footer-links{{justify-content:center}}
  .day-header{{grid-template-columns:100px 1fr 90px;padding:12px 14px}}
  .trades-summary{{grid-template-columns:1fr 1fr 1fr;gap:8px}}
  .strat-table{{font-size:0.7rem}}
  .strat-table th,.strat-table td{{padding:8px 10px}}
}}
</style>
</head>
<body>

<div class="loader" id="loader">
  <div class="loader-logo">DeltaForge</div>
  <div class="loader-bar"></div>
</div>

<canvas class="bg-canvas" id="particles"></canvas>
<div class="bg-glow">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>
  <div class="orb orb-4"></div>
</div>

<div class="app">
  <div class="header-fixed">
    <div class="topbar">
      <span class="logo">DeltaForge</span>
      <span class="meta"><span class="live-dot"></span>Paper Trading &middot; {now.strftime("%b %d, %Y")}</span>
      <span class="refresh-counter" id="refresh-counter"></span>
    </div>
    <div class="ticker"><div class="ticker-track">{ticker_html}</div></div>
  </div>

  <div class="nav" id="nav">
    <button class="nav-btn active" data-p="p-overview">Overview</button>
    <button class="nav-btn" data-p="p-trades">Trades</button>
    <button class="nav-btn" data-p="p-calendar">Calendar</button>
    <button class="nav-btn" data-p="p-analytics">Analytics</button>
  </div>

  <!-- ═══ OVERVIEW ═══ -->
  <div id="p-overview" class="panel active">
    <div class="hero-banner">
      <img src="data:image/jpeg;base64,{_bg_b64()}" alt="">
    </div>
    <div class="kpi-grid">
      <div class="glass kpi animate-in {"hero" if overall["total_pnl"] >= 0 else "hero-neg"}"><div class="kpi-info">i<span class="kpi-tip">Total realized P&amp;L</span></div><div class="kpi-label">Net P&amp;L</div><div class="kpi-val counter" data-target="{overall["total_pnl"]:.0f}">{_pnl_sign(overall["total_pnl"])}<span class="kpi-arrow {"up" if overall["total_pnl"] >= 0 else "down"}">{"&#9650;" if overall["total_pnl"] >= 0 else "&#9660;"}</span></div><div class="kpi-sub">{overall["trading_days"]} days &middot; {_pnl_sign(overall["avg_pnl"])}/trade &middot; <span class="{"pos" if overall["total_pnl"] >= 0 else "neg"}">{overall["total_pnl"] / starting_cap * 100:+.1f}%</span></div></div>
      <div class="glass kpi animate-in"><div class="kpi-info">i<span class="kpi-tip">{overall["wins"]} winners, {overall["losses"]} losers</span></div><div class="kpi-label">Win Rate</div><div class="kpi-val">{overall["wr"]}%<span class="kpi-arrow {"up" if overall["wr"] >= 50 else "down"}">{"&#9650;" if overall["wr"] >= 50 else "&#9660;"}</span></div><div class="kpi-sub">{overall["wins"]}W / {overall["losses"]}L</div></div>
      <div class="glass kpi animate-in"><div class="kpi-info">i<span class="kpi-tip">Gross profit / Gross loss</span></div><div class="kpi-label">Profit Factor</div><div class="kpi-val">{overall["profit_factor"]}<span class="kpi-arrow {"up" if overall["profit_factor"] >= 1 else "down"}">{"&#9650;" if overall["profit_factor"] >= 1 else "&#9660;"}</span></div></div>
      <div class="glass kpi animate-in"><div class="kpi-info">i<span class="kpi-tip">Current account value</span></div><div class="kpi-label">Capital</div><div class="kpi-val counter" data-target="{overall["current_capital"]:.0f}">{overall["current_capital"]:,.0f}<span class="kpi-arrow {"up" if overall["current_capital"] >= starting_cap else "down"}">{"&#9650;" if overall["current_capital"] >= starting_cap else "&#9660;"}</span></div><div class="kpi-sub">peak {overall["peak_capital"]:,.0f} &middot; <span class="{"pos" if overall["current_capital"] >= starting_cap else "neg"}">{(overall["current_capital"] - starting_cap) / starting_cap * 100:+.1f}%</span></div></div>
      <div class="glass kpi animate-in"><div class="kpi-info">i<span class="kpi-tip">Single best trade</span></div><div class="kpi-label">Best Trade</div><div class="kpi-val pos">{overall["best"]:+,.0f}<span class="kpi-arrow up">&#9650;</span></div></div>
      <div class="glass kpi animate-in"><div class="kpi-info">i<span class="kpi-tip">Single worst trade</span></div><div class="kpi-label">Worst Trade</div><div class="kpi-val neg">{overall["worst"]:+,.0f}<span class="kpi-arrow down">&#9660;</span></div></div>
      <div class="glass kpi animate-in"><div class="kpi-info">i<span class="kpi-tip">Peak to trough drawdown</span></div><div class="kpi-label">Max Drawdown</div><div class="kpi-val neg">{overall["max_dd"]:.1f}%<span class="kpi-arrow down">&#9660;</span></div></div>
    </div>

    <!-- Strategy Performance Cards -->
    <div class="strat-cards">
{_build_strat_cards(overall["strategies"])}
    </div>

    <!-- Last Day Summary -->
    <div class="last-day glass">
      <div class="ld-head">Last Session &mdash; {_last_day_label(trades)}</div>
      <div class="ld-stats">{_last_day_stats(trades)}</div>
    </div>

    <div class="glass chart-wrap">
      <div class="chart-head"><span class="chart-title">Equity Curve</span><span class="chart-sub">{starting_cap:,.0f} &rarr; {equity_data[-1]["value"] if equity_data else starting_cap:,.0f}</span></div>
      <div class="chart-container" id="equity-chart"></div>
    </div>

    <div class="glass chart-wrap">
      <div class="chart-head"><span class="chart-title">Daily P&amp;L</span><span class="chart-sub">{len(daily_bars)} sessions</span></div>
      <div class="chart-container chart-container-sm" id="pnl-chart"></div>
    </div>

    <!-- Recent Trades -->
    <div class="glass recent-trades">
      <div class="chart-head"><span class="chart-title">Recent Trades</span><span class="chart-sub">last 5</span></div>
      <div class="rt-list">{_recent_trades_html(trades)}</div>
    </div>
  </div>

  <!-- ═══ TRADES ═══ -->
  <div id="p-trades" class="panel">
    <div class="trades-summary">
      <div class="glass"><div class="ts-val">{len(trades)}</div><div class="ts-label">Total Trades</div></div>
      <div class="glass"><div class="ts-val pos">{overall["wins"]}</div><div class="ts-label">Winners</div></div>
      <div class="glass"><div class="ts-val neg">{overall["losses"]}</div><div class="ts-label">Losers</div></div>
    </div>
    <div class="filter-bar">
      <div class="search-wrap">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
        <input type="text" id="f-search" placeholder="Search trades...">
      </div>
      <select id="f-strat"><option value="">All Strategies</option>{filter_opts}</select>
      <select id="f-dir"><option value="">All Directions</option><option value="LONG">LONG</option><option value="SHORT">SHORT</option></select>
      <select id="f-result"><option value="">All Results</option><option value="win">Winners</option><option value="loss">Losers</option></select>
      <span class="filter-count" id="f-count">{len(trades)} trades</span>
    </div>
    {trade_log_html}
  </div>

  <!-- ═══ CALENDAR ═══ -->
  <div id="p-calendar" class="panel">
    <div class="cal-summary">
      <div class="glass"><div class="cs-val pos">{sum(1 for v in daily_pnl.values() if v > 0)}</div><div class="cs-label">Green Days</div></div>
      <div class="glass"><div class="cs-val neg">{sum(1 for v in daily_pnl.values() if v < 0)}</div><div class="cs-label">Red Days</div></div>
      <div class="glass"><div class="cs-val">{overall["trading_days"]}</div><div class="cs-label">Trading Days</div></div>
      <div class="glass"><div class="cs-val">{overall["wr"]}%</div><div class="cs-label">Day Win Rate</div></div>
    </div>
    <div class="glass" style="padding:24px;border-radius:16px">
      {cal_html}
    </div>
  </div>

  <!-- ═══ ANALYTICS ═══ -->
  <div id="p-analytics" class="panel">
    <div class="stats-row">
      <div class="glass stat"><div class="stat-label">Best Day</div><div class="stat-val pos">{max(daily_pnl.values(), default=0):+,.0f}</div></div>
      <div class="glass stat"><div class="stat-label">Worst Day</div><div class="stat-val neg">{min(daily_pnl.values(), default=0):+,.0f}</div></div>
      <div class="glass stat"><div class="stat-label">Trades / Day</div><div class="stat-val">{len(trades)/max(overall["trading_days"],1):.1f}</div></div>
      <div class="glass stat"><div class="stat-label">Win Streak</div><div class="stat-val pos">{streak_w} days</div></div>
      <div class="glass stat"><div class="stat-label">Loss Streak</div><div class="stat-val neg">{streak_l} days</div></div>
      <div class="glass stat"><div class="stat-label">Avg Hold</div><div class="stat-val">{avg_hold} bars</div></div>
    </div>

    <div class="chart-row">
      <div class="glass">
        <div class="chart-head"><span class="chart-title">Drawdown</span><span class="chart-sub">max {overall["max_dd"]:.1f}%</span></div>
        <div class="chart-container chart-container-sm" id="dd-chart"></div>
      </div>
      <div class="glass">
        <div class="chart-head"><span class="chart-title">Win Rate Over Time</span><span class="chart-sub">current {overall["wr"]}%</span></div>
        <div class="chart-container chart-container-sm" id="wr-chart"></div>
      </div>
    </div>

    <div class="glass chart-wrap">
      <div class="chart-head"><span class="chart-title">P&amp;L Distribution</span><span class="chart-sub">{len(trades)} trades</span></div>
      <div class="chart-container chart-container-sm" id="dist-chart"></div>
    </div>

    <div class="glass strat-wrap">
      <div class="chart-title">Strategy Breakdown</div>
      <table class="strat-table">
        <thead><tr><th>Strategy</th><th>Trades</th><th>Wins</th><th>Win Rate</th><th>Total P&L</th><th>Avg / Trade</th></tr></thead>
        <tbody>{strat_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<footer class="footer">
  <div class="footer-inner">
    <div class="footer-brand">
      <span class="logo-sm">DeltaForge</span>
      <span class="tagline">Algorithmic Trading Engine</span>
    </div>
    <div class="footer-links">
      <a href="https://github.com/mimran-khan/deltaforge" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
        Source Code
      </a>
      <a href="https://github.com/mimran-khan/deltaforge/issues" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>
        Report Issue
      </a>
    </div>
    <div class="footer-copy">
      Built with Python &middot; Powered by TradingView Lightweight Charts &middot; Generated {now.strftime("%b %d, %Y at %H:%M")}
    </div>
  </div>
</footer>

<script>
const eqData = {json.dumps(equity_data)};
const barData = {json.dumps(daily_bars)};
const tradePnls = {json.dumps([round(t.get("pnl", 0), 0) for t in trades])};
const ddData = {json.dumps(dd_data)};
const wrData = {json.dumps(wr_data_clean)};

// ── Next Refresh Counter ──
(function() {{
  const holidays = new Set({json.dumps(_load_holiday_dates())});
  const EOD_HOUR = 15, EOD_MIN = 35;

  function isTradingDay(d) {{
    const day = d.getDay();
    if (day === 0 || day === 6) return false;
    const ds = d.getFullYear() + '-' + String(d.getMonth()+1).padStart(2,'0') + '-' + String(d.getDate()).padStart(2,'0');
    return !holidays.has(ds);
  }}

  function nextRefresh() {{
    const now = new Date();
    const ist = new Date(now.toLocaleString('en-US', {{timeZone:'Asia/Kolkata'}}));
    let target = new Date(ist);
    target.setHours(EOD_HOUR, EOD_MIN, 0, 0);
    if (ist >= target || !isTradingDay(ist)) {{
      target.setDate(target.getDate() + 1);
      while (!isTradingDay(target)) target.setDate(target.getDate() + 1);
      target.setHours(EOD_HOUR, EOD_MIN, 0, 0);
    }}
    return target;
  }}

  function update() {{
    const el = document.getElementById('refresh-counter');
    if (!el) return;
    const now = new Date();
    const ist = new Date(now.toLocaleString('en-US', {{timeZone:'Asia/Kolkata'}}));
    const target = nextRefresh();
    const diff = target - ist;
    if (diff <= 0) {{ el.innerHTML = '<span class="rc-dot"></span><span class="rc-label">Refreshing...</span>'; return; }}
    const h = Math.floor(diff / 3600000);
    const m = Math.floor((diff % 3600000) / 60000);
    const s = Math.floor((diff % 60000) / 1000);
    const timeStr = h > 0 ? h + 'h ' + m + 'm' : m + 'm ' + s + 's';
    el.innerHTML = '<span class="rc-dot"></span><span class="rc-label">Next refresh</span><span class="rc-time">' + timeStr + '</span>';
  }}
  update();
  setInterval(update, 1000);
}})();

// ── TradingView Charts ──
function initCharts() {{
  const eqEl = document.getElementById('equity-chart');
  const pnlEl = document.getElementById('pnl-chart');
  if (!eqEl || !pnlEl || typeof LightweightCharts === 'undefined') return;

  const chartOpts = {{
    layout: {{ background: {{ type: 'solid', color: 'transparent' }}, textColor: '#94a3b8', fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }},
    grid: {{ vertLines: {{ color: 'rgba(255,255,255,0.03)' }}, horzLines: {{ color: 'rgba(255,255,255,0.03)' }} }},
    crosshair: {{ mode: 1 }},
    rightPriceScale: {{ borderColor: 'rgba(255,255,255,0.06)' }},
    timeScale: {{ borderColor: 'rgba(255,255,255,0.06)', timeVisible: false }},
    handleScroll: true, handleScale: true,
  }};

  // Equity curve — locked axes, no scroll/zoom
  const eq = LightweightCharts.createChart(eqEl, {{
    ...chartOpts, width: eqEl.clientWidth, height: eqEl.clientHeight,
    handleScroll: false, handleScale: false,
    timeScale: {{ ...chartOpts.timeScale, fixLeftEdge: true, fixRightEdge: true }},
    rightPriceScale: {{ ...chartOpts.rightPriceScale, autoScale: true, scaleMargins: {{ top: 0.1, bottom: 0.1 }} }},
  }});
  const eqSeries = eq.addAreaSeries({{
    lineColor: '#8b5cf6', topColor: 'rgba(139,92,246,0.4)', bottomColor: 'rgba(139,92,246,0.0)',
    lineWidth: 2, crosshairMarkerRadius: 4,
  }});
  eqSeries.setData(eqData);
  eq.timeScale().fitContent();

  // Daily P&L
  const pnl = LightweightCharts.createChart(pnlEl, {{ ...chartOpts, width: pnlEl.clientWidth, height: pnlEl.clientHeight }});
  const pnlSeries = pnl.addHistogramSeries({{
    priceFormat: {{ type: 'price', precision: 0, minMove: 1 }},
  }});
  pnlSeries.setData(barData);
  pnl.timeScale().fitContent();

  // Responsive resize
  const ro = new ResizeObserver(() => {{
    eq.applyOptions({{ width: eqEl.clientWidth }});
    pnl.applyOptions({{ width: pnlEl.clientWidth }});
  }});
  ro.observe(eqEl);
  ro.observe(pnlEl);
}}

// ── Analytics Charts (lazy — render when tab opens) ──
let analyticsDrawn = false;
function drawAnalyticsCharts() {{
  if (analyticsDrawn) return;
  analyticsDrawn = true;

  const chartOpts2 = {{
    layout: {{ background: {{ type: 'solid', color: 'transparent' }}, textColor: '#94a3b8', fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }},
    grid: {{ vertLines: {{ color: 'rgba(255,255,255,0.03)' }}, horzLines: {{ color: 'rgba(255,255,255,0.03)' }} }},
    crosshair: {{ mode: 1 }},
    rightPriceScale: {{ borderColor: 'rgba(255,255,255,0.06)' }},
    timeScale: {{ borderColor: 'rgba(255,255,255,0.06)', timeVisible: false, fixLeftEdge: true, fixRightEdge: true }},
    handleScroll: false, handleScale: false,
  }};

  // Drawdown chart
  const ddEl = document.getElementById('dd-chart');
  if (ddEl && ddEl.clientWidth > 0 && ddData.length > 0) {{
    const ddChart = LightweightCharts.createChart(ddEl, {{ ...chartOpts2, width: ddEl.clientWidth, height: ddEl.clientHeight }});
    const ddSeries = ddChart.addAreaSeries({{
      lineColor: '#ef4444', topColor: 'rgba(239,68,68,0.0)', bottomColor: 'rgba(239,68,68,0.3)',
      lineWidth: 1.5,
    }});
    ddSeries.setData(ddData);
    ddChart.timeScale().fitContent();
  }}

  // Win rate chart
  const wrEl = document.getElementById('wr-chart');
  if (wrEl && wrEl.clientWidth > 0 && wrData.length > 0) {{
    const wrChart = LightweightCharts.createChart(wrEl, {{ ...chartOpts2, width: wrEl.clientWidth, height: wrEl.clientHeight }});
    const wrSeries = wrChart.addLineSeries({{
      color: '#06b6d4', lineWidth: 2,
    }});
    wrSeries.setData(wrData);
    const baseLine = wrChart.addLineSeries({{ color: 'rgba(255,255,255,0.1)', lineWidth: 1, lineStyle: 2 }});
    baseLine.setData(wrData.map(d => ({{ time: d.time, value: 50 }})));
    wrChart.timeScale().fitContent();
  }}

  drawDistChart();
}}

function drawDistChart() {{
  const distEl = document.getElementById('dist-chart');
  if (!distEl || distEl.clientWidth === 0 || distEl.querySelector('canvas')) return;
  const canvas = document.createElement('canvas');
  const w = distEl.clientWidth, h = distEl.clientHeight;
  canvas.width = w * 2; canvas.height = h * 2;
  canvas.style.width = '100%'; canvas.style.height = '100%';
  distEl.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);
  const sorted = [...tradePnls].sort((a,b) => a - b);
  const minV = sorted[0], maxV = sorted[sorted.length-1];
  const buckets = 16;
  const step = (maxV - minV) / buckets || 1;
  const bins = Array(buckets).fill(0);
  sorted.forEach(v => {{ const i = Math.min(Math.floor((v - minV) / step), buckets - 1); bins[i]++; }});
  const maxBin = Math.max(...bins);
  const barW = (w - 60) / buckets;
  const baseY = h - 35;
  const maxH = h - 60;
  bins.forEach((count, i) => {{
    if (count === 0) return;
    const barH = (count / maxBin) * maxH;
    const x = 30 + i * barW;
    const val = minV + (i + 0.5) * step;
    const grad = ctx.createLinearGradient(x, baseY - barH, x, baseY);
    if (val >= 0) {{
      grad.addColorStop(0, 'rgba(16,185,129,0.8)');
      grad.addColorStop(1, 'rgba(16,185,129,0.3)');
    }} else {{
      grad.addColorStop(0, 'rgba(239,68,68,0.8)');
      grad.addColorStop(1, 'rgba(239,68,68,0.3)');
    }}
    ctx.fillStyle = grad;
    ctx.beginPath();
    const r = 3;
    ctx.moveTo(x + 2 + r, baseY - barH);
    ctx.lineTo(x + barW - 4 - r, baseY - barH);
    ctx.quadraticCurveTo(x + barW - 4, baseY - barH, x + barW - 4, baseY - barH + r);
    ctx.lineTo(x + barW - 4, baseY);
    ctx.lineTo(x + 2, baseY);
    ctx.lineTo(x + 2, baseY - barH + r);
    ctx.quadraticCurveTo(x + 2, baseY - barH, x + 2 + r, baseY - barH);
    ctx.fill();
    if (count > 1) {{
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = '9px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(count, x + barW/2, baseY - barH - 5);
    }}
  }});
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.beginPath(); ctx.moveTo(30, baseY); ctx.lineTo(w - 30, baseY); ctx.stroke();
  ctx.fillStyle = '#64748b';
  ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText(`${{(minV/1000).toFixed(1)}}k`, 40, h - 12);
  const zeroX = 30 + ((-minV) / (maxV - minV)) * (w - 60);
  if (zeroX > 40 && zeroX < w - 40) {{
    ctx.fillText('0', zeroX, h - 12);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(zeroX, 20); ctx.lineTo(zeroX, baseY); ctx.stroke();
    ctx.setLineDash([]);
  }}
  ctx.fillText(`+${{(maxV/1000).toFixed(1)}}k`, w - 40, h - 12);
}}

// ── Navigation ──
function switchPanel(panelId) {{
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  const btn = document.querySelector(`.nav-btn[data-p="${{panelId}}"]`);
  if (btn) btn.classList.add('active');
  const panel = document.getElementById(panelId);
  if (panel) panel.classList.add('active');
  if (panelId === 'p-analytics') setTimeout(drawAnalyticsCharts, 50);
}}
document.querySelectorAll('.nav-btn').forEach(btn => {{
  btn.addEventListener('click', () => switchPanel(btn.dataset.p));
}});

// ── Day toggle ──
document.querySelectorAll('.day-header').forEach(hdr => {{
  hdr.addEventListener('click', () => {{
    hdr.parentElement.classList.toggle('open');
  }});
}});

// ── Filters ──
const fStrat=document.getElementById('f-strat'),fDir=document.getElementById('f-dir'),fRes=document.getElementById('f-result');
function parsePnl(el) {{
  if (!el) return 0;
  const txt = el.textContent.replace(/,/g, '').replace(/[^0-9.\\-+]/g, '');
  return parseFloat(txt) || 0;
}}
function applyFilters(){{
  const searchTerm = (document.getElementById('f-search')?.value || '').toLowerCase();
  let totalVisible=0, totalPnl=0;
  document.querySelectorAll('.day-group').forEach(group => {{
    const rows = group.querySelectorAll('.trow');
    let dayVisible = 0;
    rows.forEach(r => {{
      const strat = (r.querySelector('.t-strat')?.textContent || '').trim();
      const dir = (r.querySelector('.t-dir')?.textContent || '').trim();
      const pnl = parsePnl(r.querySelector('.t-pnl'));
      const rowText = r.textContent.toLowerCase();
      let show = true;
      if (searchTerm && !rowText.includes(searchTerm)) show = false;
      if (fStrat.value && strat !== fStrat.value) show = false;
      if (fDir.value && dir !== fDir.value) show = false;
      if (fRes.value === 'win' && pnl <= 0) show = false;
      if (fRes.value === 'loss' && pnl >= 0) show = false;
      r.style.display = show ? '' : 'none';
      if (show) {{ dayVisible++; totalVisible++; totalPnl += pnl; }}
    }});
    group.style.display = dayVisible > 0 ? '' : 'none';
  }});
  const sign = totalPnl >= 0 ? '+' : '';
  document.getElementById('f-count').textContent = totalVisible + ' trades | ' + sign + Math.round(totalPnl).toLocaleString();
}}
[fStrat,fDir,fRes].forEach(e => {{
  if(e) e.addEventListener('change', applyFilters);
}});

// ── Calendar Slider Navigation ──
(function() {{
  const slides = document.querySelectorAll('.cal-slide');
  const titleEl = document.getElementById('cal-title');
  const prevBtn = document.getElementById('cal-prev');
  const nextBtn = document.getElementById('cal-next');
  if (!slides.length || !titleEl) return;

  let currentIdx = slides.length - 1;

  function showSlide(idx) {{
    slides.forEach(s => s.classList.remove('active'));
    slides[idx].classList.add('active');
    currentIdx = idx;
    const name = slides[idx].dataset.name;
    const pnl = parseFloat(slides[idx].dataset.pnl);
    const trades = slides[idx].dataset.trades;
    const pnlCls = pnl >= 0 ? 'pos' : 'neg';
    const pnlStr = (pnl >= 0 ? '+' : '') + (pnl/1000).toFixed(1) + 'k';
    titleEl.innerHTML = `<div class="cnt">${{name}}</div><div class="cnsub">${{trades}} trades &middot; <span class="${{pnlCls}}">${{pnlStr}}</span></div>`;
    prevBtn.disabled = idx === 0;
    nextBtn.disabled = idx === slides.length - 1;
  }}

  prevBtn.addEventListener('click', () => {{ if (currentIdx > 0) showSlide(currentIdx - 1); }});
  nextBtn.addEventListener('click', () => {{ if (currentIdx < slides.length - 1) showSlide(currentIdx + 1); }});
  showSlide(currentIdx);
}})();

// ── Calendar click to jump to trades ──
document.querySelectorAll('.cal-cell.traded').forEach(cell => {{
  cell.addEventListener('click', () => {{
    const date = cell.getAttribute('title');
    if (!date) return;
    switchPanel('p-trades');
    const dayGroup = document.querySelector(`.day-group[data-date="${{date}}"]`);
    if (dayGroup) {{
      dayGroup.classList.add('open');
      dayGroup.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    }}
  }});
}});

// ── Search ──
const fSearch = document.getElementById('f-search');
if (fSearch) {{
  fSearch.addEventListener('input', applyFilters);
}}

// ── Page Load Animation ──
function animateOnLoad() {{
  // Hide loader
  const loader = document.getElementById('loader');
  if (loader) setTimeout(() => loader.classList.add('hidden'), 400);

  // Animate KPI cards
  setTimeout(() => {{
    document.querySelectorAll('.animate-in').forEach(el => el.classList.add('visible'));
  }}, 500);
}}

// Init
document.addEventListener('DOMContentLoaded', () => {{
  initCharts();
  animateOnLoad();
}});
window.addEventListener('load', initCharts);

// ── Animated Background Particles ──
(function() {{
  const canvas = document.getElementById('particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, particles = [], lines = [];

  function resize() {{
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }}
  resize();
  window.addEventListener('resize', resize);

  // Floating particles
  for (let i = 0; i < 60; i++) {{
    particles.push({{
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 2 + 0.5,
      color: ['rgba(139,92,246,','rgba(6,182,212,','rgba(16,185,129,'][Math.floor(Math.random()*3)],
      alpha: Math.random() * 0.4 + 0.1,
    }});
  }}

  // Floating chart lines
  for (let i = 0; i < 3; i++) {{
    const pts = [];
    for (let j = 0; j < 8; j++) {{
      pts.push({{ x: (w / 7) * j, y: h * 0.3 + Math.random() * h * 0.4 }});
    }}
    lines.push({{ pts, speed: 0.2 + Math.random() * 0.3, offset: Math.random() * 100, color: ['rgba(139,92,246,0.08)','rgba(6,182,212,0.06)','rgba(16,185,129,0.05)'][i] }});
  }}

  let frame = 0;
  function animate() {{
    ctx.clearRect(0, 0, w, h);
    frame++;

    // Draw connecting lines between close particles
    for (let i = 0; i < particles.length; i++) {{
      for (let j = i + 1; j < particles.length; j++) {{
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 120) {{
          ctx.beginPath();
          ctx.strokeStyle = `rgba(139,92,246,${{(1 - dist/120) * 0.08}})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }}
      }}
    }}

    // Draw & move particles
    particles.forEach(p => {{
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color + p.alpha + ')';
      ctx.fill();
    }});

    // Draw flowing lines
    lines.forEach(l => {{
      ctx.beginPath();
      ctx.strokeStyle = l.color;
      ctx.lineWidth = 1.5;
      l.pts.forEach((pt, i) => {{
        const y = pt.y + Math.sin((frame * l.speed + l.offset + i * 40) * 0.02) * 20;
        if (i === 0) ctx.moveTo(pt.x, y);
        else ctx.lineTo(pt.x, y);
      }});
      ctx.stroke();
    }});

    requestAnimationFrame(animate);
  }}
  animate();
}})();
</script>
</body>
</html>'''


# ── Entry Point ──────────────────────────────────────────────

def generate(output: Optional[Path] = None) -> Path:
    out = output or OUTPUT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    trades = load_trades()
    events = load_events()
    capital = load_capital()
    html = generate_html(trades, events, capital)
    out.write_text(html)
    return out


def main():
    path = generate()
    print(f"Report generated: {path}")
    print(f"  Trades: {len(load_trades())}")
    print(f"  Open in browser: file://{path}")


if __name__ == "__main__":
    main()
