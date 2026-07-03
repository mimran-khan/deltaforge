"""Generate a single self-contained HTML trade report from trades.db + events.jsonl.

Outputs docs/index.html -- one file that accumulates all trading days.
Each run regenerates from source data so it's always consistent.

Usage:
    python -m reports.generate          # from project root
    python reports/generate.py          # direct
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, date
from html import escape
from pathlib import Path
from typing import Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")

BASE_DIR = Path(__file__).resolve().parent.parent
PROD_DATA_DIR = Path.home() / "TradingAgent" / "data"
DATA_DIR = PROD_DATA_DIR if PROD_DATA_DIR.exists() else BASE_DIR / "data"
DB_PATH = DATA_DIR / "trades.db"
EVENTS_FILE = DATA_DIR / "events.jsonl"
CAPITAL_FILE = DATA_DIR / "capital.json"
OUTPUT_FILE = BASE_DIR / "docs" / "index.html"


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
    """Load all events from events.jsonl and all events_YYYY-MM-DD.jsonl files."""
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
    """Match each trade to its PAPER_ENTRY + PAPER_EXIT events by time proximity."""
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
            sig_type = ev.get("signal_type", "")
            ev_dir = ev.get("direction", "")
            if sig_type in t_strategy and ev_dir == t_direction:
                ev_time = ev_ts[11:16] if len(ev_ts) > 16 else ""
                if not t_time or not ev_time or abs(_time_diff_mins(t_time, ev_time)) <= 10:
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

        trade_events[tid] = {
            "entry": entry_ev or {},
            "exit": exit_ev or {},
        }

    return trade_events


def _time_diff_mins(t1: str, t2: str) -> int:
    """Difference in minutes between two HH:MM strings."""
    try:
        h1, m1 = int(t1[:2]), int(t1[3:5])
        h2, m2 = int(t2[:2]), int(t2[3:5])
        return (h1 * 60 + m1) - (h2 * 60 + m2)
    except (ValueError, IndexError):
        return 0


def load_day_events(events: list[dict]) -> dict[str, dict]:
    """Extract DAY_START / DAY_END per date."""
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
            "total_pnl": 0, "best": 0, "worst": 0,
            "avg_pnl": 0, "profit_factor": 0,
            "current_capital": capital.get("current_capital", 0),
            "peak_capital": capital.get("peak_capital", 0),
            "max_dd": capital.get("max_drawdown", 0),
            "trading_days": 0, "strategies": [],
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
        {
            "name": s,
            "trades": d["n"],
            "wins": d["w"],
            "wr": round(d["w"] / d["n"] * 100, 1) if d["n"] else 0,
            "pnl": round(d["pnl"], 2),
        }
        for s, d in sorted(strats.items(), key=lambda x: x[1]["pnl"], reverse=True)
    ]

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl": round(total_pnl, 2),
        "best": round(max((t.get("pnl", 0) for t in trades), default=0), 2),
        "worst": round(min((t.get("pnl", 0) for t in trades), default=0), 2),
        "avg_pnl": round(total_pnl / len(trades), 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "current_capital": capital.get("current_capital", 0),
        "peak_capital": capital.get("peak_capital", 0),
        "max_dd": capital.get("max_drawdown", 0),
        "trading_days": len(days),
        "strategies": strategy_list,
    }


def equity_curve_points(trades: list[dict], starting_cap: float) -> list[dict]:
    points = [{"x": 0, "y": starting_cap, "label": "Start"}]
    running = starting_cap
    for i, t in enumerate(trades):
        running += t.get("pnl", 0)
        points.append({
            "x": i + 1,
            "y": round(running, 2),
            "label": f"{t['date']} #{t['id']}",
        })
    return points


# ── HTML Generation ──────────────────────────────────────────

def _pnl_class(pnl: float) -> str:
    if pnl > 0:
        return "pnl-pos"
    if pnl < 0:
        return "pnl-neg"
    return "pnl-zero"


def _pnl_sign(pnl: float) -> str:
    return f"+{pnl:,.2f}" if pnl > 0 else f"{pnl:,.2f}"


def _direction_badge(direction: str, option_type: str = "") -> str:
    cls = "badge-long" if direction == "LONG" else "badge-short"
    label = direction
    if option_type:
        label = f"{direction} {option_type}"
    return f'<span class="badge {cls}">{escape(label)}</span>'


def _strategy_badge(name: str) -> str:
    colors = {
        "PULLBACK": "#2962ff",
        "STOCH": "#ff8c00",
        "TREND_RIDE": "#00bcd4",
        "CPR": "#4caf50",
        "SUPERTREND": "#9c27b0",
    }
    base = name.split("_")[0] if name else ""
    color = "#787b86"
    for k, v in colors.items():
        if base.startswith(k.split("_")[0]):
            color = v
            break
    return f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44">{escape(name)}</span>'


def _exit_badge(reason: str) -> str:
    cls_map = {"SL": "exit-sl", "TARGET": "exit-tgt", "TRAIL": "exit-trail"}
    cls = "exit-other"
    for k, v in cls_map.items():
        if k in (reason or "").upper():
            cls = v
            break
    return f'<span class="badge {cls}">{escape(reason or "—")}</span>'


def build_svg_equity(points: list[dict], width: int = 800, height: int = 200) -> str:
    if len(points) < 2:
        return ""
    ys = [p["y"] for p in points]
    y_min, y_max = min(ys), max(ys)
    y_range = y_max - y_min or 1
    x_max = len(points) - 1 or 1
    pad = 10

    def sx(i: int) -> float:
        return pad + (i / x_max) * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - ((v - y_min) / y_range) * (height - 2 * pad)

    path_points = " ".join(f"{sx(i):.1f},{sy(p['y']):.1f}" for i, p in enumerate(points))
    fill_points = f"{sx(0):.1f},{height - pad} {path_points} {sx(len(points) - 1):.1f},{height - pad}"

    last_color = "#26a69a" if points[-1]["y"] >= points[0]["y"] else "#ef5350"

    grid_lines = ""
    for frac in (0.25, 0.5, 0.75):
        gy = pad + frac * (height - 2 * pad)
        val = y_max - frac * y_range
        grid_lines += f'<line x1="{pad}" y1="{gy:.1f}" x2="{width - pad}" y2="{gy:.1f}" stroke="#333" stroke-dasharray="4"/>'
        grid_lines += f'<text x="{pad + 2}" y="{gy - 3:.1f}" fill="#666" font-size="10">{val:,.0f}</text>'

    return f'''<svg viewBox="0 0 {width} {height}" class="equity-svg">
  {grid_lines}
  <polygon points="{fill_points}" fill="{last_color}" opacity="0.08"/>
  <polyline points="{path_points}" fill="none" stroke="{last_color}" stroke-width="2"/>
  <circle cx="{sx(len(points)-1):.1f}" cy="{sy(points[-1]['y']):.1f}" r="4" fill="{last_color}"/>
  <text x="{sx(len(points)-1) - 5:.1f}" y="{sy(points[-1]['y']) - 8:.1f}" fill="{last_color}" font-size="11" text-anchor="end">{points[-1]["y"]:,.0f}</text>
  <text x="{sx(0) + 2:.1f}" y="{sy(points[0]['y']) - 8:.1f}" fill="#888" font-size="11">{points[0]["y"]:,.0f}</text>
</svg>'''


def build_day_pnl_bars(trades: list[dict], width: int = 800, height: int = 120) -> str:
    daily: dict[str, float] = {}
    for t in trades:
        d = t.get("date", "")
        daily[d] = daily.get(d, 0) + t.get("pnl", 0)
    if not daily:
        return ""

    dates = sorted(daily.keys())
    vals = [daily[d] for d in dates]
    abs_max = max(abs(v) for v in vals) or 1
    n = len(dates)
    pad = 10
    bar_w = max(8, min(40, (width - 2 * pad) / n - 4))
    mid_y = height / 2

    bars = ""
    for i, (d, v) in enumerate(zip(dates, vals)):
        x = pad + i * ((width - 2 * pad) / n) + 2
        bar_h = abs(v) / abs_max * (mid_y - pad - 5)
        color = "#26a69a" if v >= 0 else "#ef5350"
        y = mid_y - bar_h if v >= 0 else mid_y
        label = d[5:]  # MM-DD
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2" opacity="0.85">'
        bars += f'<title>{d}: {_pnl_sign(v)}</title></rect>'
        bars += f'<text x="{x + bar_w/2:.1f}" y="{height - 2}" fill="#666" font-size="8" text-anchor="middle">{label}</text>'

    return f'''<svg viewBox="0 0 {width} {height}" class="bars-svg">
  <line x1="{pad}" y1="{mid_y}" x2="{width-pad}" y2="{mid_y}" stroke="#444" stroke-width="1"/>
  {bars}
</svg>'''


def render_trade_card(
    trade: dict, entry_ev: dict, exit_ev: dict
) -> str:
    pnl = trade.get("pnl", 0)
    cls = _pnl_class(pnl)

    option_type = entry_ev.get("option_type", "")
    nifty_at_entry = entry_ev.get("entry_index")
    signal_reason = entry_ev.get("reason", "")
    confidence = trade.get("confidence") or entry_ev.get("confidence")
    signal_type = entry_ev.get("signal_type", trade.get("strategy", ""))
    sl_premium = entry_ev.get("sl")

    costs = exit_ev.get("costs")
    exit_ts = exit_ev.get("ts", "")
    exit_time = ""
    if exit_ts and "T" in exit_ts:
        exit_time = exit_ts.split("T")[1][:8]

    entry_time = trade.get("time", "")
    if len(entry_time) > 8:
        entry_time = entry_time[:8]

    hold_text = ""
    bars = trade.get("hold_bars")
    if bars is not None:
        hold_text = f"{bars} candle{'s' if bars != 1 else ''} (~{bars * 5}m)"

    metrics_html = ""
    if confidence is not None:
        metrics_html += f'<div class="metric"><span class="metric-label">Confidence</span><span class="metric-val">{confidence}%</span></div>'
    if trade.get("htf_rsi"):
        metrics_html += f'<div class="metric"><span class="metric-label">HTF RSI</span><span class="metric-val">{trade["htf_rsi"]:.1f}</span></div>'
    if trade.get("adx"):
        metrics_html += f'<div class="metric"><span class="metric-label">ADX</span><span class="metric-val">{trade["adx"]:.1f}</span></div>'
    if nifty_at_entry:
        metrics_html += f'<div class="metric"><span class="metric-label">Nifty Spot</span><span class="metric-val">{nifty_at_entry:,.2f}</span></div>'
    if sl_premium:
        metrics_html += f'<div class="metric"><span class="metric-label">SL Premium</span><span class="metric-val">{sl_premium:.2f}</span></div>'
    if costs:
        metrics_html += f'<div class="metric"><span class="metric-label">Costs</span><span class="metric-val">{costs:.2f}</span></div>'

    return f'''
    <div class="trade-card {cls}">
      <div class="trade-header">
        <div class="trade-badges">
          {_strategy_badge(trade.get("strategy", ""))}
          {_direction_badge(trade.get("direction", ""), option_type)}
          {_exit_badge(trade.get("exit_reason", ""))}
        </div>
        <div class="trade-pnl {cls}">{_pnl_sign(pnl)}</div>
      </div>

      {"<div class='trade-reason'>" + escape(signal_reason) + "</div>" if signal_reason else ""}

      <div class="trade-grid">
        <div class="trade-col">
          <div class="metric"><span class="metric-label">Entry</span><span class="metric-val">{entry_time}</span></div>
          <div class="metric"><span class="metric-label">Entry ₹</span><span class="metric-val">{trade.get("entry_price", 0):.2f}</span></div>
          {f'<div class="metric"><span class="metric-label">Lots</span><span class="metric-val">{trade.get("lots", 1)}</span></div>' if trade.get("lots") else ""}
        </div>
        <div class="trade-col">
          <div class="metric"><span class="metric-label">Exit</span><span class="metric-val">{exit_time or "—"}</span></div>
          <div class="metric"><span class="metric-label">Exit ₹</span><span class="metric-val">{trade.get("exit_price", 0):.2f}</span></div>
          {f'<div class="metric"><span class="metric-label">Hold</span><span class="metric-val">{hold_text}</span></div>' if hold_text else ""}
        </div>
        <div class="trade-col">
          {metrics_html}
        </div>
      </div>
    </div>'''


def render_day_section(
    day_date: str,
    day_trades: list[dict],
    trade_events: dict[int, dict],
    day_ev: dict,
    is_latest: bool,
) -> str:
    total_pnl = sum(t.get("pnl", 0) for t in day_trades)
    wins = sum(1 for t in day_trades if t.get("pnl", 0) > 0)
    losses = len(day_trades) - wins
    cls = _pnl_class(total_pnl)

    start_ev = day_ev.get("start", {})
    end_ev = day_ev.get("end", {})
    start_capital = start_ev.get("capital", "—")
    signals_scored = end_ev.get("signals_scored", "—")

    try:
        dt = datetime.strptime(day_date, "%Y-%m-%d")
        day_label = dt.strftime("%A, %b %d %Y")
    except ValueError:
        day_label = day_date

    cards = ""
    for t in day_trades:
        evs = trade_events.get(t["id"], {})
        cards += render_trade_card(t, evs.get("entry", {}), evs.get("exit", {}))

    open_attr = "open" if is_latest else ""

    return f'''
    <details class="day-section" {open_attr}>
      <summary class="day-summary">
        <div class="day-left">
          <span class="day-date">{escape(day_label)}</span>
          <span class="day-stats">{len(day_trades)} trade{"s" if len(day_trades) != 1 else ""} &middot; {wins}W {losses}L</span>
        </div>
        <div class="day-right">
          <span class="day-pnl {cls}">{_pnl_sign(total_pnl)}</span>
        </div>
      </summary>
      <div class="day-body">
        <div class="day-context">
          <span>Opening Capital: <b>{start_capital if isinstance(start_capital, str) else f"{start_capital:,.2f}"}</b></span>
          <span>Signals Scored: <b>{signals_scored}</b></span>
          <span>Win Rate: <b>{round(wins/len(day_trades)*100, 1) if day_trades else 0}%</b></span>
        </div>
        {cards}
      </div>
    </details>'''


CSS = '''
:root {
  --bg: #0d1117; --surface: #161b22; --border: #21262d;
  --text: #e6edf3; --text-dim: #8b949e; --text-muted: #484f58;
  --green: #26a69a; --red: #ef5350; --blue: #2962ff;
  --accent: #58a6ff;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.6; padding: 0;
}
.container { max-width: 960px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
.header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 20px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px;
}
.header-right { font-size: 0.8rem; color: var(--text-dim); text-align: right; }
.updated { font-size: 0.7rem; color: var(--text-muted); }

/* KPI Row */
.kpi-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px; margin-bottom: 24px;
}
.kpi {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
}
.kpi-label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

/* Strategy Table */
.strat-table { width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 0.82rem; }
.strat-table th {
  text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border);
  color: var(--text-dim); font-weight: 600; font-size: 0.7rem; text-transform: uppercase;
}
.strat-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); font-family: 'JetBrains Mono', monospace; }

/* Charts */
.chart-section {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; margin-bottom: 24px;
}
.chart-title { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.equity-svg, .bars-svg { width: 100%; height: auto; }

/* Day Sections */
.day-section {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; margin-bottom: 12px; overflow: hidden;
}
.day-summary {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 18px; cursor: pointer; list-style: none;
  transition: background 0.15s;
}
.day-summary:hover { background: #1c2129; }
.day-summary::-webkit-details-marker { display: none; }
.day-left { display: flex; align-items: center; gap: 12px; }
.day-date { font-weight: 600; font-size: 0.95rem; }
.day-stats { font-size: 0.8rem; color: var(--text-dim); }
.day-pnl { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1rem; }
.day-body { padding: 0 18px 18px; }
.day-context {
  display: flex; gap: 20px; flex-wrap: wrap; padding: 10px 0;
  margin-bottom: 12px; border-bottom: 1px solid var(--border);
  font-size: 0.8rem; color: var(--text-dim);
}
.day-context b { color: var(--text); }

/* Trade Cards */
.trade-card {
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; padding: 14px 16px; margin-bottom: 10px;
  border-left: 3px solid var(--border);
}
.trade-card.pnl-pos { border-left-color: var(--green); }
.trade-card.pnl-neg { border-left-color: var(--red); }
.trade-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.trade-badges { display: flex; gap: 6px; flex-wrap: wrap; }
.trade-pnl { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.05rem; }
.trade-reason {
  font-size: 0.8rem; color: var(--accent); background: var(--accent)11;
  padding: 6px 10px; border-radius: 4px; margin-bottom: 10px;
  font-family: 'JetBrains Mono', monospace; border: 1px solid var(--accent)22;
}
.trade-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
.trade-col { display: flex; flex-direction: column; gap: 4px; }
.metric { display: flex; justify-content: space-between; gap: 8px; }
.metric-label { font-size: 0.72rem; color: var(--text-dim); }
.metric-val { font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; }

/* Badges */
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
}
.badge-long { background: var(--green)22; color: var(--green); border: 1px solid var(--green)44; }
.badge-short { background: var(--red)22; color: var(--red); border: 1px solid var(--red)44; }
.exit-sl { background: #ef535022; color: #ef5350; border: 1px solid #ef535044; }
.exit-tgt { background: #26a69a22; color: #26a69a; border: 1px solid #26a69a44; }
.exit-trail { background: #ff8c0022; color: #ff8c00; border: 1px solid #ff8c0044; }
.exit-other { background: #78787822; color: #787878; border: 1px solid #78787844; }

/* P&L Colors */
.pnl-pos { color: var(--green); }
.pnl-neg { color: var(--red); }
.pnl-zero { color: var(--text-dim); }

/* Section Labels */
.section-label {
  font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.08em; font-weight: 600; margin: 24px 0 12px;
  padding-bottom: 6px; border-bottom: 1px solid var(--border);
}

/* Calendar */
.calendar-container { margin-bottom: 24px; }
.calendar-month {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; margin-bottom: 12px;
}
.calendar-month-title {
  font-size: 0.85rem; font-weight: 700; margin-bottom: 10px;
  display: flex; justify-content: space-between; align-items: center;
}
.calendar-month-pnl { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
.calendar-grid {
  display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px;
}
.calendar-header {
  font-size: 0.65rem; color: var(--text-muted); text-align: center;
  padding: 4px 0; font-weight: 600;
}
.calendar-day {
  aspect-ratio: 1; border-radius: 6px; display: flex;
  flex-direction: column; align-items: center; justify-content: center;
  font-size: 0.7rem; position: relative; min-height: 44px;
  transition: transform 0.1s;
}
.calendar-day:hover { transform: scale(1.1); z-index: 2; }
.calendar-day-num { font-size: 0.65rem; color: var(--text-dim); margin-bottom: 1px; }
.calendar-day-pnl { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; font-weight: 700; }
.calendar-day.empty { background: transparent; }
.calendar-day.no-trade { background: var(--bg); border: 1px solid var(--border); }
.calendar-day.profit { background: rgba(38, 166, 154, 0.15); border: 1px solid rgba(38, 166, 154, 0.4); }
.calendar-day.profit .calendar-day-pnl { color: var(--green); }
.calendar-day.loss { background: rgba(239, 83, 80, 0.15); border: 1px solid rgba(239, 83, 80, 0.4); }
.calendar-day.loss .calendar-day-pnl { color: var(--red); }
.calendar-day.big-profit { background: rgba(38, 166, 154, 0.3); border: 1px solid rgba(38, 166, 154, 0.7); box-shadow: 0 0 8px rgba(38, 166, 154, 0.2); }
.calendar-day.big-loss { background: rgba(239, 83, 80, 0.3); border: 1px solid rgba(239, 83, 80, 0.7); box-shadow: 0 0 8px rgba(239, 83, 80, 0.2); }
.calendar-day.weekend { opacity: 0.3; }
.calendar-legend {
  display: flex; gap: 12px; justify-content: center; margin-top: 10px;
  font-size: 0.65rem; color: var(--text-dim);
}
.calendar-legend-item { display: flex; align-items: center; gap: 4px; }
.legend-box { width: 12px; height: 12px; border-radius: 3px; }

/* Win Streak */
.streak-bar {
  display: flex; gap: 2px; margin-bottom: 24px; flex-wrap: wrap;
}
.streak-dot {
  width: 14px; height: 14px; border-radius: 3px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.5rem; font-weight: 700; color: #fff;
}
.streak-dot.win { background: var(--green); }
.streak-dot.loss { background: var(--red); }
.streak-dot.flat { background: var(--text-muted); }

/* Donut Chart */
.donut-container {
  display: flex; gap: 24px; align-items: center; justify-content: center;
  flex-wrap: wrap; margin-bottom: 24px;
}
.donut-chart { position: relative; width: 140px; height: 140px; }
.donut-center {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  text-align: center;
}
.donut-center-val { font-size: 1.2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.donut-center-label { font-size: 0.65rem; color: var(--text-dim); }
.donut-legend { display: flex; flex-direction: column; gap: 8px; }
.donut-legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; }
.donut-legend-color { width: 12px; height: 12px; border-radius: 3px; }

/* Responsive */
@media (max-width: 640px) {
  .trade-grid { grid-template-columns: 1fr 1fr; }
  .kpi-row { grid-template-columns: repeat(2, 1fr); }
  .day-context { flex-direction: column; gap: 4px; }
  .strat-table { font-size: 0.72rem; }
  .strat-table th, .strat-table td { padding: 6px 8px; }
  .calendar-grid { gap: 2px; }
  .calendar-day { min-height: 36px; }
  .calendar-day-pnl { font-size: 0.5rem; }
}

.footer {
  text-align: center; padding: 24px 0; color: var(--text-muted);
  font-size: 0.7rem; border-top: 1px solid var(--border); margin-top: 24px;
}
'''


def build_calendar(trades: list[dict]) -> str:
    """Build a monthly calendar heatmap showing day-wise PnL."""
    import calendar
    from collections import defaultdict

    daily_pnl: dict[str, float] = defaultdict(float)
    daily_trades: dict[str, int] = defaultdict(int)
    for t in trades:
        d = t.get("date", "")
        if d:
            daily_pnl[d] += t.get("pnl", 0)
            daily_trades[d] += 1

    if not daily_pnl:
        return ""

    all_dates = sorted(daily_pnl.keys())
    first = datetime.strptime(all_dates[0], "%Y-%m-%d")
    last = datetime.strptime(all_dates[-1], "%Y-%m-%d")

    months = []
    cur = first.replace(day=1)
    while cur <= last:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    abs_max = max(abs(v) for v in daily_pnl.values()) if daily_pnl else 1

    html = '<div class="section-label">Calendar P&L</div><div class="calendar-container">'

    for year, month in months:
        month_name = calendar.month_name[month]
        cal = calendar.monthcalendar(year, month)

        month_pnl = sum(
            v for d, v in daily_pnl.items()
            if d.startswith(f"{year}-{month:02d}")
        )
        month_cls = _pnl_class(month_pnl)

        html += f'''<div class="calendar-month">
          <div class="calendar-month-title">
            <span>{month_name} {year}</span>
            <span class="calendar-month-pnl {month_cls}">{_pnl_sign(month_pnl)}</span>
          </div>
          <div class="calendar-grid">
            <div class="calendar-header">Mon</div>
            <div class="calendar-header">Tue</div>
            <div class="calendar-header">Wed</div>
            <div class="calendar-header">Thu</div>
            <div class="calendar-header">Fri</div>
            <div class="calendar-header">Sat</div>
            <div class="calendar-header">Sun</div>'''

        for week in cal:
            for dow, day_num in enumerate(week):
                if day_num == 0:
                    html += '<div class="calendar-day empty"></div>'
                    continue

                date_str = f"{year}-{month:02d}-{day_num:02d}"
                is_weekend = dow >= 5

                if is_weekend:
                    html += f'<div class="calendar-day weekend no-trade"><span class="calendar-day-num">{day_num}</span></div>'
                elif date_str in daily_pnl:
                    pnl = daily_pnl[date_str]
                    n_trades = daily_trades[date_str]
                    intensity = abs(pnl) / abs_max

                    if pnl > 0:
                        cls = "big-profit" if intensity > 0.5 else "profit"
                    else:
                        cls = "big-loss" if intensity > 0.5 else "loss"

                    pnl_str = f"+{pnl/1000:.1f}k" if pnl > 0 else f"{pnl/1000:.1f}k"
                    html += f'<div class="calendar-day {cls}" title="{date_str}: {_pnl_sign(pnl)} ({n_trades} trades)"><span class="calendar-day-num">{day_num}</span><span class="calendar-day-pnl">{pnl_str}</span></div>'
                else:
                    html += f'<div class="calendar-day no-trade"><span class="calendar-day-num">{day_num}</span></div>'

        html += '</div></div>'

    html += '''
      <div class="calendar-legend">
        <div class="calendar-legend-item"><div class="legend-box" style="background:rgba(239,83,80,0.3);border:1px solid rgba(239,83,80,0.7)"></div>Big Loss</div>
        <div class="calendar-legend-item"><div class="legend-box" style="background:rgba(239,83,80,0.15);border:1px solid rgba(239,83,80,0.4)"></div>Loss</div>
        <div class="calendar-legend-item"><div class="legend-box" style="background:var(--bg);border:1px solid var(--border)"></div>No Trade</div>
        <div class="calendar-legend-item"><div class="legend-box" style="background:rgba(38,166,154,0.15);border:1px solid rgba(38,166,154,0.4)"></div>Profit</div>
        <div class="calendar-legend-item"><div class="legend-box" style="background:rgba(38,166,154,0.3);border:1px solid rgba(38,166,154,0.7)"></div>Big Profit</div>
      </div>
    </div>'''

    return html


def build_win_streak(trades: list[dict]) -> str:
    """Build a visual win/loss streak bar from daily results."""
    from collections import defaultdict
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        d = t.get("date", "")
        if d:
            daily_pnl[d] += t.get("pnl", 0)

    if not daily_pnl:
        return ""

    html = '<div class="section-label">Daily Win/Loss Streak</div><div class="streak-bar">'
    for d in sorted(daily_pnl.keys()):
        pnl = daily_pnl[d]
        if pnl > 0:
            cls = "win"
            title = f"{d}: +{pnl:,.0f}"
        elif pnl < 0:
            cls = "loss"
            title = f"{d}: {pnl:,.0f}"
        else:
            cls = "flat"
            title = f"{d}: flat"
        html += f'<div class="streak-dot {cls}" title="{title}"></div>'

    html += '</div>'
    return html


def build_donut_chart(trades: list[dict]) -> str:
    """Build SVG donut charts for win/loss ratio and strategy split."""
    wins = len([t for t in trades if t.get("pnl", 0) > 0])
    losses = len(trades) - wins
    total = len(trades) or 1
    win_pct = wins / total
    loss_pct = losses / total

    r = 55
    cx, cy = 70, 70
    circumference = 2 * 3.14159 * r

    win_dash = win_pct * circumference
    loss_dash = loss_pct * circumference

    win_svg = f'''<svg width="140" height="140" viewBox="0 0 140 140">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="var(--border)" stroke-width="14"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="var(--green)" stroke-width="14"
        stroke-dasharray="{win_dash:.1f} {circumference:.1f}"
        stroke-dashoffset="0" transform="rotate(-90 {cx} {cy})"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="var(--red)" stroke-width="14"
        stroke-dasharray="{loss_dash:.1f} {circumference:.1f}"
        stroke-dashoffset="-{win_dash:.1f}" transform="rotate(-90 {cx} {cy})"/>
    </svg>'''

    strats: dict[str, float] = {}
    for t in trades:
        s = t.get("strategy", "OTHER")
        strats[s] = strats.get(s, 0) + abs(t.get("pnl", 0))
    strat_total = sum(strats.values()) or 1
    strat_colors = ["#2962ff", "#ff8c00", "#00bcd4", "#9c27b0", "#4caf50", "#ff5722", "#607d8b"]

    strat_legend = ""
    for i, (s, val) in enumerate(sorted(strats.items(), key=lambda x: -x[1])):
        pct = val / strat_total * 100
        color = strat_colors[i % len(strat_colors)]
        strat_legend += f'''<div class="donut-legend-item">
          <div class="donut-legend-color" style="background:{color}"></div>
          <span>{s} ({pct:.0f}%)</span>
        </div>'''

    return f'''
    <div class="section-label">Performance Overview</div>
    <div class="donut-container">
      <div class="donut-chart">
        {win_svg}
        <div class="donut-center">
          <div class="donut-center-val pnl-pos">{wins}</div>
          <div class="donut-center-label">Wins / {total}</div>
        </div>
      </div>
      <div class="donut-legend">
        <div class="donut-legend-item">
          <div class="donut-legend-color" style="background:var(--green)"></div>
          <span>Wins: {wins} ({win_pct*100:.0f}%)</span>
        </div>
        <div class="donut-legend-item">
          <div class="donut-legend-color" style="background:var(--red)"></div>
          <span>Losses: {losses} ({loss_pct*100:.0f}%)</span>
        </div>
        {strat_legend}
      </div>
    </div>'''


def generate_html(
    trades: list[dict],
    events: list[dict],
    capital: dict,
) -> str:
    trade_events = match_events_to_trades(trades, events)
    day_events = load_day_events(events)
    overall = compute_overall(trades, capital)
    starting_cap = capital.get("initial_capital") or capital.get("peak_capital") or 57402
    eq_points = equity_curve_points(trades, starting_cap)

    now = datetime.now(IST)

    # KPIs
    kpi_html = f'''
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-label">Total Trades</div><div class="kpi-value">{overall["total_trades"]}</div></div>
      <div class="kpi"><div class="kpi-label">Win Rate</div><div class="kpi-value">{overall["wr"]}%</div></div>
      <div class="kpi"><div class="kpi-label">Total P&amp;L</div><div class="kpi-value {_pnl_class(overall['total_pnl'])}">{_pnl_sign(overall["total_pnl"])}</div></div>
      <div class="kpi"><div class="kpi-label">Profit Factor</div><div class="kpi-value">{overall["profit_factor"]}</div></div>
      <div class="kpi"><div class="kpi-label">Avg Trade</div><div class="kpi-value {_pnl_class(overall['avg_pnl'])}">{_pnl_sign(overall["avg_pnl"])}</div></div>
      <div class="kpi"><div class="kpi-label">Best / Worst</div><div class="kpi-value"><span class="pnl-pos">{overall["best"]:+,.0f}</span> / <span class="pnl-neg">{overall["worst"]:+,.0f}</span></div></div>
      <div class="kpi"><div class="kpi-label">Capital</div><div class="kpi-value">{overall["current_capital"]:,.0f}</div></div>
      <div class="kpi"><div class="kpi-label">Max Drawdown</div><div class="kpi-value pnl-neg">{overall["max_dd"]:.1f}%</div></div>
    </div>'''

    # Strategy table
    strat_rows = ""
    for s in overall["strategies"]:
        strat_rows += f'''<tr>
          <td>{_strategy_badge(s["name"])}</td>
          <td>{s["trades"]}</td><td>{s["wins"]}</td><td>{s["wr"]}%</td>
          <td class="{_pnl_class(s['pnl'])}">{_pnl_sign(s["pnl"])}</td>
        </tr>'''

    strat_html = f'''
    <div class="section-label">Strategy Breakdown</div>
    <table class="strat-table">
      <thead><tr><th>Strategy</th><th>Trades</th><th>Wins</th><th>WR</th><th>P&amp;L</th></tr></thead>
      <tbody>{strat_rows}</tbody>
    </table>''' if strat_rows else ""

    # Charts
    equity_html = f'''
    <div class="chart-section">
      <div class="chart-title">Equity Curve</div>
      {build_svg_equity(eq_points)}
    </div>''' if eq_points else ""

    bars_html = f'''
    <div class="chart-section">
      <div class="chart-title">Daily P&amp;L</div>
      {build_day_pnl_bars(trades)}
    </div>''' if trades else ""

    # Calendar heatmap
    calendar_html = build_calendar(trades)

    # Win streak
    streak_html = build_win_streak(trades)

    # Donut charts
    donut_html = build_donut_chart(trades)

    # Day sections (most recent first)
    by_date: dict[str, list[dict]] = {}
    for t in trades:
        by_date.setdefault(t["date"], []).append(t)
    sorted_dates = sorted(by_date.keys(), reverse=True)

    days_html = '<div class="section-label">Trade Log</div>'
    for i, d in enumerate(sorted_dates):
        days_html += render_day_section(
            d, by_date[d], trade_events,
            day_events.get(d, {}),
            is_latest=(i == 0),
        )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeltaForge — Trade Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>DeltaForge</h1>
      <div class="updated">Trade Report &middot; {overall["trading_days"]} trading days</div>
    </div>
    <div class="header-right">
      <div>Generated {now.strftime("%b %d, %Y %H:%M IST")}</div>
      <div class="updated">Paper Trading &middot; MultiStratV11</div>
    </div>
  </div>

  {kpi_html}
  {equity_html}
  {bars_html}
  {calendar_html}
  {streak_html}
  {donut_html}
  {strat_html}
  {days_html}

  <div class="footer">
    DeltaForge &middot; Auto-generated from trades.db &middot; {now.strftime("%Y-%m-%d %H:%M")} IST
  </div>
</div>
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
