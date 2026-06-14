"""Thin read-only wrappers around every DeltaForge data source.

Each function returns plain dicts/lists -- no coupling to FastAPI.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pytz

from config import settings

IST = pytz.timezone("Asia/Kolkata")


# ── Paths ────────────────────────────────────────────────────

DATA_DIR: Path = settings.DATA_DIR
LOG_JSON_DIR: Path = settings.LOG_DIR / "json"
CAPITAL_FILE: Path = settings.CAPITAL_FILE
DB_PATH: Path = settings.DB_PATH
EVENTS_FILE: Path = DATA_DIR / "events.jsonl"
HALT_FLAG: Path = DATA_DIR / "HALT"
SESSION_LOCK: Path = DATA_DIR / "session.lock"
ENGINE_STATE_FILE: Path = DATA_DIR / "engine_state.json"


# ── capital.json ─────────────────────────────────────────────

def read_capital() -> dict:
    if not CAPITAL_FILE.exists():
        return {
            "current_capital": settings.STARTING_CAPITAL,
            "total_pnl": 0, "peak_capital": settings.STARTING_CAPITAL,
            "weekly_pnl": 0, "max_drawdown": 0, "daily_pnl": 0,
            "trades_today": 0, "consecutive_losses": 0,
            "day_start_capital": settings.STARTING_CAPITAL,
            "wins_today": 0, "losses_today": 0, "last_updated": None,
        }
    try:
        return json.loads(CAPITAL_FILE.read_text())
    except Exception:
        return {}


# ── HALT flag ────────────────────────────────────────────────

def read_halt() -> dict | None:
    if not HALT_FLAG.exists():
        return None
    try:
        return json.loads(HALT_FLAG.read_text())
    except Exception:
        return {"halted_at": None, "reason": "unknown"}


def toggle_halt() -> dict:
    """Toggle the halt flag. Returns new state."""
    if HALT_FLAG.exists():
        HALT_FLAG.unlink(missing_ok=True)
        return {"halted": False, "message": "Halt cleared"}
    HALT_FLAG.write_text(json.dumps({
        "halted_at": datetime.now(IST).isoformat(),
        "reason": "Manual kill switch from dashboard",
    }))
    return {"halted": True, "message": "Halt activated"}


# ── Session lock ─────────────────────────────────────────────

def read_session_lock() -> dict | None:
    if not SESSION_LOCK.exists():
        return None
    try:
        return json.loads(SESSION_LOCK.read_text())
    except Exception:
        return None


# ── Aggregated status (KPI bar) ──────────────────────────────

def get_status() -> dict:
    cap = read_capital()
    halt = read_halt()
    session = read_session_lock()

    session_active = False
    session_pid = None
    if session:
        session_pid = session.get("pid")
        if session_pid:
            try:
                os.kill(session_pid, 0)
                session_active = True
            except (OSError, ProcessLookupError):
                session_active = False

    trades = cap.get("trades_today", 0)
    wins = cap.get("wins_today", 0)
    wr = round(wins / trades * 100, 1) if trades > 0 else 0.0
    dd = 0.0
    peak = cap.get("peak_capital", 1)
    curr = cap.get("current_capital", 0)
    if peak > 0:
        dd = round((peak - curr) / peak * 100, 2)

    return {
        "current_capital": curr,
        "daily_pnl": cap.get("daily_pnl", 0),
        "total_pnl": cap.get("total_pnl", 0),
        "peak_capital": peak,
        "drawdown_pct": dd,
        "trades_today": trades,
        "wins_today": wins,
        "losses_today": cap.get("losses_today", 0),
        "win_rate": wr,
        "consecutive_losses": cap.get("consecutive_losses", 0),
        "halted": halt is not None,
        "halt_reason": halt.get("reason") if halt else None,
        "session_active": session_active,
        "session_pid": session_pid,
        "last_updated": cap.get("last_updated"),
    }


# ── trades.db ────────────────────────────────────────────────

def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_trades(
    target_date: Optional[str] = None,
    strategy: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = _db_conn()
    try:
        clauses, params = [], []
        if target_date:
            clauses.append("date = ?")
            params.append(target_date)
        if strategy:
            clauses.append("strategy LIKE ?")
            params.append(f"%{strategy}%")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        params += [limit, offset]
        rows = conn.execute(
            f"SELECT * FROM trades {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def get_trades_summary(target_date: Optional[str] = None) -> dict:
    d = target_date or date.today().isoformat()
    if not DB_PATH.exists():
        return {"date": d, "trades": 0, "wins": 0, "losses": 0,
                "total_pnl": 0, "wr": 0}
    conn = _db_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM trades WHERE date = ?", (d,)
        ).fetchall()
        if not rows:
            return {"date": d, "trades": 0, "wins": 0, "losses": 0,
                    "total_pnl": 0, "wr": 0}
        wins = sum(1 for r in rows if r["pnl"] > 0)
        total_pnl = sum(r["pnl"] for r in rows)
        return {
            "date": d,
            "trades": len(rows),
            "wins": wins,
            "losses": len(rows) - wins,
            "total_pnl": round(total_pnl, 2),
            "wr": round(wins / len(rows) * 100, 1),
        }
    except Exception:
        return {"date": d, "trades": 0, "wins": 0, "losses": 0,
                "total_pnl": 0, "wr": 0}
    finally:
        conn.close()


def get_strategy_stats(min_trades: int = 1) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = _db_conn()
    try:
        rows = conn.execute(
            "SELECT strategy, COUNT(*) as n, "
            "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
            "SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit, "
            "SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss, "
            "SUM(pnl) as total_pnl, AVG(pnl) as avg_pnl, "
            "MIN(pnl) as worst, MAX(pnl) as best "
            "FROM trades GROUP BY strategy HAVING n >= ?",
            (min_trades,),
        ).fetchall()
        result = []
        for r in rows:
            gl = r["gross_loss"] or 0
            pf = round(r["gross_profit"] / gl, 2) if gl > 0 else 9999.0
            result.append({
                "strategy": r["strategy"],
                "trades": r["n"],
                "wins": r["wins"],
                "wr": round(r["wins"] / r["n"] * 100, 1) if r["n"] > 0 else 0,
                "total_pnl": round(r["total_pnl"], 2),
                "avg_pnl": round(r["avg_pnl"], 2),
                "profit_factor": pf,
                "best_trade": round(r["best"], 2),
                "worst_trade": round(r["worst"], 2),
            })
        return result
    except Exception:
        return []
    finally:
        conn.close()


# ── events.jsonl ─────────────────────────────────────────────

def get_events(
    target_date: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    d = target_date or date.today().isoformat()
    if not EVENTS_FILE.exists():
        return []
    try:
        lines = EVENTS_FILE.read_text().strip().splitlines()
    except Exception:
        return []

    types = {t.strip().upper() for t in event_type.split(",")} if event_type else None
    result = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        ts = ev.get("ts", "")
        if d and not ts.startswith(d):
            continue
        if types and ev.get("event", "").upper() not in types:
            continue
        result.append(ev)
        if len(result) >= limit:
            break
    return result


# ── logs/json/*.jsonl ────────────────────────────────────────

def _log_file_for_date(d: str) -> Path:
    return LOG_JSON_DIR / f"trading_{d}.jsonl"


def get_logs(
    target_date: Optional[str] = None,
    level: Optional[str] = None,
    module: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 300,
) -> list[dict]:
    d = target_date or date.today().isoformat()
    log_file = _log_file_for_date(d)
    if not log_file.exists():
        return []

    levels = {l.strip().upper() for l in level.split(",")} if level else None
    try:
        lines = log_file.read_text().strip().splitlines()
    except Exception:
        return []

    result = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if levels and entry.get("level", "").upper() not in levels:
            continue
        if module and entry.get("module", "") != module:
            continue
        if search and search.lower() not in entry.get("message", "").lower():
            continue
        result.append(entry)
        if len(result) >= limit:
            break
    return result


# ── config/settings.py (read-only export) ────────────────────

_CONFIG_GROUPS = {
    "Capital & Sizing": [
        "STARTING_CAPITAL", "CAPITAL_PER_LOT", "MAX_LOTS_CAP",
        "NIFTY_LOT_SIZE", "BANKNIFTY_LOT_SIZE", "CAPITAL_DEPLOY_PCT",
        "COMPOUND_DAILY",
    ],
    "Risk Management": [
        "MAX_CONSECUTIVE_LOSSES", "MIN_CAPITAL_TO_TRADE",
        "DAILY_LOSS_LIMIT_PCT", "WEEKLY_LOSS_LIMIT_PCT",
        "DRAWDOWN_HALFSIZE_PCT", "DRAWDOWN_HALT_PCT",
        "MAX_VIX_THRESHOLD", "VIX_SPIKE_HALT_PCT", "SKIP_EXPIRY_DAY",
    ],
    "Options Model": [
        "PREMIUM_BASE", "PREMIUM_DELTA", "PREMIUM_THETA_PER_CANDLE",
        "PREMIUM_SL_PCT", "PREMIUM_TARGET_PCT", "PREMIUM_TARGET_POINTS",
        "SLIPPAGE_POINTS", "BID_ASK_SPREAD", "STT_SELL_PCT",
        "EXCHANGE_TXN_PCT", "MARKET_IMPACT_PCT", "BROKERAGE_PER_ORDER",
    ],
    "Timing": [
        "MARKET_OPEN", "MARKET_CLOSE", "ENTRY_START", "ENTRY_END",
        "NO_NEW_ENTRY_AFTER", "SQUARE_OFF_TIME", "SESSION_LOGIN_TIME",
        "INSTRUMENT_DOWNLOAD_TIME", "EOD_REPORT_TIME", "SESSION_LOGOUT_TIME",
    ],
    "Strategy": [
        "PULLBACK_MIN_CONFIDENCE", "PULLBACK_HOLD_CANDLES",
        "PULLBACK_MAX_SIGNALS_PER_DAY", "CONFLUENCE_THRESHOLD",
        "MIN_STRENGTH", "NIFTY_EXPIRY_DAY",
    ],
    "System": [
        "TRADING_MODE", "ALERT_METHOD",
    ],
}


def get_config() -> dict:
    grouped: dict[str, list[dict]] = {}
    for group_name, keys in _CONFIG_GROUPS.items():
        items = []
        for key in keys:
            val = getattr(settings, key, None)
            if val is None:
                continue
            items.append({"key": key, "value": _safe_value(val)})
        grouped[group_name] = items
    return grouped


def _safe_value(val):
    if isinstance(val, Path):
        return str(val)
    return val


# ── Risk gates ───────────────────────────────────────────────

def get_risk_gates() -> list[dict]:
    cap = read_capital()
    halt = read_halt()
    curr = cap.get("current_capital", 0)
    peak = cap.get("peak_capital", 1)
    dd = round((peak - curr) / peak * 100, 2) if peak > 0 else 0
    day_start = cap.get("day_start_capital", curr)
    daily_limit = day_start * (settings.DAILY_LOSS_LIMIT_PCT / 100)
    daily_used = abs(cap.get("daily_pnl", 0)) if cap.get("daily_pnl", 0) < 0 else 0
    weekly_limit = settings.STARTING_CAPITAL * (settings.WEEKLY_LOSS_LIMIT_PCT / 100)

    today_weekday = datetime.now(IST).weekday()
    is_expiry = today_weekday == settings.NIFTY_EXPIRY_DAY

    return [
        {
            "name": "Min Capital",
            "current": round(curr, 0),
            "threshold": settings.MIN_CAPITAL_TO_TRADE,
            "status": "pass" if curr >= settings.MIN_CAPITAL_TO_TRADE else "fail",
            "pct": min(100, round(curr / settings.MIN_CAPITAL_TO_TRADE * 100)),
        },
        {
            "name": "Daily Loss",
            "current": round(daily_used, 0),
            "threshold": round(daily_limit, 0),
            "status": "pass" if daily_used < daily_limit else "fail",
            "pct": round(daily_used / daily_limit * 100) if daily_limit > 0 else 0,
        },
        {
            "name": "Weekly Loss",
            "current": round(abs(cap.get("weekly_pnl", 0)), 0),
            "threshold": round(weekly_limit, 0),
            "status": "pass",
            "pct": round(abs(cap.get("weekly_pnl", 0)) / weekly_limit * 100) if weekly_limit > 0 else 0,
        },
        {
            "name": "Consec. Losses",
            "current": cap.get("consecutive_losses", 0),
            "threshold": settings.MAX_CONSECUTIVE_LOSSES,
            "status": "pass" if cap.get("consecutive_losses", 0) < settings.MAX_CONSECUTIVE_LOSSES else "fail",
            "pct": round(cap.get("consecutive_losses", 0) / settings.MAX_CONSECUTIVE_LOSSES * 100),
        },
        {
            "name": "Drawdown",
            "current": dd,
            "threshold": settings.DRAWDOWN_HALT_PCT,
            "status": "pass" if dd < settings.DRAWDOWN_HALFSIZE_PCT else ("warn" if dd < settings.DRAWDOWN_HALT_PCT else "fail"),
            "pct": round(dd / settings.DRAWDOWN_HALT_PCT * 100),
        },
        {
            "name": "VIX",
            "current": None,
            "threshold": settings.MAX_VIX_THRESHOLD,
            "status": "unknown",
            "pct": 0,
        },
        {
            "name": "Expiry Day",
            "current": "Yes" if is_expiry else "No",
            "threshold": f"Skip={settings.SKIP_EXPIRY_DAY}",
            "status": "warn" if is_expiry and settings.SKIP_EXPIRY_DAY else "pass",
            "pct": 100 if is_expiry else 0,
        },
        {
            "name": "Kill Switch",
            "current": "ON" if halt else "OFF",
            "threshold": "OFF",
            "status": "fail" if halt else "pass",
            "pct": 100 if halt else 0,
        },
    ]


# ── Broker / session info ────────────────────────────────────

def get_broker_info() -> dict:
    session = read_session_lock()
    cap = read_capital()
    session_active = False
    session_pid = None
    session_started = None
    if session:
        session_pid = session.get("pid")
        session_started = session.get("started_at")
        if session_pid:
            try:
                os.kill(session_pid, 0)
                session_active = True
            except (OSError, ProcessLookupError):
                session_active = False
    return {
        "session_active": session_active,
        "session_pid": session_pid,
        "session_started": session_started,
        "trading_mode": settings.TRADING_MODE,
        "client_id": settings.ANGEL_CLIENT_ID[:4] + "****" if settings.ANGEL_CLIENT_ID else "N/A",
        "lot_size": settings.NIFTY_LOT_SIZE,
        "max_lots": settings.MAX_LOTS_CAP,
        "alert_method": settings.ALERT_METHOD,
        "last_updated": cap.get("last_updated"),
    }


# ── Engine state (live price, indicators, positions) ─────────

def read_engine_state() -> dict | None:
    if not ENGINE_STATE_FILE.exists():
        return None
    try:
        return json.loads(ENGINE_STATE_FILE.read_text())
    except Exception:
        return None


def get_engine_state() -> dict:
    state = read_engine_state()
    if not state:
        return {
            "ts": None,
            "nifty_price": None,
            "candle_count": 0,
            "positions": [],
            "candles": [],
            "running": False,
            "signals_today": 0,
        }
    return state


def get_signals() -> list[dict]:
    """Get recent signals from engine state and events."""
    get_engine_state()
    events = get_events(event_type="PAPER_ENTRY,SIGNAL", limit=20)
    signals = []
    for ev in events:
        if ev.get("event") in ("PAPER_ENTRY", "SIGNAL"):
            data = ev.get("data") or ev
            ts = ev.get("ts", "")
            time_str = ts.split("T")[1][:8] if "T" in ts else ts[-8:]
            signals.append({
                "time": time_str,
                "type": data.get("signal_type", "UNKNOWN"),
                "direction": data.get("direction", ""),
                "confidence": data.get("confidence", 0),
                "entry_premium": data.get("entry_premium", 0),
                "reason": data.get("reason", ""),
            })
    return signals
