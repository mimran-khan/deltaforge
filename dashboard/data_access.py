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
MULTI_ASSET_STATE_FILE: Path = DATA_DIR / "multi_asset_state.json"


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
        "project_root": str(settings.BASE_DIR),
    }


# ── trades.db ────────────────────────────────────────────────

def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_trades(
    target_date: Optional[str] = None,
    strategy: Optional[str] = None,
    instrument: Optional[str] = None,
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
        if instrument:
            clauses.append("instrument = ?")
            params.append(instrument)
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


def get_trades_summary(target_date: Optional[str] = None,
                       instrument: Optional[str] = None) -> dict:
    d = target_date or date.today().isoformat()
    if not DB_PATH.exists():
        return {"date": d, "trades": 0, "wins": 0, "losses": 0,
                "total_pnl": 0, "wr": 0}
    conn = _db_conn()
    try:
        clauses = ["date = ?"]
        params: list = [d]
        if instrument:
            clauses.append("instrument = ?")
            params.append(instrument)
        where = "WHERE " + " AND ".join(clauses)
        rows = conn.execute(
            f"SELECT * FROM trades {where}", params
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


def get_strategy_stats(min_trades: int = 1,
                       instrument: Optional[str] = None) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = _db_conn()
    try:
        where_clause = ""
        params: list = []
        if instrument:
            where_clause = "WHERE instrument = ? "
            params.append(instrument)
        params.append(min_trades)
        rows = conn.execute(
            "SELECT strategy, COUNT(*) as n, "
            "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
            "SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit, "
            "SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss, "
            "SUM(pnl) as total_pnl, AVG(pnl) as avg_pnl, "
            "MIN(pnl) as worst, MAX(pnl) as best "
            f"FROM trades {where_clause}GROUP BY strategy HAVING n >= ?",
            params,
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
    "Multi-Asset": [
        "MULTI_ASSET_ENABLED",
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
            "candles_1m": [],
            "running": False,
            "signals_today": 0,
        }
    return state


def read_multi_asset_state() -> dict | None:
    if not MULTI_ASSET_STATE_FILE.exists():
        return None
    try:
        return json.loads(MULTI_ASSET_STATE_FILE.read_text())
    except Exception:
        return None


def get_multi_asset_state() -> dict:
    state = read_multi_asset_state()
    if not state:
        return {"ts": None, "running": False, "instruments": {}}
    return state


def _aggregate_candle_rows(rows: list[dict], factor: int) -> list[dict]:
    """Merge N consecutive OHLCV rows into a higher timeframe bar."""
    if factor <= 1 or not rows:
        return rows
    merged: list[dict] = []
    for i in range(0, len(rows), factor):
        chunk = rows[i:i + factor]
        if not chunk:
            continue
        last = chunk[-1]
        merged.append({
            "n": len(merged) + 1,
            "t": last.get("t"),
            "o": chunk[0].get("o"),
            "h": max(c.get("h", 0) for c in chunk),
            "l": min(c.get("l", 0) for c in chunk),
            "c": last.get("c"),
            "v": sum(c.get("v", 0) or 0 for c in chunk),
            "rsi5": last.get("rsi5"),
            "rsi15": last.get("rsi15"),
            "stoch_k": last.get("stoch_k"),
            "cci": last.get("cci"),
            "willr": last.get("willr"),
            "ema9": last.get("ema9"),
            "ema20": last.get("ema20"),
            "vwap": last.get("vwap"),
            "bb_pctb": last.get("bb_pctb"),
            "adx": last.get("adx"),
            "atr": last.get("atr"),
            "st_dir": last.get("st_dir"),
            "st_fast_dir": last.get("st_fast_dir"),
        })
    return merged


def get_candles(tf: str = "5m", instrument: str | None = None) -> list[dict]:
    """Return chart candles for the requested timeframe.

    If instrument is None or "NIFTY", returns Nifty candles from engine_state.
    For multi-asset instruments (GOLD_PETAL, CRUDEOILM, USDINR), reads from
    multi_asset_state.json.
    """
    normalized = (tf or "5m").strip().lower()

    if instrument and instrument.upper() not in ("", "NIFTY"):
        ma_state = get_multi_asset_state()
        inst_data = ma_state.get("instruments", {}).get(instrument.upper(), {})
        base = inst_data.get("candles") or []
        if normalized in ("5m", "5min"):
            return base
        if normalized in ("15m", "15min"):
            return _aggregate_candle_rows(base, 3)
        if normalized in ("1h", "60m"):
            return _aggregate_candle_rows(base, 12)
        return base

    state = get_engine_state()
    base = state.get("candles") or []

    if normalized in ("1m", "1min"):
        return state.get("candles_1m") or []
    if normalized in ("5m", "5min"):
        return base
    if normalized in ("15m", "15min"):
        return _aggregate_candle_rows(base, 3)
    if normalized in ("1h", "60m"):
        return _aggregate_candle_rows(base, 12)
    return base


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


# ── Analytics (computed from trades.db) ──────────────────────

STRATEGY_COLORS = {
    "PULLBACK": "#2962ff",
    "STOCH": "#ff8c00",
    "STOCH_CROSS": "#ff8c00",
    "SUPERTREND": "#9c27b0",
    "TREND_RIDE": "#00bcd4",
    "CPR_BREAKOUT": "#4caf50",
}

STRATEGY_CONDITIONS = {
    "PULLBACK": [
        ("RSI 5m", "40–70"),
        ("HTF RSI", "aligned"),
        ("EMA9→EMA20", "pullback"),
        ("ADX", ">10"),
        ("Supertrend", "aligned"),
        ("Cooldown", "3 bars"),
    ],
    "STOCH_CROSS": [
        ("%K×%D", "cross"),
        ("CCI", "confirm"),
        ("Williams %R", "<-80 / >-20"),
        ("ADX", ">20"),
        ("HTF RSI", "aligned"),
    ],
    "SUPERTREND": [
        ("ST(10,3)", "aligned"),
        ("ST(7,2)", "aligned"),
        ("Price vs VWAP", "above/below"),
        ("ADX", ">25"),
        ("RSI", "45–65"),
    ],
}

DISABLED_STRATEGY_BASES = {
    "EMA_MOMENTUM", "SUPERTREND", "VWAP_MOMENTUM", "VWAP_MEAN_REV",
    "RSI_REVERSION", "GAP_TRADE", "ADX_BREAKOUT", "ORB_BREAKOUT",
    "BB_SQUEEZE", "VWAP_BOUNCE", "RSI_DIVERGENCE",
}

HEATMAP_HOURS = [
    "9:30", "10:00", "10:30", "11:00", "11:30",
    "12:00", "12:30", "13:00", "13:30", "14:00", "14:30",
]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _strategy_base(name: str) -> str:
    base = (name or "").split("_")[0].upper()
    if base.startswith("STOCH"):
        return "STOCH_CROSS"
    return base


def _strategy_color(name: str) -> str:
    base = _strategy_base(name)
    for key, color in STRATEGY_COLORS.items():
        if base.startswith(key.split("_")[0]):
            return color
    return "#787b86"


def _strategy_active(name: str) -> bool:
    return _strategy_base(name) not in DISABLED_STRATEGY_BASES


def _get_all_trades() -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = _db_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY date ASC, time ASC, id ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _hour_slot(time_str: str) -> str:
    try:
        parts = time_str.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        slot_min = 30 if minute >= 30 else 0
        return f"{hour}:{slot_min:02d}"
    except (ValueError, IndexError):
        return "—"


def _weekday_index(date_str: str) -> int:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday()
    except ValueError:
        return 0


def _compute_streaks(trades: list[dict]) -> dict:
    max_win = max_loss = 0
    cur_win = cur_loss = 0
    series: list[int] = []
    for t in trades:
        pnl = t.get("pnl") or 0
        if pnl > 0:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
            series.append(cur_win)
        elif pnl < 0:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
            series.append(-cur_loss)
        else:
            cur_win = cur_loss = 0
            series.append(0)
    return {"max_win_streak": max_win, "max_loss_streak": max_loss, "series": series}


def get_strategy_details() -> list[dict]:
    trades = _get_all_trades()
    by_strategy: dict[str, list[dict]] = {}
    for t in trades:
        key = t.get("strategy") or "UNKNOWN"
        by_strategy.setdefault(key, []).append(t)

    result = []
    for strategy, strat_trades in sorted(by_strategy.items()):
        wins = [t for t in strat_trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in strat_trades if (t.get("pnl") or 0) < 0]
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 9999.0
        avg_win = round(gross_profit / len(wins), 2) if wins else 0
        avg_loss = round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0

        cumulative = []
        running = 0.0
        for i, t in enumerate(strat_trades):
            running += t.get("pnl") or 0
            cumulative.append({"i": i + 1, "pnl": round(running, 2)})

        heatmap: dict[str, dict] = {}
        for wd in range(5):
            for hr in HEATMAP_HOURS:
                heatmap[f"{wd}-{hr}"] = {"wins": 0, "losses": 0}
        for t in strat_trades:
            wd = _weekday_index(t.get("date", ""))
            if wd > 4:
                continue
            slot = _hour_slot(t.get("time", ""))
            if slot not in HEATMAP_HOURS:
                continue
            key = f"{wd}-{slot}"
            if (t.get("pnl") or 0) >= 0:
                heatmap[key]["wins"] += 1
            else:
                heatmap[key]["losses"] += 1

        base = _strategy_base(strategy)
        conditions = STRATEGY_CONDITIONS.get(base, [])

        result.append({
            "strategy": strategy,
            "base": base,
            "color": _strategy_color(strategy),
            "active": _strategy_active(strategy),
            "trades": len(strat_trades),
            "wins": len(wins),
            "wr": round(len(wins) / len(strat_trades) * 100, 1) if strat_trades else 0,
            "total_pnl": round(sum(t.get("pnl") or 0 for t in strat_trades), 2),
            "avg_pnl": round(sum(t.get("pnl") or 0 for t in strat_trades) / len(strat_trades), 2) if strat_trades else 0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": pf,
            "best_trade": round(max((t.get("pnl") or 0 for t in strat_trades), default=0), 2),
            "worst_trade": round(min((t.get("pnl") or 0 for t in strat_trades), default=0), 2),
            "conditions": [{"label": a, "value": b} for a, b in conditions],
            "cumulative_pnl": cumulative,
            "heatmap": heatmap,
        })
    return result


def get_analytics() -> dict:
    trades = _get_all_trades()
    if not trades:
        return {
            "summary": {
                "sharpe": None, "sortino": None, "calmar": None,
                "max_win_streak": 0, "max_loss_streak": 0, "expectancy": 0,
                "total_trades": 0, "total_pnl": 0,
            },
            "daily_pnl": {},
            "monthly": [],
            "hourly_avg_pnl": {},
            "recent_daily_pnl": [],
            "hold_vs_pnl": [],
            "streak_series": [],
            "starting_capital": settings.STARTING_CAPITAL,
        }

    daily: dict[str, float] = {}
    monthly: dict[str, dict] = {}
    hourly: dict[str, list[float]] = {}
    hold_vs_pnl = [{"hold": t.get("hold_bars"), "pnl": t.get("pnl")} for t in trades if t.get("hold_bars")]

    for t in trades:
        d = t.get("date", "")
        pnl = t.get("pnl") or 0
        daily[d] = daily.get(d, 0) + pnl
        month = d[:7] if len(d) >= 7 else "unknown"
        bucket = monthly.setdefault(month, {"trades": 0, "wins": 0, "pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0})
        bucket["trades"] += 1
        bucket["pnl"] += pnl
        if pnl > 0:
            bucket["wins"] += 1
            bucket["gross_profit"] += pnl
        elif pnl < 0:
            bucket["gross_loss"] += abs(pnl)
        slot = _hour_slot(t.get("time", ""))
        hourly.setdefault(slot, []).append(pnl)

    hourly_avg = {
        hr: round(sum(vals) / len(vals), 2)
        for hr, vals in hourly.items()
        if vals
    }

    monthly_rows = []
    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = round(m["wins"] / m["trades"] * 100, 1) if m["trades"] else 0
        pf = round(m["gross_profit"] / m["gross_loss"], 2) if m["gross_loss"] > 0 else 9999.0
        monthly_rows.append({
            "month": month,
            "trades": m["trades"],
            "wins": m["wins"],
            "wr": wr,
            "pnl": round(m["pnl"], 2),
            "pf": pf,
        })

    streaks = _compute_streaks(trades)
    total_pnl = sum(t.get("pnl") or 0 for t in trades)
    expectancy = round(total_pnl / len(trades), 2) if trades else 0

    start_cap = settings.STARTING_CAPITAL
    cap = read_capital()
    if cap.get("day_start_capital"):
        start_cap = cap.get("day_start_capital", start_cap)

    daily_returns = []
    running_cap = settings.STARTING_CAPITAL
    for d in sorted(daily.keys()):
        pnl = daily[d]
        if running_cap > 0:
            daily_returns.append(pnl / running_cap)
        running_cap += pnl

    sharpe = sortino = calmar = None
    if len(daily_returns) >= 2:
        mean_r = sum(daily_returns) / len(daily_returns)
        var = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std = var ** 0.5
        if std > 0:
            sharpe = round(mean_r / std * (252 ** 0.5), 2)
        downside = [r for r in daily_returns if r < 0]
        if downside:
            down_std = (sum(r ** 2 for r in downside) / len(downside)) ** 0.5
            if down_std > 0:
                sortino = round(mean_r / down_std * (252 ** 0.5), 2)

    peak = settings.STARTING_CAPITAL
    max_dd = 0.0
    running = settings.STARTING_CAPITAL
    for d in sorted(daily.keys()):
        running += daily[d]
        peak = max(peak, running)
        if peak > 0:
            max_dd = max(max_dd, (peak - running) / peak * 100)
    if max_dd > 0 and total_pnl > 0:
        ann_return = total_pnl / settings.STARTING_CAPITAL
        calmar = round(ann_return / (max_dd / 100), 2)

    recent_daily = [
        {"date": d, "pnl": round(daily[d], 2)}
        for d in sorted(daily.keys())[-14:]
    ]

    return {
        "summary": {
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_win_streak": streaks["max_win_streak"],
            "max_loss_streak": streaks["max_loss_streak"],
            "expectancy": expectancy,
            "total_trades": len(trades),
            "total_pnl": round(total_pnl, 2),
        },
        "daily_pnl": {d: round(v, 2) for d, v in daily.items()},
        "monthly": monthly_rows,
        "hourly_avg_pnl": hourly_avg,
        "recent_daily_pnl": recent_daily,
        "hold_vs_pnl": hold_vs_pnl,
        "streak_series": streaks["series"],
        "starting_capital": settings.STARTING_CAPITAL,
    }


# ── Instruments (multi-asset) ────────────────────────────────

def get_instruments() -> list[dict]:
    """Return status of all registered instruments.

    Merges live data from both engine_state.json (Nifty) and
    multi_asset_state.json (Gold/Crude/USDINR) with DB trade counts.
    """
    try:
        from config.instruments import ALL_INSTRUMENTS
    except ImportError:
        return []

    nifty_state = read_engine_state() or {}
    ma_state = read_multi_asset_state() or {}
    ma_instruments = ma_state.get("instruments", {})
    today = date.today().isoformat()

    result = []
    for name, inst in ALL_INSTRUMENTS.items():
        if name == "NIFTY":
            live_price = nifty_state.get("nifty_price")
            positions = nifty_state.get("positions", [])
        else:
            ma_inst = ma_instruments.get(name, {})
            live_price = ma_inst.get("price")
            positions = ma_inst.get("positions", [])

        daily_pnl = None
        trades_today = None
        if DB_PATH.exists():
            try:
                conn = _db_conn()
                row = conn.execute(
                    "SELECT COUNT(*) as n, "
                    "COALESCE(SUM(pnl), 0) as total "
                    "FROM trades WHERE date = ? AND instrument = ?",
                    (today, name),
                ).fetchone()
                if row:
                    trades_today = row["n"]
                    daily_pnl = round(row["total"], 2)
                conn.close()
            except Exception:
                pass

        if name != "NIFTY":
            ma_inst = ma_instruments.get(name, {})
            if ma_inst.get("daily_pnl") and not daily_pnl:
                daily_pnl = round(ma_inst["daily_pnl"], 2)
            if ma_inst.get("trades_today") and not trades_today:
                trades_today = ma_inst["trades_today"]

        result.append({
            "name": name,
            "display_name": inst.display_name,
            "exchange": inst.exchange,
            "asset_type": inst.asset_type,
            "enabled": inst.enabled and inst.capital_alloc_pct > 0,
            "current_price": live_price,
            "daily_pnl": daily_pnl,
            "trades_today": trades_today,
            "open_positions": len(positions),
            "capital_allocated": inst.capital_alloc_pct,
        })
    return result
