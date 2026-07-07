"""Tests for the DeltaForge dashboard API.

Uses temp directories with fixture data so tests run independently of
live trading state.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from config import settings

# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture()
def data_dir(tmp_path):
    """Create a temp data directory with sample files."""
    data = tmp_path / "data"
    data.mkdir()

    # capital.json
    cap = {
        "current_capital": 11234.50,
        "total_pnl": 1234.50,
        "peak_capital": 12000.00,
        "weekly_pnl": 800.00,
        "max_drawdown": 5.2,
        "daily_pnl": 234.50,
        "trades_today": 3,
        "consecutive_losses": 0,
        "day_start_capital": 11000.00,
        "wins_today": 2,
        "losses_today": 1,
        "last_updated": "2026-06-02T13:05:00+05:30",
    }
    (data / "capital.json").write_text(json.dumps(cap))

    # trades.db
    db_path = data / "trades.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
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
    trades = [
        ("2026-06-02", "10:45", "PULLBACK_3", "LONG", 72, 58, 28,
         102.3, 111.72, 612.0, 7, "TGT", 1, 11234.50),
        ("2026-06-02", "11:35", "SUPERTREND_1", "LONG", 74, 61, 30,
         95.2, 109.37, 921.0, 7, "TGT", 1, 10400.0),
        ("2026-06-02", "12:20", "STOCH_CROSS_2", "LONG", 68, 60, 27,
         98.5, 111.33, 834.0, 7, "TRAIL", 1, 11234.50),
        ("2026-06-01", "10:30", "PULLBACK_3", "SHORT", 65, 42, 18,
         104.6, 96.6, -520.0, 7, "SL", 1, 9480.0),
    ]
    for t in trades:
        conn.execute(
            "INSERT INTO trades (date, time, strategy, direction, confidence, "
            "htf_rsi, adx, entry_price, exit_price, pnl, hold_bars, "
            "exit_reason, lots, capital_after) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            t,
        )
    conn.commit()
    conn.close()

    # events.jsonl -- use today's date so get_events() default filter matches
    today_str = datetime.now().strftime("%Y-%m-%d")
    events = [
        {"ts": f"{today_str}T09:15:00+05:30", "event": "DAY_START",
         "capital": 10000, "mode": "paper", "strategy": "MultiStratV11"},
        {"ts": f"{today_str}T10:45:00+05:30", "event": "PAPER_ENTRY",
         "direction": "LONG", "signal_type": "PULLBACK", "entry_premium": 102.3},
        {"ts": f"{today_str}T12:10:00+05:30", "event": "PAPER_EXIT",
         "direction": "LONG", "signal_type": "SUPERTREND", "pnl": 921.0},
    ]
    with open(data / "events.jsonl", "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    return data


@pytest.fixture()
def log_dir(tmp_path):
    """Create a temp log directory with sample JSON log."""
    log_root = tmp_path / "logs"
    log_json = log_root / "json"
    log_json.mkdir(parents=True)

    lines = [
        {"ts": "2026-06-02T09:15:00+05:30", "level": "INFO",
         "module": "scheduler", "function": "run", "line": 10,
         "message": "Market open -- starting trading loop"},
        {"ts": "2026-06-02T10:45:00+05:30", "level": "INFO",
         "module": "engine", "function": "_tick", "line": 274,
         "message": "PULLBACK LONG signal fired: conf=72"},
        {"ts": "2026-06-02T10:45:01+05:30", "level": "DEBUG",
         "module": "engine", "function": "_tick", "line": 280,
         "message": "Indicators: RSI5=58.2 ADX=28.5 ST=BULL"},
        {"ts": "2026-06-02T11:00:00+05:30", "level": "WARNING",
         "module": "risk", "function": "check_realtime", "line": 50,
         "message": "Daily loss 26% of limit"},
        {"ts": "2026-06-02T12:00:00+05:30", "level": "ERROR",
         "module": "broker", "function": "get_ltp", "line": 120,
         "message": "REST fallback: WebSocket stale > 10s"},
    ]
    with open(log_json / "trading_2026-06-02.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    return log_root


@pytest.fixture()
def patched_app(data_dir, log_dir, tmp_path):
    """Create a FastAPI TestClient with data_access pointing at temp dirs."""
    from dashboard import data_access as da

    patches = {
        "DATA_DIR": data_dir,
        "LOG_JSON_DIR": log_dir / "json",
        "CAPITAL_FILE": data_dir / "capital.json",
        "DB_PATH": data_dir / "trades.db",
        "EVENTS_FILE": data_dir / "events.jsonl",
        "HALT_FLAG": data_dir / "HALT",
        "SESSION_LOCK": data_dir / "session.lock",
        "ENGINE_STATE_FILE": data_dir / "engine_state.json",
    }

    with mock.patch.multiple(da, **patches):
        from dashboard.server import app
        client = TestClient(app)
        yield client, data_dir


# ── Status ───────────────────────────────────────────────────

class TestStatus:
    def test_returns_correct_shape(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/status")
        assert r.status_code == 200
        d = r.json()
        assert d["current_capital"] == 11234.50
        assert d["daily_pnl"] == 234.50
        assert d["trades_today"] == 3
        assert d["wins_today"] == 2
        assert d["halted"] is False
        assert d["session_active"] is False

    def test_reflects_halt_flag(self, patched_app):
        client, data_dir = patched_app
        halt = {"halted_at": "2026-06-02T11:00:00+05:30",
                "reason": "Daily loss limit"}
        (data_dir / "HALT").write_text(json.dumps(halt))
        r = client.get("/api/status")
        d = r.json()
        assert d["halted"] is True
        assert d["halt_reason"] == "Daily loss limit"

    def test_reflects_session_lock(self, patched_app):
        client, data_dir = patched_app
        lock = {"pid": 12345, "started_at": "2026-06-02T08:30:00+05:30"}
        (data_dir / "session.lock").write_text(json.dumps(lock))
        with mock.patch("os.kill", return_value=None):
            r = client.get("/api/status")
        d = r.json()
        assert d["session_active"] is True
        assert d["session_pid"] == 12345


# ── Capital ──────────────────────────────────────────────────

class TestCapital:
    def test_returns_full_state(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/capital")
        assert r.status_code == 200
        d = r.json()
        assert d["current_capital"] == 11234.50
        assert d["peak_capital"] == 12000.00
        assert d["trades_today"] == 3


# ── Trades ───────────────────────────────────────────────────

class TestTrades:
    def test_returns_all_trades(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades")
        assert r.status_code == 200
        trades = r.json()
        assert len(trades) == 4

    def test_filters_by_date(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades?date=2026-06-02")
        trades = r.json()
        assert len(trades) == 3
        assert all(t["date"] == "2026-06-02" for t in trades)

    def test_filters_by_strategy(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades?strategy=PULLBACK")
        trades = r.json()
        assert len(trades) == 2
        assert all("PULLBACK" in t["strategy"] for t in trades)

    def test_respects_limit(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades?limit=2")
        trades = r.json()
        assert len(trades) == 2

    def test_newest_first(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades")
        trades = r.json()
        ids = [t["id"] for t in trades]
        assert ids == sorted(ids, reverse=True)


class TestTradesSummary:
    def test_daily_summary(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades/summary?date=2026-06-02")
        assert r.status_code == 200
        d = r.json()
        assert d["trades"] == 3
        assert d["wins"] == 3
        assert d["total_pnl"] == 2367.0

    def test_empty_date(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades/summary?date=2020-01-01")
        d = r.json()
        assert d["trades"] == 0


class TestStrategyStats:
    def test_returns_per_strategy(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades/strategy-stats")
        assert r.status_code == 200
        stats = r.json()
        assert len(stats) >= 1
        names = {s["strategy"] for s in stats}
        assert "PULLBACK_3" in names

    def test_includes_profit_factor(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades/strategy-stats")
        for s in r.json():
            assert "profit_factor" in s
            assert "best_trade" in s
            assert "worst_trade" in s


class TestStrategyDetails:
    def test_returns_strategy_details(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades/strategy-details")
        assert r.status_code == 200
        details = r.json()
        assert len(details) >= 1
        d = details[0]
        assert "strategy" in d
        assert "heatmap" in d
        assert "cumulative_pnl" in d
        assert "conditions" in d

    def test_includes_avg_win_loss(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/trades/strategy-details")
        for d in r.json():
            assert "avg_win" in d
            assert "avg_loss" in d
            assert "active" in d


class TestAnalytics:
    def test_returns_analytics_shape(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/analytics")
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        assert "daily_pnl" in data
        assert "monthly" in data
        assert "hourly_avg_pnl" in data
        assert "recent_daily_pnl" in data

    def test_summary_has_trade_counts(self, patched_app):
        client, _ = patched_app
        s = r.json()["summary"] if (r := client.get("/api/analytics")).status_code == 200 else {}
        assert s["total_trades"] == 4
        assert "expectancy" in s

    def test_monthly_breakdown(self, patched_app):
        client, _ = patched_app
        monthly = client.get("/api/analytics").json()["monthly"]
        assert len(monthly) >= 1
        assert "month" in monthly[0]
        assert "pnl" in monthly[0]


# ── Events ───────────────────────────────────────────────────

class TestEvents:
    @staticmethod
    def _today():
        return datetime.now().strftime("%Y-%m-%d")

    def test_returns_events(self, patched_app):
        client, _ = patched_app
        r = client.get(f"/api/events?date={self._today()}")
        assert r.status_code == 200
        events = r.json()
        assert len(events) == 3

    def test_filters_by_type(self, patched_app):
        client, _ = patched_app
        r = client.get(f"/api/events?date={self._today()}&type=PAPER_ENTRY")
        events = r.json()
        assert len(events) == 1
        assert events[0]["event"] == "PAPER_ENTRY"

    def test_newest_first(self, patched_app):
        client, _ = patched_app
        r = client.get(f"/api/events?date={self._today()}")
        events = r.json()
        assert events[0]["event"] == "PAPER_EXIT"

    def test_respects_limit(self, patched_app):
        client, _ = patched_app
        r = client.get(f"/api/events?date={self._today()}&limit=1")
        assert len(r.json()) == 1


# ── Logs ─────────────────────────────────────────────────────

class TestLogs:
    def test_returns_logs(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/logs?date=2026-06-02")
        assert r.status_code == 200
        logs = r.json()
        assert len(logs) == 5

    def test_filters_by_level(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/logs?date=2026-06-02&level=ERROR")
        logs = r.json()
        assert len(logs) == 1
        assert logs[0]["level"] == "ERROR"

    def test_filters_by_module(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/logs?date=2026-06-02&module=engine")
        logs = r.json()
        assert len(logs) == 2
        assert all(l["module"] == "engine" for l in logs)

    def test_text_search(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/logs?date=2026-06-02&search=PULLBACK")
        logs = r.json()
        assert len(logs) == 1
        assert "PULLBACK" in logs[0]["message"]

    def test_combined_filters(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/logs?date=2026-06-02&level=INFO,DEBUG&module=engine")
        logs = r.json()
        assert len(logs) == 2


# ── Config ───────────────────────────────────────────────────

class TestConfig:
    def test_returns_grouped_settings(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/config")
        assert r.status_code == 200
        cfg = r.json()
        assert "Capital & Sizing" in cfg
        assert "Risk Management" in cfg
        assert "Timing" in cfg
        keys = {item["key"] for group in cfg.values() for item in group}
        assert "STARTING_CAPITAL" in keys
        assert "PREMIUM_SL_PCT" in keys

    def test_no_secrets(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/config")
        cfg = r.json()
        all_keys = {item["key"] for group in cfg.values() for item in group}
        assert "ANGEL_API_KEY" not in all_keys
        assert "SLACK_BOT_TOKEN" not in all_keys


# ── Risk ─────────────────────────────────────────────────────

class TestRisk:
    def test_returns_all_gates(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/risk")
        assert r.status_code == 200
        gates = r.json()
        assert len(gates) == 8
        names = {g["name"] for g in gates}
        assert "Min Capital" in names
        assert "Daily Loss" in names
        assert "Kill Switch" in names

    def test_gate_fields(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/risk")
        for gate in r.json():
            assert "name" in gate
            assert "current" in gate
            assert "threshold" in gate
            assert "status" in gate
            assert "pct" in gate


# ── Broker ───────────────────────────────────────────────────

class TestBroker:
    def test_returns_broker_info(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/broker")
        assert r.status_code == 200
        d = r.json()
        assert "trading_mode" in d
        assert "lot_size" in d
        assert d["session_active"] is False


# ── Halt toggle ──────────────────────────────────────────────

class TestHaltToggle:
    def test_toggle_halt_on(self, patched_app):
        """POST /api/halt should set halt when not halted."""
        client, data_dir = patched_app
        r = client.post("/api/halt")
        assert r.status_code == 200
        d = r.json()
        assert d["halted"] is True
        assert (data_dir / "HALT").exists()

    def test_toggle_halt_off(self, patched_app):
        """POST /api/halt should clear halt when already halted."""
        client, data_dir = patched_app
        (data_dir / "HALT").write_text('{"reason":"test","halted_at":"2026-06-02T11:00:00"}')
        r = client.post("/api/halt")
        assert r.status_code == 200
        d = r.json()
        assert d["halted"] is False
        assert not (data_dir / "HALT").exists()


# ── Signals ──────────────────────────────────────────────────

class TestSignals:
    def test_returns_signals_from_events(self, patched_app):
        """GET /api/signals should parse PAPER_ENTRY events."""
        client, _ = patched_app
        r = client.get("/api/signals")
        assert r.status_code == 200
        signals = r.json()
        assert isinstance(signals, list)
        paper_entries = [s for s in signals if s.get("type") != "UNKNOWN"]
        assert len(paper_entries) >= 1


# ── Engine ───────────────────────────────────────────────────

class TestEngine:
    def test_engine_state_returns_default(self, patched_app):
        """GET /api/engine should return default state when no engine_state.json."""
        client, _ = patched_app
        r = client.get("/api/engine")
        assert r.status_code == 200
        d = r.json()
        assert d["candle_count"] == 0
        assert d["positions"] == []


# ── Candles ──────────────────────────────────────────────────

class TestCandles:
    def test_candles_returns_list(self, patched_app):
        """GET /api/candles should return empty list by default."""
        client, _ = patched_app
        r = client.get("/api/candles")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_candles_accepts_timeframe(self, patched_app):
        client, _ = patched_app
        r = client.get("/api/candles?tf=15m")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ── CORS ─────────────────────────────────────────────────────

class TestCORS:
    def test_cors_headers(self, patched_app):
        client, _ = patched_app
        origin = f"http://localhost:{settings.DASHBOARD_PORT}"
        r = client.options(
            "/api/status",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.headers.get("access-control-allow-origin") == origin


# ── Static Serving ───────────────────────────────────────────

class TestStatic:
    def test_dashboard_html_served(self, patched_app):
        client, _ = patched_app
        r = client.get("/")
        assert r.status_code == 200
        assert "DeltaForge" in r.text


# ── WebSocket ────────────────────────────────────────────────

class TestWebSocket:
    def test_ws_connects(self, patched_app):
        client, _ = patched_app
        with client.websocket_connect("/ws/live"):
            pass

    def test_ws_receives_capital_update(self, patched_app):
        client, data_dir = patched_app
        with client.websocket_connect("/ws/live") as ws:
            import asyncio

            from dashboard.websocket import manager

            async def _broadcast():
                await manager.broadcast({
                    "type": "capital",
                    "data": {"current_capital": 99999},
                })

            loop = asyncio.new_event_loop()
            loop.run_until_complete(_broadcast())
            loop.close()

            msg = ws.receive_json()
            assert msg["type"] == "capital"
            assert msg["data"]["current_capital"] == 99999
