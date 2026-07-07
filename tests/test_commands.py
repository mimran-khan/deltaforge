"""Tests for the iMessage command system (CommandRouter + CommandPoller).

Covers:
  - All 6 command handlers (/status, /stop, /start, /market, /paper, /help)
  - Unknown/help commands
  - Router dispatch edge cases
  - Poller lifecycle (start/stop)
  - Poller chat.db query simulation
  - Engine not ready (offline) scenarios
  - SessionManager lifecycle
  - Backtest command (/paper N)

Usage:
    ./venv/bin/python -m pytest tests/test_commands.py -v
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alerts.command_listener import (
    HELP_TEXT,
    CommandPoller,
    CommandRouter,
    SessionManager,
)
from config import settings


def _mock_engine():
    """Build a mock TradingEngine with realistic attributes."""
    engine = MagicMock()
    engine._running = True
    engine._paper_positions = []
    engine.position_mgr = None

    engine.get_status_dict.return_value = {
        "capital": 12500.0,
        "daily_pnl": 350.0,
        "daily_pnl_pct": 2.8,
        "trades": 5,
        "wins": 4,
        "losses": 1,
        "drawdown": 1.2,
        "consecutive_losses": 0,
        "open_positions": 0,
        "running": True,
    }

    engine.get_market_snapshot.return_value = {
        "nifty": 24150.50,
        "banknifty": 51200.75,
    }
    engine.broker = MagicMock()
    engine.broker.is_active = True
    engine.risk = MagicMock()

    return engine


# ─── CommandRouter ────────────────────────────────────────────

class TestCommandRouter(unittest.TestCase):
    """Test CommandRouter dispatch and all handlers."""

    def setUp(self):
        self.engine = _mock_engine()
        self.router = CommandRouter(engine=self.engine)

    def test_status_command_with_engine(self):
        resp = self.router.dispatch("/status")
        self.assertIn("DELTAFORGE STATUS", resp)
        self.assertIn("Rs 12,500", resp)
        self.assertIn("+350", resp)
        self.assertIn("80%", resp)
        self.assertIn("W:4 L:1", resp)

    def test_status_no_engine_no_file(self):
        router = CommandRouter(engine=None)
        with patch.object(settings, "CAPITAL_FILE", Path("/nonexistent/capital.json")):
            resp = router.dispatch("/status")
        self.assertIn("offline", resp.lower())

    def test_status_no_engine_with_file(self):
        """Offline status reads capital from disk."""
        import json
        import os
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump({
            "current_capital": 11000,
            "peak_capital": 12000,
            "daily_pnl": -200,
            "trades_today": 3,
            "wins_today": 1,
            "losses_today": 2,
            "consecutive_losses": 2,
            "last_updated": "2026-05-30T15:30:00",
        }, tmp)
        tmp.close()

        router = CommandRouter(engine=None)
        with patch.object(settings, "CAPITAL_FILE", Path(tmp.name)):
            resp = router.dispatch("/status")

        os.unlink(tmp.name)

        self.assertIn("OFFLINE", resp)
        self.assertIn("Rs 11,000", resp)
        self.assertIn("-200", resp)

    @patch("alerts.command_listener.set_halt")
    def test_stop_command(self, mock_halt):
        resp = self.router.dispatch("/stop")
        mock_halt.assert_called_once_with("Manual stop via command")
        self.assertIn("TRADING STOPPED", resp)

    @patch("alerts.command_listener.set_halt")
    def test_stop_squares_off_paper(self, mock_halt):
        self.engine._paper_positions = [MagicMock()]
        self.router.dispatch("/stop")
        self.engine._square_off_all.assert_called_once_with("MANUAL_STOP")

    @patch("alerts.command_listener.is_halted", return_value=False)
    def test_start_already_running_engine(self, _):
        resp = self.router.dispatch("/start")
        self.assertIn("already running", resp)

    @patch("alerts.command_listener.is_session_running", return_value=(False, 0))
    @patch("alerts.command_listener.clear_halt")
    @patch("alerts.command_listener.is_halted", return_value=True)
    def test_start_clears_halt_and_launches(self, _, mock_clear, __):
        self.engine._running = False
        self.router.session_mgr = MagicMock()
        self.router.session_mgr.is_running = False
        self.router.session_mgr.start_session.return_value = "TRADING SESSION STARTED"
        resp = self.router.dispatch("/start")
        mock_clear.assert_called_once()
        self.engine.risk.resume.assert_called_once()
        self.assertIn("TRADING SESSION STARTED", resp)

    def test_market_command(self):
        resp = self.router.dispatch("/market")
        self.assertIn("24,150.50", resp)
        self.assertIn("51,200.75", resp)

    def test_market_no_engine(self):
        router = CommandRouter(engine=None)
        resp = router.dispatch("/market")
        self.assertIn("N/A", resp)

    def test_help_command(self):
        resp = self.router.dispatch("/help")
        self.assertEqual(resp, HELP_TEXT)
        self.assertIn("status", resp)
        self.assertIn("stop", resp)
        self.assertIn("start", resp)
        self.assertIn("market", resp)
        self.assertIn("paper", resp)
        self.assertIn("help", resp)

    def test_unknown_command(self):
        resp = self.router.dispatch("/foo")
        self.assertIn("Unknown command", resp)
        self.assertIn("help", resp)

    def test_empty_command(self):
        resp = self.router.dispatch("")
        self.assertIn("Unknown command", resp)

    def test_command_case_insensitive(self):
        resp = self.router.dispatch("/STATUS")
        self.assertIn("DELTAFORGE STATUS", resp)

    def test_command_with_extra_args(self):
        resp = self.router.dispatch("/status extra args here")
        self.assertIn("DELTAFORGE STATUS", resp)


# ─── /paper command ──────────────────────────────────────────

class TestPaperCommand(unittest.TestCase):
    """Test /paper <days> backtest command."""

    def setUp(self):
        self.router = CommandRouter(engine=None)

    @patch("alerts.command_listener.send_alert")
    def test_paper_invalid_days(self, mock_send):
        resp = self.router.dispatch("/paper abc")
        self.assertIn("Invalid days", resp)

    @patch("alerts.command_listener.send_alert")
    def test_paper_days_too_small(self, mock_send):
        resp = self.router.dispatch("/paper 2")
        self.assertIn("between 5 and 500", resp)

    @patch("alerts.command_listener.send_alert")
    def test_paper_days_too_large(self, mock_send):
        resp = self.router.dispatch("/paper 999")
        self.assertIn("between 5 and 500", resp)

    @patch("alerts.command_listener.send_alert")
    def test_paper_default_60(self, mock_send):
        resp = self.router.dispatch("/paper")
        # /paper with no args runs backtest in background, returns None
        self.assertIsNone(resp)
        # First call should be the "Running..." message
        mock_send.assert_called_once()
        self.assertIn("60-day", mock_send.call_args[0][0])

    @patch("alerts.command_listener.send_alert")
    def test_paper_runs_backtest_async(self, mock_send):
        """Test that /paper 10 triggers an async backtest."""
        resp = self.router.dispatch("/paper 10")
        self.assertIsNone(resp)
        mock_send.assert_called_once()
        self.assertIn("10-day", mock_send.call_args[0][0])

        # Wait for async backtest thread to finish (engine takes ~30s)
        for _ in range(12):
            time.sleep(5)
            if mock_send.call_count > 1:
                break

        # Should have been called again with results
        self.assertGreater(mock_send.call_count, 1)
        result_msg = mock_send.call_args_list[-1][0][0]
        self.assertIn("BACKTEST", result_msg)
        self.assertIn("PF:", result_msg)
        self.assertIn("Verdict:", result_msg)


# ─── SessionManager ─────────────────────────────────────────

class TestSessionManager(unittest.TestCase):
    """Test SessionManager process lifecycle."""

    def test_not_running_initially(self):
        sm = SessionManager()
        self.assertFalse(sm.is_running)

    def test_stop_when_not_running(self):
        sm = SessionManager()
        result = sm.stop_session()
        self.assertFalse(result)

    def test_is_running_dead_proc(self):
        sm = SessionManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already exited
        sm._proc = mock_proc
        self.assertFalse(sm.is_running)

    def test_is_running_live_proc(self):
        sm = SessionManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        sm._proc = mock_proc
        self.assertTrue(sm.is_running)

    def test_start_already_running(self):
        sm = SessionManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        sm._proc = mock_proc
        result = sm.start_session()
        self.assertIn("already running", result)


# ─── CommandPoller Lifecycle ─────────────────────────────────

class TestCommandPollerLifecycle(unittest.TestCase):
    """Test CommandPoller start/stop and watermark init."""

    @patch.object(settings, "ALERT_METHOD", "imessage")
    @patch.object(settings, "IMESSAGE_RECIPIENT", "")
    def test_no_recipient_disables(self):
        poller = CommandPoller()
        poller.start()
        self.assertFalse(poller._backend._running)

    @patch("alerts.command_listener.CHAT_DB", Path("/nonexistent/chat.db"))
    @patch.object(settings, "ALERT_METHOD", "imessage")
    @patch.object(settings, "IMESSAGE_RECIPIENT", "+919876543210")
    def test_no_chatdb_disables(self):
        poller = CommandPoller()
        poller.start()
        self.assertFalse(poller._backend._running)

    def test_stop_idempotent(self):
        poller = CommandPoller()
        poller.stop()
        poller.stop()

    def test_engine_property(self):
        engine = _mock_engine()
        poller = CommandPoller(engine=engine)
        self.assertEqual(poller.engine, engine)

        new_engine = _mock_engine()
        poller.engine = new_engine
        self.assertEqual(poller.router.engine, new_engine)


# ─── CommandPoller with Mock DB ──────────────────────────────

class TestCommandPollerWithMockDB(unittest.TestCase):
    """Test CommandPoller against a real SQLite database mimicking chat.db schema."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "chat.db"
        self._create_mock_chatdb()
        self.engine = _mock_engine()

    def _create_mock_chatdb(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY,
                chat_identifier TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                date INTEGER,
                is_from_me INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE chat_message_join (
                chat_id INTEGER,
                message_id INTEGER
            )
        """)
        conn.execute("INSERT INTO chat (ROWID, chat_identifier) VALUES (1, '+919876543210')")
        conn.commit()
        conn.close()

    def _insert_message(self, text, is_from_me=0, chat_id=1):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "INSERT INTO message (text, date, is_from_me) VALUES (?, 0, ?)",
            (text, is_from_me),
        )
        msg_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
            (chat_id, msg_id),
        )
        conn.commit()
        conn.close()
        return msg_id

    @patch("alerts.command_listener.send_alert")
    @patch("alerts.command_listener.CHAT_DB")
    @patch.object(settings, "IMESSAGE_RECIPIENT", "+919876543210")
    def test_processes_slash_commands(self, mock_db_path, mock_send):
        mock_db_path.__str__ = lambda _: str(self.db_path)
        mock_db_path.exists.return_value = True

        self._insert_message("hello world")
        self._insert_message("random text")

        poller = CommandPoller(engine=self.engine, poll_interval=1)
        poller._last_rowid = 0

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.execute("SELECT MAX(ROWID) FROM message")
        max_row = cursor.fetchone()[0]
        conn.close()

        poller._last_rowid = max_row

        self._insert_message("/status")

        poller._recipient = "+919876543210"
        poller._check_new_commands = lambda: self._manual_check(poller, mock_send)
        self._manual_check(poller, mock_send)

        mock_send.assert_called()
        status_calls = [c for c in mock_send.call_args_list if "DELTAFORGE STATUS" in c[0][0]]
        self.assertTrue(len(status_calls) >= 1, "Expected at least one STATUS response")
        self.assertIn("DELTAFORGE STATUS", status_calls[0][0][0])

    def _manual_check(self, poller, mock_send):
        """Manually run the check using the test DB path."""
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        except Exception:
            return

        try:
            query = """
                SELECT m.ROWID, m.text
                FROM message m
                JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                JOIN chat c ON cmj.chat_id = c.ROWID
                WHERE m.is_from_me = 0
                  AND m.ROWID > ?
                  AND c.chat_identifier = ?
                ORDER BY m.ROWID ASC
                LIMIT 10
            """
            cursor = conn.execute(query, (poller._last_rowid, poller._recipient))
            rows = cursor.fetchall()

            for rowid, text in rows:
                poller._last_rowid = rowid
                if not text or not text.strip().startswith("/"):
                    continue
                response = poller.router.dispatch(text.strip())
                if response:
                    mock_send(response)

        finally:
            conn.close()

    @patch("alerts.command_listener.send_alert")
    @patch("alerts.command_listener.CHAT_DB")
    @patch.object(settings, "IMESSAGE_RECIPIENT", "+919876543210")
    def test_skips_non_slash_messages(self, mock_db_path, mock_send):
        mock_db_path.__str__ = lambda _: str(self.db_path)
        mock_db_path.exists.return_value = True

        self._insert_message("just chatting")
        self._insert_message("hey there")

        poller = CommandPoller(engine=self.engine, poll_interval=1)
        poller._last_rowid = 0
        poller._recipient = "+919876543210"
        self._manual_check(poller, mock_send)

        mock_send.assert_not_called()

    @patch("alerts.command_listener.send_alert")
    @patch("alerts.command_listener.CHAT_DB")
    @patch.object(settings, "IMESSAGE_RECIPIENT", "+919876543210")
    def test_watermark_prevents_reprocessing(self, mock_db_path, mock_send):
        mock_db_path.__str__ = lambda _: str(self.db_path)
        mock_db_path.exists.return_value = True

        msg_id = self._insert_message("/status")

        poller = CommandPoller(engine=self.engine, poll_interval=1)
        poller._last_rowid = msg_id
        poller._recipient = "+919876543210"

        # Reset mock to clear any calls from async threads in prior tests
        mock_send.reset_mock()
        self._manual_check(poller, mock_send)

        mock_send.assert_not_called()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ─── TradingEngine methods ───────────────────────────────────

class TestTradingEngineStatusMethods(unittest.TestCase):
    """Test get_status_dict and get_market_snapshot on real TradingEngine."""

    def test_get_status_dict(self):
        from engine.trading_engine import TradingEngine
        with patch.object(TradingEngine, '__init__', lambda self: None):
            engine = TradingEngine()
            engine.capital = MagicMock()
            engine.capital.current_capital = 10000
            engine.capital.daily_pnl = 500
            engine.capital.day_start_capital = 10000
            engine.capital.trades_today = 3
            engine.capital.wins_today = 2
            engine.capital.losses_today = 1
            engine.capital.drawdown_pct = 0.5
            engine.capital.consecutive_losses = 0
            engine._paper_positions = []
            engine._running = True

            status = engine.get_status_dict()
            self.assertEqual(status["capital"], 10000)
            self.assertEqual(status["daily_pnl"], 500)
            self.assertAlmostEqual(status["daily_pnl_pct"], 5.0)
            self.assertEqual(status["trades"], 3)
            self.assertEqual(status["wins"], 2)
            self.assertEqual(status["losses"], 1)
            self.assertEqual(status["open_positions"], 0)

    def test_get_market_snapshot(self):
        from engine.trading_engine import TradingEngine
        with patch.object(TradingEngine, '__init__', lambda self: None):
            engine = TradingEngine()
            engine.broker = MagicMock()
            engine.broker.is_active = True
            engine.broker.get_ltp.side_effect = lambda ex, sym, tok: {
                "99926000": 24100.0,
                "99926009": 51000.0,
            }.get(tok)
            engine._nifty_spot = 0

            snap = engine.get_market_snapshot()
            self.assertEqual(snap["nifty"], 24100.0)
            self.assertEqual(snap["banknifty"], 51000.0)

    def test_get_market_snapshot_broker_down(self):
        from engine.trading_engine import TradingEngine
        with patch.object(TradingEngine, '__init__', lambda self: None):
            engine = TradingEngine()
            engine.broker = MagicMock()
            engine.broker.is_active = False
            engine._nifty_spot = 24200.0

            snap = engine.get_market_snapshot()
            self.assertEqual(snap["nifty"], 24200.0)
            self.assertIsNone(snap["banknifty"])


# ─── DailyScheduler integration ─────────────────────────────

class TestDailySchedulerCommandIntegration(unittest.TestCase):
    """Test that DailyScheduler wires CommandPoller correctly."""

    def test_scheduler_init_with_commands(self):
        from automation.daily_scheduler import DailyScheduler
        s = DailyScheduler(enable_commands=True)
        self.assertTrue(s._enable_commands)
        self.assertIsNone(s._command_poller)

    def test_scheduler_init_without_commands(self):
        from automation.daily_scheduler import DailyScheduler
        s = DailyScheduler(enable_commands=False)
        self.assertFalse(s._enable_commands)

    def test_cleanup_with_no_poller(self):
        from automation.daily_scheduler import DailyScheduler
        s = DailyScheduler(enable_commands=False)
        s._watchdog_proc = None
        s.scheduler = MagicMock()
        s._cleanup()
        s.scheduler.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
