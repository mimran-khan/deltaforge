"""Tests for alerts/imessage_bot.py

Unit tests mock subprocess -- no actual messages sent.
Live test (--live flag) sends a real iMessage to IMESSAGE_RECIPIENT from .env.

Usage:
    python -m tests.test_imessage          # unit tests only
    python -m tests.test_imessage --live   # sends a real iMessage
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestStripHtml(unittest.TestCase):

    def test_removes_bold_tags(self):
        from alerts.imessage_bot import _strip_html
        self.assertEqual(_strip_html("<b>bold</b>"), "bold")

    def test_removes_nested_tags(self):
        from alerts.imessage_bot import _strip_html
        self.assertEqual(
            _strip_html("<b>PnL: <i>Rs 500</i></b>"),
            "PnL: Rs 500"
        )

    def test_preserves_plain_text(self):
        from alerts.imessage_bot import _strip_html
        self.assertEqual(_strip_html("no tags here"), "no tags here")

    def test_empty_string(self):
        from alerts.imessage_bot import _strip_html
        self.assertEqual(_strip_html(""), "")


class TestSendImessage(unittest.TestCase):

    @patch("alerts.imessage_bot.settings")
    @patch("alerts.imessage_bot.subprocess.run")
    def test_sends_with_valid_recipient(self, mock_run, mock_settings):
        mock_settings.IMESSAGE_RECIPIENT = "+919999999999"
        mock_run.return_value = MagicMock(returncode=0)

        from alerts.imessage_bot import _send_imessage
        result = _send_imessage("Test message")

        self.assertTrue(result)
        mock_run.assert_called_once()
        args = mock_run.call_args
        self.assertEqual(args[0][0][0], "osascript")
        self.assertIn("+919999999999", args[0][0][2])
        self.assertIn("Test message", args[0][0][2])

    @patch("alerts.imessage_bot.settings")
    def test_skips_when_no_recipient(self, mock_settings):
        mock_settings.IMESSAGE_RECIPIENT = ""

        from alerts.imessage_bot import _send_imessage
        result = _send_imessage("Test")

        self.assertFalse(result)

    @patch("alerts.imessage_bot.settings")
    @patch("alerts.imessage_bot.subprocess.run")
    def test_handles_osascript_failure(self, mock_run, mock_settings):
        mock_settings.IMESSAGE_RECIPIENT = "+919999999999"
        mock_run.return_value = MagicMock(returncode=1, stderr="error")

        from alerts.imessage_bot import _send_imessage
        result = _send_imessage("Test")

        self.assertFalse(result)

    @patch("alerts.imessage_bot.settings")
    @patch("alerts.imessage_bot.subprocess.run")
    def test_handles_timeout(self, mock_run, mock_settings):
        mock_settings.IMESSAGE_RECIPIENT = "+919999999999"
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="osascript", timeout=15)

        from alerts.imessage_bot import _send_imessage
        result = _send_imessage("Test")

        self.assertFalse(result)

    @patch("alerts.imessage_bot.settings")
    @patch("alerts.imessage_bot.subprocess.run")
    def test_escapes_quotes(self, mock_run, mock_settings):
        mock_settings.IMESSAGE_RECIPIENT = "+919999999999"
        mock_run.return_value = MagicMock(returncode=0)

        from alerts.imessage_bot import _send_imessage
        _send_imessage('He said "hello"')

        script = mock_run.call_args[0][0][2]
        self.assertIn('\\"hello\\"', script)


class TestSendAlert(unittest.TestCase):

    @patch("alerts.imessage_bot._send_imessage")
    def test_strips_html_before_sending(self, mock_send):
        mock_send.return_value = True

        from alerts.imessage_bot import send_alert
        send_alert("<b>ALERT</b>: market <i>crashed</i>")

        mock_send.assert_called_once_with("ALERT: market crashed")


class TestSendTradeAlert(unittest.TestCase):

    @patch("alerts.imessage_bot.send_alert")
    def test_entry_format(self, mock_alert):
        from alerts.imessage_bot import send_trade_alert
        send_trade_alert("ENTRY", "pullback_1", "NIFTY_LONG",
                         price=100.50, quantity=75, sl=50.25, target=150.75)

        msg = mock_alert.call_args[0][0]
        self.assertIn("TRADE OPENED", msg)
        self.assertIn("100.50", msg)
        self.assertIn("50.25", msg)

    @patch("alerts.imessage_bot.send_alert")
    def test_exit_format(self, mock_alert):
        from alerts.imessage_bot import send_trade_alert
        send_trade_alert("SL_HIT", "stoch_cross_0", "NIFTY_SHORT",
                         price=80.00, quantity=75, pnl=-1500)

        msg = mock_alert.call_args[0][0]
        self.assertIn("TRADE CLOSED", msg)
        self.assertIn("-1500", msg)
        self.assertIn("Stop Loss", msg)


class TestSendEodReport(unittest.TestCase):

    @patch("alerts.imessage_bot.send_alert")
    def test_eod_format(self, mock_alert):
        from alerts.imessage_bot import send_eod_report
        send_eod_report({
            "capital": 11000,
            "daily_pnl": 1000,
            "daily_pnl_pct": 10.0,
            "trades": 2,
            "wins": 2,
            "losses": 0,
            "win_rate": 100,
            "total_pnl": 1000,
            "max_drawdown": 2.5,
        })

        msg = mock_alert.call_args[0][0]
        self.assertIn("EOD REPORT", msg)
        self.assertIn("11000", msg)
        self.assertIn("+1000", msg)
        self.assertIn("100%", msg)


class TestSendSystemAlert(unittest.TestCase):

    @patch("alerts.imessage_bot.send_alert")
    def test_system_alert_format(self, mock_alert):
        from alerts.imessage_bot import send_system_alert
        send_system_alert("KILL SWITCH", "Daily loss exceeded")

        msg = mock_alert.call_args[0][0]
        self.assertIn("KILL SWITCH", msg)
        self.assertIn("Daily loss exceeded", msg)


def run_live_test():
    """Send a real iMessage -- requires IMESSAGE_RECIPIENT in .env."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    import importlib

    import config.settings
    from alerts.imessage_bot import send_system_alert
    importlib.reload(config.settings)

    recipient = config.settings.IMESSAGE_RECIPIENT
    if not recipient:
        print("IMESSAGE_RECIPIENT not set in .env -- cannot run live test.")
        print("Add IMESSAGE_RECIPIENT=+91XXXXXXXXXX to your .env file.")
        return

    print(f"Sending test iMessage to {recipient}...")
    ok = send_system_alert(
        "Trading Agent Test",
        "If you see this, iMessage alerts are working."
    )
    print(f"Result: {'sent' if ok else 'failed -- check Messages.app permissions'}")


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live_test()
    else:
        unittest.main(argv=[sys.argv[0]])
