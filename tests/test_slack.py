"""Tests for alerts/slack_bot.py

Unit tests mock slack_sdk -- no actual messages sent.
Live test (--live flag) sends a real Slack message using SLACK_BOT_TOKEN from .env.

Usage:
    python -m tests.test_slack          # unit tests only
    python -m tests.test_slack --live   # sends a real Slack message
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestStripHtml(unittest.TestCase):

    def test_removes_bold_tags(self):
        from alerts.slack_bot import _strip_html
        self.assertEqual(_strip_html("<b>bold</b>"), "bold")

    def test_removes_nested_tags(self):
        from alerts.slack_bot import _strip_html
        self.assertEqual(
            _strip_html("<b>PnL: <i>Rs 500</i></b>"),
            "PnL: Rs 500"
        )

    def test_preserves_plain_text(self):
        from alerts.slack_bot import _strip_html
        self.assertEqual(_strip_html("no tags here"), "no tags here")

    def test_empty_string(self):
        from alerts.slack_bot import _strip_html
        self.assertEqual(_strip_html(""), "")


class TestSendAlert(unittest.TestCase):

    @patch("alerts.slack_bot._get_client")
    @patch("alerts.slack_bot.settings")
    def test_sends_with_valid_config(self, mock_settings, mock_get_client):
        mock_settings.SLACK_CHANNEL_ID = "C123"
        mock_client = MagicMock()
        mock_client.chat_postMessage.return_value = {"ok": True}
        mock_get_client.return_value = mock_client

        from alerts.slack_bot import send_alert
        result = send_alert("Test message")

        self.assertTrue(result)
        mock_client.chat_postMessage.assert_called_once_with(
            channel="C123", text="Test message"
        )

    @patch("alerts.slack_bot.settings")
    def test_skips_when_no_channel(self, mock_settings):
        mock_settings.SLACK_CHANNEL_ID = ""

        from alerts.slack_bot import send_alert
        result = send_alert("Test")

        self.assertFalse(result)

    @patch("alerts.slack_bot._get_client")
    @patch("alerts.slack_bot.settings")
    def test_skips_when_no_token(self, mock_settings, mock_get_client):
        mock_settings.SLACK_CHANNEL_ID = "C123"
        mock_get_client.return_value = None

        from alerts.slack_bot import send_alert
        result = send_alert("Test")

        self.assertFalse(result)

    @patch("alerts.slack_bot._get_client")
    @patch("alerts.slack_bot.settings")
    def test_handles_api_error(self, mock_settings, mock_get_client):
        mock_settings.SLACK_CHANNEL_ID = "C123"
        mock_client = MagicMock()
        mock_client.chat_postMessage.return_value = {
            "ok": False, "error": "channel_not_found"
        }
        mock_get_client.return_value = mock_client

        from alerts.slack_bot import send_alert
        result = send_alert("Test")

        self.assertFalse(result)

    @patch("alerts.slack_bot._get_client")
    @patch("alerts.slack_bot.settings")
    def test_handles_exception(self, mock_settings, mock_get_client):
        mock_settings.SLACK_CHANNEL_ID = "C123"
        mock_client = MagicMock()
        mock_client.chat_postMessage.side_effect = Exception("network error")
        mock_get_client.return_value = mock_client

        from alerts.slack_bot import send_alert
        result = send_alert("Test")

        self.assertFalse(result)

    @patch("alerts.slack_bot._get_client")
    @patch("alerts.slack_bot.settings")
    def test_strips_html(self, mock_settings, mock_get_client):
        mock_settings.SLACK_CHANNEL_ID = "C123"
        mock_client = MagicMock()
        mock_client.chat_postMessage.return_value = {"ok": True}
        mock_get_client.return_value = mock_client

        from alerts.slack_bot import send_alert
        send_alert("<b>ALERT</b>: market <i>crashed</i>")

        sent_text = mock_client.chat_postMessage.call_args[1]["text"]
        self.assertEqual(sent_text, "ALERT: market crashed")


class TestSendTradeAlert(unittest.TestCase):

    @patch("alerts.slack_bot.send_alert")
    def test_entry_format(self, mock_alert):
        mock_alert.return_value = True
        from alerts.slack_bot import send_trade_alert
        send_trade_alert("ENTRY", "pullback_1", "NIFTY_LONG",
                         price=100.50, quantity=75, sl=50.25, target=150.75)

        msg = mock_alert.call_args[0][0]
        self.assertIn("TRADE OPENED", msg)
        self.assertIn("100.50", msg)
        self.assertIn("50.25", msg)

    @patch("alerts.slack_bot.send_alert")
    def test_exit_format(self, mock_alert):
        mock_alert.return_value = True
        from alerts.slack_bot import send_trade_alert
        send_trade_alert("SL_HIT", "stoch_cross_0", "NIFTY_SHORT",
                         price=80.00, quantity=75, pnl=-1500)

        msg = mock_alert.call_args[0][0]
        self.assertIn("TRADE CLOSED", msg)
        self.assertIn("-1500", msg)
        self.assertIn("Stop Loss", msg)


class TestSendEodReport(unittest.TestCase):

    @patch("alerts.slack_bot.send_alert")
    def test_eod_format(self, mock_alert):
        mock_alert.return_value = True
        from alerts.slack_bot import send_eod_report
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

    @patch("alerts.slack_bot.send_alert")
    def test_system_alert_format(self, mock_alert):
        mock_alert.return_value = True
        from alerts.slack_bot import send_system_alert
        send_system_alert("KILL SWITCH", "Daily loss exceeded")

        msg = mock_alert.call_args[0][0]
        self.assertIn("KILL SWITCH", msg)
        self.assertIn("Daily loss exceeded", msg)


def run_live_test():
    """Send a real Slack message -- requires SLACK_BOT_TOKEN and SLACK_CHANNEL_ID in .env."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    import importlib, config.settings
    importlib.reload(config.settings)

    from alerts.slack_bot import send_system_alert

    token = config.settings.SLACK_BOT_TOKEN
    channel = config.settings.SLACK_CHANNEL_ID
    if not token or not channel:
        print("SLACK_BOT_TOKEN and SLACK_CHANNEL_ID must be set in .env")
        print("See .env.example for setup instructions.")
        return

    print(f"Sending test Slack message to channel {channel}...")
    ok = send_system_alert(
        "Trading Agent Test",
        "If you see this, Slack alerts are working."
    )
    print(f"Result: {'sent' if ok else 'failed -- check bot token and channel ID'}")


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live_test()
    else:
        unittest.main(argv=[sys.argv[0]])
