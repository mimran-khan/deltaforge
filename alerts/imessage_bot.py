"""iMessage notification service for trade alerts and EOD reports.

Drop-in replacement for telegram_bot.py -- same API surface.
Uses macOS AppleScript via osascript to send via Messages.app.

Requires: System Settings -> Privacy & Security -> Automation
          -> enable Terminal for Messages.app (one-time setup).
"""
from __future__ import annotations

import re
import subprocess

from loguru import logger

from config import settings


def _strip_html(text: str) -> str:
    """Remove HTML tags used in Telegram formatting."""
    return re.sub(r'<[^>]+>', '', text)


def _send_imessage(message: str) -> bool:
    """Send a message via iMessage using AppleScript.

    Uses 'service' (not 'account') which works on macOS Ventura/Sonoma/Sequoia.
    Newlines are converted to AppleScript's linefeed character since AppleScript
    string literals cannot span multiple lines.
    """
    recipient = getattr(settings, 'IMESSAGE_RECIPIENT', '')
    if not recipient:
        logger.debug("iMessage not configured -- skipping alert")
        return False

    safe_msg = (message
                .replace('\\', '\\\\')
                .replace('"', '\\"')
                .replace('\n', '" & linefeed & "'))

    script = (
        'tell application "Messages"\n'
        '  set targetService to 1st service whose service type = iMessage\n'
        f'  set targetBuddy to buddy "{recipient}" of targetService\n'
        f'  send ("{safe_msg}") to targetBuddy\n'
        'end tell'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            return True
        logger.error("iMessage error: {}", result.stderr.strip())
    except subprocess.TimeoutExpired:
        logger.error("iMessage send timed out")
    except Exception as e:
        logger.error("iMessage send failed: {}", e)
    return False


def send_alert(message: str, parse_mode: str = "") -> bool:
    """Send an alert message. parse_mode is ignored (Telegram compat)."""
    clean = _strip_html(message)
    return _send_imessage(clean)


_EXIT_REASON_LABELS = {
    "SL": "Stop Loss",
    "TIME": "Time Limit",
    "EOD": "End of Day",
    "RISK_HALT": "Risk Halt",
    "MANUAL_STOP": "Manual Stop",
    "SCHEDULED_SQUARE_OFF": "Scheduled Close",
    "END_OF_DAY": "End of Day",
}


def _clean_action(action: str) -> str:
    """Convert internal action codes to user-friendly labels."""
    a = action.upper()
    for code, label in _EXIT_REASON_LABELS.items():
        if code in a:
            return label
    return action


def send_trade_alert(action: str, strategy: str, symbol: str,
                     price: float, quantity: int, sl: float = 0,
                     target: float = 0, pnl: float = 0):
    """Send formatted trade entry/exit alert."""
    is_entry = "ENTRY" in action.upper()

    if is_entry:
        msg = (
            f"TRADE OPENED\n"
            f"---\n"
            f"{symbol}\n"
            f"Entry: Rs {price:.2f}\n"
            f"Qty: {quantity}\n"
            f"SL: Rs {sl:.2f}"
        )
    else:
        sign = "+" if pnl >= 0 else ""
        reason = _clean_action(action)
        msg = (
            f"TRADE CLOSED\n"
            f"---\n"
            f"{symbol}\n"
            f"Exit: Rs {price:.2f}\n"
            f"P&L: Rs {sign}{pnl:.0f}\n"
            f"Reason: {reason}"
        )
    return send_alert(msg)


def send_eod_report(summary: dict):
    """Send end-of-day P&L summary."""
    win_rate = summary.get("win_rate", 0)
    daily_pnl = summary.get("daily_pnl", 0)
    sign = "+" if daily_pnl >= 0 else ""

    msg = (
        f"EOD REPORT\n"
        f"---\n"
        f"Capital: Rs {summary['capital']:.0f}\n"
        f"Day PnL: Rs {sign}{daily_pnl:.0f} ({summary['daily_pnl_pct']:.1f}%)\n"
        f"Trades: {summary['trades']}\n"
        f"Wins: {summary['wins']} | Losses: {summary['losses']}\n"
        f"Win Rate: {win_rate:.0f}%\n"
        f"Total PnL: Rs {summary['total_pnl']:.0f}\n"
        f"Max DD: {summary['max_drawdown']:.1f}%"
    )
    return send_alert(msg)


def send_system_alert(title: str, message: str):
    """Send a system alert."""
    msg = f"{title}\n\n{message}"
    return send_alert(msg)
