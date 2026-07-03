"""Slack notification service for trade alerts and EOD reports.

Drop-in replacement for imessage_bot.py / telegram_bot.py -- same API surface.
Uses slack_sdk WebClient to post messages to a configured channel.

Setup:
  1. Create a Slack App at https://api.slack.com/apps
  2. Add Bot Token Scopes: chat:write, channels:history, channels:read
  3. Install to workspace, copy Bot User OAuth Token
  4. Invite the bot to your channel: /invite @YourBot
  5. Set SLACK_BOT_TOKEN and SLACK_CHANNEL_ID in .env
"""
from __future__ import annotations

import re
from functools import lru_cache

from loguru import logger

from config import settings


def _get_client():
    """Lazily initialise the Slack WebClient (avoids import cost at startup)."""
    token = getattr(settings, "SLACK_BOT_TOKEN", "")
    if not token:
        return None
    from slack_sdk import WebClient
    return WebClient(token=token)


def _strip_html(text: str) -> str:
    """Remove HTML tags used in Telegram formatting."""
    return re.sub(r"<[^>]+>", "", text)


def _mrkdwn_block(text: str) -> list[dict]:
    """Wrap text in a Slack mrkdwn section block."""
    return [{"type": "section", "block_id": "msg",
             "text": {"type": "mrkdwn", "text": text}}]


def send_alert(message: str, parse_mode: str = "") -> bool:
    """Send a plain-text alert to the configured Slack channel."""
    channel = getattr(settings, "SLACK_CHANNEL_ID", "")
    if not channel:
        logger.debug("Slack not configured -- skipping alert")
        return False

    client = _get_client()
    if client is None:
        logger.debug("Slack bot token missing -- skipping alert")
        return False

    clean = _strip_html(message)
    try:
        resp = client.chat_postMessage(channel=channel, text=clean)
        if resp["ok"]:
            return True
        logger.error("Slack error: {}", resp.get("error", "unknown"))
    except Exception as e:
        logger.error("Slack send failed: {}", e)
    return False


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
    a = action.upper()
    for code, label in _EXIT_REASON_LABELS.items():
        if code in a:
            return label
    return action


def send_trade_alert(action: str, strategy: str, symbol: str,
                     price: float, quantity: int, sl: float = 0,
                     target: float = 0, pnl: float = 0,
                     confidence: float = 0, capital: float = 0,
                     daily_pnl: float = 0, win_count: int = 0,
                     loss_count: int = 0):
    """Send formatted trade entry/exit alert with Slack rich formatting."""
    is_entry = "ENTRY" in action.upper()

    if is_entry:
        sl_pct = ((price - sl) / price * 100) if price > 0 and sl > 0 else 0
        tgt_pct = ((target - price) / price * 100) if price > 0 and target > 0 else 0
        rr = abs(tgt_pct / sl_pct) if sl_pct != 0 else 0
        msg = (
            f":chart_with_upwards_trend: *TRADE OPENED*\n"
            f"{'─' * 20}\n"
            f"*{symbol}* | `{strategy}`\n"
            f"Entry: `Rs {price:.2f}` | Qty: `{quantity}`\n"
            f"SL: `Rs {sl:.2f}` ({sl_pct:.1f}%)\n"
            f"Target: `Rs {target:.2f}` ({tgt_pct:.1f}%)\n"
            f"R:R `{rr:.1f}:1` | Conf: `{confidence:.0f}`"
        )
        if capital > 0:
            msg += f"\nCapital: `Rs {capital:,.0f}`"
    else:
        sign = "+" if pnl >= 0 else ""
        icon = ":white_check_mark:" if pnl >= 0 else ":x:"
        reason = _clean_action(action)
        wr = (win_count / (win_count + loss_count) * 100) if (win_count + loss_count) > 0 else 0
        msg = (
            f"{icon} *TRADE CLOSED*\n"
            f"{'─' * 20}\n"
            f"*{symbol}* | `{strategy}`\n"
            f"Exit: `Rs {price:.2f}` | Reason: {reason}\n"
            f"P&L: *Rs {sign}{pnl:.0f}*\n"
            f"Today: `Rs {'+' if daily_pnl >= 0 else ''}{daily_pnl:.0f}` | "
            f"W/L: `{win_count}/{loss_count}` ({wr:.0f}%)"
        )
        if capital > 0:
            msg += f"\nCapital: `Rs {capital:,.0f}`"
    return send_alert(msg)


def send_eod_report(summary: dict):
    """Send end-of-day P&L summary with Slack formatting."""
    win_rate = summary.get("win_rate", 0)
    daily_pnl = summary.get("daily_pnl", 0)
    sign = "+" if daily_pnl >= 0 else ""
    icon = ":green_book:" if daily_pnl >= 0 else ":closed_book:"

    daily_ret_pct = summary.get("daily_pnl_pct", 0)
    cap = summary.get("capital", 0)
    target_7pct = cap * 0.07 if cap > 0 else 0
    on_track = daily_pnl >= target_7pct

    streak = summary.get("streak", "")
    streak_line = f"\nStreak: {streak}" if streak else ""

    strategies = summary.get("strategy_breakdown", "")
    strat_line = f"\nStrategies: {strategies}" if strategies else ""

    msg = (
        f"{icon} *EOD REPORT*\n"
        f"{'═' * 24}\n"
        f"Capital: `Rs {cap:,.0f}`\n"
        f"Day P&L: *Rs {sign}{daily_pnl:,.0f}* ({daily_ret_pct:.1f}%)\n"
        f"Target (7%): Rs {target_7pct:,.0f} {'*HIT*' if on_track else '_missed_'}\n"
        f"{'─' * 24}\n"
        f"Trades: {summary.get('trades', 0)} | "
        f"Wins: {summary.get('wins', 0)} | Losses: {summary.get('losses', 0)}\n"
        f"Win Rate: {win_rate:.0f}%\n"
        f"Total P&L: `Rs {summary.get('total_pnl', 0):,.0f}`\n"
        f"Max DD: {summary.get('max_drawdown', 0):.1f}%"
        f"{streak_line}{strat_line}"
    )
    return send_alert(msg)


def send_system_alert(title: str, message: str):
    """Send a system alert."""
    msg = f":warning: *{title}*\n\n{message}"
    return send_alert(msg)
