"""Telegram notification service for trade alerts and EOD reports."""

from __future__ import annotations

import requests
from loguru import logger

from config import settings


def send_alert(message: str, parse_mode: str = "HTML") -> bool:
    """Send a Telegram message. Returns True on success."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured -- skipping alert")
        return False

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        logger.error("Telegram error {}: {}", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Telegram send failed: {}", e)
    return False


def send_trade_alert(action: str, strategy: str, symbol: str,
                     price: float, quantity: int, sl: float = 0,
                     target: float = 0, pnl: float = 0):
    """Send formatted trade entry/exit alert."""
    if action == "ENTRY":
        msg = (
            f"<b>NEW TRADE</b>\n"
            f"Strategy: {strategy}\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Entry: Rs {price:.2f}\n"
            f"Qty: {quantity}\n"
            f"SL: Rs {sl:.2f}\n"
            f"Target: Rs {target:.2f}"
        )
    else:
        emoji = "+" if pnl >= 0 else ""
        msg = (
            f"<b>TRADE CLOSED</b>\n"
            f"Strategy: {strategy}\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Exit: Rs {price:.2f}\n"
            f"PnL: <b>Rs {emoji}{pnl:.0f}</b>\n"
            f"Reason: {action}"
        )
    send_alert(msg)


def send_eod_report(summary: dict):
    """Send end-of-day P&L summary."""
    win_rate = summary.get("win_rate", 0)
    daily_pnl = summary.get("daily_pnl", 0)
    emoji = "+" if daily_pnl >= 0 else ""

    msg = (
        f"<b>EOD REPORT</b>\n"
        f"{'=' * 30}\n"
        f"Capital: Rs {summary['capital']:.0f}\n"
        f"Day PnL: <b>Rs {emoji}{daily_pnl:.0f}</b> ({summary['daily_pnl_pct']:.1f}%)\n"
        f"Trades: {summary['trades']}\n"
        f"Wins: {summary['wins']} | Losses: {summary['losses']}\n"
        f"Win Rate: {win_rate:.0f}%\n"
        f"Total PnL: Rs {summary['total_pnl']:.0f}\n"
        f"Max DD: {summary['max_drawdown']:.1f}%\n"
        f"{'=' * 30}"
    )
    send_alert(msg)


def send_system_alert(title: str, message: str):
    msg = f"<b>{title}</b>\n\n{message}"
    send_alert(msg)
