"""Alert channel factory.

Usage:
    from alerts import get_alert_sender, send_alert, send_system_alert

    send_alert("Trade entered: NIFTY CE 24500")
    send_system_alert("Kill Switch", "Daily loss limit breached")
"""
from __future__ import annotations

from config import settings


def get_alert_module():
    """Return the alert module for the configured ALERT_METHOD."""
    method = getattr(settings, "ALERT_METHOD", "slack")
    if method == "imessage":
        from alerts import imessage_bot
        return imessage_bot
    elif method == "telegram":
        from alerts import telegram_bot
        return telegram_bot
    else:
        from alerts import slack_bot
        return slack_bot


def send_alert(message: str) -> bool:
    """Send an alert via the configured channel."""
    mod = get_alert_module()
    return mod.send_alert(message)


def send_system_alert(title: str, body: str) -> bool:
    """Send a system alert via the configured channel."""
    mod = get_alert_module()
    fn = getattr(mod, "send_system_alert", None)
    if fn:
        return fn(title, body)
    return mod.send_alert(f"[{title}] {body}")
