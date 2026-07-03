"""Real-time market feed via Angel One SmartWebSocketV2.

Primary: WebSocket for sub-second Nifty/BankNifty ticks with real volume.
Fallback: REST LTP polling (existing broker.get_ltp()) when WS is down.

The feed_token is already fetched at broker login but was never used.
This module activates it.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

import pytz
from loguru import logger

from config import settings

IST = pytz.timezone("Asia/Kolkata")


class MarketFeed:
    """WebSocket feed with automatic REST fallback."""

    NIFTY_TOKEN = settings.NIFTY_INDEX_TOKEN
    BANKNIFTY_TOKEN = settings.BANKNIFTY_INDEX_TOKEN

    def __init__(self, api_key: str, client_code: str,
                 feed_token: str, auth_token: str):
        self._api_key = api_key
        self._client_code = client_code
        self._feed_token = feed_token
        self._auth_token = auth_token

        self._prices: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._reconnect_count = 0

    def start(self, tokens: list[str] | None = None,
              token_groups: list[dict] | None = None):
        """Start WebSocket in background thread.

        Args:
            tokens: legacy single-segment token list (exchangeType 1 / NSE).
                    Used by the existing TradingEngine -- behavior unchanged.
            token_groups: multi-segment subscription list for multi-asset mode.
                    Format: [{"exchangeType": 5, "tokens": ["t1","t2"]}, ...]
                    When provided, `tokens` is ignored.
        """
        if self._running:
            return
        self._running = True

        if token_groups:
            subscribe_list = token_groups
        else:
            subscribe_list = [
                {"exchangeType": 1, "tokens": tokens or [self.NIFTY_TOKEN]}
            ]

        self._thread = threading.Thread(
            target=self._run_ws,
            args=(subscribe_list,),
            daemon=True,
        )
        self._thread.start()
        logger.info("MarketFeed started (WebSocket thread, {} segment(s))",
                     len(subscribe_list))

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._connected = False
        logger.info("MarketFeed stopped")

    def get_ltp(self, token: str = "") -> Optional[dict]:
        """Get latest tick for a token. Returns {price, volume, time} or None."""
        token = token or self.NIFTY_TOKEN
        with self._lock:
            data = self._prices.get(token)
        if data:
            return data.copy()
        return None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _run_ws(self, subscribe_list: list[dict]):
        """WebSocket loop with unlimited reconnection and exponential backoff."""
        while self._running:
            try:
                self._connect_and_subscribe(subscribe_list)
            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                backoff = min(60, 5 * (2 ** min(self._reconnect_count - 1, 4)))
                logger.warning(
                    "WebSocket error (attempt {}): {} -- retry in {}s",
                    self._reconnect_count, e, backoff)
                if self._running:
                    time.sleep(backoff)

        self._connected = False

    def _connect_and_subscribe(self, subscribe_list: list[dict]):
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        except ImportError:
            logger.error("SmartWebSocketV2 not available -- REST only mode")
            return

        self._ws = SmartWebSocketV2(
            self._auth_token,
            self._api_key,
            self._client_code,
            self._feed_token,
        )

        def on_data(ws, message):
            try:
                token = str(message.get("token", ""))
                ltp = message.get("last_traded_price", 0)
                vol = message.get("exchange_feed_time_epoch_volume", 0)
                if token and ltp:
                    price = ltp / 100.0
                    with self._lock:
                        self._prices[token] = {
                            "price": price,
                            "volume": vol,
                            "time": datetime.now(IST),
                        }
            except Exception as e:
                logger.debug("WS data parse error: {}", e)

        def on_open(ws):
            self._connected = True
            self._reconnect_count = 0
            total_tokens = sum(len(g.get("tokens", [])) for g in subscribe_list)
            logger.info("WebSocket connected -- subscribing {} token(s) across {} segment(s)",
                         total_tokens, len(subscribe_list))
            self._ws.subscribe("abc123", 1, subscribe_list)

        def on_error(ws, error):
            logger.warning("WebSocket error: {}", error)

        def on_close(ws, code, reason):
            self._connected = False
            logger.info("WebSocket closed: {} {}", code, reason)

        self._ws.on_data = on_data
        self._ws.on_open = on_open
        self._ws.on_error = on_error
        self._ws.on_close = on_close

        self._ws.connect()
