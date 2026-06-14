"""Angel One SmartAPI broker connection and order management."""

from __future__ import annotations
import json
import time
from datetime import datetime, timedelta
from typing import Optional

import pyotp
import requests
from loguru import logger

from config import settings

# Defer SmartApi import to avoid startup DNS issues in sandboxed envs
SmartConnect = None


def _get_smart_connect():
    global SmartConnect
    if SmartConnect is None:
        from SmartApi.smartConnect import SmartConnect as SC
        SmartConnect = SC
    return SmartConnect


class BrokerConnection:
    """Manages Angel One SmartAPI session lifecycle."""

    def __init__(self):
        self.api: Optional[object] = None
        self.auth_token: str = ""
        self.refresh_token: str = ""
        self.feed_token: str = ""
        self.client_code: str = settings.ANGEL_CLIENT_ID
        self._session_active = False

    def login(self) -> bool:
        try:
            SC = _get_smart_connect()
            self.api = SC(api_key=settings.ANGEL_API_KEY)
            totp = pyotp.TOTP(settings.ANGEL_TOTP_SECRET).now()

            data = self.api.generateSession(
                clientCode=settings.ANGEL_CLIENT_ID,
                password=settings.ANGEL_PASSWORD,
                totp=totp,
            )

            if data.get("status"):
                self.auth_token = data["data"]["jwtToken"]
                self.refresh_token = data["data"]["refreshToken"]
                self.feed_token = self.api.getfeedToken()
                self._session_active = True
                logger.info("Broker login successful for {}", self.client_code)
                return True

            logger.error("Login failed: {}", data.get("message", "unknown"))
            return False

        except Exception as e:
            logger.exception("Login exception: {}", e)
            return False

    def logout(self) -> bool:
        try:
            if self.api and self._session_active:
                self.api.terminateSession(self.client_code)
                self._session_active = False
                logger.info("Session terminated")
            return True
        except Exception as e:
            logger.error("Logout error: {}", e)
            return False

    @property
    def is_active(self) -> bool:
        return self._session_active and self.api is not None

    # ── Market Data ─────────────────────────────────────────────────

    def get_ltp(self, exchange: str, symbol: str, token: str) -> Optional[float]:
        try:
            data = self.api.ltpData(exchange, symbol, token)
            if data.get("status"):
                return data["data"]["ltp"]
        except Exception as e:
            logger.error("LTP error for {}: {}", symbol, e)
        return None

    def get_historical(self, exchange: str, token: str,
                       interval: str, from_date: str, to_date: str) -> list:
        delays = [10, 30, 60]
        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }
        for attempt in range(len(delays) + 1):
            try:
                data = self.api.getCandleData(params)
                if data.get("status") and data.get("data"):
                    return data["data"]
                msg = data.get("message", "empty response")
                if attempt < len(delays):
                    logger.warning(
                        "Historical API attempt {}/{}: {} -- retrying in {}s",
                        attempt + 1, len(delays) + 1, msg, delays[attempt])
                    time.sleep(delays[attempt])
                else:
                    logger.error("Historical API failed after {} attempts: {}",
                                 len(delays) + 1, msg)
            except Exception as e:
                if attempt < len(delays):
                    logger.warning(
                        "Historical API attempt {}/{} error: {} -- retrying in {}s",
                        attempt + 1, len(delays) + 1, e, delays[attempt])
                    time.sleep(delays[attempt])
                else:
                    logger.error("Historical data error after {} attempts: {}",
                                 len(delays) + 1, e)
        return []

    def get_option_chain_ltp(self, exchange: str, symbol: str, token: str) -> Optional[dict]:
        """Get quote data for an option symbol."""
        try:
            data = self.api.ltpData(exchange, symbol, token)
            if data.get("status"):
                return data["data"]
        except Exception as e:
            logger.error("Option LTP error for {}: {}", symbol, e)
        return None

    # ── Orders ──────────────────────────────────────────────────────

    def place_order(self, symbol: str, token: str, exchange: str,
                    transaction_type: str, quantity: int,
                    order_type: str = "LIMIT", price: float = 0,
                    trigger_price: float = 0,
                    product: str = "INTRADAY",
                    variety: str = "NORMAL",
                    tag: str = "") -> Optional[str]:
        """Place order. Returns order_id or None."""
        if settings.TRADING_MODE == "paper":
            return self._paper_order(symbol, token, exchange,
                                     transaction_type, quantity,
                                     order_type, price, tag)
        try:
            params = {
                "variety": variety,
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": transaction_type,
                "exchange": exchange,
                "ordertype": order_type,
                "producttype": product,
                "duration": "DAY",
                "quantity": str(quantity),
                "price": str(price),
                "triggerprice": str(trigger_price),
            }
            if tag:
                params["ordertag"] = tag[:20]

            data = self.api.placeOrder(params)
            if data:
                logger.info("Order placed: {} {} {} qty={} @ {} -> {}",
                            transaction_type, symbol, order_type, quantity, price, data)
                return str(data)
            logger.error("Order placement failed for {}", symbol)
        except Exception as e:
            logger.exception("Order exception: {}", e)
        return None

    def modify_order(self, order_id: str, symbol: str, token: str,
                     exchange: str, order_type: str, quantity: int,
                     price: float, trigger_price: float = 0,
                     variety: str = "NORMAL") -> bool:
        if settings.TRADING_MODE == "paper":
            logger.info("[PAPER] Modified order {} -> price={}", order_id, price)
            return True
        try:
            params = {
                "variety": variety,
                "orderid": order_id,
                "tradingsymbol": symbol,
                "symboltoken": token,
                "exchange": exchange,
                "ordertype": order_type,
                "producttype": "INTRADAY",
                "duration": "DAY",
                "quantity": str(quantity),
                "price": str(price),
                "triggerprice": str(trigger_price),
            }
            data = self.api.modifyOrder(params)
            logger.info("Order modified: {} -> {}", order_id, data)
            return True
        except Exception as e:
            logger.error("Modify error: {}", e)
            return False

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> bool:
        if settings.TRADING_MODE == "paper":
            logger.info("[PAPER] Cancelled order {}", order_id)
            return True
        try:
            self.api.cancelOrder(order_id, variety)
            logger.info("Order cancelled: {}", order_id)
            return True
        except Exception as e:
            logger.error("Cancel error: {}", e)
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        if settings.TRADING_MODE == "paper":
            logger.info("[PAPER] Cancel all orders")
            return 0
        try:
            order_book = self.api.orderBook()
            if not order_book or not order_book.get("data"):
                return 0
            cancelled = 0
            for order in order_book["data"]:
                if order["orderstatus"] in ("open", "trigger pending"):
                    self.cancel_order(order["orderid"], order.get("variety", "NORMAL"))
                    cancelled += 1
            return cancelled
        except Exception as e:
            logger.error("Cancel all error: {}", e)
            return 0

    def get_positions(self) -> list:
        if settings.TRADING_MODE == "paper":
            return []
        try:
            data = self.api.position()
            if data and data.get("data"):
                return data["data"]
        except Exception as e:
            logger.error("Position error: {}", e)
        return []

    def get_order_book(self) -> list:
        if settings.TRADING_MODE == "paper":
            return []
        try:
            data = self.api.orderBook()
            if data and data.get("data"):
                return data["data"]
        except Exception as e:
            logger.error("Order book error: {}", e)
        return []

    # ── Instruments ─────────────────────────────────────────────────

    def download_instruments(self) -> list:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        try:
            resp = requests.get(url, timeout=30)
            instruments = resp.json()
            tmp = settings.INSTRUMENTS_FILE.with_suffix('.tmp')
            with open(tmp, "w") as f:
                json.dump(instruments, f)
            import os as _os
            _os.replace(str(tmp), str(settings.INSTRUMENTS_FILE))
            logger.info("Downloaded {} instruments", len(instruments))
            return instruments
        except Exception as e:
            logger.error("Instrument download error: {}", e)
            if settings.INSTRUMENTS_FILE.exists():
                with open(settings.INSTRUMENTS_FILE) as f:
                    return json.load(f)
            return []

    # ── Paper Trading ───────────────────────────────────────────────

    _paper_order_counter = 0

    def _paper_order(self, symbol, token, exchange, txn_type,
                     quantity, order_type, price, tag) -> str:
        BrokerConnection._paper_order_counter += 1
        oid = f"PAPER-{BrokerConnection._paper_order_counter:06d}"
        logger.info("[PAPER] {} {} {} qty={} @ {} tag={} -> {}",
                    txn_type, symbol, order_type, quantity, price, tag, oid)
        return oid
