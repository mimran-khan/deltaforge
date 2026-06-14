"""Select the optimal option strike for Nifty/BankNifty trading."""

from __future__ import annotations
import json
import math
from datetime import datetime, date, timedelta
from typing import Optional, Tuple

import pytz
from loguru import logger

from config import settings

IST = pytz.timezone("Asia/Kolkata")

_EXPIRY_FORMATS = ("%d%b%Y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y")


def _parse_expiry(exp_str: str):
    """Parse expiry string to date for reliable sorting."""
    if not exp_str:
        return None
    cleaned = exp_str.strip().upper()
    for fmt in _EXPIRY_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


class StrikeSelector:
    """Selects ATM/ITM strikes from the instrument master."""

    def __init__(self):
        self.instruments: list = []
        self._nifty_options: list = []
        self._banknifty_options: list = []

    def load_instruments(self, instruments: Optional[list] = None):
        if instruments:
            self.instruments = instruments
        elif settings.INSTRUMENTS_FILE.exists():
            with open(settings.INSTRUMENTS_FILE) as f:
                self.instruments = json.load(f)
        else:
            logger.error("No instruments file found")
            return

        self._nifty_options = [
            i for i in self.instruments
            if i.get("name") == "NIFTY"
            and i.get("instrumenttype") in ("OPTIDX",)
            and i.get("exch_seg") == "NFO"
        ]
        self._banknifty_options = [
            i for i in self.instruments
            if i.get("name") == "BANKNIFTY"
            and i.get("instrumenttype") in ("OPTIDX",)
            and i.get("exch_seg") == "NFO"
        ]
        logger.info("Loaded {} NIFTY options, {} BANKNIFTY options",
                     len(self._nifty_options), len(self._banknifty_options))

    def get_nifty_token(self) -> Optional[dict]:
        """Get NIFTY 50 index instrument."""
        for i in self.instruments:
            if (i.get("name") == "Nifty 50" and
                    i.get("exch_seg") == "NSE" and
                    i.get("symbol") == "Nifty 50"):
                return i
        for i in self.instruments:
            if i.get("token") == settings.NIFTY_INDEX_TOKEN and i.get("exch_seg") == "NSE":
                return i
        return None

    def get_banknifty_token(self) -> Optional[dict]:
        """Get BANKNIFTY index instrument."""
        for i in self.instruments:
            if (i.get("name") == "Nifty Bank" and
                    i.get("exch_seg") == "NSE" and
                    i.get("symbol") == "Nifty Bank"):
                return i
        for i in self.instruments:
            if i.get("token") == settings.BANKNIFTY_INDEX_TOKEN and i.get("exch_seg") == "NSE":
                return i
        return None

    def find_strike(self, spot_price: float, underlying: str = "NIFTY",
                    option_type: str = "CE",
                    offset: int = 0,
                    expiry_date: Optional[date] = None) -> Optional[dict]:
        """Find the option instrument matching criteria.

        Args:
            spot_price: current index price
            underlying: "NIFTY" or "BANKNIFTY"
            option_type: "CE" or "PE"
            offset: 0=ATM, 1=ITM1, -1=OTM1
            expiry_date: specific expiry or None for nearest
        """
        options = self._nifty_options if underlying == "NIFTY" else self._banknifty_options
        step = 50 if underlying == "NIFTY" else 100

        atm_strike = round(spot_price / step) * step

        if option_type == "CE":
            target_strike = atm_strike - (offset * step)
        else:
            target_strike = atm_strike + (offset * step)

        candidates = [
            o for o in options
            if float(o.get("strike", 0)) / 100 == target_strike
            and o.get("symbol", "").endswith(option_type)
        ]

        if not candidates:
            candidates = [
                o for o in options
                if abs(float(o.get("strike", 0)) / 100 - target_strike) <= step
                and o.get("symbol", "").endswith(option_type)
            ]

        if not candidates:
            logger.warning("No strike found for {} {} {} @ {}",
                           underlying, option_type, target_strike, spot_price)
            return None

        if expiry_date:
            expiry_str = expiry_date.strftime("%d%b%Y").upper()
            dated = [c for c in candidates if expiry_str in c.get("expiry", "").upper()]
            if dated:
                candidates = dated

        candidates.sort(
            key=lambda x: _parse_expiry(x.get("expiry", "")) or datetime.max.date()
        )
        return candidates[0]

    def get_nearest_expiry(self, underlying: str = "NIFTY") -> Optional[str]:
        """Get nearest expiry date string for weekly/monthly options."""
        options = self._nifty_options if underlying == "NIFTY" else self._banknifty_options
        today = datetime.now(IST).date()

        expiries = set()
        for o in options:
            exp = o.get("expiry", "")
            if exp:
                try:
                    exp_date = datetime.strptime(exp, "%d%b%Y").date()
                    if exp_date >= today:
                        expiries.add(exp_date)
                except ValueError:
                    pass

        if not expiries:
            return None

        nearest = min(expiries)
        return nearest.strftime("%d%b%Y").upper()

    def get_lot_size(self, underlying: str = "NIFTY") -> int:
        if underlying == "NIFTY":
            return settings.NIFTY_LOT_SIZE
        return settings.BANKNIFTY_LOT_SIZE
