"""Manages open positions -- trailing SL, partial exit, time-based exit."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pytz
from loguru import logger

from config import settings

IST = pytz.timezone("Asia/Kolkata")
from engine.broker import BrokerConnection
from risk.capital_tracker import CapitalTracker


@dataclass
class Position:
    position_id: str
    strategy: str
    symbol: str
    token: str
    exchange: str
    option_type: str       # CE or PE
    direction: str         # BUY
    quantity: int
    entry_price: float     # premium price per unit
    entry_time: datetime
    stop_loss: float       # premium price for SL
    target: float          # premium price for target
    sl_order_id: Optional[str] = None
    status: str = "OPEN"   # OPEN, CLOSED, SL_HIT, TARGET_HIT, TIMED_OUT
    exit_price: float = 0
    exit_time: Optional[datetime] = None
    pnl: float = 0
    trailing_activated: bool = False
    index_entry: float = 0   # index price at entry (for reference)
    index_sl: float = 0
    index_target: float = 0


class PositionManager:
    """Track and manage all open positions."""

    def __init__(self, broker: BrokerConnection, capital: CapitalTracker):
        self.broker = broker
        self.capital = capital
        self.positions: dict[str, Position] = {}  # position_id -> Position

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.status == "OPEN"]

    @property
    def has_open_positions(self) -> bool:
        return len(self.open_positions) > 0

    def open_position(self, signal, strike_info: dict,
                      lots: int, premium_price: float,
                      premium_sl_pct: float) -> Optional[Position]:
        """Open a new position from a signal."""
        lot_size = int(strike_info.get("lotsize", settings.NIFTY_LOT_SIZE))
        quantity = lots * lot_size
        symbol = strike_info["symbol"]
        token = strike_info["token"]

        order_id = self.broker.place_order(
            symbol=symbol,
            token=token,
            exchange="NFO",
            transaction_type="BUY",
            quantity=quantity,
            order_type="LIMIT",
            price=premium_price + 0.5,
            product="INTRADAY",
            tag=signal.strategy_name[:20],
        )

        if not order_id:
            logger.error("Failed to place entry order for {}", symbol)
            return None

        sl_price = round(premium_price * (1 - premium_sl_pct / 100), 2)
        target_price = round(premium_price + settings.PREMIUM_TARGET_POINTS, 2)

        pos = Position(
            position_id=order_id,
            strategy=signal.strategy_name,
            symbol=symbol,
            token=token,
            exchange="NFO",
            option_type=signal.option_type,
            direction="BUY",
            quantity=quantity,
            entry_price=premium_price,
            entry_time=datetime.now(IST),
            stop_loss=sl_price,
            target=target_price,
            index_entry=signal.entry_price,
            index_sl=signal.stop_loss_index,
            index_target=signal.target_index,
        )

        sl_order_id = self.broker.place_order(
            symbol=symbol,
            token=token,
            exchange="NFO",
            transaction_type="SELL",
            quantity=quantity,
            order_type="STOPLOSS_LIMIT",
            price=max(sl_price - 1, 0.05),
            trigger_price=sl_price,
            product="INTRADAY",
            tag="SL",
        )
        pos.sl_order_id = sl_order_id

        self.positions[order_id] = pos
        logger.info("Position opened: {} {} qty={} @ {:.2f} SL={:.2f} T={:.2f}",
                     signal.strategy_name, symbol, quantity,
                     premium_price, sl_price, target_price)
        return pos

    def update_positions(self, current_prices: dict[str, float]):
        """Update all open positions with current premium prices.
        current_prices: {symbol: current_premium_price}
        """
        for pos in self.open_positions:
            if pos.symbol not in current_prices:
                continue

            current_price = current_prices[pos.symbol]
            self._check_exit_conditions(pos, current_price)

    def _check_exit_conditions(self, pos: Position, current_price: float):
        # Target hit
        if current_price >= pos.target:
            self._close_position(pos, current_price, "TARGET_HIT")
            return

        # SL hit (backup check -- SL order should handle this)
        if current_price <= pos.stop_loss:
            self._close_position(pos, current_price, "SL_HIT")
            return

        # Trail SL to breakeven at 1:1
        if not pos.trailing_activated:
            entry_risk = pos.entry_price - pos.stop_loss
            current_profit = current_price - pos.entry_price
            if current_profit >= entry_risk and entry_risk > 0:
                new_sl = pos.entry_price + 0.5
                pos.stop_loss = new_sl
                pos.trailing_activated = True
                if pos.sl_order_id:
                    self.broker.modify_order(
                        order_id=pos.sl_order_id,
                        symbol=pos.symbol,
                        token=pos.token,
                        exchange="NFO",
                        order_type="STOPLOSS_LIMIT",
                        quantity=pos.quantity,
                        price=max(new_sl - 1, 0.05),
                        trigger_price=new_sl,
                    )
                logger.info("Trailing SL activated for {} -> {:.2f}",
                            pos.symbol, new_sl)

    def _close_position(self, pos: Position, exit_price: float, reason: str):
        if pos.sl_order_id:
            self.broker.cancel_order(pos.sl_order_id)

        self.broker.place_order(
            symbol=pos.symbol,
            token=pos.token,
            exchange="NFO",
            transaction_type="SELL",
            quantity=pos.quantity,
            order_type="LIMIT",
            price=exit_price - 0.5,
            product="INTRADAY",
            tag="EXIT",
        )

        pos.exit_price = exit_price
        pos.exit_time = datetime.now(IST)
        pos.status = reason
        pos.pnl = (exit_price - pos.entry_price) * pos.quantity

        self.capital.record_trade(
            pnl=pos.pnl,
            strategy=pos.strategy,
            symbol=pos.symbol,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            reason=reason,
        )

        logger.info("Position closed: {} {} PnL=Rs {:.0f} ({})",
                     pos.strategy, pos.symbol, pos.pnl, reason)

    def square_off_all(self, reason: str = "SQUARE_OFF"):
        """Force close all open positions."""
        for pos in self.open_positions:
            current_ltp = self.broker.get_ltp("NFO", pos.symbol, pos.token)
            price = current_ltp if current_ltp else pos.entry_price * 0.8
            self._close_position(pos, price, reason)
        logger.info("All positions squared off: {}", reason)
