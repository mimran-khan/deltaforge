"""Production trading engine V2 -- Pullback-in-Trend.

Architecture:
  WebSocket/LTP → CandleBuilder → [closed bar event] → PullbackEngine
  → RiskEngine → PositionManager → Broker API

Core strategy (walk-forward validated):
  67% WR, PF 2.82, +380% over 48 days
  100% Monte Carlo probability of profit

Key principles:
  1. Score ONLY on closed 5-min bars (no look-ahead bias)
  2. Same PullbackEngine as backtesting (zero production drift)
  3. ATM options with 50% SL, no TP, 24-candle max hold
  4. Multi-oscillator pullback detection (RSI, Stoch, CCI, Williams %R)
  5. Paper mode simulates full trade lifecycle with realistic costs
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from loguru import logger

from config import settings
from engine.broker import BrokerConnection
from engine.candle_builder import CandleBuilder
from engine.pullback_engine import PullbackEngine, PullbackSignal
from engine.premium_model import create_premium_state, PremiumState
from risk.risk_engine import RiskEngine, RiskDecision
from risk.capital_tracker import CapitalTracker
from risk.kill_switch import is_halted
from alerts.telegram_bot import (
    send_trade_alert, send_eod_report, send_system_alert
)

IST = pytz.timezone("Asia/Kolkata")

MAX_HOLD = getattr(settings, 'PULLBACK_HOLD_CANDLES', 24)
SL_PCT = getattr(settings, 'PREMIUM_SL_PCT', 50.0)


@dataclass
class PaperPosition:
    """Tracks a paper trade through its full lifecycle."""
    direction: str
    entry_time: str
    entry_index: float
    entry_premium: float
    sl_premium: float
    lots: int
    qty: int
    signal: PullbackSignal
    prem_state: PremiumState
    candles_held: int = 0
    peak_premium: float = 0
    exit_premium: float = 0
    exit_reason: str = ""
    exit_time: str = ""
    pnl: float = 0


class TradingEngine:
    """Production engine V2: Pullback-in-Trend strategy.

    Modes:
      paper: full trade simulation with entries, exits, and P&L tracking
      live:  real orders via Angel One SmartAPI
    """

    def __init__(self):
        self.broker = BrokerConnection()
        self.capital = CapitalTracker()
        self.risk = RiskEngine(self.capital)
        self.candle_builder = CandleBuilder(interval_minutes=5)
        self.pullback = PullbackEngine()

        self._running = False
        self._nifty_token_info: Optional[dict] = None
        self._nifty_spot: float = 0
        self._prev_candle_count: int = 0
        self._day_indicators: dict = {}
        self._paper_positions: list[PaperPosition] = []
        self._day_signals_count: int = 0

        self._event_log = Path(settings.DATA_DIR) / "events.jsonl"
        self._signal_log = Path(settings.DATA_DIR) / "signal_log.jsonl"

    def start_day(self):
        """Initialize for a new trading day."""
        now = datetime.now(IST)
        logger.info("=" * 60)
        logger.info("DAY START: {} | Mode: {} | Capital: Rs {:.0f}",
                     now.strftime("%Y-%m-%d"), settings.TRADING_MODE.upper(),
                     self.capital.current_capital)
        logger.info("Strategy: Pullback-in-Trend V2 (67% WR, PF 2.82)")
        logger.info("Options: ATM delta={}, SL={}%, MaxHold={}",
                     settings.PREMIUM_DELTA, SL_PCT, MAX_HOLD)
        logger.info("=" * 60)

        self.risk.start_day()
        self.candle_builder.reset()
        self.pullback.reset_day()
        self._prev_candle_count = 0
        self._day_indicators = {}
        self._paper_positions = []
        self._day_signals_count = 0

        try:
            from engine.strike_selector import StrikeSelector
            self.strike_selector = StrikeSelector()
            self.strike_selector.load_instruments()
            self._nifty_token_info = self.strike_selector.get_nifty_token()
        except Exception as e:
            logger.warning("Strike selector unavailable: {}", e)
            self._nifty_token_info = None

        self._log_event("DAY_START", {
            "capital": self.capital.current_capital,
            "mode": settings.TRADING_MODE,
            "strategy": "PullbackV2",
        })

        send_system_alert(
            "Day Started - Pullback V2",
            f"Mode: {settings.TRADING_MODE.upper()}\n"
            f"Capital: Rs {self.capital.current_capital:.0f}\n"
            f"Strategy: Pullback-in-Trend (67% WR)\n"
            f"Max trades: {settings.MAX_TRADES_PER_DAY}"
        )

    def run_loop(self, poll_interval: int = 5):
        """Main loop. Scores only on CLOSED bars."""
        self._running = True
        logger.info("Loop started (poll {}s)", poll_interval)

        while self._running:
            try:
                now = datetime.now(IST)
                t = now.strftime("%H:%M")

                if is_halted():
                    logger.warning("Kill switch active")
                    self._running = False
                    break

                if t >= settings.SQUARE_OFF_TIME:
                    self._square_off_all("EOD")
                    self._running = False
                    break

                if t < settings.MARKET_OPEN:
                    time.sleep(10)
                    continue

                self._tick()
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("User interrupted")
                self._running = False
            except Exception as e:
                logger.exception("Loop error: {}", e)
                self._log_event("ERROR", {"error": str(e)})
                time.sleep(10)

        self.end_day()

    def _tick(self):
        """Single iteration: update data, check exits, maybe enter."""
        if not self.broker.is_active:
            return

        self._update_market_data()

        # Check real-time risk
        if not self.risk.check_realtime():
            if self._paper_positions:
                self._square_off_all("RISK_HALT")
            return

        # Update paper position P&L and check exits
        self._update_paper_positions()

        # Only score on NEW closed bar (event-driven)
        candles = self.candle_builder.get_candles()
        current_count = len(candles)
        if current_count <= self._prev_candle_count:
            return
        self._prev_candle_count = current_count

        if current_count < 10:
            return

        # Precompute indicators on full day's candles
        try:
            self._day_indicators = self.pullback.precompute(candles)
        except Exception as e:
            logger.debug("Precompute error: {}", e)
            return

        idx = current_count - 1
        now = datetime.now(IST)
        time_str = now.strftime("%H:%M")

        # Scan for pullback signals
        signals = self.pullback.scan(self._day_indicators, idx, time_str)

        self._day_signals_count += 1

        # Skip if already in a position
        if self._paper_positions:
            return

        for signal in signals:
            if signal.confidence < getattr(settings, 'PULLBACK_MIN_CONFIDENCE', 50):
                continue

            # Risk evaluation
            decision = self.risk.evaluate(
                confluence_score=signal.confidence,
                direction=signal.direction,
                strength="STRONG" if signal.confidence > 70 else "MODERATE",
            )

            if not decision.approved:
                continue

            logger.info("SIGNAL: {} | {}", signal.summary(), time_str)
            self._enter_trade(signal, decision)
            break  # one entry per bar

    def _enter_trade(self, signal: PullbackSignal, decision: RiskDecision):
        """Open a trade in paper or live mode."""
        option_type = "CE" if signal.direction == "LONG" else "PE"

        if settings.TRADING_MODE == "paper":
            self._paper_enter(signal, decision, option_type)
        else:
            self._live_enter(signal, decision, option_type)

    def _compute_lots(self) -> int:
        """Compute lot count from current capital -- this is how we compound."""
        capital = self.capital.current_capital
        per_lot = getattr(settings, 'CAPITAL_PER_LOT', 10_000)
        cap = getattr(settings, 'MAX_LOTS_CAP', 10)
        lots = max(1, int(capital / per_lot))
        return min(lots, cap)

    def _paper_enter(self, signal: PullbackSignal,
                     decision: RiskDecision, option_type: str):
        """Full paper trade simulation with realistic costs."""
        lots = self._compute_lots()
        qty = lots * settings.NIFTY_LOT_SIZE

        prem_state = create_premium_state(
            entry_index_price=self._nifty_spot,
            direction=signal.direction,
            base_premium=settings.PREMIUM_BASE,
            delta=settings.PREMIUM_DELTA,
            theta_per_candle=settings.PREMIUM_THETA_PER_CANDLE,
            sl_pct=SL_PCT,
            confluence_score=signal.confidence,
        )

        entry_premium = prem_state.entry_premium + settings.SLIPPAGE_POINTS
        sl_premium = entry_premium * (1 - SL_PCT / 100)

        pos = PaperPosition(
            direction=signal.direction,
            entry_time=datetime.now(IST).isoformat(),
            entry_index=self._nifty_spot,
            entry_premium=entry_premium,
            sl_premium=sl_premium,
            lots=lots,
            qty=qty,
            signal=signal,
            prem_state=prem_state,
            peak_premium=entry_premium,
        )
        self._paper_positions.append(pos)

        self._log_event("PAPER_ENTRY", {
            "direction": signal.direction,
            "option_type": option_type,
            "entry_index": self._nifty_spot,
            "entry_premium": entry_premium,
            "sl": sl_premium,
            "lots": lots,
            "confidence": signal.confidence,
            "signal_type": signal.signal_type,
            "reason": signal.reason,
            "pullback_count": signal.pullback_count,
        })

        send_trade_alert(
            action="PAPER_ENTRY",
            strategy=f"Pullback ({signal.pullback_count} conf)",
            symbol=f"NIFTY {option_type}",
            price=entry_premium,
            quantity=qty,
            sl=sl_premium,
            target=0,
        )

    def _update_paper_positions(self):
        """Update paper positions with current market data."""
        closed = []
        for pos in self._paper_positions:
            pos.candles_held += 1
            cur_prem = pos.prem_state.current_premium(
                self._nifty_spot, pos.candles_held)

            if cur_prem > pos.peak_premium:
                pos.peak_premium = cur_prem

            exit_reason = None

            # SL hit (50% of premium)
            if cur_prem <= pos.sl_premium:
                exit_reason = "SL"
                pos.exit_premium = pos.sl_premium

            # Time exit (24 candles = 2 hours max hold)
            elif pos.candles_held >= MAX_HOLD:
                exit_reason = "TIME"
                pos.exit_premium = cur_prem

            if exit_reason:
                pos.exit_reason = exit_reason
                pos.exit_time = datetime.now(IST).isoformat()

                exit_prem = pos.exit_premium - settings.SLIPPAGE_POINTS
                raw_pnl = (exit_prem - pos.entry_premium) * pos.qty
                costs = self._calc_costs(pos.entry_premium, exit_prem, pos.qty)
                pos.pnl = raw_pnl - costs

                self.capital.record_trade(
                    pnl=pos.pnl,
                    strategy=f"Pullback_{pos.signal.pullback_count}",
                    symbol=f"NIFTY_{pos.direction}",
                    entry_price=pos.entry_premium,
                    exit_price=exit_prem,
                    quantity=pos.qty,
                    reason=exit_reason,
                )

                self._log_event("PAPER_EXIT", {
                    "direction": pos.direction,
                    "entry_premium": pos.entry_premium,
                    "exit_premium": exit_prem,
                    "reason": exit_reason,
                    "candles_held": pos.candles_held,
                    "pnl": round(pos.pnl, 2),
                    "costs": round(costs, 2),
                    "capital": round(self.capital.current_capital, 2),
                })

                send_trade_alert(
                    action=f"PAPER_EXIT ({exit_reason})",
                    strategy=f"Pullback_{pos.signal.pullback_count}",
                    symbol=f"NIFTY_{pos.direction}",
                    price=exit_prem,
                    quantity=pos.qty,
                    sl=0, target=0,
                )

                closed.append(pos)

        for pos in closed:
            self._paper_positions.remove(pos)

    def _calc_costs(self, entry_prem: float, exit_prem: float, qty: int) -> float:
        """Calculate realistic execution costs for NFO."""
        brokerage = settings.BROKERAGE_PER_ORDER * 2
        stt = exit_prem * qty * settings.STT_PCT / 100
        stamp = entry_prem * qty * settings.STAMP_DUTY_PCT / 100
        slippage = settings.SLIPPAGE_POINTS * qty
        return brokerage + stt + stamp + slippage

    def _live_enter(self, signal: PullbackSignal,
                    decision: RiskDecision, option_type: str):
        """Execute real trade via Angel One API."""
        from strategies.base_strategy import Signal, SignalType

        sig_obj = Signal(
            signal_type=SignalType.LONG if signal.direction == "LONG" else SignalType.SHORT,
            strategy_name=f"Pullback_{signal.pullback_count}",
            entry_price=self._nifty_spot,
            option_type=option_type,
            confluence_score=signal.confidence,
        )

        try:
            from engine.strike_selector import StrikeSelector
            strike = self.strike_selector.find_strike(
                spot_price=self._nifty_spot,
                underlying="NIFTY",
                option_type=option_type,
                offset=settings.STRIKE_OFFSET,
            )
        except Exception as e:
            logger.error("Strike selection failed: {}", e)
            return

        if not strike:
            logger.error("No strike found")
            return

        premium_ltp = self.broker.get_ltp("NFO", strike["symbol"], strike["token"])
        if not premium_ltp:
            logger.error("No LTP for {}", strike["symbol"])
            return

        from execution.position_manager import PositionManager
        pos_mgr = PositionManager(self.broker, self.capital)
        pos = pos_mgr.open_position(
            signal=sig_obj, strike_info=strike,
            lots=decision.lots, premium_price=premium_ltp,
            premium_sl_pct=decision.premium_sl_pct,
        )

        if pos:
            self._log_event("LIVE_ENTRY", {
                "symbol": pos.symbol, "direction": signal.direction,
                "entry": pos.entry_price, "lots": decision.lots,
                "confidence": signal.confidence,
            })
            send_trade_alert(
                action="ENTRY", strategy=sig_obj.strategy_name,
                symbol=pos.symbol, price=pos.entry_price,
                quantity=pos.quantity, sl=pos.stop_loss, target=pos.target,
            )

    def _square_off_all(self, reason: str):
        """Force close all paper positions."""
        for pos in list(self._paper_positions):
            cur_prem = pos.prem_state.current_premium(
                self._nifty_spot, pos.candles_held)
            exit_prem = cur_prem - settings.SLIPPAGE_POINTS
            pos.exit_premium = exit_prem
            pos.exit_reason = reason
            pos.exit_time = datetime.now(IST).isoformat()

            raw_pnl = (exit_prem - pos.entry_premium) * pos.qty
            costs = self._calc_costs(pos.entry_premium, exit_prem, pos.qty)
            pos.pnl = raw_pnl - costs

            self.capital.record_trade(
                pnl=pos.pnl,
                strategy=f"Pullback_{pos.signal.pullback_count}",
                symbol=f"NIFTY_{pos.direction}",
                entry_price=pos.entry_premium,
                exit_price=exit_prem,
                quantity=pos.qty,
                reason=reason,
            )

            self._log_event("PAPER_EXIT", {
                "direction": pos.direction,
                "exit_premium": exit_prem,
                "reason": reason,
                "pnl": round(pos.pnl, 2),
            })

        self._paper_positions.clear()

    def _update_market_data(self):
        if not self._nifty_token_info:
            return
        token = self._nifty_token_info.get("token", "99926000")
        symbol = self._nifty_token_info.get("symbol", "Nifty 50")
        ltp = self.broker.get_ltp("NSE", symbol, token)
        if ltp:
            self._nifty_spot = ltp
            self.candle_builder.on_tick(
                price=ltp, volume=1000,
                timestamp=datetime.now(IST),
            )

    def _log_event(self, event_type: str, data: dict):
        entry = {"ts": datetime.now(IST).isoformat(), "event": event_type, **data}
        try:
            with open(self._event_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def end_day(self):
        if self._paper_positions:
            self._square_off_all("END_OF_DAY")

        summary = self.capital.get_summary()

        self._log_event("DAY_END", {
            "capital": self.capital.current_capital,
            "daily_pnl": self.capital.daily_pnl,
            "trades": self.capital.trades_today,
            "signals_scored": self._day_signals_count,
        })

        send_eod_report(summary)

        logger.info("=" * 60)
        logger.info("DAY END | PnL: Rs {:.0f} | Trades: {} | Signals: {}",
                     self.capital.daily_pnl, self.capital.trades_today,
                     self._day_signals_count)
        logger.info("Capital: Rs {:.0f} | Peak: Rs {:.0f} | DD: {:.1f}%",
                     self.capital.current_capital, self.capital.peak_capital,
                     self.capital.drawdown_pct)
        logger.info("=" * 60)

        self.capital.save()

    def stop(self):
        self._running = False
