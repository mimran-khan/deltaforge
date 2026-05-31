"""Production trading engine -- event-driven, institutional-grade.

Architecture:
  WebSocket/LTP → CandleBuilder → [closed bar event] → ConfluenceEngine
  → RiskEngine (9 gates) → PositionManager → Broker API

Principles:
  1. Score ONLY on closed 5-min bars (no look-ahead bias)
  2. Same ConfluenceEngine as backtesting (zero drift)
  3. Risk engine operates independently of external systems
  4. Every event logged to append-only JSONL (audit trail)
  5. Paper mode simulates full trade lifecycle with exits
  6. Realistic execution costs modeled in both modes
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
from engine.confluence import ConfluenceEngine, ConfluenceResult
from engine.confluence_pruned import PrunedConfluenceEngine, PrunedResult
from engine.premium_model import create_premium_state, PremiumState
from risk.risk_engine import RiskEngine, RiskDecision
from risk.capital_tracker import CapitalTracker
from risk.kill_switch import is_halted
from alerts.telegram_bot import (
    send_trade_alert, send_eod_report, send_system_alert
)

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class PaperPosition:
    """Tracks a paper trade through its full lifecycle."""
    direction: str
    entry_time: str
    entry_index: float
    entry_premium: float
    sl_premium: float
    target_premium: float
    lots: int
    qty: int
    confluence_score: float
    strength: str
    prem_state: PremiumState
    candles_held: int = 0
    peak_premium: float = 0
    exit_premium: float = 0
    exit_reason: str = ""
    exit_time: str = ""
    pnl: float = 0


class TradingEngine:
    """Production engine with 200+ indicator confluence.

    Modes:
      paper: full trade simulation with entries, exits, and P&L tracking
      live:  real orders via Angel One SmartAPI
    """

    def __init__(self):
        self.broker = BrokerConnection()
        self.capital = CapitalTracker()
        self.risk = RiskEngine(self.capital)
        self.candle_builder = CandleBuilder(interval_minutes=5)
        self.confluence = PrunedConfluenceEngine()
        self.confluence_full = ConfluenceEngine()  # kept for logging

        self._running = False
        self._nifty_token_info: Optional[dict] = None
        self._nifty_spot: float = 0
        self._prev_candle_count: int = 0
        self._day_indicators: dict = {}
        self._paper_positions: list[PaperPosition] = []
        self._day_signals_count: int = 0

        self._event_log = Path(settings.DATA_DIR) / "events.jsonl"
        self._signal_log = Path(settings.DATA_DIR) / "signal_log.jsonl"
        self._trade_log = Path(settings.DATA_DIR) / "trade_log.jsonl"

    def start_day(self):
        """Initialize for a new trading day."""
        now = datetime.now(IST)
        logger.info("=" * 60)
        logger.info("DAY START: {} | Mode: {} | Capital: Rs {:.0f}",
                     now.strftime("%Y-%m-%d"), settings.TRADING_MODE.upper(),
                     self.capital.current_capital)
        logger.info("Risk: {}% daily SL, {} trades/day, DD={:.1f}%",
                     settings.DAILY_LOSS_LIMIT_PCT, settings.MAX_TRADES_PER_DAY,
                     self.capital.drawdown_pct)
        logger.info("=" * 60)

        self.risk.start_day()
        self.candle_builder.reset()
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
            "drawdown": self.capital.drawdown_pct,
        })

        send_system_alert(
            "Day Started",
            f"Mode: {settings.TRADING_MODE.upper()}\n"
            f"Capital: Rs {self.capital.current_capital:.0f}\n"
            f"Drawdown: {self.capital.drawdown_pct:.1f}%\n"
            f"Risk: {settings.DAILY_LOSS_LIMIT_PCT}% daily, {settings.MAX_TRADES_PER_DAY} trades"
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
            self._day_indicators = self.confluence.precompute(candles)
        except Exception as e:
            logger.debug("Precompute error: {}", e)
            return

        idx = current_count - 1
        result = self.confluence.score(self._day_indicators, idx)

        # Log every closed-bar score
        self._log_signal(result)

        # Skip if already in a position
        if self._paper_positions:
            return

        # Risk evaluation (all 9 gates)
        decision = self.risk.evaluate(
            confluence_score=result.score,
            direction=result.direction,
            strength=result.strength,
        )

        if not decision.approved:
            return

        logger.info("SIGNAL: {} | {}", result.summary(),
                     datetime.now(IST).strftime("%H:%M"))

        self._enter_trade(result, decision)

    def _enter_trade(self, result: ConfluenceResult, decision: RiskDecision):
        """Open a trade in paper or live mode."""
        option_type = "CE" if result.direction == "LONG" else "PE"

        if settings.TRADING_MODE == "paper":
            self._paper_enter(result, decision, option_type)
        else:
            self._live_enter(result, decision, option_type)

    def _paper_enter(self, result, decision: RiskDecision, option_type: str):
        """Full paper trade simulation with realistic costs."""
        lots = min(decision.lots, getattr(settings, "MAX_LOTS", 1))
        qty = lots * settings.NIFTY_LOT_SIZE

        prem_state = create_premium_state(
            entry_index_price=self._nifty_spot,
            direction=result.direction,
            base_premium=settings.PREMIUM_BASE,
            delta=settings.PREMIUM_DELTA,
            theta_per_candle=settings.PREMIUM_THETA_PER_CANDLE,
            sl_pct=decision.premium_sl_pct,
            confluence_score=result.score,
        )

        # Add slippage to entry
        entry_premium = prem_state.entry_premium + settings.SLIPPAGE_POINTS

        pos = PaperPosition(
            direction=result.direction,
            entry_time=datetime.now(IST).isoformat(),
            entry_index=self._nifty_spot,
            entry_premium=entry_premium,
            sl_premium=prem_state.sl_premium,
            target_premium=prem_state.target_premium,
            lots=lots,
            qty=qty,
            confluence_score=result.score,
            strength=result.strength,
            prem_state=prem_state,
            peak_premium=entry_premium,
        )
        self._paper_positions.append(pos)

        self._log_event("PAPER_ENTRY", {
            "direction": result.direction,
            "option_type": option_type,
            "entry_index": self._nifty_spot,
            "entry_premium": entry_premium,
            "sl": prem_state.sl_premium,
            "target": prem_state.target_premium,
            "lots": decision.lots,
            "confluence": result.score,
            "strength": result.strength,
            "indicators": result.total_indicators,
        })

        send_trade_alert(
            action="PAPER_ENTRY",
            strategy=f"Confluence ({result.strength})",
            symbol=f"NIFTY {option_type}",
            price=entry_premium,
            quantity=qty,
            sl=prem_state.sl_premium,
            target=prem_state.target_premium,
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

            # Check exits
            exit_reason = None

            # SL hit
            if cur_prem <= pos.sl_premium:
                exit_reason = "SL"
                pos.exit_premium = pos.sl_premium

            # Target hit
            elif cur_prem >= pos.target_premium:
                exit_reason = "TGT"
                pos.exit_premium = pos.target_premium

            # Time exit (hold period complete)
            elif pos.candles_held >= settings.CONFLUENCE_HOLD_CANDLES:
                exit_reason = "TIME"
                pos.exit_premium = cur_prem

            if exit_reason:
                pos.exit_reason = exit_reason
                pos.exit_time = datetime.now(IST).isoformat()

                # Calculate P&L with costs
                raw_pnl = (pos.exit_premium - pos.entry_premium) * pos.qty
                costs = self._calc_costs(pos.entry_premium, pos.exit_premium, pos.qty)
                pos.pnl = raw_pnl - costs

                self.capital.record_trade(
                    pnl=pos.pnl,
                    strategy=f"Confluence_{pos.strength}",
                    symbol=f"NIFTY_{pos.direction}",
                    entry_price=pos.entry_premium,
                    exit_price=pos.exit_premium,
                    quantity=pos.qty,
                    reason=exit_reason,
                )

                self._log_event("PAPER_EXIT", {
                    "direction": pos.direction,
                    "entry_premium": pos.entry_premium,
                    "exit_premium": pos.exit_premium,
                    "reason": exit_reason,
                    "candles_held": pos.candles_held,
                    "pnl": round(pos.pnl, 2),
                    "costs": round(costs, 2),
                    "capital": round(self.capital.current_capital, 2),
                })

                send_trade_alert(
                    action=f"PAPER_EXIT ({exit_reason})",
                    strategy=f"Confluence_{pos.strength}",
                    symbol=f"NIFTY_{pos.direction}",
                    price=pos.exit_premium,
                    quantity=pos.qty,
                    sl=0, target=0,
                )

                closed.append(pos)

        for pos in closed:
            self._paper_positions.remove(pos)

    def _calc_costs(self, entry_prem: float, exit_prem: float, qty: int) -> float:
        """Calculate realistic execution costs for NFO."""
        brokerage = settings.BROKERAGE_PER_ORDER * 2  # entry + exit
        stt = exit_prem * qty * settings.STT_PCT / 100
        stamp = entry_prem * qty * settings.STAMP_DUTY_PCT / 100
        slippage = settings.SLIPPAGE_POINTS * qty
        return brokerage + stt + stamp + slippage

    def _live_enter(self, result: ConfluenceResult,
                     decision: RiskDecision, option_type: str):
        """Execute real trade via Angel One API."""
        from strategies.base_strategy import Signal, SignalType

        signal = Signal(
            signal_type=SignalType.LONG if result.direction == "LONG" else SignalType.SHORT,
            strategy_name=f"Confluence_{result.strength}",
            entry_price=self._nifty_spot,
            option_type=option_type,
            confluence_score=result.score,
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
            signal=signal, strike_info=strike,
            lots=decision.lots, premium_price=premium_ltp,
            premium_sl_pct=decision.premium_sl_pct,
        )

        if pos:
            self._log_event("LIVE_ENTRY", {
                "symbol": pos.symbol, "direction": result.direction,
                "entry": pos.entry_price, "lots": decision.lots,
                "confluence": result.score,
            })
            send_trade_alert(
                action="ENTRY", strategy=signal.strategy_name,
                symbol=pos.symbol, price=pos.entry_price,
                quantity=pos.quantity, sl=pos.stop_loss, target=pos.target,
            )

    def _square_off_all(self, reason: str):
        """Force close all paper positions."""
        for pos in list(self._paper_positions):
            cur_prem = pos.prem_state.current_premium(
                self._nifty_spot, pos.candles_held)
            pos.exit_premium = cur_prem
            pos.exit_reason = reason
            pos.exit_time = datetime.now(IST).isoformat()

            raw_pnl = (cur_prem - pos.entry_premium) * pos.qty
            costs = self._calc_costs(pos.entry_premium, cur_prem, pos.qty)
            pos.pnl = raw_pnl - costs

            self.capital.record_trade(
                pnl=pos.pnl,
                strategy=f"Confluence_{pos.strength}",
                symbol=f"NIFTY_{pos.direction}",
                entry_price=pos.entry_premium,
                exit_price=cur_prem,
                quantity=pos.qty,
                reason=reason,
            )

            self._log_event("PAPER_EXIT", {
                "direction": pos.direction,
                "exit_premium": cur_prem,
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

    def _log_signal(self, result):
        entry = {
            "ts": datetime.now(IST).isoformat(),
            "nifty": self._nifty_spot,
            "score": result.score,
            "dir": result.direction,
            "str": result.strength,
            "bull": result.bullish_count,
            "bear": result.bearish_count,
            "total": result.total_indicators,
        }
        try:
            with open(self._signal_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        self._day_signals_count += 1

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
