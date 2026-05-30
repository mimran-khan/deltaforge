"""Main trading engine -- orchestrates data, signals, risk, and execution."""

from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz
from loguru import logger

from config import settings
from engine.broker import BrokerConnection
from engine.candle_builder import CandleBuilder
from engine.indicators import add_all_indicators
from engine.strike_selector import StrikeSelector
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from strategies.orb_options import ORBStrategy
from strategies.vwap_momentum import VWAPMomentumStrategy
from risk.risk_engine import RiskEngine
from risk.capital_tracker import CapitalTracker
from risk.kill_switch import is_halted
from execution.position_manager import PositionManager
from alerts.telegram_bot import (
    send_trade_alert, send_eod_report, send_system_alert
)

IST = pytz.timezone("Asia/Kolkata")


class TradingEngine:
    """The heart of the system. Runs the intraday trading loop."""

    def __init__(self):
        self.broker = BrokerConnection()
        self.capital = CapitalTracker()
        self.risk = RiskEngine(self.capital)
        self.strike_selector = StrikeSelector()
        self.position_mgr = PositionManager(self.broker, self.capital)
        self.candle_builder = CandleBuilder(interval_minutes=5)

        self.strategies: list[BaseStrategy] = [
            ORBStrategy(),
            VWAPMomentumStrategy(),
        ]

        self._running = False
        self._nifty_token_info: Optional[dict] = None
        self._nifty_spot: float = 0

    # ── Lifecycle ───────────────────────────────────────────────────

    def start_day(self):
        """Initialize everything for a new trading day."""
        logger.info("=" * 60)
        logger.info("TRADING DAY START -- {}", datetime.now(IST).strftime("%Y-%m-%d"))
        logger.info("Mode: {}", settings.TRADING_MODE.upper())
        logger.info("Capital: Rs {:.0f}", self.capital.current_capital)
        logger.info("=" * 60)

        self.risk.start_day()
        self.candle_builder.reset()
        for s in self.strategies:
            s.reset()

        self.strike_selector.load_instruments()
        self._nifty_token_info = self.strike_selector.get_nifty_token()

        send_system_alert(
            "Trading Day Started",
            f"Mode: {settings.TRADING_MODE.upper()}\n"
            f"Capital: Rs {self.capital.current_capital:.0f}\n"
            f"Max lots: {self.capital.get_max_lots()}"
        )

    def run_loop(self, poll_interval: int = 5):
        """Main trading loop -- polls data and processes signals.

        For production with WebSocket, this would be event-driven.
        For initial deployment, polling is simpler and more reliable.
        """
        self._running = True
        logger.info("Trading loop started (poll every {}s)", poll_interval)

        while self._running:
            try:
                now = datetime.now(IST)
                time_str = now.strftime("%H:%M")

                if is_halted():
                    logger.warning("Kill switch active -- stopping loop")
                    self._running = False
                    break

                if time_str >= settings.SQUARE_OFF_TIME:
                    self.position_mgr.time_based_exit()
                    logger.info("Past square-off time -- ending loop")
                    self._running = False
                    break

                if time_str < settings.MARKET_OPEN:
                    time.sleep(10)
                    continue

                self._tick()
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("Loop interrupted by user")
                self._running = False
            except Exception as e:
                logger.exception("Loop error: {}", e)
                time.sleep(10)

        self.end_day()

    def _tick(self):
        """Single iteration of the trading loop."""
        if not self.broker.is_active:
            return

        now = datetime.now(IST)
        time_str = now.strftime("%H:%M")

        self._update_market_data()

        if self.risk.is_halted:
            return

        self.position_mgr.time_based_exit()
        self.risk.check_daily_limits()

        if self.risk.is_halted:
            if self.position_mgr.has_open_positions:
                self.position_mgr.square_off_all("RISK_HALT")
            return

        if time_str >= settings.NO_NEW_ENTRY_AFTER:
            return

        candles = self.candle_builder.get_candles()
        if len(candles) < 5:
            return

        candles_with_indicators = add_all_indicators(candles)

        for strategy in self.strategies:
            signal = strategy.on_candle(candles_with_indicators, now)
            if signal:
                self._process_signal(signal)

    def _update_market_data(self):
        """Fetch latest Nifty price and build candles."""
        if not self._nifty_token_info:
            return

        token = self._nifty_token_info.get("token", "99926000")
        symbol = self._nifty_token_info.get("symbol", "Nifty 50")

        ltp = self.broker.get_ltp("NSE", symbol, token)
        if ltp:
            self._nifty_spot = ltp
            self.candle_builder.on_tick(
                price=ltp,
                volume=1000,
                timestamp=datetime.now(IST),
            )

        self._update_position_prices()

    def _update_position_prices(self):
        """Update open position prices for P&L tracking."""
        prices = {}
        for pos in self.position_mgr.open_positions:
            ltp = self.broker.get_ltp("NFO", pos.symbol, pos.token)
            if ltp:
                prices[pos.symbol] = ltp
        if prices:
            self.position_mgr.update_positions(prices)

    def _process_signal(self, signal: Signal):
        """Risk-check a signal and execute if approved."""
        decision = self.risk.evaluate(signal)

        if not decision.approved:
            logger.info("Signal REJECTED: {} -- {}", signal.strategy_name, decision.reason)
            return

        strike = self.strike_selector.find_strike(
            spot_price=self._nifty_spot,
            underlying="NIFTY",
            option_type=signal.option_type,
            offset=0,
        )

        if not strike:
            logger.error("No strike found for signal")
            return

        premium_ltp = self.broker.get_ltp("NFO", strike["symbol"], strike["token"])

        if settings.TRADING_MODE == "paper" and not premium_ltp:
            premium_ltp = 100.0

        if not premium_ltp:
            logger.error("Cannot get premium LTP for {}", strike["symbol"])
            return

        pos = self.position_mgr.open_position(
            signal=signal,
            strike_info=strike,
            lots=decision.lots,
            premium_price=premium_ltp,
            premium_sl_pct=decision.premium_sl_pct,
        )

        if pos:
            send_trade_alert(
                action="ENTRY",
                strategy=signal.strategy_name,
                symbol=pos.symbol,
                price=pos.entry_price,
                quantity=pos.quantity,
                sl=pos.stop_loss,
                target=pos.target,
            )

    def end_day(self):
        """End-of-day cleanup."""
        if self.position_mgr.has_open_positions:
            self.position_mgr.square_off_all("END_OF_DAY")

        summary = self.capital.get_summary()
        send_eod_report(summary)

        logger.info("=" * 60)
        logger.info("DAY ENDED -- PnL: Rs {:.0f} ({:.1f}%)",
                     summary["daily_pnl"], summary["daily_pnl_pct"])
        logger.info("Trades: {} | Wins: {} | Losses: {}",
                     summary["trades"], summary["wins"], summary["losses"])
        logger.info("Capital: Rs {:.0f}", summary["capital"])
        logger.info("=" * 60)

        self.capital.save()

    def stop(self):
        self._running = False

    # ── Backtest Mode ───────────────────────────────────────────────

    def backtest(self, historical_df: pd.DataFrame) -> dict:
        """Run strategies on historical data. Returns performance summary.

        historical_df must have: open, high, low, close, volume
        indexed by datetime (IST, intraday candles)
        """
        self.risk.start_day()
        for s in self.strategies:
            s.reset()

        df = add_all_indicators(historical_df)
        signals = []
        trades = []

        current_pos = None
        current_day = None

        for i in range(20, len(df)):
            ts = df.index[i]
            day = ts.date() if hasattr(ts, 'date') else ts

            if day != current_day:
                if current_pos:
                    exit_price = current_pos["entry_premium"] * 0.9
                    pnl = (exit_price - current_pos["entry_premium"]) * current_pos["quantity"]
                    trades.append({**current_pos, "exit_premium": exit_price,
                                   "pnl": pnl, "exit_reason": "EOD"})
                    current_pos = None

                current_day = day
                for s in self.strategies:
                    s.reset()
                self.risk.start_day()

            window = df.iloc[:i + 1]

            for strategy in self.strategies:
                signal = strategy.on_candle(window, ts)
                if signal and current_pos is None:
                    decision = self.risk.evaluate(signal)
                    if decision.approved:
                        premium = 100.0
                        sl = premium * (1 - settings.PREMIUM_SL_PCT / 100)
                        target = premium + settings.PREMIUM_TARGET_POINTS
                        current_pos = {
                            "strategy": signal.strategy_name,
                            "signal": signal.signal_type.value,
                            "entry_time": ts,
                            "entry_index": signal.entry_price,
                            "entry_premium": premium,
                            "sl": sl,
                            "target": target,
                            "quantity": decision.lots * settings.NIFTY_LOT_SIZE,
                        }
                        signals.append(signal)

            if current_pos:
                row = df.iloc[i]
                idx_move = abs(row["close"] - current_pos["entry_index"])
                premium_change = idx_move * 0.5  # rough delta ~ 0.5

                if current_pos["signal"] == "LONG":
                    sim_premium = current_pos["entry_premium"] + (
                        (row["close"] - current_pos["entry_index"]) * 0.5
                    )
                else:
                    sim_premium = current_pos["entry_premium"] + (
                        (current_pos["entry_index"] - row["close"]) * 0.5
                    )

                if sim_premium <= current_pos["sl"]:
                    pnl = (current_pos["sl"] - current_pos["entry_premium"]) * current_pos["quantity"]
                    trades.append({**current_pos, "exit_premium": current_pos["sl"],
                                   "pnl": pnl, "exit_reason": "SL_HIT",
                                   "exit_time": ts})
                    self.capital.record_trade(pnl, current_pos["strategy"],
                                             "BACKTEST", current_pos["entry_premium"],
                                             current_pos["sl"], current_pos["quantity"], "SL")
                    current_pos = None
                elif sim_premium >= current_pos["target"]:
                    pnl = (current_pos["target"] - current_pos["entry_premium"]) * current_pos["quantity"]
                    trades.append({**current_pos, "exit_premium": current_pos["target"],
                                   "pnl": pnl, "exit_reason": "TARGET_HIT",
                                   "exit_time": ts})
                    self.capital.record_trade(pnl, current_pos["strategy"],
                                             "BACKTEST", current_pos["entry_premium"],
                                             current_pos["target"], current_pos["quantity"], "TARGET")
                    current_pos = None

                time_str = ts.strftime("%H:%M") if hasattr(ts, 'strftime') else ""
                if time_str >= "15:15" and current_pos:
                    pnl = (sim_premium - current_pos["entry_premium"]) * current_pos["quantity"]
                    trades.append({**current_pos, "exit_premium": sim_premium,
                                   "pnl": pnl, "exit_reason": "EOD",
                                   "exit_time": ts})
                    self.capital.record_trade(pnl, current_pos["strategy"],
                                             "BACKTEST", current_pos["entry_premium"],
                                             sim_premium, current_pos["quantity"], "EOD")
                    current_pos = None

        total_pnl = sum(t["pnl"] for t in trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        profit_factor = (sum(t["pnl"] for t in wins) /
                         abs(sum(t["pnl"] for t in losses))) if losses and sum(t["pnl"] for t in losses) != 0 else float("inf")

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "trades": trades,
            "signals_generated": len(signals),
        }
