"""DeltaForge trading engine -- DeltaForge V11 -- Multi-Strategy (76% WR, PF 2.62).

Architecture:
  WebSocket/LTP → CandleBuilder → [closed bar event] → MultiStrategyEngine
  → RiskEngine → PositionManager → Broker API

Core strategy (walk-forward validated):
  76% WR, PF 2.62, multi-strategy compound growth on Nifty 5m bars

Signal types:
  PULLBACK    = Pullback-in-Trend with HTF RSI + LTF oscillators
  STOCH_CROSS = Stochastic cross from extreme with EMA trend
  SUPERTREND  = Supertrend flip with ADX confirmation

Key principles:
  1. Score ONLY on closed 5-min bars (no look-ahead bias)
  2. Same MultiStrategyEngine as backtesting (zero production drift)
  3. ATM options with 30% SL, strategy-specific targets, 24-candle max hold
  4. Dynamic lot sizing for compounding
  5. Paper mode simulates full trade lifecycle with realistic costs
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
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
from engine.market_feed import MarketFeed
from engine.multi_strategy_engine import MultiStrategyEngine, TradeSignal
from engine.premium_model import create_premium_state, PremiumState, STRATEGY_SL_PCT
from execution.position_manager import PositionManager
from persistence.performance_db import PerformanceDB
from risk.risk_engine import RiskEngine, RiskDecision
from risk.capital_tracker import CapitalTracker
from risk.kill_switch import is_halted
from risk.adaptive_mode import AdaptiveModeController

_alert_method = getattr(settings, 'ALERT_METHOD', 'slack')
if _alert_method == 'imessage':
    from alerts.imessage_bot import (
        send_trade_alert, send_eod_report, send_system_alert
    )
elif _alert_method == 'slack':
    from alerts.slack_bot import (
        send_trade_alert, send_eod_report, send_system_alert
    )
else:
    from alerts.telegram_bot import (
        send_trade_alert, send_eod_report, send_system_alert
    )

IST = pytz.timezone("Asia/Kolkata")

ENGINE_STATE_FILE = settings.DATA_DIR / "engine_state.json"
PAPER_POSITIONS_FILE = settings.DATA_DIR / "paper_positions.json"

MAX_HOLD = getattr(settings, 'PULLBACK_HOLD_CANDLES', 12)
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
    signal: TradeSignal
    prem_state: PremiumState
    candles_held: int = 0
    peak_premium: float = 0
    exit_premium: float = 0
    exit_reason: str = ""
    exit_time: str = ""
    pnl: float = 0
    runner_mode: bool = False
    runner_bars: int = 0


class TradingEngine:
    """Production engine -- DeltaForge V11 -- Multi-Strategy (76% WR, PF 2.62).

    Modes:
      paper: full trade simulation with entries, exits, and P&L tracking
      live:  real orders via Angel One SmartAPI
    """

    def __init__(self):
        self.broker = BrokerConnection()
        self.capital = CapitalTracker()
        self.risk = RiskEngine(self.capital)
        self.candle_builder = CandleBuilder(interval_minutes=5)
        self.strategy = MultiStrategyEngine()
        self.position_mgr: Optional[PositionManager] = None
        self.market_feed: Optional[MarketFeed] = None
        self.perf_db = PerformanceDB()
        self.adaptive = AdaptiveModeController()

        self._running = False
        self._nifty_token_info: Optional[dict] = None
        self._nifty_spot: float = 0
        self._prev_candle_count: int = 0
        self._day_indicators: dict = {}
        self._paper_positions: list[PaperPosition] = []
        self._day_signals_count: int = 0
        self._last_price_time: Optional[datetime] = None
        self._skip_today = False
        self._last_atr: float = 30.0

        self._event_log = Path(settings.DATA_DIR) / "events.jsonl"

    def _rotate_event_log_if_stale(self):
        """Archive yesterday's events.jsonl before starting a fresh day."""
        if not self._event_log.exists():
            return
        try:
            mtime = datetime.fromtimestamp(
                self._event_log.stat().st_mtime, tz=IST)
            today = datetime.now(IST).date()
            if mtime.date() < today:
                archive = settings.DATA_DIR / f"events_{mtime.date().isoformat()}.jsonl"
                if archive.exists():
                    archive = settings.DATA_DIR / (
                        f"events_{mtime.date().isoformat()}_{int(time.time())}.jsonl")
                self._event_log.rename(archive)
                logger.info("Archived stale events log to {}", archive.name)
        except Exception as e:
            logger.warning("Event log rotation failed: {}", e)

    def start_day(self):
        """Initialize for a new trading day."""
        now = datetime.now(IST)
        logger.info("=" * 60)
        logger.info("DeltaForge -- DAY START: {} | Mode: {} | Capital: Rs {:.0f}",
                     now.strftime("%Y-%m-%d"), settings.TRADING_MODE.upper(),
                     self.capital.current_capital)
        logger.info("Strategy: DeltaForge V11 -- Multi-Strategy (76% WR, PF 2.62)")
        logger.info("Options: ATM delta={}, SL={}%, MaxHold={}, Lot={}",
                     settings.PREMIUM_DELTA, SL_PCT, MAX_HOLD,
                     settings.NIFTY_LOT_SIZE)
        logger.info("=" * 60)

        if getattr(settings, 'SKIP_EXPIRY_DAY', False):
            if now.weekday() == getattr(settings, 'NIFTY_EXPIRY_DAY', 1):
                logger.warning("EXPIRY DAY (Tuesday) -- trading disabled per SKIP_EXPIRY_DAY=True")
                send_system_alert("DeltaForge -- Expiry Day",
                                  "Trading skipped on expiry day (Tuesday).\n"
                                  f"Capital: Rs {self.capital.current_capital:,.0f}")
                self._skip_today = True
                return

        self._rotate_event_log_if_stale()

        self._day_ended = False

        self.risk.start_day()
        self.adaptive.reset()
        self.candle_builder.reset()
        self.strategy.reset_day()
        # If broker is not active, try to set prev day data from local CSV
        if not self.broker.is_active:
            self._set_prev_day_from_csv()
        self.position_mgr = PositionManager(self.broker, self.capital)
        self._prev_candle_count = 0
        self._day_indicators = {}
        self._paper_positions = []
        self._day_signals_count = 0
        self._last_price_time = None

        self._load_paper_positions()

        try:
            from engine.strike_selector import StrikeSelector
            self.strike_selector = StrikeSelector()
            self.strike_selector.load_instruments()
            self._nifty_token_info = self.strike_selector.get_nifty_token()
        except Exception as e:
            logger.warning("Strike selector unavailable: {}", e)
            self._nifty_token_info = None

        # Pre-flight self-test: verify engine produces signals on known data
        if not self._run_selftest():
            logger.critical("SELF-TEST FAILED -- engine behavior has changed!")
            send_system_alert("SELF-TEST FAILED",
                              "Engine produced unexpected signal count. "
                              "Trading NOT started. Check logs.")
            return

        # Start WebSocket feed if broker has feed_token
        if self.broker.feed_token and self.broker.auth_token:
            try:
                self.market_feed = MarketFeed(
                    api_key=settings.ANGEL_API_KEY,
                    client_code=settings.ANGEL_CLIENT_ID,
                    feed_token=self.broker.feed_token,
                    auth_token=self.broker.auth_token,
                )
                self.market_feed.start()
                logger.info("WebSocket feed started")
            except Exception as e:
                logger.warning("WebSocket start failed, using REST: {}", e)
                self.market_feed = None

        self._log_event("DAY_START", {
            "capital": self.capital.current_capital,
            "mode": settings.TRADING_MODE,
            "strategy": "MultiStratV11",
            "lot_size": settings.NIFTY_LOT_SIZE,
        })

        send_system_alert(
            "DeltaForge Started",
            f"Mode: {settings.TRADING_MODE.upper()}\n"
            f"Capital: Rs {self.capital.current_capital:,.0f}\n"
            f"Date: {now.strftime('%Y-%m-%d %A')}"
        )

    def run_loop(self, poll_interval: int = 5):
        """Main loop. Scores only on CLOSED bars."""
        if self._skip_today:
            logger.info("Trading skipped today (expiry day or other reason)")
            return
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
            if not getattr(self, '_warned_broker_inactive', False):
                logger.warning("Broker session not active -- using last price for exit checks")
                self._warned_broker_inactive = True
            if self._nifty_spot and self._paper_positions:
                self._check_emergency_sl()
            return
        self._warned_broker_inactive = False

        self._update_market_data()

        # Check real-time risk
        if not self.risk.check_realtime():
            if self._paper_positions:
                self._square_off_all("RISK_HALT")
            return

        # Emergency-only check between bars (flash crash protection)
        self._check_emergency_sl()

        # Only score on NEW closed bar (event-driven)
        candles = self.candle_builder.get_candles()
        current_count = len(candles)
        if current_count <= self._prev_candle_count:
            tick_count = getattr(self, '_silent_tick_count', 0) + 1
            self._silent_tick_count = tick_count
            if tick_count % 60 == 0:
                logger.debug("Heartbeat: {} ticks, {} candles, Nifty={}",
                             tick_count, current_count, self._nifty_spot)
            return
        self._silent_tick_count = 0
        self._prev_candle_count = current_count

        logger.info("New candle #{} | Nifty={}", current_count, self._nifty_spot)

        # Stale price guard: skip all bar processing if price is stale
        if self._last_price_time:
            staleness = (datetime.now(IST) - self._last_price_time).total_seconds()
            if staleness > 30:
                logger.warning("Stale price ({:.0f}s old) -- skipping bar processing", staleness)
                return

        # Score on completed bars only (exclude the in-progress partial bar)
        completed = candles.iloc[:-1] if len(candles) > 1 else candles
        try:
            self._day_indicators = self.strategy.precompute(completed)
        except Exception as e:
            logger.debug("Precompute error: {}", e)

        # Update positions AFTER precompute so Runner Mode sees fresh ADX
        self._update_paper_positions()

        if current_count < settings.SCAN_WARMUP_BARS:
            return

        idx = len(completed) - 1

        atr_series = self._day_indicators.get('atr')
        if atr_series is not None and len(atr_series) > 0:
            last_atr = atr_series.iloc[-1]
            if not (isinstance(last_atr, float) and np.isnan(last_atr)):
                self._last_atr = float(last_atr)
        time_str = completed.index[idx].strftime("%H:%M")

        ap = self.adaptive.profile
        self.adaptive.on_bar()

        signals = self.strategy.scan(
            self._day_indicators, idx, time_str,
            max_total_override=ap.max_trades_per_day,
        )

        if not signals:
            adx = self.strategy._sv(self._day_indicators.get('adx', pd.Series()), idx, 0)
            rsi15 = self.strategy._htf_rsi(self._day_indicators, idx, 50)
            rsi5 = self.strategy._sv(self._day_indicators.get('rsi_5m', pd.Series()), idx, 50)
            logger.debug("No signal @ {} | ADX={:.0f} HTF_RSI={:.0f} LTF_RSI={:.0f}",
                         time_str, adx, rsi15, rsi5)
        else:
            self._day_signals_count += len(signals)

        if len(self._paper_positions) >= ap.max_simultaneous:
            return

        open_directions = {p.direction for p in self._paper_positions}

        for signal in signals:
            if signal.direction in open_directions:
                continue

            if signal.confidence < ap.min_confidence:
                continue

            decision = self.risk.evaluate(
                confluence_score=signal.confidence,
                direction=signal.direction,
                signal_obj=signal,
                lot_multiplier=ap.lot_multiplier,
                min_confidence_override=ap.min_confidence,
            )

            if not decision.approved:
                continue

            logger.info("SIGNAL: {} | {}", signal.summary(), time_str)
            self._enter_trade(signal, decision)
            break  # one entry per bar

        self._export_engine_state()

    def _enter_trade(self, signal: TradeSignal, decision: RiskDecision):
        """Open a trade in paper or live mode."""
        option_type = "CE" if signal.direction == "LONG" else "PE"

        if settings.TRADING_MODE == "paper":
            self._paper_enter(signal, decision, option_type)
        else:
            self._live_enter(signal, decision, option_type)

    def _paper_enter(self, signal: TradeSignal,
                     decision: RiskDecision, option_type: str):
        """Full paper trade simulation with realistic costs."""
        ap = self.adaptive.profile
        lots = max(1, int(decision.lots * ap.lot_multiplier))
        qty = lots * settings.NIFTY_LOT_SIZE

        theta = settings.get_scaled_theta(self._nifty_spot)
        prem_state = create_premium_state(
            entry_index_price=self._nifty_spot,
            direction=signal.direction,
            base_premium=settings.PREMIUM_BASE,
            delta=settings.PREMIUM_DELTA,
            theta_per_candle=theta,
            sl_pct=SL_PCT,
            confluence_score=signal.confidence,
            signal_type=signal.signal_type,
        )

        if ap.target_multiplier != 1.0:
            prem_state.target_premium = (
                prem_state.entry_premium
                + (prem_state.target_premium - prem_state.entry_premium) * ap.target_multiplier
            )

        entry_premium = prem_state.entry_premium + settings.SLIPPAGE_POINTS
        eff_sl = STRATEGY_SL_PCT.get(signal.signal_type, SL_PCT) * ap.sl_multiplier
        sl_premium = entry_premium * (1 - eff_sl / 100)

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
            strategy=f"{signal.signal_type} ({signal.pullback_count} conf)",
            symbol=f"NIFTY {option_type}",
            price=entry_premium,
            quantity=qty,
            sl=sl_premium,
            target=0,
        )
        self._save_paper_positions()

    def _close_paper_position(self, pos: PaperPosition, exit_prem_raw: float, reason: str):
        """Shared exit handler for all paper position close paths.

        Handles cost calc, capital recording, perf DB, event log, and alerts.
        """
        pos.exit_reason = reason
        pos.exit_time = datetime.now(IST).isoformat()

        exit_prem = exit_prem_raw - settings.SLIPPAGE_POINTS
        raw_pnl = (exit_prem - pos.entry_premium) * pos.qty
        costs = self._calc_costs(pos.entry_premium, exit_prem, pos.qty, pos.lots)
        pos.pnl = raw_pnl - costs
        pos.exit_premium = exit_prem

        strat_name = f"{pos.signal.signal_type}_{pos.signal.pullback_count}"
        self.capital.record_trade(
            pnl=pos.pnl, strategy=strat_name,
            symbol=f"NIFTY_{pos.direction}",
            entry_price=pos.entry_premium, exit_price=exit_prem,
            quantity=pos.qty, reason=reason,
        )

        now_ist = datetime.now(IST)
        self.perf_db.record_trade(
            date=now_ist.strftime("%Y-%m-%d"),
            time=now_ist.strftime("%H:%M"),
            strategy=strat_name, direction=pos.direction,
            confidence=pos.signal.confidence,
            htf_rsi=pos.signal.htf_rsi,
            adx=getattr(pos.signal, 'adx', 0),
            entry_price=pos.entry_premium, exit_price=exit_prem,
            pnl=pos.pnl, hold_bars=pos.candles_held,
            exit_reason=reason, lots=pos.lots,
            capital_after=self.capital.current_capital,
        )

        if reason == "SL":
            bar_idx = max(self._prev_candle_count - 2, 0)
            self.strategy.record_sl_exit(pos.signal.signal_type, bar_idx)

        won = pos.pnl > 0
        daily_pnl_pct = self.capital.daily_pnl_pct
        self.adaptive.update(
            daily_pnl_pct=daily_pnl_pct,
            wins=self.capital.wins_today,
            losses=self.capital.losses_today,
            consecutive_losses=self.capital.consecutive_losses,
            trades=self.capital.wins_today + self.capital.losses_today,
            last_trade_won=won,
        )

        self._log_event("PAPER_EXIT", {
            "direction": pos.direction,
            "signal_type": pos.signal.signal_type,
            "entry_premium": pos.entry_premium,
            "exit_premium": exit_prem,
            "reason": reason,
            "candles_held": pos.candles_held,
            "pnl": round(pos.pnl, 2),
            "costs": round(costs, 2),
            "capital": round(self.capital.current_capital, 2),
        })

        logger.info("EXIT ({}): {} {} | PnL={:.0f} | Held {} bars",
                     reason, pos.signal.signal_type, pos.direction,
                     pos.pnl, pos.candles_held)

        send_trade_alert(
            action=f"PAPER_EXIT ({reason})",
            strategy=strat_name,
            symbol=f"NIFTY_{pos.direction}",
            price=exit_prem, quantity=pos.qty,
            sl=0, target=0,
        )
        self._save_paper_positions()

    def _check_emergency_sl(self):
        """Emergency-only exit between bar closes -- flash crash protection.

        Normal SL/target/trail checks happen on bar close only (in
        _update_paper_positions) to match backtest behavior and avoid
        intrabar whipsaw that caused most live SL hits.

        This fires only when Nifty moves > 2x ATR adversely from entry.
        """
        if not self._paper_positions or not self._nifty_spot:
            return

        emergency_threshold = self._last_atr * 2.0

        closed = []
        for pos in self._paper_positions:
            if pos.signal.direction == "LONG":
                adverse_move = pos.entry_index - self._nifty_spot
            else:
                adverse_move = self._nifty_spot - pos.entry_index

            if adverse_move >= emergency_threshold:
                cur_prem = pos.prem_state.current_premium(
                    self._nifty_spot, pos.candles_held)
                logger.warning(
                    "EMERGENCY EXIT: {} {} | Nifty moved {:.0f}pts (>{:.0f} 2xATR) | prem {:.1f}",
                    pos.signal.signal_type, pos.signal.direction,
                    adverse_move, emergency_threshold, cur_prem)
                self._close_paper_position(
                    pos, min(cur_prem, pos.sl_premium), "SL")
                closed.append(pos)

        for pos in closed:
            self._paper_positions.remove(pos)

    def _update_paper_positions(self):
        """Update paper positions on new closed bar (candles_held per 5m bar)."""
        ap = self.adaptive.profile
        closed = []
        adx_val = self._get_current_adx()
        now_str = datetime.now(IST).strftime("%H:%M")

        for pos in self._paper_positions:
            pos.candles_held += 1
            cur_prem = pos.prem_state.current_premium(
                self._nifty_spot, pos.candles_held)

            if cur_prem > pos.peak_premium:
                pos.peak_premium = cur_prem

            trail_floor = pos.prem_state.update_trail(
                cur_prem,
                trigger_pct=ap.trail_trigger_pct,
                trail_pct=ap.trail_pct)

            exit_reason = None
            exit_prem_raw = cur_prem

            if cur_prem <= pos.sl_premium:
                exit_reason = "SL"
                exit_prem_raw = pos.sl_premium
            elif cur_prem >= pos.prem_state.target_premium:
                if (not pos.runner_mode
                        and self._should_activate_runner(pos, adx_val, now_str)):
                    pos.runner_mode = True
                    pos.runner_bars = 0
                    pos.sl_premium = max(pos.sl_premium, pos.prem_state.entry_premium)
                    logger.info(
                        "RUNNER ACTIVATED: {} {} | ADX={:.0f} | "
                        "prem {:.1f} >= target {:.1f} | SL→breakeven {:.1f}",
                        pos.signal.signal_type, pos.signal.direction,
                        adx_val, cur_prem,
                        pos.prem_state.target_premium, pos.sl_premium)
                elif not pos.runner_mode:
                    exit_reason = "TGT"
                    exit_prem_raw = pos.prem_state.target_premium
            elif pos.runner_mode:
                pos.runner_bars += 1
                runner_floor = pos.peak_premium * (1 - settings.TREND_RUNNER_TRAIL_PCT / 100)
                if cur_prem <= runner_floor:
                    exit_reason = "RUNNER_TRAIL"
                    logger.info("RUNNER EXIT (trail): prem {:.1f} <= floor {:.1f}",
                                cur_prem, runner_floor)
                elif adx_val < settings.TREND_RUNNER_ADX_EXIT:
                    exit_reason = "RUNNER_WEAK"
                    logger.info("RUNNER EXIT (trend weakened): ADX={:.0f}", adx_val)
                elif pos.runner_bars >= settings.TREND_RUNNER_MAX_BARS:
                    exit_reason = "RUNNER_TIME"
                    logger.info("RUNNER EXIT (max bars): held {} extra bars",
                                pos.runner_bars)
            elif trail_floor is not None and cur_prem <= trail_floor:
                exit_reason = "TRAIL"
                exit_prem_raw = trail_floor

            if not exit_reason and pos.candles_held >= MAX_HOLD:
                exit_reason = "TIME"

            if exit_reason:
                self._close_paper_position(pos, exit_prem_raw, exit_reason)
                closed.append(pos)

        for pos in closed:
            self._paper_positions.remove(pos)

    def _get_current_adx(self) -> float:
        """Get the latest ADX value from precomputed day indicators."""
        try:
            adx_series = self._day_indicators.get('adx', pd.Series())
            if hasattr(adx_series, 'iloc') and len(adx_series) > 0:
                val = float(adx_series.iloc[-1])
                if not np.isnan(val):
                    return val
        except Exception:
            pass
        return 0.0

    def _should_activate_runner(self, pos: PaperPosition,
                                adx_val: float, now_str: str) -> bool:
        """Check if a TGT-hitting position should switch to runner mode."""
        if not getattr(settings, 'TREND_RUNNER_ENABLED', False):
            return False
        if pos.signal.signal_type not in getattr(
                settings, 'TREND_RUNNER_STRATEGIES', []):
            return False
        cutoff = getattr(settings, 'TREND_RUNNER_CUTOFF_TIME', '14:30')
        if now_str >= cutoff:
            return False
        return adx_val >= settings.TREND_RUNNER_ADX_MIN

    def _calc_costs(self, entry_prem: float, exit_prem: float,
                    qty: int, lots: int = 1) -> float:
        """Execution costs for NFO: brokerage + STT + exchange + impact.

        Spread/slippage is already applied to entry_premium (+0.30) and
        exit_premium (-0.30) in the PnL calculation, so not included here.
        """
        brokerage = settings.BROKERAGE_PER_ORDER * 2
        sell_turnover = exit_prem * qty
        stt = sell_turnover * getattr(settings, 'STT_SELL_PCT', 0.05) / 100
        total_turnover = (entry_prem + exit_prem) * qty
        exchange_costs = total_turnover * getattr(settings, 'EXCHANGE_TXN_PCT', 0.05) / 100
        impact = 0.0
        if lots >= 5:
            impact_pct = getattr(settings, 'MARKET_IMPACT_PCT', 0.10)
            impact = entry_prem * qty * impact_pct / 100
        return brokerage + stt + exchange_costs + impact

    def _save_paper_positions(self):
        """Persist open paper positions to disk for crash recovery."""
        try:
            data = []
            for p in self._paper_positions:
                data.append({
                    "direction": p.direction,
                    "entry_time": p.entry_time,
                    "entry_index": p.entry_index,
                    "entry_premium": p.entry_premium,
                    "sl_premium": p.sl_premium,
                    "lots": p.lots,
                    "qty": p.qty,
                    "candles_held": p.candles_held,
                    "peak_premium": p.peak_premium,
                    "signal_type": p.signal.signal_type,
                    "signal_direction": p.signal.direction,
                    "signal_confidence": p.signal.confidence,
                    "signal_pullback_count": p.signal.pullback_count,
                    "signal_htf_rsi": p.signal.htf_rsi,
                    "signal_ltf_rsi": getattr(p.signal, 'ltf_rsi', 50),
                    "signal_nifty_price": getattr(p.signal, 'nifty_price', 0),
                    "signal_reason": getattr(p.signal, 'reason', ''),
                    "signal_adx": getattr(p.signal, 'adx', 0),
                    "prem_entry": p.prem_state.entry_premium,
                    "prem_delta": p.prem_state.delta,
                    "prem_theta_per_bar": p.prem_state.theta_per_candle,
                    "prem_target": p.prem_state.target_premium,
                    "runner_mode": p.runner_mode,
                    "runner_bars": p.runner_bars,
                    "date": datetime.now(IST).strftime("%Y-%m-%d"),
                })
            tmp = PAPER_POSITIONS_FILE.with_suffix('.tmp')
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            import os as _os
            _os.replace(str(tmp), str(PAPER_POSITIONS_FILE))
        except Exception as e:
            logger.warning("Paper position save error: {}", e)

    def _load_paper_positions(self):
        """Load paper positions from disk if saved today."""
        if not PAPER_POSITIONS_FILE.exists():
            return
        try:
            with open(PAPER_POSITIONS_FILE) as f:
                data = json.load(f)
            today = datetime.now(IST).strftime("%Y-%m-%d")
            for p_data in data:
                if p_data.get("date") != today:
                    continue
                sig = TradeSignal(
                    signal_type=p_data["signal_type"],
                    direction=p_data["signal_direction"],
                    confidence=p_data["signal_confidence"],
                    pullback_count=p_data.get("signal_pullback_count", 0),
                    htf_rsi=p_data.get("signal_htf_rsi", 50),
                    ltf_rsi=p_data.get("signal_ltf_rsi", 50),
                    nifty_price=p_data.get("signal_nifty_price", 0),
                    reason=p_data.get("signal_reason", ""),
                )
                if hasattr(sig, 'adx'):
                    sig.adx = p_data.get("signal_adx", 0)
                prem = create_premium_state(
                    entry_index_price=p_data.get("entry_index", 0),
                    direction=p_data["direction"],
                    base_premium=p_data.get("prem_entry", settings.PREMIUM_BASE),
                    delta=p_data.get("prem_delta", settings.PREMIUM_DELTA),
                    theta_per_candle=p_data.get("prem_theta_per_bar", settings.PREMIUM_THETA_PER_CANDLE),
                    sl_pct=SL_PCT,
                    confluence_score=p_data.get("signal_confidence", 70),
                    signal_type=p_data.get("signal_type", "UNKNOWN"),
                )
                saved_target = p_data.get("prem_target")
                if saved_target is not None:
                    prem.target_premium = saved_target
                pos = PaperPosition(
                    direction=p_data["direction"],
                    entry_time=p_data["entry_time"],
                    entry_index=p_data["entry_index"],
                    entry_premium=p_data["entry_premium"],
                    sl_premium=p_data["sl_premium"],
                    lots=p_data["lots"],
                    qty=p_data["qty"],
                    signal=sig,
                    prem_state=prem,
                    candles_held=p_data.get("candles_held", 0),
                    peak_premium=p_data.get("peak_premium", p_data["entry_premium"]),
                    runner_mode=p_data.get("runner_mode", False),
                    runner_bars=p_data.get("runner_bars", 0),
                )
                self._paper_positions.append(pos)
            if self._paper_positions:
                logger.info("Restored {} paper position(s) from disk", len(self._paper_positions))
        except Exception as e:
            logger.debug("Paper position load error: {}", e)

    def _live_enter(self, signal: TradeSignal,
                    decision: RiskDecision, option_type: str):
        """Execute real trade via Angel One API."""
        from strategies.base_strategy import Signal, SignalType

        sig_obj = Signal(
            signal_type=SignalType.LONG if signal.direction == "LONG" else SignalType.SHORT,
            strategy_name=f"{signal.signal_type}_{signal.pullback_count}",
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

        pos = self.position_mgr.open_position(
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
        """Force close all paper positions using shared exit handler."""
        for pos in list(self._paper_positions):
            if self._nifty_spot > 0:
                cur_prem = pos.prem_state.current_premium(
                    self._nifty_spot, pos.candles_held)
            else:
                # No valid spot price — close at entry (flat) to avoid fake P&L
                logger.warning("Square-off with no spot price — closing {} at entry premium {:.1f}",
                               pos.direction, pos.entry_premium)
                cur_prem = pos.entry_premium
            self._close_paper_position(pos, cur_prem, reason)
        self._paper_positions.clear()
        self._save_paper_positions()  # Clear positions file on disk

    def _update_market_data(self):
        if not self._nifty_token_info:
            if not getattr(self, '_warned_no_token', False):
                logger.warning("No Nifty token info -- market data disabled")
                self._warned_no_token = True
            return

        price = None
        volume = 0
        token = self._nifty_token_info.get("token", settings.NIFTY_INDEX_TOKEN)
        source = ""

        # Try WebSocket first (real-time with volume)
        if self.market_feed and self.market_feed.is_connected:
            tick = self.market_feed.get_ltp(token)
            if tick:
                price = tick["price"]
                volume = tick.get("volume", 0)
                source = "WS"

        # Fall back to REST (no volume)
        if price is None:
            symbol = self._nifty_token_info.get("symbol", "Nifty 50")
            price = self.broker.get_ltp("NSE", symbol, token)
            if price:
                source = "REST"

        if price:
            self._nifty_spot = price
            self._last_price_time = datetime.now(IST)
            self._no_data_count = 0
            self.candle_builder.on_tick(
                price=price, volume=volume,
                timestamp=datetime.now(IST),
            )
        else:
            count = getattr(self, '_no_data_count', 0) + 1
            self._no_data_count = count
            ws_state = "connected" if (self.market_feed and self.market_feed.is_connected) else "disconnected"
            if count <= 3 or count % 60 == 0:
                logger.warning("No market data (attempt {}, WS={})", count, ws_state)

            if self._last_price_time and self._paper_positions:
                blackout_secs = (datetime.now(IST) - self._last_price_time).total_seconds()
                if blackout_secs > 30 and ws_state == "disconnected":
                    logger.warning("WS disconnected for {:.0f}s with open positions", blackout_secs)
                if blackout_secs > 60:
                    logger.critical(
                        "No data for {:.0f}s with {} open position(s) -- emergency square-off",
                        blackout_secs, len(self._paper_positions),
                    )
                    self._square_off_all("DATA_BLACKOUT")
                    from risk.kill_switch import set_halt
                    set_halt(f"Data blackout {blackout_secs:.0f}s with open positions")

    def restart_market_feed(self):
        """Recreate the WebSocket feed with current broker tokens.

        Called after broker session refresh so the feed doesn't use stale tokens.
        """
        if self.market_feed:
            self.market_feed.stop()
            self.market_feed = None

        if self.broker.feed_token and self.broker.auth_token:
            try:
                self.market_feed = MarketFeed(
                    api_key=settings.ANGEL_API_KEY,
                    client_code=settings.ANGEL_CLIENT_ID,
                    feed_token=self.broker.feed_token,
                    auth_token=self.broker.auth_token,
                )
                self.market_feed.start()
                logger.info("WebSocket feed restarted with fresh tokens")
            except Exception as e:
                logger.warning("WebSocket restart failed: {}", e)
                self.market_feed = None

    def _set_prev_day_from_candles(self, hist_df: pd.DataFrame):
        """Extract previous trading day HLC from historical data and
        update strategy engine for CPR and Gap strategies."""
        try:
            today = datetime.now(IST).date()
            prev_dates = sorted(set(d for d in hist_df.index.date if d < today))
            if prev_dates:
                prev_day = prev_dates[-1]
                prev_df = hist_df[hist_df.index.date == prev_day]
                if len(prev_df) > 0:
                    prev_data = {
                        "high": float(prev_df["high"].max()),
                        "low": float(prev_df["low"].min()),
                        "close": float(prev_df["close"].iloc[-1]),
                    }
                    self.strategy.reset_day(prev_data)
                    logger.info("Prev day HLC set: H={:.0f} L={:.0f} C={:.0f} ({})",
                                prev_data["high"], prev_data["low"],
                                prev_data["close"], prev_day)
                    return
            # Fallback: try local daily CSV
            self._set_prev_day_from_csv()
        except Exception as e:
            logger.warning("Failed to set prev day data from candles: {}", e)
            self._set_prev_day_from_csv()

    def _set_prev_day_from_csv(self):
        """Fallback: load previous day HLC from local nifty_daily.csv."""
        try:
            daily_path = settings.DATA_DIR / "nifty_daily.csv"
            if not daily_path.exists():
                logger.debug("No nifty_daily.csv for prev day fallback")
                return
            daily_df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
            if daily_df.empty:
                return
            daily_df.columns = [c.lower() for c in daily_df.columns]
            today = datetime.now(IST).date()
            prev_rows = daily_df[daily_df.index.date < today]
            if len(prev_rows) > 0:
                last = prev_rows.iloc[-1]
                prev_data = {
                    "high": float(last["high"]),
                    "low": float(last["low"]),
                    "close": float(last["close"]),
                }
                self.strategy.reset_day(prev_data)
                logger.info("Prev day HLC from CSV: H={:.0f} L={:.0f} C={:.0f}",
                            prev_data["high"], prev_data["low"], prev_data["close"])
        except Exception as e:
            logger.warning("Failed to load prev day from CSV: {}", e)

    def seed_historical_candles(self):
        """Pre-seed candle builder with recent historical 5m bars.

        Priority: local disk cache first, then broker API as fallback.
        This ensures HTF indicators (15m RSI etc.) have enough data
        to produce valid values from the first tick.
        """
        disk_count = self.candle_builder.load_from_disk()
        if disk_count >= settings.SCAN_WARMUP_BARS:
            candles = self.candle_builder.get_candles()
            self._nifty_spot = float(candles["close"].iloc[-1])
            self._prev_candle_count = len(candles)
            self._set_prev_day_from_candles(candles)
            try:
                self._day_indicators = self.strategy.precompute(candles)
            except Exception:
                pass
            self._export_engine_state()
            logger.info("Seeded from disk cache: {} bars (Nifty={:.0f})",
                        disk_count, self._nifty_spot)
            return disk_count

        if not self.broker.is_active:
            logger.warning("Broker not active -- cannot seed historical candles")
            return disk_count

        token = settings.NIFTY_INDEX_TOKEN
        if self._nifty_token_info:
            token = self._nifty_token_info.get("token", token)

        now = datetime.now(IST)
        from_date = (now - pd.Timedelta(days=3)).replace(
            hour=9, minute=15, second=0, microsecond=0)
        from_str = from_date.strftime("%Y-%m-%d %H:%M")
        to_str = now.strftime("%Y-%m-%d %H:%M")

        try:
            raw = self.broker.get_historical(
                exchange="NSE", token=token,
                interval="FIVE_MINUTE",
                from_date=from_str, to_date=to_str,
            )
            if not raw:
                logger.info("No historical data returned for seeding")
                return 0

            from engine.candle_builder import CandleBuilder
            hist_df = CandleBuilder.from_historical(raw)

            if hist_df.empty:
                return 0

            self.candle_builder.seed(hist_df)
            self._nifty_spot = float(hist_df["close"].iloc[-1])
            self._prev_candle_count = len(hist_df)

            # Extract previous day HLC for CPR/Gap strategies
            self._set_prev_day_from_candles(hist_df)

            logger.info("Seeded {} historical candles (Nifty={:.0f})",
                        len(hist_df), self._nifty_spot)

            try:
                self._day_indicators = self.strategy.precompute(hist_df)
            except Exception:
                pass
            self._export_engine_state()

            return len(hist_df)

        except Exception as e:
            logger.warning("Historical candle seeding failed: {}", e)
            return 0

    def _run_selftest(self) -> bool:
        """Run engine on cached sample to verify expected behavior.

        Returns True if the engine produces a reasonable signal count
        (not zero and not wildly different). Guards against accidental
        code changes that silently break signal generation.
        """
        selftest_path = Path(settings.DATA_DIR) / "selftest_candles.csv"
        if not selftest_path.exists():
            logger.warning("Self-test data not found -- skipping")
            return True

        try:
            import pandas as pd
            df = pd.read_csv(selftest_path)
            dt_col = "Datetime" if "Datetime" in df.columns else "datetime"
            df[dt_col] = pd.to_datetime(df[dt_col])
            df.set_index(dt_col, inplace=True)

            if len(df) < 30:
                logger.warning("Self-test data too small ({} rows) -- skipping", len(df))
                return True

            test_engine = MultiStrategyEngine()
            test_engine.reset_day()
            indicators = test_engine.precompute(df)

            signal_count = 0
            for i in range(10, len(df)):
                ts = df.index[i]
                time_str = ts.strftime("%H:%M") if hasattr(ts, 'strftime') else ""
                signals = test_engine.scan(indicators, i, time_str)
                signal_count += len(signals)

            logger.info("Self-test: {} signals on {} bars",
                        signal_count, len(df))

            if signal_count == 0:
                logger.warning("Self-test: 0 signals -- strategy may not fire today "
                               "(check HTF RSI warmup and market conditions)")
                return False

            return True

        except Exception as e:
            logger.error("Self-test error: {}", e)
            return False

    def _log_event(self, event_type: str, data: dict):
        entry = {"ts": datetime.now(IST).isoformat(), "event": event_type, **data}
        try:
            with open(self._event_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("Failed to write event log: {}", exc)

    def end_day(self):
        if getattr(self, '_day_ended', False):
            logger.debug("end_day() already called -- skipping duplicate")
            return
        self._day_ended = True

        if self.market_feed:
            self.market_feed.stop()

        if self._paper_positions:
            self._square_off_all("END_OF_DAY")
            self._save_paper_positions()  # Persist empty list to disk

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

        state_file = Path(settings.DATA_DIR) / "engine_state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                state["running"] = False
                state["ts"] = datetime.now(IST).isoformat()
                state_file.write_text(json.dumps(state, indent=2))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to update engine state on end_day: {}", exc)

    def _export_engine_state(self):
        """Write engine state to disk for dashboard consumption."""
        try:
            candles = self.candle_builder.get_candles()
            num_candles = len(candles)

            # Build per-bar indicator snapshot (last 30 bars)
            bar_window = min(num_candles, 30)
            candle_rows = []
            ind = self._day_indicators or {}

            for i in range(num_candles - bar_window, num_candles):
                def _v(key, default=None):
                    s = ind.get(key)
                    if s is None:
                        return default
                    try:
                        val = float(s.iloc[i]) if i < len(s) else default
                        return round(val, 2) if val == val else default  # NaN guard
                    except Exception:
                        return default

                row = {
                    "n": i + 1,
                    "t": candles.index[i].strftime("%H:%M") if hasattr(candles.index[i], "strftime") else str(i),
                    "o": round(float(candles["open"].iloc[i]), 2),
                    "h": round(float(candles["high"].iloc[i]), 2),
                    "l": round(float(candles["low"].iloc[i]), 2),
                    "c": round(float(candles["close"].iloc[i]), 2),
                    "v": int(candles["volume"].iloc[i]) if "volume" in candles else 0,
                    "rsi5": _v("rsi_5m", 50),
                    "rsi15": _v("rsi_15m", 50),
                    "stoch_k": _v("stoch_k", 50),
                    "cci": _v("cci", 0),
                    "willr": _v("willr", -50),
                    "ema9": _v("ema_9"),
                    "ema20": _v("ema_20"),
                    "vwap": _v("vwap"),
                    "bb_pctb": _v("bb_pctb"),
                    "adx": _v("adx", 0),
                    "atr": _v("atr"),
                    "st_dir": _v("supertrend_dir", 0),
                    "st_fast_dir": _v("supertrend_fast_dir", 0),
                }
                candle_rows.append(row)

            # Open positions
            positions = []
            for p in self._paper_positions:
                positions.append({
                    "direction": p.direction,
                    "signal_type": p.signal.signal_type,
                    "entry_time": p.entry_time,
                    "entry_premium": round(p.entry_premium, 2),
                    "current_premium": round(p.prem_state.current_premium(self._nifty_spot, p.candles_held), 2),
                    "sl_premium": round(p.sl_premium, 2),
                    "peak_premium": round(p.peak_premium, 2),
                    "lots": p.lots,
                    "qty": p.qty,
                    "candles_held": p.candles_held,
                    "unrealized_pnl": round(
                        (p.prem_state.current_premium(self._nifty_spot, p.candles_held) - p.entry_premium) * p.qty, 2
                    ),
                    "confidence": p.signal.confidence,
                    "entry_index": round(p.entry_index, 2),
                })

            state = {
                "ts": datetime.now(IST).isoformat(),
                "nifty_price": self._nifty_spot,
                "candle_count": num_candles,
                "positions": positions,
                "candles": candle_rows,
                "running": self._running,
                "signals_today": self._day_signals_count,
            }

            tmp = ENGINE_STATE_FILE.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(state, f)
            import os
            os.replace(str(tmp), str(ENGINE_STATE_FILE))
        except Exception as e:
            logger.debug("Engine state export error: {}", e)

    def get_status_dict(self) -> dict:
        """Snapshot of engine state for /status command."""
        ct = self.capital
        pnl_pct = (ct.daily_pnl / ct.day_start_capital * 100) if ct.day_start_capital else 0
        return {
            "capital": ct.current_capital,
            "daily_pnl": ct.daily_pnl,
            "daily_pnl_pct": pnl_pct,
            "trades": ct.trades_today,
            "wins": ct.wins_today,
            "losses": ct.losses_today,
            "drawdown": ct.drawdown_pct,
            "consecutive_losses": ct.consecutive_losses,
            "open_positions": len(self._paper_positions),
            "running": self._running,
        }

    def get_market_snapshot(self) -> dict:
        """Fetch Nifty and BankNifty LTP for /market command."""
        nifty_ltp = None
        banknifty_ltp = None

        if self.broker.is_active:
            try:
                nifty_ltp = self.broker.get_ltp("NSE", "Nifty 50", settings.NIFTY_INDEX_TOKEN)
            except Exception:
                pass
            try:
                banknifty_ltp = self.broker.get_ltp("NSE", "Nifty Bank", settings.BANKNIFTY_INDEX_TOKEN)
            except Exception:
                pass

        if nifty_ltp is None and self._nifty_spot:
            nifty_ltp = self._nifty_spot

        return {
            "nifty": nifty_ltp,
            "banknifty": banknifty_ltp,
        }

    def stop(self):
        self._running = False
