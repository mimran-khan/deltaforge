"""Multi-Asset Trading Engine -- runs futures strategies on Gold, Crude, USDINR.

This is a SEPARATE engine from TradingEngine.  It does NOT modify or replace
the existing Nifty options flow.  It can run in the same process or as a
standalone script.

Architecture:
  - One CandleBuilder per instrument
  - One MultiStrategyEngine per instrument (with instrument-specific overrides)
  - Shared broker connection (Angel One supports MCX + CDS)
  - Shared PerformanceDB (trades tagged with instrument/exchange)
  - Independent capital pool per instrument

Activation: MULTI_ASSET_ENABLED=true in .env
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytz
from loguru import logger

from config import settings
from config.instruments import Instrument, get_futures_instruments

_alert_method = getattr(settings, 'ALERT_METHOD', 'slack')
if _alert_method == 'imessage':
    from alerts.imessage_bot import send_trade_alert, send_alert
elif _alert_method == 'slack':
    from alerts.slack_bot import send_trade_alert, send_alert
else:
    from alerts.telegram_bot import send_trade_alert, send_alert
from engine.broker import BrokerConnection
from engine.candle_builder import CandleBuilder
from engine.multi_strategy_engine import MultiStrategyEngine, TradeSignal
from engine.futures_model import FuturesPosition, create_futures_position
from persistence.performance_db import PerformanceDB
from risk.futures_capital_tracker import FuturesCapitalTracker
from risk.futures_risk_engine import FuturesRiskEngine
from risk.kill_switch import is_halted

IST = pytz.timezone("Asia/Kolkata")

MULTI_ASSET_STATE_FILE = settings.DATA_DIR / "multi_asset_state.json"


@dataclass
class FuturesPaperPosition:
    """Tracks a paper futures trade through its lifecycle."""
    instrument: Instrument
    direction: str
    entry_time: str
    entry_price: float
    lots: int
    qty: int
    signal: TradeSignal
    futures_pos: FuturesPosition
    candles_held: int = 0
    exit_price: float = 0
    exit_reason: str = ""
    exit_time: str = ""
    pnl: float = 0


class InstrumentState:
    """Per-instrument runtime state."""

    def __init__(self, instrument: Instrument):
        self.instrument = instrument
        self.candle_builder = CandleBuilder(interval_minutes=5)
        self.strategy = MultiStrategyEngine(
            enabled_strategies=instrument.strategy.enabled_strategies,
            disabled_strategies_override=instrument.strategy.disabled_strategies,
            max_adx_override=instrument.strategy.max_adx,
        )
        self.indicators: dict = {}
        self.positions: list[FuturesPaperPosition] = []
        self.spot_price: float = 0
        self.prev_candle_count: int = 0
        self.signals_today: int = 0
        self.trades_today: int = 0
        self.wins_today: int = 0
        self.losses_today: int = 0
        self.daily_pnl: float = 0
        self.seeded: bool = False
        self.consecutive_ltp_failures: int = 0
        self.token: str = instrument.index_token
        self.trading_symbol: str = ""
        self.last_price_time: Optional[datetime] = None

    def is_within_trading_hours(self, now: datetime) -> bool:
        t = now.strftime("%H:%M")
        return self.instrument.hours.market_open <= t < self.instrument.hours.market_close

    def is_within_entry_window(self, now: datetime) -> bool:
        t = now.strftime("%H:%M")
        return self.instrument.hours.entry_start <= t <= self.instrument.hours.entry_end

    def should_square_off(self, now: datetime) -> bool:
        t = now.strftime("%H:%M")
        return t >= self.instrument.hours.square_off


class MultiAssetEngine:
    """Runs futures strategies on MCX Gold, MCX Crude, NSE USDINR.

    Completely isolated from the existing TradingEngine / Nifty options flow.
    """

    def __init__(self, broker: Optional[BrokerConnection] = None):
        self.broker = broker or BrokerConnection()
        self.perf_db = PerformanceDB()
        self.capital = FuturesCapitalTracker()
        self.risk = FuturesRiskEngine(self.capital)
        self._instruments: dict[str, InstrumentState] = {}
        self._running = False

        for inst in get_futures_instruments():
            self._instruments[inst.name] = InstrumentState(inst)
            logger.info("MultiAsset: registered {} ({})", inst.display_name, inst.exchange)

    @property
    def instrument_names(self) -> list[str]:
        return list(self._instruments.keys())

    def start_day(self):
        now = datetime.now(IST)
        logger.info("=" * 60)
        logger.info("MultiAsset Engine -- DAY START: {} | Instruments: {}",
                     now.strftime("%Y-%m-%d"),
                     ", ".join(self._instruments.keys()))
        logger.info("=" * 60)

        self.capital.start_day()
        self.risk.reset_day()

        try:
            self._resolve_tokens()
        except Exception as e:
            logger.error("MultiAsset: token resolution crashed (non-fatal): {}", e)

        for state in self._instruments.values():
            state.candle_builder.reset()
            try:
                prev_day = self._fetch_prev_day_data(state)
            except Exception as e:
                logger.warning("MultiAsset: prev-day fetch crashed for {} (non-fatal): {}",
                               state.instrument.name, e)
                prev_day = None
            state.strategy.reset_day(prev_day_data=prev_day)
            state.positions = []
            state.prev_candle_count = 0
            state.signals_today = 0
            state.trades_today = 0
            state.wins_today = 0
            state.losses_today = 0
            state.daily_pnl = 0
            state.seeded = False
            state.consecutive_ltp_failures = 0

        try:
            self._seed_historical()
        except Exception as e:
            logger.error("MultiAsset: historical seeding crashed (non-fatal): {}", e)

    def _fetch_prev_day_data(self, state: InstrumentState) -> dict | None:
        """Fetch previous trading day OHLC for CPR/gap calculations."""
        if not self.broker.is_active:
            logger.warning("MultiAsset: broker not active for prev-day fetch")
            return None
        if not state.token:
            logger.warning("MultiAsset: no token for {} -- skipping prev-day", state.instrument.name)
            return None
        inst = state.instrument
        now = datetime.now(IST)
        prev_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        prev_day -= timedelta(days=1)
        if prev_day.weekday() == 6:
            prev_day -= timedelta(days=2)
        elif prev_day.weekday() == 5:
            prev_day -= timedelta(days=1)
        from_str = prev_day.strftime("%Y-%m-%d") + f" {inst.hours.market_open}"
        to_str = prev_day.strftime("%Y-%m-%d") + f" {inst.hours.market_close}"
        try:
            raw = self.broker.get_historical(
                exchange=inst.exchange,
                token=state.token,
                interval="FIVE_MINUTE",
                from_date=from_str,
                to_date=to_str,
            )
            if not raw:
                logger.warning("MultiAsset: no prev-day data for {}", inst.name)
                return None
            hist_df = CandleBuilder.from_historical(raw)
            if hist_df.empty:
                return None
            result = {
                "high": float(hist_df["high"].max()),
                "low": float(hist_df["low"].min()),
                "close": float(hist_df["close"].iloc[-1]),
            }
            atr_range = result["high"] - result["low"]
            result["atr"] = atr_range if atr_range > 0 else 100.0
            logger.info("MultiAsset: {} prev-day H={:.2f} L={:.2f} C={:.2f}",
                         inst.display_name, result["high"], result["low"], result["close"])
            time.sleep(2)
            return result
        except Exception as e:
            logger.warning("MultiAsset: prev-day fetch failed for {}: {}", inst.name, e)
            return None

    def _seed_historical(self):
        """Pre-seed candle builders with today's historical data from broker API.

        Fetches 5-minute candles from market open until now so indicators
        warm up immediately instead of waiting hours.
        """
        if not self.broker.is_active:
            logger.warning("MultiAsset: broker not active -- cannot seed historical")
            return

        now = datetime.now(IST)
        for name, state in self._instruments.items():
            if not state.token:
                continue
            inst = state.instrument
            from_dt = now.replace(
                hour=int(inst.hours.market_open.split(":")[0]),
                minute=int(inst.hours.market_open.split(":")[1]),
                second=0, microsecond=0,
            )
            from_str = from_dt.strftime("%Y-%m-%d %H:%M")
            to_str = now.strftime("%Y-%m-%d %H:%M")

            try:
                raw = self.broker.get_historical(
                    exchange=inst.exchange,
                    token=state.token,
                    interval="FIVE_MINUTE",
                    from_date=from_str,
                    to_date=to_str,
                )
                if not raw:
                    logger.warning("MultiAsset: no historical data for {}", name)
                    continue

                hist_df = CandleBuilder.from_historical(raw)
                state.candle_builder.seed(hist_df)
                state.prev_candle_count = len(state.candle_builder.get_candles())
                state.spot_price = float(hist_df["close"].iloc[-1])
                state.seeded = True
                logger.info("MultiAsset: seeded {} with {} bars (price={:.2f})",
                            inst.display_name, len(hist_df), state.spot_price)

                if state.prev_candle_count >= settings.SCAN_WARMUP_BARS:
                    completed = state.candle_builder.get_candles().iloc[:-1]
                    try:
                        state.indicators = state.strategy.precompute(completed)
                        logger.info("MultiAsset: {} indicators pre-warmed", name)
                    except Exception as e:
                        logger.debug("MultiAsset: {} precompute error: {}", name, e)

                    self._warmup_strategy_state(state, completed)
            except Exception as e:
                logger.warning("MultiAsset: historical seed failed for {}: {}", name, e)

            time.sleep(15)

    def _warmup_strategy_state(self, state: InstrumentState, completed):
        """Replay scan() on historical bars to classify regime and warm up strategy state.

        Without this, the MultiStrategyEngine's regime is never classified and
        internal counters stay at zero, causing the first live bar to miss
        signals that require an established regime.
        """
        inst = state.instrument
        warmup_start = max(20, settings.SCAN_WARMUP_BARS)
        scanned = 0
        for i in range(warmup_start, len(completed)):
            window = completed.iloc[:i + 1]
            try:
                inds = state.strategy.precompute(window)
            except Exception:
                continue
            ts = window.index[i].strftime("%H:%M")
            state.strategy.scan(
                inds, i, time_str=ts,
                entry_start_override=inst.hours.entry_start,
                entry_end_override=inst.hours.entry_end,
            )
            scanned += 1
        state.indicators = state.strategy.precompute(completed)
        logger.info("MultiAsset: {} strategy state warmed ({} bars replayed)", inst.display_name, scanned)

    def _resolve_tokens(self):
        """Look up Angel One tokens for MCX/CDS instruments from scrip master."""
        instruments_path = settings.DATA_DIR / "instruments.json"
        if not instruments_path.exists():
            logger.critical("MultiAsset: instruments.json NOT FOUND -- cannot resolve tokens")
            send_alert(":warning: *Multi-Asset*: instruments.json not found. "
                       "Token resolution failed. No futures trading possible.")
            return

        try:
            with open(instruments_path) as f:
                all_scrips = json.load(f)
        except Exception as e:
            logger.critical("MultiAsset: failed to load instruments.json: {}", e)
            send_alert(f":warning: *Multi-Asset*: instruments.json load failed: {e}")
            return

        def _parse_expiry(expiry_str: str) -> datetime:
            """Parse Angel One expiry format like '20JUL2026' to datetime."""
            try:
                return IST.localize(datetime.strptime(expiry_str, "%d%b%Y"))
            except (ValueError, TypeError):
                return IST.localize(datetime(2099, 12, 31))

        failed_instruments = []
        for name, state in self._instruments.items():
            if state.token:
                continue
            inst = state.instrument
            matches = [
                s for s in all_scrips
                if s.get("exch_seg") == inst.exchange
                and s.get("name", "").startswith(inst.symbol_prefix)
                and s.get("instrumenttype") in ("FUTCOM", "FUTCUR", "FUTSTK")
            ]
            if not matches:
                logger.critical("MultiAsset: NO scrip match for {} on {} -- instrument DISABLED",
                                inst.symbol_prefix, inst.exchange)
                failed_instruments.append(name)
                continue

            now = datetime.now(IST)
            future_matches = [
                s for s in matches
                if _parse_expiry(s.get("expiry", "")) >= now
            ]
            pool = future_matches if future_matches else matches
            pool.sort(key=lambda s: _parse_expiry(s.get("expiry", "")))
            nearest = pool[0]
            state.token = str(nearest.get("token", ""))
            state.trading_symbol = nearest.get("symbol", inst.symbol_prefix)
            logger.info("MultiAsset: {} -> token={} symbol={} expiry={}",
                         name, state.token,
                         state.trading_symbol,
                         nearest.get("expiry", ""))

        if failed_instruments:
            send_alert(f":warning: *Multi-Asset*: Token resolution FAILED for: "
                       f"{', '.join(failed_instruments)}. These instruments will not trade.")

    def run_loop(self, poll_interval: int = 5):
        """Main loop -- iterates over all instruments, respects per-instrument hours."""
        self._running = True
        logger.info("MultiAsset loop started (poll {}s)", poll_interval)

        while self._running:
            try:
                now = datetime.now(IST)

                if is_halted():
                    logger.warning("MultiAsset: kill switch active")
                    self._running = False
                    break

                any_active = False
                for name, state in self._instruments.items():
                    if not state.is_within_trading_hours(now):
                        if state.positions and state.should_square_off(now):
                            self._square_off_instrument(state, "EOD")
                        continue

                    any_active = True

                    if state.positions and state.should_square_off(now):
                        self._square_off_instrument(state, "EOD")
                    else:
                        self._tick_instrument(state)

                self._export_state()

                if not any_active:
                    all_closed = all(
                        now.strftime("%H:%M") >= s.instrument.hours.market_close
                        for s in self._instruments.values()
                    )
                    if all_closed:
                        logger.info("MultiAsset: all markets closed")
                        self._running = False
                        break

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("MultiAsset: user interrupted")
                self._running = False
            except Exception as e:
                logger.exception("MultiAsset loop error: {}", e)
                time.sleep(10)

        self.end_day()

    def _try_late_seed(self, state: InstrumentState):
        """Retry historical seeding for instruments that failed at startup."""
        inst = state.instrument
        now = datetime.now(IST)
        from_dt = now.replace(
            hour=int(inst.hours.market_open.split(":")[0]),
            minute=int(inst.hours.market_open.split(":")[1]),
            second=0, microsecond=0,
        )
        if now <= from_dt:
            return
        from_str = from_dt.strftime("%Y-%m-%d %H:%M")
        to_str = now.strftime("%Y-%m-%d %H:%M")
        try:
            raw = self.broker.get_historical(
                exchange=inst.exchange,
                token=state.token,
                interval="FIVE_MINUTE",
                from_date=from_str,
                to_date=to_str,
            )
            if raw:
                hist_df = CandleBuilder.from_historical(raw)
                state.candle_builder.seed(hist_df)
                state.prev_candle_count = len(state.candle_builder.get_candles())
                state.spot_price = float(hist_df["close"].iloc[-1])
                state.seeded = True
                logger.info("MultiAsset: late-seeded {} with {} bars (price={:.2f})",
                            inst.display_name, len(hist_df), state.spot_price)

                if state.prev_candle_count >= settings.SCAN_WARMUP_BARS:
                    completed = state.candle_builder.get_candles().iloc[:-1]
                    state.indicators = state.strategy.precompute(completed)
                    self._warmup_strategy_state(state, completed)
        except Exception as e:
            logger.warning("MultiAsset: late-seed failed for {}: {}", inst.name, e)

    def _tick_instrument(self, state: InstrumentState):
        """Single tick for one instrument."""
        if not self.broker.is_active or not state.token:
            return

        if not state.seeded:
            self._try_late_seed(state)

        symbol = state.trading_symbol or state.instrument.symbol_prefix
        raw_price = self.broker.get_ltp(
            state.instrument.exchange,
            symbol,
            state.token,
        )
        if not raw_price:
            state.consecutive_ltp_failures += 1
            if state.consecutive_ltp_failures == 6:
                logger.warning("MultiAsset: {} LTP unavailable for 6 consecutive ticks (~30s)",
                               state.instrument.display_name)
            elif state.consecutive_ltp_failures % 60 == 0:
                logger.warning("MultiAsset: {} LTP down for {} ticks (~{}m)",
                               state.instrument.display_name,
                               state.consecutive_ltp_failures,
                               state.consecutive_ltp_failures * 5 // 60)
            return

        state.consecutive_ltp_failures = 0
        price = raw_price / state.instrument.price_divisor
        state.spot_price = price
        state.last_price_time = datetime.now(IST)
        state.candle_builder.on_tick(
            price=price, volume=0, timestamp=datetime.now(IST),
        )

        candles = state.candle_builder.get_candles()
        current_count = len(candles)
        if current_count <= state.prev_candle_count:
            return
        state.prev_candle_count = current_count

        logger.info("MultiAsset candle #{} for {} | price={:.2f}",
                     current_count, state.instrument.display_name, price)

        completed = candles.iloc[:-1] if len(candles) > 1 else candles
        if len(completed) < settings.SCAN_WARMUP_BARS:
            logger.info("MultiAsset: {} warmup not ready ({} < {})",
                        state.instrument.display_name, len(completed), settings.SCAN_WARMUP_BARS)
            return

        try:
            state.indicators = state.strategy.precompute(completed)
        except Exception as e:
            logger.warning("MultiAsset PRECOMPUTE FAILED for {}: {}", state.instrument.name, e)
            return

        idx = len(completed) - 1
        time_str = completed.index[idx].strftime("%H:%M")

        self._update_positions(state, time_str)

        now = datetime.now(IST)
        if not state.is_within_entry_window(now):
            return
        if state.positions:
            return

        allowed, reason = self.risk.can_trade(state.instrument)
        if not allowed:
            if "Max trades" not in reason:
                logger.info("MultiAsset RISK BLOCK {}: {}", state.instrument.display_name, reason)
            return

        overrides = state.instrument.strategy
        max_trades = overrides.max_total_per_day or 8

        adx_val = None
        adx_series = state.indicators.get('adx')
        if adx_series is not None and idx < len(adx_series):
            adx_val = float(adx_series.iloc[idx])

        signals = state.strategy.scan(
            state.indicators, idx, time_str,
            max_total_override=max_trades,
            entry_start_override=state.instrument.hours.entry_start,
            entry_end_override=state.instrument.hours.entry_end,
        )

        if not signals:
            if idx % 6 == 0:
                rsi_series = state.indicators.get('rsi_5m')
                rsi_val = float(rsi_series.iloc[idx]) if rsi_series is not None and idx < len(rsi_series) else None
                logger.info("MultiAsset SCAN: {} bar {} | ADX={} RSI={} regime={}",
                             state.instrument.display_name, idx,
                             f"{adx_val:.1f}" if adx_val else "?",
                             f"{rsi_val:.1f}" if rsi_val else "?",
                             state.strategy.regime.regime)
            return

        for signal in signals:
            min_conf = overrides.min_confidence or 68
            if signal.confidence < min_conf:
                continue

            if self._enter_futures_trade(state, signal):
                state.signals_today += 1
            break

        self._export_state()

    def _enter_futures_trade(self, state: InstrumentState, signal: TradeSignal) -> bool:
        """Open a paper futures position. Returns True if entry succeeded."""
        inst = state.instrument
        overrides = inst.strategy

        sl_pct = overrides.sl_pct or 2.0
        target_mult = overrides.target_mult or 1.5
        lots = self.capital.get_lots(inst, state.spot_price)
        if lots <= 0:
            logger.warning("MultiAsset: insufficient capital for {} lot", inst.display_name)
            return False

        futures_pos = create_futures_position(
            instrument=inst,
            direction=signal.direction,
            entry_price=state.spot_price,
            lots=lots,
            sl_pct=sl_pct,
            target_mult=target_mult,
        )

        pos = FuturesPaperPosition(
            instrument=inst,
            direction=signal.direction,
            entry_time=datetime.now(IST).isoformat(),
            entry_price=state.spot_price,
            lots=lots,
            qty=inst.lot_size * lots,
            signal=signal,
            futures_pos=futures_pos,
        )
        state.positions.append(pos)

        logger.info("MultiAsset ENTRY: {} {} {} @ {:.2f} | SL={:.2f} TGT={:.2f} | conf={}",
                     inst.display_name, signal.signal_type, signal.direction,
                     state.spot_price, futures_pos.sl_price, futures_pos.target_price,
                     signal.confidence)

        send_trade_alert(
            action="ENTRY",
            strategy=signal.signal_type,
            symbol=f"{inst.display_name} ({inst.exchange})",
            price=state.spot_price,
            quantity=inst.lot_size * lots,
            sl=futures_pos.sl_price,
            target=futures_pos.target_price,
            confidence=signal.confidence,
        )
        return True

    def _update_positions(self, state: InstrumentState, time_str: str):
        """Check exits for all open positions of an instrument."""
        inst = state.instrument
        overrides = inst.strategy
        closed = []

        for pos in state.positions:
            pos.candles_held += 1
            fp = pos.futures_pos

            trigger_pct = overrides.trail_trigger_pct or 3.0
            trail_pct = overrides.trail_pct or 1.5
            trail_level = fp.update_trail(state.spot_price, trigger_pct, trail_pct)

            exit_reason = fp.check_exit(state.spot_price, trail_level)

            max_hold = overrides.max_hold_candles or 36
            if not exit_reason and pos.candles_held >= max_hold:
                exit_reason = "TIME"

            if exit_reason:
                self._close_position(state, pos, state.spot_price, exit_reason)
                closed.append(pos)

        for pos in closed:
            state.positions.remove(pos)

    def _close_position(self, state: InstrumentState,
                        pos: FuturesPaperPosition,
                        exit_price: float, reason: str):
        inst = state.instrument
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.exit_time = datetime.now(IST).isoformat()

        raw_pnl = pos.futures_pos.current_pnl(exit_price)
        costs = inst.costs.total_costs(
            pos.entry_price, exit_price, pos.qty, pos.lots,
            price_divisor=1.0,
        )
        pos.pnl = raw_pnl - costs
        state.daily_pnl += pos.pnl
        state.trades_today += 1
        if pos.pnl >= 0:
            state.wins_today += 1
        else:
            state.losses_today += 1

        self.capital.record_trade(inst.name, pos.pnl)
        pool = self.capital.pools.get(inst.name)
        capital_after = pool.current if pool else 0

        if reason in ("SL", "TRAIL"):
            state.strategy.record_sl_exit(pos.signal.signal_type, state.prev_candle_count)

        now_ist = datetime.now(IST)
        self.perf_db.record_trade(
            date=now_ist.strftime("%Y-%m-%d"),
            time=now_ist.strftime("%H:%M"),
            strategy=f"{pos.signal.signal_type}_{pos.signal.pullback_count}",
            direction=pos.direction,
            confidence=pos.signal.confidence,
            htf_rsi=pos.signal.htf_rsi,
            adx=getattr(pos.signal, 'adx', 0),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl=pos.pnl,
            hold_bars=pos.candles_held,
            exit_reason=reason,
            lots=pos.lots,
            capital_after=capital_after,
            instrument=inst.name,
            exchange=inst.exchange,
        )

        logger.info("MultiAsset EXIT ({}): {} {} {} | PnL={:.2f} | Held {} bars",
                     reason, inst.display_name, pos.signal.signal_type,
                     pos.direction, pos.pnl, pos.candles_held)

        send_trade_alert(
            action=f"EXIT_{reason}",
            strategy=pos.signal.signal_type,
            symbol=f"{inst.display_name} ({inst.exchange})",
            price=exit_price,
            quantity=pos.qty,
            pnl=pos.pnl,
            daily_pnl=state.daily_pnl,
            win_count=state.wins_today,
            loss_count=state.losses_today,
        )

    def _square_off_instrument(self, state: InstrumentState, reason: str):
        for pos in list(state.positions):
            self._close_position(state, pos, state.spot_price, reason)
        state.positions.clear()

    @staticmethod
    def _build_candle_rows(state: InstrumentState, window: int = 120) -> list[dict]:
        """Serialize recent OHLCV + indicator bars for dashboard charts."""
        candles = state.candle_builder.get_candles()
        if candles is None or candles.empty:
            return []
        ind = state.indicators or {}
        n = len(candles)
        start = max(0, n - window)
        rows = []
        for i in range(start, n):
            def _v(key, default=None):
                s = ind.get(key)
                if s is None:
                    return default
                try:
                    val = float(s.iloc[i]) if i < len(s) else default
                    return round(val, 2) if val == val else default
                except Exception:
                    return default

            idx = candles.index[i]
            t = idx.strftime("%H:%M") if hasattr(idx, "strftime") else str(i)
            rows.append({
                "n": i + 1, "t": t,
                "o": round(float(candles["open"].iloc[i]), 2),
                "h": round(float(candles["high"].iloc[i]), 2),
                "l": round(float(candles["low"].iloc[i]), 2),
                "c": round(float(candles["close"].iloc[i]), 2),
                "v": int(candles["volume"].iloc[i]) if "volume" in candles else 0,
                "rsi5": _v("rsi_5m", 50),
                "ema9": _v("ema_9"),
                "ema20": _v("ema_20"),
                "vwap": _v("vwap"),
                "adx": _v("adx", 0),
            })
        return rows

    def _export_state(self):
        """Write multi-asset engine state to separate JSON file."""
        try:
            instruments_data = {}
            for name, state in self._instruments.items():
                candles = state.candle_builder.get_candles()
                positions = []
                for p in state.positions:
                    positions.append({
                        "direction": p.direction,
                        "signal_type": p.signal.signal_type,
                        "entry_time": p.entry_time,
                        "entry_price": round(p.entry_price, 4),
                        "current_pnl": round(p.futures_pos.current_pnl(state.spot_price), 2),
                        "sl_price": round(p.futures_pos.sl_price, 4),
                        "target_price": round(p.futures_pos.target_price, 4),
                        "lots": p.lots,
                        "candles_held": p.candles_held,
                        "confidence": p.signal.confidence,
                    })
                candle_rows = self._build_candle_rows(state, window=120)
                instruments_data[name] = {
                    "display_name": state.instrument.display_name,
                    "exchange": state.instrument.exchange,
                    "price": state.spot_price,
                    "candle_count": len(candles),
                    "positions": positions,
                    "signals_today": state.signals_today,
                    "trades_today": state.trades_today,
                    "daily_pnl": round(state.daily_pnl, 2),
                    "candles": candle_rows,
                }

            output = {
                "ts": datetime.now(IST).isoformat(),
                "running": self._running,
                "instruments": instruments_data,
            }

            tmp = MULTI_ASSET_STATE_FILE.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(output, f)
            os.replace(str(tmp), str(MULTI_ASSET_STATE_FILE))
        except Exception as e:
            logger.debug("MultiAsset state export error: {}", e)

    def end_day(self):
        for name, state in self._instruments.items():
            if state.positions:
                self._square_off_instrument(state, "END_OF_DAY")

        total_pnl = sum(s.daily_pnl for s in self._instruments.values())
        total_trades = sum(s.trades_today for s in self._instruments.values())

        logger.info("=" * 60)
        logger.info("MultiAsset DAY END | Total PnL: Rs {:.0f} | Trades: {}",
                     total_pnl, total_trades)
        lines = []
        for name, state in self._instruments.items():
            if state.trades_today > 0 or state.daily_pnl != 0:
                logger.info("  {} | PnL: Rs {:.0f} | Trades: {}",
                             state.instrument.display_name,
                             state.daily_pnl, state.trades_today)
                sign = "+" if state.daily_pnl >= 0 else ""
                lines.append(f"  {state.instrument.display_name}: Rs {sign}{state.daily_pnl:.0f} ({state.trades_today} trades)")
        logger.info("=" * 60)

        sign = "+" if total_pnl >= 0 else ""
        icon = ":chart_with_upwards_trend:" if total_pnl >= 0 else ":chart_with_downwards_trend:"
        eod_msg = (
            f"{icon} *Multi-Asset EOD*\n"
            f"{'═' * 24}\n"
            f"Total P&L: *Rs {sign}{total_pnl:,.0f}* | Trades: {total_trades}\n"
        )
        if lines:
            eod_msg += "\n".join(lines)
        send_alert(eod_msg)

        self._export_state()

    def stop(self):
        self._running = False
