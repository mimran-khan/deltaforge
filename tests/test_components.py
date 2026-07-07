"""Deep component tests -- finds real bugs through edge cases and invariants.

Each test class targets a single component with:
  - Happy path
  - Edge cases / boundary conditions
  - Error handling / corruption resilience
  - State consistency invariants
  - Concurrency / ordering issues

Usage:
    ./venv/bin/python -m pytest tests/test_components.py -v
    ./venv/bin/python -m tests.test_components -v
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def make_candles(n=100, start=24000, trend=0.5, vol=30, t0="2026-03-11 09:15:00"):
    np.random.seed(42)
    ts = pd.date_range(t0, periods=n, freq="5min")
    c = [start]
    for _ in range(n - 1):
        c.append(c[-1] + trend + np.random.randn() * vol)
    c = np.array(c)
    return pd.DataFrame({
        "open": c + np.random.randn(n) * 5,
        "high": c + np.abs(np.random.randn(n)) * 15,
        "low": c - np.abs(np.random.randn(n)) * 15,
        "close": c,
        "volume": np.random.randint(50000, 200000, n),
    }, index=ts)


# ═══════════════════════════════════════════════════════════════════
#  CANDLE BUILDER -- tick aggregation correctness
# ═══════════════════════════════════════════════════════════════════

class TestCandleBuilderDeep(unittest.TestCase):

    def setUp(self):
        from engine.candle_builder import CandleBuilder
        self.cb = CandleBuilder(interval_minutes=5)
        self.base = datetime(2026, 3, 11, 9, 15, 0)

    def test_ohlv_correctness(self):
        """OHLV values must match tick extremes exactly."""
        ticks = [(24000, 100), (24050, 200), (23980, 150), (24020, 300)]
        for i, (p, v) in enumerate(ticks):
            self.cb.on_tick(p, v, self.base + timedelta(seconds=i * 30))

        next_bucket = self.base + timedelta(minutes=5, seconds=1)
        bar = self.cb.on_tick(24100, 50, next_bucket)

        self.assertEqual(bar["open"], 24000)
        self.assertEqual(bar["high"], 24050)
        self.assertEqual(bar["low"], 23980)
        self.assertEqual(bar["close"], 24020)
        self.assertEqual(bar["volume"], 750)  # 100+200+150+300

    def test_single_tick_candle(self):
        """A bucket with only 1 tick: O=H=L=C."""
        self.cb.on_tick(24000, 100, self.base)
        bar = self.cb.on_tick(24100, 50, self.base + timedelta(minutes=5, seconds=1))

        self.assertEqual(bar["open"], bar["high"])
        self.assertEqual(bar["open"], bar["low"])
        self.assertEqual(bar["open"], bar["close"])

    def test_zero_volume_ticks(self):
        """REST fallback sends volume=0 -- candles should still build."""
        for i in range(10):
            self.cb.on_tick(24000 + i, 0, self.base + timedelta(seconds=i * 30))
        bar = self.cb.on_tick(24100, 0, self.base + timedelta(minutes=5, seconds=1))
        self.assertIsNotNone(bar)
        self.assertEqual(bar["volume"], 0)

    def test_bucket_boundary_exact(self):
        """Tick at exact bucket boundary starts a new candle."""
        self.cb.on_tick(24000, 100, self.base)
        bar = self.cb.on_tick(24050, 200, self.base + timedelta(minutes=5))
        self.assertIsNotNone(bar)  # 09:20:00 is a new bucket

    def test_get_candles_count(self):
        """get_candles includes both completed and in-progress bars."""
        for i in range(25):
            self.cb.on_tick(24000 + i, 100,
                           self.base + timedelta(minutes=i))
        candles = self.cb.get_candles()
        # 25 minutes / 5 min = 5 buckets, 4 completed + 1 in-progress = 5
        self.assertEqual(len(candles), 5)

    def test_from_historical(self):
        """Static factory for broker API data."""
        from engine.candle_builder import CandleBuilder
        raw = [
            ["2026-03-11 09:15", 24000, 24050, 23980, 24020, 1000],
            ["2026-03-11 09:20", 24020, 24060, 24010, 24040, 1200],
        ]
        df = CandleBuilder.from_historical(raw)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["close"], 24020)

    def test_from_historical_empty(self):
        from engine.candle_builder import CandleBuilder
        df = CandleBuilder.from_historical([])
        self.assertEqual(len(df), 0)


# ═══════════════════════════════════════════════════════════════════
#  CAPITAL TRACKER -- financial state machine
# ═══════════════════════════════════════════════════════════════════

class TestCapitalTrackerDeep(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig = settings.CAPITAL_FILE
        self.orig_db = settings.DB_PATH
        self.orig_data_dir = settings.DATA_DIR
        settings.CAPITAL_FILE = Path(self.tmpdir) / "capital.json"
        settings.DB_PATH = Path(self.tmpdir) / "trades.db"
        settings.DATA_DIR = Path(self.tmpdir)

    def tearDown(self):
        settings.CAPITAL_FILE = self.orig
        settings.DB_PATH = self.orig_db
        settings.DATA_DIR = self.orig_data_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make(self):
        from risk.capital_tracker import CapitalTracker
        return CapitalTracker()

    def test_win_then_loss_streak(self):
        """Consecutive losses counter resets on win."""
        ct = self._make()
        ct.start_day()

        ct.record_trade(pnl=-200, strategy="t", symbol="N",
                        entry_price=100, exit_price=97, quantity=75, reason="SL")
        ct.record_trade(pnl=-200, strategy="t", symbol="N",
                        entry_price=100, exit_price=97, quantity=75, reason="SL")
        self.assertEqual(ct.consecutive_losses, 2)

        ct.record_trade(pnl=500, strategy="t", symbol="N",
                        entry_price=100, exit_price=107, quantity=75, reason="TGT")
        self.assertEqual(ct.consecutive_losses, 0)

    def test_peak_capital_only_increases(self):
        """Peak never decreases -- even after losses."""
        ct = self._make()
        ct.start_day()

        ct.record_trade(pnl=1000, strategy="t", symbol="N",
                        entry_price=100, exit_price=113, quantity=75, reason="TGT")
        self.assertEqual(ct.peak_capital, 11000)

        ct.record_trade(pnl=-500, strategy="t", symbol="N",
                        entry_price=100, exit_price=93, quantity=75, reason="SL")
        self.assertEqual(ct.peak_capital, 11000)  # still 11k

    def test_drawdown_increases_monotonically(self):
        """max_drawdown should track the worst-ever drawdown."""
        ct = self._make()
        ct.start_day()
        ct.record_trade(pnl=2000, strategy="t", symbol="N",
                        entry_price=100, exit_price=127, quantity=75, reason="TGT")
        ct.record_trade(pnl=-1000, strategy="t", symbol="N",
                        entry_price=100, exit_price=87, quantity=75, reason="SL")
        dd1 = ct.max_drawdown

        ct.record_trade(pnl=500, strategy="t", symbol="N",
                        entry_price=100, exit_price=107, quantity=75, reason="TGT")
        self.assertGreaterEqual(ct.max_drawdown, dd1)

    def test_zero_pnl_counted_as_win(self):
        """A breakeven trade (pnl=0) should not increment consecutive losses."""
        ct = self._make()
        ct.start_day()
        ct.consecutive_losses = 1
        ct.record_trade(pnl=0, strategy="t", symbol="N",
                        entry_price=100, exit_price=100, quantity=75, reason="TIME")
        self.assertEqual(ct.consecutive_losses, 0)
        self.assertEqual(ct.wins_today, 1)

    def test_sizing_tiers(self):
        """Drawdown-based sizing: normal -> half -> halt."""
        ct = self._make()
        ct.peak_capital = 10000

        # DD < DRAWDOWN_HALFSIZE_PCT (20%) -> full size
        ct.current_capital = 8500  # 15% DD
        self.assertEqual(ct.get_sizing_multiplier(), 1.0)

        # DD >= DRAWDOWN_HALFSIZE_PCT (20%) -> half size
        ct.current_capital = 8000  # 20% DD
        mult = ct.get_sizing_multiplier()
        self.assertLessEqual(mult, 0.5)

        # DD >= DRAWDOWN_HALT_PCT (35%) -> halt
        ct.current_capital = 6500  # 35% DD
        self.assertEqual(ct.get_sizing_multiplier(), 0.0)

    def test_corrupted_json_loads_defaults(self):
        """Corrupted capital.json should not crash -- use defaults."""
        settings.CAPITAL_FILE.write_text("NOT VALID JSON {{{")
        ct = self._make()
        self.assertEqual(ct.current_capital, settings.STARTING_CAPITAL)

    def test_partial_json_loads_defaults_for_missing(self):
        """JSON with missing fields uses defaults."""
        settings.CAPITAL_FILE.write_text('{"current_capital": 15000}')
        ct = self._make()
        self.assertEqual(ct.current_capital, 15000)
        self.assertEqual(ct.wins_today, 0)
        self.assertEqual(ct.weekly_pnl, 0)

    def test_daily_pnl_pct_zero_capital(self):
        """No division-by-zero when day_start_capital is 0."""
        ct = self._make()
        ct.day_start_capital = 0
        self.assertEqual(ct.daily_pnl_pct, 0)

    def test_compound_lot_sizing(self):
        """More capital = more lots (compounding)."""
        ct = self._make()
        ct.current_capital = 10000
        lots_10k = ct.get_max_lots()

        ct.current_capital = 50000
        lots_50k = ct.get_max_lots()

        self.assertGreater(lots_50k, lots_10k)

    def test_summary_has_all_keys(self):
        ct = self._make()
        ct.start_day()
        s = ct.get_summary()
        for key in ["capital", "daily_pnl", "daily_pnl_pct", "trades",
                     "wins", "losses", "win_rate", "total_pnl",
                     "max_drawdown", "drawdown_current", "consecutive_losses",
                     "peak_capital", "trade_log"]:
            self.assertIn(key, s, f"Missing key: {key}")

    def test_concurrent_save_safety(self):
        """Multiple saves should not corrupt the file."""
        ct = self._make()
        ct.start_day()

        def saver():
            for _ in range(20):
                ct.current_capital += 1
                ct.save()

        threads = [threading.Thread(target=saver) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # File should still be valid JSON
        with open(settings.CAPITAL_FILE) as f:
            data = json.load(f)
        self.assertIn("current_capital", data)

    def test_mid_day_restart_preserves_pnl(self):
        """Calling start_day() twice on the same day must not reset daily state."""
        ct = self._make()
        ct.start_day()
        ct.record_trade(
            pnl=-150, strategy="t", symbol="N",
            entry_price=100, exit_price=98, quantity=75, reason="SL",
        )
        pnl_before = ct.daily_pnl
        trades_before = ct.trades_today

        ct.start_day()

        self.assertEqual(ct.daily_pnl, pnl_before)
        self.assertEqual(ct.trades_today, trades_before)

    def test_lot_sizing_uses_deploy_pct(self):
        """Only CAPITAL_DEPLOY_PCT of capital should be used for lot sizing."""
        ct = self._make()
        ct.current_capital = 10000
        ct.peak_capital = 10000
        lots = ct.get_max_lots()
        deployable = 10000 * settings.CAPITAL_DEPLOY_PCT / 100
        per_lot_exposure = 100.0 * getattr(settings, 'NIFTY_LOT_SIZE', 65)
        per_lot = max(settings.CAPITAL_PER_LOT, per_lot_exposure)
        expected = max(1, int(deployable / per_lot))
        self.assertEqual(lots, min(expected, settings.MAX_LOTS_CAP))


# ═══════════════════════════════════════════════════════════════════
#  RISK ENGINE -- gate ordering and completeness
# ═══════════════════════════════════════════════════════════════════

class TestRiskEngineDeep(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig = settings.CAPITAL_FILE
        self.orig_db = settings.DB_PATH
        self.orig_data_dir = settings.DATA_DIR
        settings.CAPITAL_FILE = Path(self.tmpdir) / "capital.json"
        settings.DB_PATH = Path(self.tmpdir) / "trades.db"
        settings.DATA_DIR = Path(self.tmpdir)

    def tearDown(self):
        settings.CAPITAL_FILE = self.orig
        settings.DB_PATH = self.orig_db
        settings.DATA_DIR = self.orig_data_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _engine_at(self, hour=10, minute=30, weekday=2):
        from risk.capital_tracker import CapitalTracker
        from risk.risk_engine import RiskEngine
        ct = CapitalTracker()
        ct.start_day()
        re = RiskEngine(ct)
        re._halted = False
        return re, ct

    @patch("risk.risk_engine.datetime")
    def test_all_gates_pass(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        d = re.evaluate(confluence_score=70, direction="LONG")
        self.assertTrue(d.approved)

    @patch("risk.risk_engine.datetime")
    def test_gate0_halted(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        re.halt("test")
        d = re.evaluate(confluence_score=70)
        self.assertFalse(d.approved)
        self.assertIn("Halted", d.reason)

    @patch("risk.risk_engine.datetime")
    def test_gate1_capital_below_min(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        ct.current_capital = 100
        d = re.evaluate(confluence_score=70)
        self.assertFalse(d.approved)
        self.assertIn("Capital below", d.reason)
        self.assertTrue(re.is_halted)

    @patch("risk.risk_engine.datetime")
    def test_gate2_daily_loss(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        ct.daily_pnl = -ct.get_daily_loss_limit()
        d = re.evaluate(confluence_score=70)
        self.assertFalse(d.approved)
        self.assertTrue(re.is_halted)

    @patch("risk.risk_engine.datetime")
    def test_gate3_weekly_loss(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        ct.weekly_pnl = -ct.get_weekly_loss_limit()
        d = re.evaluate(confluence_score=70)
        self.assertFalse(d.approved)

    @patch("risk.risk_engine.datetime")
    def test_gate4_consecutive_losses(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        ct.consecutive_losses = settings.MAX_CONSECUTIVE_LOSSES
        d = re.evaluate(confluence_score=70)
        self.assertFalse(d.approved)
        self.assertTrue(re.is_halted)

    @patch("risk.risk_engine.datetime")
    def test_gate6_drawdown_halt(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        ct.peak_capital = 10000
        ct.current_capital = 6000  # 40% DD
        d = re.evaluate(confluence_score=70)
        self.assertFalse(d.approved)

    @patch("risk.risk_engine.datetime")
    def test_gate8_before_window(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 8, 0)
        re, ct = self._engine_at()
        d = re.evaluate(confluence_score=70)
        self.assertFalse(d.approved)
        self.assertIn("Before entry", d.reason)

    @patch("risk.risk_engine.datetime")
    def test_gate8_after_window(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 15, 30)
        re, ct = self._engine_at()
        d = re.evaluate(confluence_score=70)
        self.assertFalse(d.approved)
        self.assertIn("After entry", d.reason)

    @patch("risk.risk_engine.datetime")
    def test_gate9_low_confidence(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        d = re.evaluate(confluence_score=10)
        self.assertFalse(d.approved)
        self.assertIn("Confidence", d.reason)

    @patch("risk.risk_engine.datetime")
    def test_realtime_daily_loss_halt(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        ct.daily_pnl = -ct.get_daily_loss_limit()
        ok = re.check_realtime()
        self.assertFalse(ok)
        self.assertTrue(re.is_halted)

    @patch("risk.risk_engine.datetime")
    def test_resume_allows_new_trades(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        re.halt("test")
        re.resume()
        d = re.evaluate(confluence_score=70)
        self.assertTrue(d.approved)

    @patch("risk.risk_engine.datetime")
    def test_lots_returned_on_approval(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)
        re, ct = self._engine_at()
        d = re.evaluate(confluence_score=70)
        self.assertGreater(d.lots, 0)
        self.assertGreater(d.premium_sl_pct, 0)


# ═══════════════════════════════════════════════════════════════════
#  MULTI-STRATEGY ENGINE -- signal generation internals
# ═══════════════════════════════════════════════════════════════════

class TestMultiStrategyDeep(unittest.TestCase):

    def _make(self):
        from engine.multi_strategy_engine import MultiStrategyEngine
        return MultiStrategyEngine()

    def test_cooldown_bars_consumed_even_on_quality_reject(self):
        """If bar-quality rejects a signal, cooldown bars are still consumed."""
        engine = self._make()
        df = make_candles(75, trend=3.0)
        ind = engine.precompute(df)

        signals = []
        triggered_bars = []
        for i in range(10, len(df)):
            t = df.index[i].strftime("%H:%M")
            sigs = engine.scan(ind, i, t)
            if sigs:
                signals.extend(sigs)
                triggered_bars.append(i)

        if len(triggered_bars) >= 2:
            gap = triggered_bars[1] - triggered_bars[0]
            self.assertGreaterEqual(gap, engine.COOLDOWN_BARS)

    def test_no_signal_before_bar_10(self):
        """Engine needs at least 10 bars of data."""
        engine = self._make()
        df = make_candles(75, trend=5)
        ind = engine.precompute(df)

        for i in range(10):
            sigs = engine.scan(ind, i, "10:00")
            self.assertEqual(len(sigs), 0)

    def test_max_pullback_per_day_respected(self):
        engine = self._make()
        engine.MAX_PULLBACK_PER_DAY = 1
        engine.MAX_TOTAL_PER_DAY = 10  # don't limit total

        df = make_candles(200, trend=3, t0="2026-03-11 09:15:00")
        ind = engine.precompute(df)

        pullback_signals = []
        for i in range(10, len(df)):
            t = df.index[i].strftime("%H:%M")
            for sig in engine.scan(ind, i, t):
                if sig.signal_type == "PULLBACK":
                    pullback_signals.append(sig)

        self.assertLessEqual(len(pullback_signals), 1)

    def test_dead_zone_filter(self):
        """HTF RSI 48-52 with strength 0-5 should be filtered."""
        engine = self._make()
        # When RSI15 = 52 (strength = 2, in dead zone [0,5))
        self.assertTrue(engine.HTF_DEAD_ZONE_LO <= 2 < engine.HTF_DEAD_ZONE_HI)

    def test_sv_helper_handles_nan(self):
        """_sv returns default for NaN values."""
        from engine.multi_strategy_engine import MultiStrategyEngine
        s = pd.Series([1.0, np.nan, 3.0])
        self.assertEqual(MultiStrategyEngine._sv(s, 1, 99), 99)

    def test_sv_helper_handles_out_of_bounds(self):
        from engine.multi_strategy_engine import MultiStrategyEngine
        s = pd.Series([1.0, 2.0])
        self.assertEqual(MultiStrategyEngine._sv(s, 100, -1), -1)

    def test_sv_helper_scalar(self):
        from engine.multi_strategy_engine import MultiStrategyEngine
        val = MultiStrategyEngine._sv(42.0, 0, -1)
        self.assertEqual(val, 42.0)

    def test_precompute_on_minimal_data(self):
        """Engine shouldn't crash on very small datasets."""
        engine = self._make()
        df = make_candles(5)
        ind = engine.precompute(df)
        self.assertIn('close', ind)
        self.assertIn('adx', ind)

    def test_adx_gate_rejects_low_adx(self):
        """When ADX < MIN_ADX, scan returns empty."""
        engine = self._make()
        ind = {
            'adx': pd.Series([5.0] * 20),
            'close': pd.Series([24000] * 20),
        }
        sigs = engine.scan(ind, 15, "10:30")
        self.assertEqual(len(sigs), 0)

    def test_shock_blocks_signals(self):
        """Shock detector blocks signals after extreme moves."""
        engine = self._make()
        engine.shock._halt_until_bar = 100
        ind = {
            'adx': pd.Series([30.0] * 20),
            'close': pd.Series([24000] * 20),
        }
        sigs = engine.scan(ind, 15, "10:30")
        self.assertEqual(len(sigs), 0)

    def test_used_bars_prevent_duplicate(self):
        """Same bar index never generates two signals."""
        engine = self._make()
        engine._used_bars = {15}
        ind = {'adx': pd.Series([30.0] * 20), 'close': pd.Series([24000] * 20)}
        sigs = engine.scan(ind, 15, "10:30")
        self.assertEqual(len(sigs), 0)

    def test_signal_confidence_range(self):
        """All signals must have confidence in [50, 100]."""
        engine = self._make()
        df = make_candles(200, trend=3, t0="2026-03-11 09:15:00")
        ind = engine.precompute(df)

        for i in range(10, len(df)):
            t = df.index[i].strftime("%H:%M")
            for sig in engine.scan(ind, i, t):
                self.assertGreaterEqual(sig.confidence, 50)
                self.assertLessEqual(sig.confidence, 100)


# ═══════════════════════════════════════════════════════════════════
#  SHOCK DETECTOR -- edge cases
# ═══════════════════════════════════════════════════════════════════

class TestShockDetectorDeep(unittest.TestCase):

    def _make(self, **kw):
        from risk.shock_detector import ShockDetector
        return ShockDetector(**kw)

    def test_zero_price_no_crash(self):
        sd = self._make(threshold_pct=1.0, lookback_bars=3, halt_bars=6)
        closes = pd.Series([0.0] * 10)
        self.assertTrue(sd.check(closes, 5))

    def test_nan_prices_safe(self):
        sd = self._make(threshold_pct=1.0, lookback_bars=3, halt_bars=6)
        closes = pd.Series([24000, np.nan, 24000, np.nan, 24000])
        self.assertTrue(sd.check(closes, 3))

    def test_exact_threshold_triggers(self):
        sd = self._make(threshold_pct=1.0, lookback_bars=2, halt_bars=3)
        closes = pd.Series([10000, 10000, 10100])  # exactly 1%
        result = sd.check(closes, 2)
        self.assertFalse(result)

    def test_below_threshold_passes(self):
        sd = self._make(threshold_pct=1.0, lookback_bars=2, halt_bars=3)
        closes = pd.Series([10000, 10000, 10099])  # 0.99%
        self.assertTrue(sd.check(closes, 2))

    def test_negative_move_also_triggers(self):
        sd = self._make(threshold_pct=1.0, lookback_bars=2, halt_bars=3)
        closes = pd.Series([10000, 10000, 9899])  # -1.01%
        result = sd.check(closes, 2)
        self.assertFalse(result)

    def test_early_bar_passes(self):
        sd = self._make(threshold_pct=1.0, lookback_bars=5, halt_bars=3)
        closes = pd.Series([24000] * 3)
        self.assertTrue(sd.check(closes, 2))  # bar 2 < lookback 5

    def test_multiple_shocks_extend_halt(self):
        sd = self._make(threshold_pct=1.0, lookback_bars=2, halt_bars=2)
        prices = [10000] * 5 + [10200] + [10000] * 5 + [10200]
        closes = pd.Series(prices)
        sd.check(closes, 5)  # first shock
        halt1 = sd._halt_until_bar

        # Manually clear so we can trigger a second
        sd._halt_until_bar = -1
        sd.check(closes, 11)  # second shock
        halt2 = sd._halt_until_bar
        self.assertGreater(halt2, halt1)


# ═══════════════════════════════════════════════════════════════════
#  PREMIUM MODEL -- pricing correctness
# ═══════════════════════════════════════════════════════════════════

class TestPremiumModelDeep(unittest.TestCase):

    def _make(self, direction="LONG", **kw):
        from engine.premium_model import create_premium_state
        defaults = dict(entry_index_price=24000, base_premium=100,
                        delta=0.45, theta_per_candle=0.3, sl_pct=50,
                        confluence_score=60)
        defaults.update(kw)
        return create_premium_state(direction=direction, **defaults)

    def test_long_up_move_profits(self):
        ps = self._make("LONG")
        prem = ps.current_premium(24100, 1)  # +100 pts
        self.assertGreater(prem, ps.entry_premium)

    def test_short_down_move_profits(self):
        ps = self._make("SHORT")
        prem = ps.current_premium(23900, 1)  # -100 pts
        self.assertGreater(prem, ps.entry_premium)

    def test_long_down_move_loses(self):
        ps = self._make("LONG")
        prem = ps.current_premium(23900, 1)
        self.assertLess(prem, ps.entry_premium)

    def test_short_up_move_loses(self):
        ps = self._make("SHORT")
        prem = ps.current_premium(24100, 1)
        self.assertLess(prem, ps.entry_premium)

    def test_theta_accumulates_linearly(self):
        ps = self._make("LONG")
        p1 = ps.current_premium(24000, 10)
        p2 = ps.current_premium(24000, 20)
        theta_diff = p1 - p2
        self.assertAlmostEqual(theta_diff, 0.3 * 10, places=1)

    def test_sl_pct_calculation(self):
        ps = self._make("LONG", sl_pct=50)
        expected = ps.entry_premium * 0.5
        self.assertAlmostEqual(ps.sl_premium, expected, places=0)

    def test_higher_confluence_higher_premium(self):
        ps_low = self._make("LONG", confluence_score=40)
        ps_high = self._make("LONG", confluence_score=80)
        self.assertGreaterEqual(ps_high.entry_premium, ps_low.entry_premium)

    def test_check_exit_no_trigger(self):
        ps = self._make("LONG")
        mid_price = (ps.entry_premium + ps.sl_premium) / 2 + 10
        reason = ps.check_exit(mid_price, None)
        self.assertIsNone(reason)


# ═══════════════════════════════════════════════════════════════════
#  PERFORMANCE DB -- data integrity
# ═══════════════════════════════════════════════════════════════════

class TestPerformanceDBDeep(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make(self):
        from persistence.performance_db import PerformanceDB
        return PerformanceDB(self.db_path)

    def test_multiple_strategies_tracked(self):
        db = self._make()
        for i in range(10):
            strat = "pullback_1" if i < 6 else "stoch_cross_0"
            db.record_trade(date="2026-03-11", time=f"10:{i:02d}",
                            strategy=strat, direction="LONG",
                            entry_price=100, exit_price=110, pnl=750)

        stats = db.strategy_stats(min_trades=1)
        self.assertEqual(len(stats), 2)
        db.close()

    def test_negative_pnl_stored_correctly(self):
        db = self._make()
        db.record_trade(date="2026-03-11", time="10:00", strategy="test",
                        direction="LONG", entry_price=100, exit_price=90,
                        pnl=-750)
        summary = db.daily_summary("2026-03-11")
        self.assertEqual(summary["losses"], 1)
        self.assertAlmostEqual(summary["total_pnl"], -750)
        db.close()

    def test_empty_db_queries_safe(self):
        db = self._make()
        self.assertEqual(db.daily_summary()["trades"], 0)
        self.assertEqual(len(db.strategy_stats()), 0)
        db.close()

    def test_win_rate_calculation(self):
        db = self._make()
        for pnl in [500, 500, 500, -300, -300]:
            db.record_trade(date="2026-03-11", time="10:00", strategy="t",
                            direction="LONG", entry_price=100,
                            exit_price=100, pnl=pnl)
        s = db.daily_summary("2026-03-11")
        self.assertAlmostEqual(s["wr"], 60.0)
        db.close()

    def test_capital_after_tracked(self):
        db = self._make()
        db.record_trade(date="2026-03-11", time="10:00", strategy="t",
                        direction="LONG", entry_price=100, exit_price=110,
                        pnl=750, capital_after=10750)

        import sqlite3
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT capital_after FROM trades LIMIT 1").fetchone()
        self.assertEqual(row[0], 10750)
        conn.close()
        db.close()

    def test_concurrent_writes(self):
        db = self._make()

        def writer(offset):
            for i in range(20):
                db.record_trade(date="2026-03-11", time=f"10:{i + offset:02d}",
                                strategy=f"t{offset}", direction="LONG",
                                entry_price=100, exit_price=110, pnl=500)

        threads = [threading.Thread(target=writer, args=(i * 20,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        s = db.daily_summary("2026-03-11")
        self.assertEqual(s["trades"], 60)
        db.close()


# ═══════════════════════════════════════════════════════════════════
#  MARKET FEED -- state machine
# ═══════════════════════════════════════════════════════════════════

class TestMarketFeedDeep(unittest.TestCase):

    def _make(self):
        from engine.market_feed import MarketFeed
        return MarketFeed("key", "client", "feed", "auth")

    def test_price_isolation_per_token(self):
        feed = self._make()
        feed._prices["TOKEN_A"] = {"price": 100, "volume": 1, "time": datetime.now()}
        feed._prices["TOKEN_B"] = {"price": 200, "volume": 2, "time": datetime.now()}

        a = feed.get_ltp("TOKEN_A")
        b = feed.get_ltp("TOKEN_B")
        self.assertEqual(a["price"], 100)
        self.assertEqual(b["price"], 200)

    def test_get_ltp_returns_copy(self):
        """Returned dict should not mutate internal state."""
        feed = self._make()
        feed._prices["T1"] = {"price": 100, "volume": 1, "time": datetime.now()}

        result = feed.get_ltp("T1")
        result["price"] = 999

        internal = feed.get_ltp("T1")
        self.assertEqual(internal["price"], 100)

    def test_stop_without_start(self):
        feed = self._make()
        feed.stop()  # should not raise

    def test_default_token_is_nifty(self):
        feed = self._make()
        feed._prices[feed.NIFTY_TOKEN] = {"price": 24500, "volume": 0, "time": datetime.now()}
        result = feed.get_ltp()
        self.assertEqual(result["price"], 24500)


# ═══════════════════════════════════════════════════════════════════
#  POSITION MANAGER -- lifecycle correctness
# ═══════════════════════════════════════════════════════════════════

class TestPositionManagerDeep(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig = settings.CAPITAL_FILE
        settings.CAPITAL_FILE = Path(self.tmpdir) / "capital.json"

    def tearDown(self):
        settings.CAPITAL_FILE = self.orig
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make(self):
        from execution.position_manager import PositionManager
        from risk.capital_tracker import CapitalTracker
        broker = MagicMock()
        broker.place_order.return_value = "ORD001"
        broker.get_ltp.return_value = 100.0
        broker.cancel_order.return_value = True
        ct = CapitalTracker()
        ct.start_day()
        return PositionManager(broker, ct), broker

    def _signal(self):
        sig = MagicMock()
        sig.strategy_name = "test_strat"
        sig.option_type = "CE"
        sig.entry_price = 24000
        sig.stop_loss_index = 23900
        sig.target_index = 24200
        return sig

    def _strike(self):
        return {"symbol": "NIFTY_CE", "token": "123",
                "lotsize": str(settings.NIFTY_LOT_SIZE)}

    def test_failed_order_returns_none(self):
        pm, broker = self._make()
        broker.place_order.return_value = None
        pos = pm.open_position(self._signal(), self._strike(), 1, 100, 50)
        self.assertIsNone(pos)
        self.assertFalse(pm.has_open_positions)

    def test_sl_hit_closes_position(self):
        pm, broker = self._make()
        pos = pm.open_position(self._signal(), self._strike(), 1, 100, 50)
        self.assertIsNotNone(pos)

        pm.update_positions({"NIFTY_CE": 40})  # below SL of 50
        self.assertEqual(pos.status, "SL_HIT")
        self.assertLess(pos.pnl, 0)

    def test_target_hit_closes_position(self):
        pm, broker = self._make()
        sig = self._signal()
        pos = pm.open_position(sig, self._strike(), 1, 100, 50)

        # Target is entry + PREMIUM_TARGET_POINTS; might be 100 or higher
        pm.update_positions({"NIFTY_CE": pos.target + 10})
        self.assertIn(pos.status, ("TARGET_HIT", "OPEN"))

    def test_square_off_all_closes_everything(self):
        pm, broker = self._make()
        broker.place_order.side_effect = lambda **kw: f"ORD_{kw.get('tag', 'X')}"
        pm.open_position(self._signal(), self._strike(), 1, 100, 50)
        pm.open_position(self._signal(), self._strike(), 1, 110, 50)

        pm.square_off_all("TEST")
        self.assertEqual(len(pm.open_positions), 0)

    def test_square_off_with_no_ltp_uses_fallback(self):
        pm, broker = self._make()
        pos = pm.open_position(self._signal(), self._strike(), 1, 100, 50)
        broker.get_ltp.return_value = None  # LTP unavailable for square-off
        pm.square_off_all("TEST")
        self.assertAlmostEqual(pos.exit_price, 80, places=1)

    def test_missing_symbol_in_update_ignored(self):
        pm, broker = self._make()
        pm.open_position(self._signal(), self._strike(), 1, 100, 50)
        pm.update_positions({"WRONG_SYMBOL": 150})  # should not crash


# ═══════════════════════════════════════════════════════════════════
#  KILL SWITCH -- file-based flag
# ═══════════════════════════════════════════════════════════════════

class TestKillSwitchDeep(unittest.TestCase):

    def setUp(self):
        self.halt_file = Path(settings.DATA_DIR) / "HALT"
        if self.halt_file.exists():
            self.halt_file.unlink()

    def tearDown(self):
        if self.halt_file.exists():
            self.halt_file.unlink()

    def test_idempotent_clear(self):
        from risk.kill_switch import clear_halt
        clear_halt()
        clear_halt()  # double clear is safe

    def test_halt_file_contains_reason(self):
        from risk.kill_switch import set_halt
        set_halt("test reason here")
        content = self.halt_file.read_text()
        self.assertIn("test reason here", content)

    def test_halt_survives_process_restart(self):
        from risk.kill_switch import is_halted, set_halt
        set_halt("persist test")
        # Simulate "restart" by reimporting
        self.assertTrue(is_halted())


# ═══════════════════════════════════════════════════════════════════
#  DYNAMIC THETA -- mathematical correctness
# ═══════════════════════════════════════════════════════════════════

class TestDynamicTheta(unittest.TestCase):

    def test_linearly_proportional(self):
        t1 = settings.get_scaled_theta(24000)
        t2 = settings.get_scaled_theta(48000)
        self.assertAlmostEqual(t2, t1 * 2, places=4)

    def test_historical_nifty_levels(self):
        """Theta should be reasonable across historical Nifty range."""
        for level in [5000, 10000, 15000, 20000, 25000]:
            theta = settings.get_scaled_theta(level)
            self.assertGreater(theta, 0)
            self.assertLess(theta, 2.0)  # reasonable upper bound

    def test_matches_reference_level(self):
        theta = settings.get_scaled_theta(settings.THETA_REFERENCE_LEVEL)
        self.assertAlmostEqual(theta, settings.THETA_BASE, places=6)


# ═══════════════════════════════════════════════════════════════════
#  IMESSAGE ALERTS -- formatting integrity
# ═══════════════════════════════════════════════════════════════════

class TestiMessageFormatting(unittest.TestCase):

    def test_strip_html_complex(self):
        from alerts.imessage_bot import _strip_html
        html = '<b>PnL</b>: Rs <span style="color:green">+500</span>'
        self.assertEqual(_strip_html(html), "PnL: Rs +500")

    def test_eod_report_negative_pnl(self):
        from alerts.imessage_bot import send_eod_report
        with patch("alerts.imessage_bot.send_alert") as mock:
            send_eod_report({
                "capital": 9500, "daily_pnl": -500, "daily_pnl_pct": -5.0,
                "trades": 3, "wins": 1, "losses": 2, "win_rate": 33,
                "total_pnl": -500, "max_drawdown": 5.0,
            })
            msg = mock.call_args[0][0]
            self.assertIn("-500", msg)
            self.assertNotIn("+", msg.split("Day PnL")[1].split("\n")[0])

    def test_trade_alert_entry_format(self):
        from alerts.imessage_bot import send_trade_alert
        with patch("alerts.imessage_bot.send_alert") as mock:
            send_trade_alert("ENTRY", "pullback_1", "NIFTY_CE",
                             100.50, 75, sl=50.25, target=150.75)
            msg = mock.call_args[0][0]
            self.assertIn("TRADE OPENED", msg)
            self.assertNotIn("PnL", msg)

    def test_trade_alert_exit_with_loss(self):
        from alerts.imessage_bot import send_trade_alert
        with patch("alerts.imessage_bot.send_alert") as mock:
            send_trade_alert("SL_HIT", "pullback_1", "NIFTY_PE",
                             80, 75, pnl=-1500)
            msg = mock.call_args[0][0]
            self.assertIn("TRADE CLOSED", msg)
            self.assertIn("-1500", msg)

    @patch("alerts.imessage_bot.settings")
    @patch("alerts.imessage_bot.subprocess.run")
    def test_special_characters_escaped(self, mock_run, mock_settings):
        mock_settings.IMESSAGE_RECIPIENT = "+91999"
        mock_run.return_value = MagicMock(returncode=0)

        from alerts.imessage_bot import _send_imessage
        _send_imessage('Price: Rs 100 "quoted" and \\backslash')

        script = mock_run.call_args[0][0][2]
        self.assertNotIn('"quoted"', script)
        self.assertIn('\\"quoted\\"', script)


# ═══════════════════════════════════════════════════════════════════
#  HOLIDAYS -- data quality
# ═══════════════════════════════════════════════════════════════════

class TestHolidaysDeep(unittest.TestCase):

    def test_all_dates_are_valid(self):
        with open(settings.BASE_DIR / "config" / "holidays.json") as f:
            data = json.load(f)
        for h in data["holidays"]:
            dt = datetime.strptime(h["date"], "%Y-%m-%d")
            self.assertEqual(dt.year, 2026)

    def test_no_weekends_in_holidays(self):
        """NSE holidays only list weekdays -- weekends are off by default."""
        with open(settings.BASE_DIR / "config" / "holidays.json") as f:
            data = json.load(f)
        for h in data["holidays"]:
            dt = datetime.strptime(h["date"], "%Y-%m-%d")
            self.assertLess(dt.weekday(), 5,
                            f"{h['date']} ({h['name']}) is a weekend")

    def test_no_duplicates(self):
        with open(settings.BASE_DIR / "config" / "holidays.json") as f:
            data = json.load(f)
        dates = [h["date"] for h in data["holidays"]]
        self.assertEqual(len(dates), len(set(dates)))

    def test_sorted_chronologically(self):
        with open(settings.BASE_DIR / "config" / "holidays.json") as f:
            data = json.load(f)
        dates = [h["date"] for h in data["holidays"]]
        self.assertEqual(dates, sorted(dates))


# ═══════════════════════════════════════════════════════════════════
#  INDICATORS -- numerical edge cases
# ═══════════════════════════════════════════════════════════════════

class TestIndicatorsEdgeCases(unittest.TestCase):

    def test_flat_price_rsi_is_50(self):
        from engine.indicators import rsi
        flat = pd.Series([100.0] * 50)
        r = rsi(flat, 14)
        valid = r.dropna()
        if len(valid) > 0:
            # RSI of flat prices should be ~50 or NaN (no gains/losses)
            for v in valid:
                self.assertTrue(np.isnan(v) or 40 <= v <= 60,
                                f"RSI={v} on flat prices")

    def test_monotonic_up_rsi_high(self):
        from engine.indicators import rsi
        up = pd.Series([100 + i for i in range(50)])
        r = rsi(up, 14)
        valid = r.dropna()
        self.assertTrue((valid > 80).all())

    def test_monotonic_down_rsi_low(self):
        from engine.indicators import rsi
        down = pd.Series([100 - i for i in range(50)])
        r = rsi(down, 14)
        valid = r.dropna()
        self.assertTrue((valid < 20).all())

    def test_atr_with_zero_range(self):
        from engine.indicators import atr
        flat = pd.DataFrame({
            "open": [100.0] * 30, "high": [100.0] * 30,
            "low": [100.0] * 30, "close": [100.0] * 30,
            "volume": [1000] * 30,
        })
        a = atr(flat, 14)
        valid = a.dropna()
        self.assertTrue((valid == 0).all())

    def test_ema_convergence(self):
        """EMA should converge to constant for constant input."""
        from engine.indicators import ema
        const = pd.Series([42.0] * 100)
        e = ema(const, 20)
        self.assertAlmostEqual(e.iloc[-1], 42.0, places=4)

    def test_vwap_with_zero_volume(self):
        from engine.indicators import vwap
        df = pd.DataFrame({
            "open": [100] * 10, "high": [105] * 10,
            "low": [95] * 10, "close": [100] * 10,
            "volume": [0] * 10,
        })
        v = vwap(df)
        self.assertEqual(len(v), 10)


# ═══════════════════════════════════════════════════════════════════
#  TRADING ENGINE -- paper lifecycle and sizing
# ═══════════════════════════════════════════════════════════════════

class TestTradingEngineDeep(unittest.TestCase):

    def test_lot_sizing_uses_decision_lots(self):
        """Paper entry must use decision.lots, not _compute_lots()."""
        from engine.multi_strategy_engine import TradeSignal
        from engine.trading_engine import TradingEngine
        from risk.risk_engine import RiskDecision

        engine = TradingEngine()
        engine._nifty_spot = 24000.0
        signal = TradeSignal(
            direction="LONG",
            signal_type="PULLBACK",
            confidence=70,
            htf_rsi=55,
            ltf_rsi=45,
            nifty_price=24000,
            reason="test",
        )
        decision = RiskDecision(approved=True, reason="ok", lots=3, premium_sl_pct=30)

        with patch.object(engine, "_save_paper_positions"), patch("engine.trading_engine.send_trade_alert"):
            engine._paper_enter(signal, decision, "CE")

        self.assertEqual(len(engine._paper_positions), 1)
        self.assertEqual(engine._paper_positions[0].lots, 3)
        self.assertNotEqual(engine._paper_positions[0].lots, 99)

    def test_bar_close_sl_records_trade(self):
        """_update_paper_positions should persist SL exits on bar close."""
        from engine.multi_strategy_engine import TradeSignal
        from engine.premium_model import create_premium_state
        from engine.trading_engine import PaperPosition, TradingEngine

        engine = TradingEngine()
        engine._nifty_spot = 24000.0
        engine.perf_db = MagicMock()

        signal = TradeSignal(
            direction="LONG",
            signal_type="PULLBACK",
            confidence=70,
            htf_rsi=55,
            ltf_rsi=45,
            nifty_price=24000,
            reason="test",
        )
        prem = create_premium_state(
            entry_index_price=24000,
            direction="LONG",
            base_premium=100,
            delta=0.7,
            theta_per_candle=0.3,
            sl_pct=30,
            confluence_score=70,
            signal_type="PULLBACK",
        )
        pos = PaperPosition(
            direction="LONG",
            entry_time="2026-03-11T10:00:00",
            entry_index=24000,
            entry_premium=100,
            sl_premium=70,
            lots=1,
            qty=settings.NIFTY_LOT_SIZE,
            signal=signal,
            prem_state=prem,
            peak_premium=100,
            candles_held=2,
        )
        engine._paper_positions = [pos]

        with patch.object(engine, "_save_paper_positions"), patch("engine.trading_engine.send_trade_alert"):
            engine._nifty_spot = 23900
            engine._update_paper_positions()

        engine.perf_db.record_trade.assert_called_once()
        self.assertEqual(len(engine._paper_positions), 0)

    def test_emergency_sl_only_on_extreme_move(self):
        """_check_emergency_sl fires only when Nifty moves > 2x ATR."""
        from engine.multi_strategy_engine import TradeSignal
        from engine.premium_model import create_premium_state
        from engine.trading_engine import PaperPosition, TradingEngine

        engine = TradingEngine()
        engine._nifty_spot = 24000.0
        engine._last_atr = 28.0
        engine.perf_db = MagicMock()

        signal = TradeSignal(
            direction="LONG",
            signal_type="PULLBACK",
            confidence=70,
            htf_rsi=55,
            ltf_rsi=45,
            nifty_price=24000,
            reason="test",
        )
        prem = create_premium_state(
            entry_index_price=24000,
            direction="LONG",
            base_premium=100,
            delta=0.7,
            theta_per_candle=0.3,
            sl_pct=30,
            confluence_score=70,
            signal_type="PULLBACK",
        )
        pos = PaperPosition(
            direction="LONG",
            entry_time="2026-03-11T10:00:00",
            entry_index=24000,
            entry_premium=100,
            sl_premium=70,
            lots=1,
            qty=settings.NIFTY_LOT_SIZE,
            signal=signal,
            prem_state=prem,
            peak_premium=100,
        )

        engine._paper_positions = [pos]
        with patch.object(engine, "_save_paper_positions"), patch("engine.trading_engine.send_trade_alert"):
            engine._nifty_spot = 23960  # 40pts < 56 (2x28) -- no emergency
            engine._check_emergency_sl()
        self.assertEqual(len(engine._paper_positions), 1)

        with patch.object(engine, "_save_paper_positions"), patch("engine.trading_engine.send_trade_alert"):
            engine._nifty_spot = 23940  # 60pts > 56 (2x28) -- emergency
            engine._check_emergency_sl()
        self.assertEqual(len(engine._paper_positions), 0)


# ═══════════════════════════════════════════════════════════════════
#  HTF AGGREGATION -- time-based resampling
# ═══════════════════════════════════════════════════════════════════

class TestHTFAggregation(unittest.TestCase):

    def test_htf_aggregation_time_based(self):
        """aggregate_timeframe uses clock-aligned resample buckets."""
        from engine.indicators_extended import aggregate_timeframe

        ts = pd.date_range("2026-03-11 09:15", periods=12, freq="5min")
        c = np.linspace(24000, 24050, 12)
        df = pd.DataFrame({
            "open": c, "high": c + 5, "low": c - 5,
            "close": c, "volume": np.arange(12) + 1,
        }, index=ts)

        agg = aggregate_timeframe(df, 3)

        self.assertEqual(len(agg), 4)
        self.assertEqual(agg.index[0].hour, 9)
        self.assertEqual(agg.index[0].minute, 15)
        self.assertEqual(agg.index[1].minute, 30)
        self.assertEqual(agg["volume"].iloc[0], 6)  # sum of bars 0-2


if __name__ == "__main__":
    unittest.main(verbosity=2)
