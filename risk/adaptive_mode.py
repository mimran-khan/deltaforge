"""Intra-day adaptive mode controller.

Shifts between AGGRESSIVE / NORMAL / DEFENSIVE / HALT based on
real-time performance metrics (daily P&L, win rate, consecutive
results). Emits an AdaptiveProfile consumed by the trading engine,
strategy scanner, and risk engine to modulate signal quality,
position sizing, and exit parameters throughout the trading day.

Hysteresis rules prevent flip-flopping:
  HALT -> DEFENSIVE (after 1 bar)
  DEFENSIVE -> NORMAL (after 1 win)
  NORMAL -> AGGRESSIVE (performance triggers)
  AGGRESSIVE -> NORMAL (on any loss)
  Never skip tiers.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from config import settings


class Mode(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"
    HALT = "HALT"


@dataclass
class AdaptiveProfile:
    """Trading parameters for the current adaptive mode."""
    mode: Mode
    min_confidence: int
    max_trades_per_day: int
    max_simultaneous: int
    lot_multiplier: float
    sl_multiplier: float
    target_multiplier: float
    trail_lock_tiers: tuple
    trail_trigger_pct: float = 12.0
    trail_pct: float = 8.0


_PROFILES = {
    Mode.AGGRESSIVE: AdaptiveProfile(
        mode=Mode.AGGRESSIVE,
        min_confidence=68,
        max_trades_per_day=10,
        max_simultaneous=2,
        lot_multiplier=1.0,
        sl_multiplier=1.0,
        target_multiplier=1.3,
        trail_lock_tiers=(10, 14, 18, 22, 26, 30),
        trail_trigger_pct=10.0,
        trail_pct=6.0,
    ),
    Mode.NORMAL: AdaptiveProfile(
        mode=Mode.NORMAL,
        min_confidence=72,
        max_trades_per_day=8,
        max_simultaneous=2,
        lot_multiplier=1.0,
        sl_multiplier=1.0,
        target_multiplier=1.0,
        trail_lock_tiers=(10, 14, 18, 22, 26, 30),
        trail_trigger_pct=12.0,
        trail_pct=8.0,
    ),
    Mode.DEFENSIVE: AdaptiveProfile(
        mode=Mode.DEFENSIVE,
        min_confidence=78,
        max_trades_per_day=4,
        max_simultaneous=1,
        lot_multiplier=0.5,
        sl_multiplier=0.7,
        target_multiplier=0.8,
        trail_lock_tiers=(7, 10, 13, 16, 19, 22),
        trail_trigger_pct=8.0,
        trail_pct=5.0,
    ),
    Mode.HALT: AdaptiveProfile(
        mode=Mode.HALT,
        min_confidence=100,
        max_trades_per_day=0,
        max_simultaneous=0,
        lot_multiplier=0.0,
        sl_multiplier=1.0,
        target_multiplier=1.0,
        trail_lock_tiers=(4, 6, 8, 10, 12, 14, 16),
        trail_trigger_pct=12.0,
        trail_pct=8.0,
    ),
}


class AdaptiveModeController:
    """Stateful controller that tracks intra-day performance and emits profiles.

    Call ``update()`` after every trade close and ``reset()`` at day start.
    Read ``profile`` at entry time to get current adaptive parameters.
    """

    def __init__(self):
        self._mode: Mode = Mode.NORMAL
        self._prev_mode: Mode = Mode.NORMAL
        self._bars_in_halt: int = 0
        self._consecutive_wins: int = 0

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def profile(self) -> AdaptiveProfile:
        return _PROFILES[self._mode]

    def reset(self):
        """Call at day start."""
        self._mode = Mode.NORMAL
        self._prev_mode = Mode.NORMAL
        self._bars_in_halt = 0
        self._consecutive_wins = 0

    def update(self, daily_pnl_pct: float, wins: int, losses: int,
               consecutive_losses: int, trades: int,
               last_trade_won: bool | None = None):
        """Recompute mode after a trade close.

        Args:
            daily_pnl_pct: current day P&L as % of day-start capital
            wins: total wins today
            losses: total losses today
            consecutive_losses: current consecutive loss streak
            trades: total trades today
            last_trade_won: True if the just-closed trade was a win
        """
        if not getattr(settings, 'ADAPTIVE_MODE_ENABLED', True):
            return

        prev = self._mode
        wr = (wins / trades * 100) if trades > 0 else 50.0

        halt_consec = getattr(settings, 'ADAPTIVE_HALT_CONSECUTIVE', 3)
        halt_loss_pct = getattr(settings, 'ADAPTIVE_HALT_LOSS_PCT', -10)
        def_loss_pct = getattr(settings, 'ADAPTIVE_DEFENSIVE_PNL_PCT', -3)
        def_consec = getattr(settings, 'ADAPTIVE_DEFENSIVE_CONSECUTIVE', 2)
        def_wr = getattr(settings, 'ADAPTIVE_DEFENSIVE_WR', 40)
        def_wr_min_trades = getattr(settings, 'ADAPTIVE_DEFENSIVE_WR_MIN_TRADES', 3)
        agg_pnl_pct = getattr(settings, 'ADAPTIVE_AGGRESSIVE_PNL_PCT', 5)
        agg_consec_wins = getattr(settings, 'ADAPTIVE_AGGRESSIVE_CONSEC_WINS', 2)
        agg_min_wr = getattr(settings, 'ADAPTIVE_AGGRESSIVE_MIN_WR', 60)

        if last_trade_won is True:
            self._consecutive_wins += 1
        elif last_trade_won is False:
            self._consecutive_wins = 0

        # --- Evaluate from most restrictive to least ---

        # HALT conditions (absolute)
        if consecutive_losses >= halt_consec:
            self._mode = Mode.HALT
        elif daily_pnl_pct <= halt_loss_pct:
            self._mode = Mode.HALT

        # DEFENSIVE conditions
        elif daily_pnl_pct <= def_loss_pct:
            self._mode = Mode.DEFENSIVE
        elif consecutive_losses >= def_consec:
            self._mode = Mode.DEFENSIVE
        elif trades >= def_wr_min_trades and wr < def_wr:
            self._mode = Mode.DEFENSIVE

        # AGGRESSIVE conditions (only promote from NORMAL, never skip)
        elif (self._mode in (Mode.NORMAL, Mode.AGGRESSIVE)
              and daily_pnl_pct >= agg_pnl_pct
              and self._consecutive_wins >= agg_consec_wins
              and wr >= agg_min_wr):
            self._mode = Mode.AGGRESSIVE

        # Recovery: HALT -> DEFENSIVE (after last_trade_won or 1 bar)
        elif prev == Mode.HALT and last_trade_won is True:
            self._mode = Mode.DEFENSIVE

        # Recovery: DEFENSIVE -> NORMAL (after 1 win)
        elif prev == Mode.DEFENSIVE and last_trade_won is True:
            self._mode = Mode.NORMAL

        # AGGRESSIVE -> NORMAL on loss
        elif prev == Mode.AGGRESSIVE and last_trade_won is False:
            self._mode = Mode.NORMAL

        # No change -- stay in current mode
        else:
            pass

        if self._mode != prev:
            self._prev_mode = prev
            logger.info(
                "ADAPTIVE MODE: {} -> {} | PnL={:.1f}% W={} L={} CL={} CW={} WR={:.0f}%",
                prev.value, self._mode.value,
                daily_pnl_pct, wins, losses,
                consecutive_losses, self._consecutive_wins, wr,
            )

    def on_bar(self):
        """Call on each new bar -- handles HALT -> DEFENSIVE after 1 bar."""
        if self._mode == Mode.HALT:
            self._bars_in_halt += 1
            if self._bars_in_halt >= 2:
                self._mode = Mode.DEFENSIVE
                self._bars_in_halt = 0
                logger.info("ADAPTIVE: HALT -> DEFENSIVE (cooldown elapsed)")
        else:
            self._bars_in_halt = 0
