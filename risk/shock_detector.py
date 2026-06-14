"""Shock detector -- circuit breaker for extreme price moves.

Halts new entries when Nifty moves > threshold% within lookback bars.
Does not generate signals; only blocks them. Pure safety mechanism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class ShockDetector:

    def __init__(self, threshold_pct: float = 1.5,
                 lookback_bars: int = 3, halt_bars: int = 6):
        self.threshold = threshold_pct / 100
        self.lookback = lookback_bars
        self.halt_bars = halt_bars
        self._halt_until_bar = -1

    def reset(self):
        self._halt_until_bar = -1

    def check(self, closes: pd.Series, bar_idx: int) -> bool:
        """Returns True if safe to trade, False if shocked/halted."""
        if bar_idx <= self._halt_until_bar:
            return False

        if bar_idx < self.lookback:
            return True

        try:
            current = closes.iloc[bar_idx]
            past = closes.iloc[bar_idx - self.lookback]
        except (IndexError, KeyError):
            return True

        if np.isnan(current) or np.isnan(past) or past == 0:
            return True

        pct_move = abs(current - past) / past
        if pct_move >= self.threshold:
            self._halt_until_bar = bar_idx + self.halt_bars
            logger.warning(
                "SHOCK: {:.2f}% move in {} bars (bar {}). Halting until bar {}",
                pct_move * 100, self.lookback, bar_idx, self._halt_until_bar)
            return False

        return True
