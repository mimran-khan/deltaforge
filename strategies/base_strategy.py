"""Abstract base for all trading strategies."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Signal:
    signal_type: SignalType
    strategy_name: str
    entry_price: float          # index price at signal
    option_type: str            # "CE" or "PE"
    stop_loss_index: float = 0  # SL in index points (0 = use premium SL)
    target_index: float = 0     # target in index points (0 = use premium target)
    confidence: float = 1.0     # 0-1 scale
    confluence_score: float = 0 # raw confluence score
    reason: str = ""
    timestamp: Optional[pd.Timestamp] = None


class BaseStrategy(ABC):
    """Every strategy must implement these methods."""

    name: str = "base"

    @abstractmethod
    def on_candle(self, df: pd.DataFrame, current_time: pd.Timestamp) -> Optional[Signal]:
        """Called on each new candle. Return Signal or None."""
        ...

    @abstractmethod
    def reset(self):
        """Reset strategy state for a new trading day."""
        ...
