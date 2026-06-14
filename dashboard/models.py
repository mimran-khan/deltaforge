"""Pydantic response models for the dashboard API."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class StatusResponse(BaseModel):
    current_capital: float
    daily_pnl: float
    total_pnl: float
    peak_capital: float
    drawdown_pct: float
    trades_today: int
    wins_today: int
    losses_today: int
    win_rate: float
    consecutive_losses: int
    halted: bool
    halt_reason: Optional[str] = None
    session_active: bool
    session_pid: Optional[int] = None
    last_updated: Optional[str] = None


class CapitalResponse(BaseModel):
    current_capital: float = 0
    total_pnl: float = 0
    peak_capital: float = 0
    weekly_pnl: float = 0
    max_drawdown: float = 0
    daily_pnl: float = 0
    trades_today: int = 0
    consecutive_losses: int = 0
    day_start_capital: float = 0
    wins_today: int = 0
    losses_today: int = 0
    last_updated: Optional[str] = None


class TradeRecord(BaseModel):
    id: int
    date: str
    time: str
    strategy: str
    direction: str
    confidence: Optional[float] = None
    htf_rsi: Optional[float] = None
    adx: Optional[float] = None
    entry_price: float
    exit_price: float
    pnl: float
    hold_bars: Optional[int] = None
    exit_reason: Optional[str] = None
    lots: Optional[int] = None
    capital_after: Optional[float] = None
    created_at: Optional[str] = None


class TradeSummary(BaseModel):
    date: str
    trades: int
    wins: int
    losses: int
    total_pnl: float
    wr: float


class StrategyStats(BaseModel):
    strategy: str
    trades: int
    wins: int
    wr: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    best_trade: float
    worst_trade: float


class RiskGate(BaseModel):
    name: str
    current: Any
    threshold: Any
    status: str
    pct: int


class BrokerInfo(BaseModel):
    session_active: bool
    session_pid: Optional[int] = None
    session_started: Optional[str] = None
    trading_mode: str
    client_id: str
    lot_size: int
    max_lots: int
    alert_method: str
    last_updated: Optional[str] = None
