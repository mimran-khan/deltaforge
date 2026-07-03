"""Instrument registry for multi-asset trading.

Defines specifications for every tradeable instrument (Nifty options,
MCX Gold, MCX Crude, NSE Currency).  The registry is read-only at runtime;
instruments are toggled via MULTI_ASSET_ENABLED and per-instrument `enabled`
flags in settings / .env.

This module is purely additive -- importing it has zero side effects on the
existing Nifty-only trading path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TradingHours:
    """Per-instrument session boundaries (IST, 24h format)."""
    market_open: str
    market_close: str
    entry_start: str
    entry_end: str
    square_off: str
    eod_report: str


@dataclass(frozen=True)
class CostModel:
    """Transaction cost parameters per exchange segment."""
    brokerage_per_order: float = 20.0
    tax_sell_pct: float = 0.05       # STT (NFO) / CTT (MCX) / 0 (CDS)
    exchange_txn_pct: float = 0.05
    stamp_duty_pct: float = 0.003
    gst_pct: float = 18.0           # GST on brokerage + exchange txn
    slippage: float = 0.30
    market_impact_pct: float = 0.10  # applied when lots >= impact_lot_threshold
    impact_lot_threshold: int = 5

    def total_costs(self, entry_price: float, exit_price: float,
                    qty: int, lots: int,
                    price_divisor: float = 1.0) -> float:
        entry_norm = entry_price / price_divisor if price_divisor > 1.0 else entry_price
        exit_norm = exit_price / price_divisor if price_divisor > 1.0 else exit_price

        brokerage = self.brokerage_per_order * 2
        sell_turnover = exit_norm * qty
        tax = sell_turnover * self.tax_sell_pct / 100
        total_turnover = (entry_norm + exit_norm) * qty
        exchange = total_turnover * self.exchange_txn_pct / 100
        stamp = total_turnover * self.stamp_duty_pct / 100
        gst = (brokerage + exchange) * self.gst_pct / 100
        impact = 0.0
        if lots >= self.impact_lot_threshold:
            impact = entry_norm * qty * self.market_impact_pct / 100
        return brokerage + tax + exchange + stamp + gst + impact


# ── Cost model presets ──────────────────────────────────────────

NFO_COSTS = CostModel(
    brokerage_per_order=20.0,
    tax_sell_pct=0.05,       # STT on options sell
    exchange_txn_pct=0.05,
    stamp_duty_pct=0.003,
    slippage=0.30,
)

MCX_COSTS = CostModel(
    brokerage_per_order=20.0,
    tax_sell_pct=0.01,       # CTT (Commodity Transaction Tax)
    exchange_txn_pct=0.04,
    stamp_duty_pct=0.002,
    slippage=0.50,           # wider spreads on MCX
)

CDS_COSTS = CostModel(
    brokerage_per_order=20.0,
    tax_sell_pct=0.0,        # no STT/CTT on currency
    exchange_txn_pct=0.03,
    stamp_duty_pct=0.001,
    slippage=0.10,           # tight spreads on USDINR
)


@dataclass
class StrategyOverrides:
    """Per-instrument strategy parameter overrides.

    Any field set to None falls through to the engine's default.
    """
    enabled_strategies: Optional[set[str]] = None
    disabled_strategies: Optional[set[str]] = None
    min_adx: Optional[float] = None
    max_adx: Optional[float] = None
    min_confidence: Optional[int] = None
    max_hold_candles: Optional[int] = None
    sl_pct: Optional[float] = None
    target_mult: Optional[float] = None
    trail_trigger_pct: Optional[float] = None
    trail_pct: Optional[float] = None
    cooldown_bars: Optional[int] = None
    max_total_per_day: Optional[int] = None


@dataclass
class Instrument:
    """Full specification for a tradeable instrument."""
    name: str
    display_name: str
    exchange: str              # Angel One exchange string: "NFO", "MCX", "CDS"
    ws_exchange_type: int      # WebSocket segment: 1=NSE, 2=NFO, 5=MCX, 8=CDS
    asset_type: str            # "options" or "futures"

    symbol_prefix: str         # prefix for scrip master filter
    index_token: str           # Angel One token for spot/futures price feed

    lot_size: int
    tick_size: float           # minimum price movement
    tick_value: float          # P&L per tick per lot

    margin_pct: float          # approx initial margin as % of contract value
    capital_alloc_pct: float   # % of total capital allocated to this instrument

    hours: TradingHours
    costs: CostModel
    strategy: StrategyOverrides = field(default_factory=StrategyOverrides)

    enabled: bool = True
    price_divisor: float = 1.0


# ═══════════════════════════════════════════════════════════════
#  INSTRUMENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════

NIFTY = Instrument(
    name="NIFTY",
    display_name="Nifty 50",
    exchange="NFO",
    ws_exchange_type=1,        # NSE Cash for index price
    asset_type="options",
    symbol_prefix="NIFTY",
    index_token="99926000",
    lot_size=65,
    tick_size=0.05,
    tick_value=0.05 * 65,      # Rs 3.25 per tick
    margin_pct=100.0,          # premium-based, not margin
    capital_alloc_pct=40.0,
    hours=TradingHours(
        market_open="09:15",
        market_close="15:30",
        entry_start="09:30",
        entry_end="14:30",
        square_off="15:15",
        eod_report="15:35",
    ),
    costs=NFO_COSTS,
    strategy=StrategyOverrides(
        disabled_strategies={
            "EMA_MOMENTUM", "VWAP_MOMENTUM", "VWAP_MEAN_REV",
            "RSI_REVERSION", "ADX_BREAKOUT", "ORB_BREAKOUT",
            "BB_SQUEEZE", "VWAP_BOUNCE", "RSI_DIVERGENCE",
            "CPR_RANGE", "CPR_BREAKOUT", "GAP_TRADE",
            "FIRST_HOUR_MOM", "NARROW_CPR_BO",
            "TRIPLE_CONFIRM", "VWAP_2SD_REV",
        },
    ),
    enabled=True,
)

GOLD_PETAL = Instrument(
    name="GOLD_PETAL",
    display_name="Gold Petal",
    exchange="MCX",
    ws_exchange_type=5,
    asset_type="futures",
    symbol_prefix="GOLDPETAL",
    index_token="",            # resolved at runtime from scrip master
    lot_size=1,                # 1 gram
    tick_size=1.0,             # Rs 1 per gram
    tick_value=1.0,            # Rs 1 per tick per lot
    margin_pct=7.0,            # ~6% SPAN + 1% ELM
    capital_alloc_pct=30.0,
    hours=TradingHours(
        market_open="09:00",
        market_close="23:30",
        entry_start="09:15",
        entry_end="23:00",
        square_off="23:25",
        eod_report="23:35",
    ),
    costs=MCX_COSTS,
    strategy=StrategyOverrides(
        enabled_strategies={
            "PULLBACK", "TREND_RIDE", "SUPERTREND",
        },
        max_adx=50.0,
        min_confidence=65,
        max_hold_candles=48,
        sl_pct=2.5,            # 2.5% -- Gold Petal daily range is 1-2%, need room for noise
        target_mult=2.0,       # 2x risk -- let winners run
        trail_trigger_pct=3.0,
        trail_pct=1.5,
        max_total_per_day=3,
    ),
    enabled=False,             # disabled: too slow for 5m momentum strategies
)

GOLD_MINI = Instrument(
    name="GOLD_MINI",
    display_name="Gold Mini",
    exchange="MCX",
    ws_exchange_type=5,
    asset_type="futures",
    symbol_prefix="GOLDM",
    index_token="",
    lot_size=100,              # 100 grams
    tick_size=1.0,
    tick_value=100.0,          # Rs 100 per tick per lot
    margin_pct=7.0,
    capital_alloc_pct=0.0,     # disabled by default (needs ~Rs 50K margin)
    hours=TradingHours(
        market_open="09:00",
        market_close="23:30",
        entry_start="09:15",
        entry_end="23:00",
        square_off="23:25",
        eod_report="23:35",
    ),
    costs=MCX_COSTS,
    strategy=StrategyOverrides(
        enabled_strategies={"PULLBACK", "TREND_RIDE", "CPR_BREAKOUT"},
        max_adx=40.0,
        min_confidence=65,
        sl_pct=2.0,
        target_mult=1.5,
    ),
    enabled=False,             # too expensive for current capital
)

CRUDE_OIL_MINI = Instrument(
    name="CRUDEOILM",
    display_name="Crude Oil Mini",
    exchange="MCX",
    ws_exchange_type=5,
    asset_type="futures",
    symbol_prefix="CRUDEOILM",
    index_token="",
    lot_size=10,               # 10 barrels
    tick_size=1.0,             # Rs 1 per barrel
    tick_value=10.0,           # Rs 10 per tick per lot
    margin_pct=9.5,
    capital_alloc_pct=20.0,
    hours=TradingHours(
        market_open="09:00",
        market_close="23:30",
        entry_start="09:15",
        entry_end="23:00",
        square_off="23:25",
        eod_report="23:35",
    ),
    costs=MCX_COSTS,
    strategy=StrategyOverrides(
        enabled_strategies={
            "PULLBACK", "TREND_RIDE", "SUPERTREND",
        },
        max_adx=50.0,
        min_confidence=65,
        max_hold_candles=36,
        sl_pct=2.0,            # 2% -- Crude is volatile, needs breathing room
        target_mult=2.0,       # 2x risk
        trail_trigger_pct=3.0,
        trail_pct=1.5,
        cooldown_bars=3,
        max_total_per_day=4,
    ),
    enabled=True,
)

USDINR = Instrument(
    name="USDINR",
    display_name="USD/INR",
    exchange="CDS",
    ws_exchange_type=8,
    asset_type="futures",
    symbol_prefix="USDINR",
    index_token="",
    lot_size=1000,             # 1000 USD
    tick_size=0.0025,          # 0.25 paise
    tick_value=2.50,           # Rs 2.50 per tick per lot
    margin_pct=2.5,
    capital_alloc_pct=10.0,
    price_divisor=1000.0,      # Angel One CDS returns paise (84250 → 84.250)
    enabled=True,
    hours=TradingHours(
        market_open="09:00",
        market_close="17:00",
        entry_start="09:15",
        entry_end="16:30",
        square_off="16:55",
        eod_report="17:05",
    ),
    costs=CDS_COSTS,
    strategy=StrategyOverrides(
        enabled_strategies={
            "PULLBACK", "TREND_RIDE", "SUPERTREND",
        },
        max_adx=40.0,
        min_confidence=65,
        max_hold_candles=30,
        sl_pct=0.25,           # ~25 pips -- USDINR daily range is ~0.3-0.5%
        target_mult=2.0,       # 2x risk -- currency trends well
        trail_trigger_pct=0.35,
        trail_pct=0.15,
        max_total_per_day=3,
    ),
)


# ── Registry ────────────────────────────────────────────────────

ALL_INSTRUMENTS: dict[str, Instrument] = {
    inst.name: inst
    for inst in [NIFTY, GOLD_PETAL, GOLD_MINI, CRUDE_OIL_MINI, USDINR]
}


def get_enabled_instruments() -> list[Instrument]:
    """Return instruments that are both enabled and have capital allocated."""
    return [
        inst for inst in ALL_INSTRUMENTS.values()
        if inst.enabled and inst.capital_alloc_pct > 0
    ]


def get_futures_instruments() -> list[Instrument]:
    """Return enabled futures-only instruments (excludes Nifty options)."""
    return [
        inst for inst in get_enabled_instruments()
        if inst.asset_type == "futures"
    ]


def get_instrument(name: str) -> Optional[Instrument]:
    return ALL_INSTRUMENTS.get(name)


def get_latest_square_off_time() -> str:
    """Return the latest square-off time across all enabled instruments.

    Used by the scheduler to know when the engine can safely shut down.
    """
    enabled = get_enabled_instruments()
    if not enabled:
        return "15:15"
    return max(inst.hours.square_off for inst in enabled)


def get_latest_eod_time() -> str:
    """Return the latest EOD report time across all enabled instruments."""
    enabled = get_enabled_instruments()
    if not enabled:
        return "15:35"
    return max(inst.hours.eod_report for inst in enabled)
