"""REST API routes for the DeltaForge dashboard."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from config import settings
from dashboard import data_access as da
from dashboard.models import (
    BrokerInfo,
    CapitalResponse,
    InstrumentStatus,
    RiskGate,
    StatusResponse,
    StrategyStats,
    TradeRecord,
    TradeSummary,
)

router = APIRouter(prefix="/api")

DASHBOARD_API_TOKEN: str = getattr(settings, "DASHBOARD_API_TOKEN", "")


def _require_auth(authorization: Optional[str]):
    """Verify Bearer token for mutating endpoints. No-op when token is unset."""
    if not DASHBOARD_API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization[7:] != DASHBOARD_API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")


@router.get("/status", response_model=StatusResponse)
def status():
    return da.get_status()


@router.get("/capital", response_model=CapitalResponse)
def capital():
    return da.read_capital()


@router.get("/trades", response_model=list[TradeRecord])
def trades(
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    strategy: Optional[str] = Query(None),
    instrument: Optional[str] = Query(None, description="NIFTY, GOLD_PETAL, CRUDEOILM, USDINR"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return da.get_trades(target_date=date, strategy=strategy,
                         instrument=instrument, limit=limit, offset=offset)


@router.get("/trades/summary", response_model=TradeSummary)
def trades_summary(
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    instrument: Optional[str] = Query(None),
):
    return da.get_trades_summary(target_date=date, instrument=instrument)


@router.get("/trades/strategy-stats", response_model=list[StrategyStats])
def strategy_stats(
    min_trades: int = Query(1, ge=1),
    instrument: Optional[str] = Query(None),
):
    return da.get_strategy_stats(min_trades=min_trades, instrument=instrument)


@router.get("/events")
def events(
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    type: Optional[str] = Query(None, description="Comma-separated event types"),
    limit: int = Query(200, ge=1, le=2000),
):
    return da.get_events(target_date=date, event_type=type, limit=limit)


@router.get("/logs")
def logs(
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    level: Optional[str] = Query(None, description="Comma-separated: INFO,WARNING,ERROR"),
    module: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(300, ge=1, le=5000),
):
    return da.get_logs(target_date=date, level=level, module=module,
                       search=search, limit=limit)


@router.get("/config")
def config():
    return da.get_config()


@router.get("/risk", response_model=list[RiskGate])
def risk():
    return da.get_risk_gates()


@router.get("/broker", response_model=BrokerInfo)
def broker():
    return da.get_broker_info()


@router.get("/engine")
def engine_state():
    return da.get_engine_state()


@router.get("/engine/multi-asset")
def multi_asset_state():
    return da.get_multi_asset_state()


@router.get("/candles")
def candles(
    tf: Optional[str] = Query("5m", description="1m, 5m, 15m, or 1H"),
    instrument: Optional[str] = Query(None, description="NIFTY, GOLD_PETAL, CRUDEOILM, USDINR"),
):
    return da.get_candles(tf=tf, instrument=instrument)


@router.get("/signals")
def signals():
    return da.get_signals()


@router.get("/analytics")
def analytics():
    return da.get_analytics()


@router.get("/trades/strategy-details")
def strategy_details():
    return da.get_strategy_details()


@router.get("/instruments", response_model=list[InstrumentStatus])
def instruments():
    return da.get_instruments()


@router.post("/halt")
def toggle_halt(authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    return da.toggle_halt()
