"""Backtest endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import BacktestRequest
from ..services import backtest as bt
from ..services.market_data import MarketDataError

router = APIRouter(prefix="/api", tags=["backtest"])


@router.post("/backtest")
async def run_backtest(req: BacktestRequest) -> dict:
    try:
        return await bt.run_backtest(req)
    except (ValueError, MarketDataError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
