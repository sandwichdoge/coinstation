"""Analysis endpoint: rule-based technicals (as-of) -> action + confidence."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..schemas import AnalysisResult, AnalyzeRequest
from ..services import analysis, indicators, market_data
from ..services import news as news_svc
from ..timeutil import now_ms, to_ms

router = APIRouter(prefix="/api", tags=["analysis"])

# A neighbour timeframe only needs enough bars for its own trend read (EMA200 +
# recent-window context); the warm-up below it covers the rest. Kept small so
# the two extra fetches add little latency to a request.
_CONTEXT_LOOKBACK = 60


async def _context_snapshot(symbol: str, interval: str, as_of_ms: int) -> dict | None:
    """Compact indicator snapshot for a neighbouring timeframe, or None if the
    data can't be fetched — multi-timeframe context is best-effort and must
    never fail the primary analysis."""
    step = market_data.interval_ms(interval)
    fetch_start = as_of_ms - (_CONTEXT_LOOKBACK + indicators.WARMUP_BARS) * step
    try:
        df = await market_data.fetch_ohlcv(
            symbol, interval, fetch_start, as_of_ms, limit=_CONTEXT_LOOKBACK
        )
    except market_data.MarketDataError:
        return None
    if df.empty:
        return None
    return indicators.latest_snapshot(indicators.compute_indicator_frame(df))


@router.post("/analyze", response_model=AnalysisResult)
async def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)) -> AnalysisResult:
    as_of_ms = to_ms(req.as_of) or now_ms()
    step = market_data.interval_ms(req.interval)
    fetch_start = as_of_ms - (req.lookback + indicators.WARMUP_BARS) * step

    try:
        df = await market_data.fetch_ohlcv(
            req.symbol, req.interval, fetch_start, as_of_ms, limit=req.lookback
        )
    except market_data.MarketDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if df.empty:
        raise HTTPException(status_code=400, detail="No market data for that symbol/date.")

    ind = indicators.compute_indicator_frame(df)
    snapshot = indicators.latest_snapshot(ind)
    items = news_svc.get_news(db, symbol=req.symbol, before_ms=as_of_ms, limit=req.news_limit)
    headlines = [news_svc.to_dict(i) for i in items]

    # Multi-timeframe context: read the trend on the next coarser ("the tide")
    # and finer ("the ripples") timeframe so the scorer can confirm alignment
    # and temper calls that fight the bigger picture. Fetched in parallel and
    # treated as best-effort — a missing neighbour just drops that term.
    higher_int, lower_int = market_data.neighbor_intervals(req.interval)
    higher_snap, lower_snap = await asyncio.gather(
        _context_snapshot(req.symbol, higher_int, as_of_ms) if higher_int else _none(),
        _context_snapshot(req.symbol, lower_int, as_of_ms) if lower_int else _none(),
    )
    context = {
        "higher_interval": higher_int, "higher": higher_snap,
        "lower_interval": lower_int, "lower": lower_snap,
    }

    return await analysis.analyze(
        symbol=req.symbol.upper(),
        interval=req.interval,
        as_of_ms=as_of_ms,
        snapshot=snapshot,
        headlines=headlines,
        context=context,
    )


async def _none() -> None:
    """Awaitable returning None, so the gather above can skip absent neighbours."""
    return None
