"""Analysis endpoint: rule-based technicals (as-of) -> action + confidence."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..schemas import AnalysisResult, AnalyzeRequest
from ..services import analysis, indicators, market_data
from ..services import news as news_svc
from ..timeutil import now_ms, to_ms

router = APIRouter(prefix="/api", tags=["analysis"])


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

    return await analysis.analyze(
        symbol=req.symbol.upper(),
        interval=req.interval,
        as_of_ms=as_of_ms,
        snapshot=snapshot,
        headlines=headlines,
    )
