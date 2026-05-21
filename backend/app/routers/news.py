"""News endpoints: query the archive (with an as-of cutoff), trigger ingestion."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..services import news as news_svc
from ..timeutil import to_ms

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("")
def list_news(
    symbol: str | None = Query(None),
    before: str | None = Query(None, description="ISO-8601 as-of cutoff."),
    after: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict:
    items = news_svc.get_news(
        db, symbol=symbol, before_ms=to_ms(before), after_ms=to_ms(after), limit=limit
    )
    return {"count": len(items), "items": [news_svc.to_dict(i) for i in items]}


@router.post("/ingest")
async def ingest_news(db: Session = Depends(get_db)) -> dict:
    return await news_svc.ingest(db)


@router.get("/stats")
def stats(db: Session = Depends(get_db)) -> dict:
    return {"total": news_svc.count(db)}
