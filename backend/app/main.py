"""CoinStation FastAPI application."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import SessionLocal, init_db
from .routers import analysis, backtest, market, news
from .services import news as news_svc


async def _startup_ingest() -> None:
    """Best-effort initial news pull so analysis has data on first run."""
    try:
        db = SessionLocal()
        try:
            await news_svc.ingest(db)
        finally:
            db.close()
    except Exception:
        pass  # never block startup on flaky feeds


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    asyncio.create_task(_startup_ingest())
    yield


app = FastAPI(title="CoinStation API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "ai_enabled": settings.ai_enabled,
        "model": settings.openai_model if settings.ai_enabled else None,
    }


app.include_router(market.router)
app.include_router(news.router)
app.include_router(analysis.router)
app.include_router(backtest.router)
