"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    as_of: str | None = Field(
        default=None, description="ISO-8601 instant to analyze as-of; defaults to now."
    )
    lookback: int = Field(default=300, ge=50, le=1000, description="Candles of history.")
    news_limit: int = Field(default=12, ge=0, le=50)


class AnalysisResult(BaseModel):
    symbol: str
    as_of: str
    interval: str
    action: str  # strong_buy | buy | hold | sell | strong_sell
    confidence: float  # 0..100
    horizon: str
    summary: str
    rationale: str
    key_factors: list[str] = []
    risks: list[str] = []
    technical_snapshot: dict = {}
    news_considered: int = 0
    headlines: list[dict] = []
    source: str = "rule-based"  # "openai" | "rule-based"
    model: str | None = None


class BacktestRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1d"
    start: str = Field(description="ISO-8601 start date (inclusive).")
    end: str | None = Field(default=None, description="ISO-8601 end date; defaults to now.")
    strategy: str = Field(default="macd", description="rsi | macd | ema_cross | ai | buy_hold")
    initial_capital: float = Field(default=10_000.0, gt=0)
    fee_pct: float = Field(default=0.1, ge=0, le=5, description="Per-trade fee, percent.")
    rsi_buy: float = Field(default=30.0, ge=1, le=99)
    rsi_sell: float = Field(default=70.0, ge=1, le=99)
    ai_rebalance_every: int = Field(default=14, ge=1, description="Bars between AI evals.")
    ai_confidence_threshold: float = Field(default=60.0, ge=0, le=100)
