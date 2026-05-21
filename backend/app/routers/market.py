"""Market data endpoints: coin list, intervals, and klines (with indicators)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..services import indicators, market_data
from ..timeutil import to_ms

router = APIRouter(prefix="/api", tags=["market"])

ORDERED_INTERVALS = list(market_data.INTERVAL_MS.keys()) + ["1M"]


@router.get("/coins")
def coins() -> dict:
    return {"coins": market_data.get_symbols()}


@router.get("/intervals")
def intervals() -> dict:
    return {"intervals": ORDERED_INTERVALS}


@router.get("/klines")
async def klines(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1h"),
    start: str | None = Query(None, description="ISO-8601 start; omit for most-recent."),
    end: str | None = Query(None, description="ISO-8601 end; defaults to now."),
    limit: int = Query(500, ge=1, le=1000),
    with_indicators: bool = Query(True),
    warmup: bool = Query(
        False,
        description="When a start date is given, also fetch lead-in bars so "
        "indicators are valid at the window's left edge.",
    ),
) -> dict:
    start_ms, end_ms = to_ms(start), to_ms(end)
    fetch_start = start_ms
    if warmup and start_ms is not None:
        fetch_start = start_ms - indicators.WARMUP_BARS * market_data.interval_ms(interval)

    try:
        df = await market_data.fetch_ohlcv(symbol, interval, fetch_start, end_ms, limit)
    except market_data.MarketDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    ind = indicators.compute_indicator_frame(df) if (with_indicators and not df.empty) else None

    # If we fetched warm-up bars, trim the *output* back to the requested window.
    if warmup and start_ms is not None and not df.empty:
        keep = (df["time"] * 1000) >= start_ms
        df_view = df[keep].reset_index(drop=True)
        ind_view = ind[keep].reset_index(drop=True) if ind is not None else None
    else:
        df_view, ind_view = df, ind

    resp: dict = {
        "symbol": symbol.upper(),
        "interval": interval,
        "candles": market_data.to_candles(df_view),
    }
    if ind_view is not None:
        resp["indicators"] = indicators.build_payload(ind_view)
        resp["snapshot"] = indicators.latest_snapshot(ind)  # snapshot uses full history
    else:
        resp["indicators"] = {}
        resp["snapshot"] = {}
    return resp
