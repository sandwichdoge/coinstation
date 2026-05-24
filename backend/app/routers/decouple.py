"""Decouple meter endpoint: how independently a coin trades from BTC, the
market's king / global-sentiment proxy. As-of aware like the rest of the app."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query

from ..services import decouple as decouple_svc
from ..services import market_data
from ..timeutil import ms_to_iso, now_ms, to_ms

router = APIRouter(prefix="/api", tags=["decouple"])

REFERENCE_SYMBOL = "BTCUSDT"


@router.get("/decouple")
async def decouple(
    symbol: str = Query("ETHUSDT"),
    interval: str = Query("1h"),
    as_of: str | None = Query(None, description="ISO-8601 instant; defaults to now."),
    window: int = Query(
        decouple_svc.DECOUPLE_WINDOW, ge=decouple_svc.MIN_SAMPLES, le=200,
        description="Candles of returns the regression reads.",
    ),
) -> dict:
    symbol = symbol.upper().strip()
    as_of_ms = to_ms(as_of) or now_ms()
    base = {
        "symbol": symbol, "reference": REFERENCE_SYMBOL, "interval": interval,
        "as_of": ms_to_iso(as_of_ms), "window": window,
    }

    # BTC against itself is the reference frame — coupled by definition.
    if symbol == REFERENCE_SYMBOL:
        return {**base, "is_reference": True, "meter": {
            "available": False,
            "reason": "BTC is the reference; it can't decouple from itself.",
        }}

    # Fetch enough trailing candles for `window` returns (+1 for the first diff,
    # a little slack for any bars that don't align between the two symbols).
    step = market_data.interval_ms(interval)
    pad = window + 5
    start_ms = as_of_ms - pad * step
    try:
        coin_df, btc_df = await asyncio.gather(
            market_data.fetch_ohlcv(symbol, interval, start_ms, as_of_ms, limit=pad),
            market_data.fetch_ohlcv(REFERENCE_SYMBOL, interval, start_ms, as_of_ms, limit=pad),
        )
    except market_data.MarketDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    meter = decouple_svc.decouple_meter(coin_df, btc_df, window=window)
    return {**base, "is_reference": False, "meter": meter}
