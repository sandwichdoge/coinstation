"""Market data provider. Currently Binance public REST, wrapped so other
exchanges can be swapped in. Date control is explicit (startTime/endTime in ms)
because precise historical ranges are the backbone of backtesting."""
from __future__ import annotations

import httpx
import pandas as pd

from ..config import settings
from ..timeutil import now_ms

BINANCE_MAX_LIMIT = 1000

# Milliseconds per interval (1M handled separately — calendar months vary).
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}
VALID_INTERVALS = set(INTERVAL_MS) | {"1M"}

# Ordered, fine -> coarse, including the calendar-month bar (~30 days) so the
# multi-timeframe context can reach the very top of the ladder.
INTERVAL_LADDER: list[tuple[str, int]] = sorted(
    {**INTERVAL_MS, "1M": 2_592_000_000}.items(), key=lambda kv: kv[1]
)

# A "higher" / "lower" timeframe is only useful if it is materially coarser /
# finer than the base — stepping 1h -> 2h tells you almost nothing new. We pick
# the nearest neighbour at least this many times the base interval away, so the
# context genuinely zooms out (the tide) and in (the ripples).
NEIGHBOR_RATIO = 3.5


def neighbor_intervals(interval: str) -> tuple[str | None, str | None]:
    """Return ``(higher, lower)`` timeframes flanking `interval` on the ladder.

    `higher` is the finest interval at least ``NEIGHBOR_RATIO``x coarser than the
    base; `lower` the coarsest interval at least that much finer. Either is
    ``None`` at the ends of the ladder (no 1m has a meaningful lower; 1M has no
    higher)."""
    base = dict(INTERVAL_LADDER).get(interval)
    if base is None:
        return None, None
    higher = next((name for name, ms in INTERVAL_LADDER if ms >= base * NEIGHBOR_RATIO), None)
    lower = next(
        (name for name, ms in reversed(INTERVAL_LADDER) if ms * NEIGHBOR_RATIO <= base), None
    )
    return higher, lower

# Curated popular USDT pairs for the picker. Users may request any valid symbol;
# this is just a convenient default list.
POPULAR_SYMBOLS: list[dict[str, str]] = [
    {"symbol": "BTCUSDT", "base": "BTC", "name": "Bitcoin"},
    {"symbol": "ETHUSDT", "base": "ETH", "name": "Ethereum"},
    {"symbol": "BNBUSDT", "base": "BNB", "name": "BNB"},
    {"symbol": "SOLUSDT", "base": "SOL", "name": "Solana"},
    {"symbol": "XRPUSDT", "base": "XRP", "name": "XRP"},
    {"symbol": "ADAUSDT", "base": "ADA", "name": "Cardano"},
    {"symbol": "DOGEUSDT", "base": "DOGE", "name": "Dogecoin"},
    {"symbol": "AVAXUSDT", "base": "AVAX", "name": "Avalanche"},
    {"symbol": "DOTUSDT", "base": "DOT", "name": "Polkadot"},
    {"symbol": "LINKUSDT", "base": "LINK", "name": "Chainlink"},
    {"symbol": "LTCUSDT", "base": "LTC", "name": "Litecoin"},
    {"symbol": "TRXUSDT", "base": "TRX", "name": "TRON"},
    {"symbol": "ATOMUSDT", "base": "ATOM", "name": "Cosmos"},
    {"symbol": "UNIUSDT", "base": "UNI", "name": "Uniswap"},
    {"symbol": "ETCUSDT", "base": "ETC", "name": "Ethereum Classic"},
    {"symbol": "XLMUSDT", "base": "XLM", "name": "Stellar"},
    {"symbol": "NEARUSDT", "base": "NEAR", "name": "NEAR Protocol"},
    {"symbol": "APTUSDT", "base": "APT", "name": "Aptos"},
    {"symbol": "ARBUSDT", "base": "ARB", "name": "Arbitrum"},
    {"symbol": "OPUSDT", "base": "OP", "name": "Optimism"},
    {"symbol": "FILUSDT", "base": "FIL", "name": "Filecoin"},
    {"symbol": "SUIUSDT", "base": "SUI", "name": "Sui"},
    {"symbol": "SHIBUSDT", "base": "SHIB", "name": "Shiba Inu"},
    {"symbol": "PEPEUSDT", "base": "PEPE", "name": "Pepe"},
    {"symbol": "TONUSDT", "base": "TON", "name": "Toncoin"},
    {"symbol": "ICPUSDT", "base": "ICP", "name": "Internet Computer"},
    {"symbol": "AAVEUSDT", "base": "AAVE", "name": "Aave"},
    {"symbol": "BCHUSDT", "base": "BCH", "name": "Bitcoin Cash"},
    {"symbol": "ZECUSDT", "base": "ZEC", "name": "Zcash"},
]

OHLCV_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


class MarketDataError(Exception):
    """Raised on provider/HTTP errors with a human-readable message."""


def get_symbols() -> list[dict[str, str]]:
    return POPULAR_SYMBOLS


def interval_ms(interval: str) -> int:
    return INTERVAL_MS.get(interval, INTERVAL_MS["1d"])


async def _get_klines(client: httpx.AsyncClient, params: dict) -> list:
    try:
        resp = await client.get("/api/v3/klines", params=params)
    except httpx.HTTPError as exc:  # network/timeout
        raise MarketDataError(f"Market data request failed: {exc}") from exc
    if resp.status_code != 200:
        try:
            msg = resp.json().get("msg", resp.text)
        except Exception:
            msg = resp.text
        raise MarketDataError(f"Binance error {resp.status_code}: {msg}")
    return resp.json()


def _to_df(rows: list, tail: int | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbav", "tqav", "ignore",
        ],
    )
    df["time"] = (df["open_time"].astype("int64") // 1000)  # seconds, UTC
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (
        df[OHLCV_COLUMNS]
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )
    if tail:
        df = df.tail(tail).reset_index(drop=True)
    return df


def to_candles(df: pd.DataFrame) -> list[dict]:
    """Serialize OHLC for lightweight-charts (volume is sent separately)."""
    return [
        {"time": int(t), "open": float(o), "high": float(h), "low": float(lo), "close": float(c)}
        for t, o, h, lo, c in zip(df["time"], df["open"], df["high"], df["low"], df["close"])
        if pd.notna(o)
    ]


async def fetch_ohlcv(
    symbol: str,
    interval: str = "1h",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Return an OHLCV DataFrame [time, open, high, low, close, volume].

    - No start_ms: most recent `limit` candles (optionally ending at end_ms).
    - With start_ms: every candle in [start_ms, end_ms], paginated as needed.
    """
    symbol = symbol.upper().strip()
    if interval not in VALID_INTERVALS:
        raise MarketDataError(
            f"Unsupported interval '{interval}'. Valid: {sorted(VALID_INTERVALS)}"
        )

    rows: list = []
    async with httpx.AsyncClient(
        base_url=settings.binance_base_url, timeout=20.0
    ) as client:
        if start_ms is None:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, BINANCE_MAX_LIMIT),
            }
            if end_ms is not None:
                params["endTime"] = end_ms
            rows = await _get_klines(client, params)
            return _to_df(rows, tail=limit)

        hard_end = end_ms if end_ms is not None else now_ms()
        step = interval_ms(interval)
        cursor = start_ms
        # Cap pages so a tiny interval over a huge range can't loop forever.
        for _ in range(500):
            if cursor > hard_end:
                break
            batch = await _get_klines(
                client,
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": hard_end,
                    "limit": BINANCE_MAX_LIMIT,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < BINANCE_MAX_LIMIT:
                break
            next_cursor = int(batch[-1][0]) + step
            if next_cursor <= cursor:
                break
            cursor = next_cursor

    return _to_df(rows, tail=None)
