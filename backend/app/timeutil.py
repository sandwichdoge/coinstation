"""Date/time helpers. Everything internal is epoch-millisecond UTC so that the
`as_of` date threads cleanly through market data, news and backtests."""
from __future__ import annotations

from datetime import datetime, timezone

from dateutil import parser as dtparser


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def to_ms(value: str | int | float | datetime | None) -> int | None:
    """Parse an ISO-8601 string / epoch / datetime into epoch milliseconds (UTC).
    Returns None for empty input."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return int(v if v > 1e12 else v * 1000)  # heuristically treat as ms vs s
    dt = value if isinstance(value, datetime) else dtparser.parse(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
