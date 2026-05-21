"""News aggregation: fetch RSS feeds, tag items with the coins they mention,
and store them in a timestamped archive. Backtests read this archive with an
`as_of` cutoff so they only ever see news that existed at that point in time.

Limitation worth knowing: RSS feeds expose only their most recent entries, so
historical depth equals what has been ingested over time. The schema is ready
for a paid historical-news API to backfill older rows."""
from __future__ import annotations

import asyncio
import calendar
import re
from datetime import datetime, timezone

import feedparser
import httpx
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import NewsItem
from .market_data import POPULAR_SYMBOLS

_HTML_RE = re.compile(r"<[^>]+>")
_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD", "BTC", "ETH", "BNB")


def _build_keyword_map() -> dict[str, set[str]]:
    """base ticker -> set of lowercase keywords that imply it."""
    mapping: dict[str, set[str]] = {}
    for s in POPULAR_SYMBOLS:
        mapping[s["base"]] = {s["base"].lower(), s["name"].lower()}
    mapping.setdefault("ETH", set()).add("ether")
    return mapping


_KEYWORD_MAP = _build_keyword_map()


def base_of(symbol: str) -> str:
    s = symbol.upper().strip()
    for entry in POPULAR_SYMBOLS:
        if entry["symbol"] == s:
            return entry["base"]
    for q in _QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def _clean(text: str | None) -> str:
    return _HTML_RE.sub("", text or "").strip()


def _detect_symbols(text: str) -> list[str]:
    low = text.lower()
    found: list[str] = []
    for base, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", low):
                found.append(base)
                break
    return found


def _published_dt(entry) -> datetime:
    """feedparser's *_parsed structs are UTC; convert with timegm. Returns naive UTC."""
    for key in ("published_parsed", "updated_parsed"):
        val = entry.get(key)
        if val:
            return datetime.fromtimestamp(calendar.timegm(val), tz=timezone.utc).replace(tzinfo=None)
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)


async def _fetch_feed(client: httpx.AsyncClient, url: str) -> bytes | None:
    try:
        resp = await client.get(url, follow_redirects=True)
        return resp.content if resp.status_code == 200 else None
    except httpx.HTTPError:
        return None


async def ingest(db: Session, feeds: list[str] | None = None) -> dict:
    """Fetch all feeds, parse, dedupe by link, and store new items."""
    feeds = feeds or settings.rss_feeds_list
    async with httpx.AsyncClient(
        headers={"User-Agent": "CoinStation/0.1 (+https://localhost)"}, timeout=20.0
    ) as client:
        contents = await asyncio.gather(*[_fetch_feed(client, u) for u in feeds])

    parsed: dict[str, dict] = {}  # link -> item (dedupes within this batch)
    feeds_ok = 0
    for content in contents:
        if not content:
            continue
        feeds_ok += 1
        fp = feedparser.parse(content)
        source = _clean(fp.feed.get("title")) or "RSS"
        for entry in fp.entries:
            link = (entry.get("link") or "").strip()
            if not link:
                continue
            title = _clean(entry.get("title"))
            summary = _clean(entry.get("summary") or entry.get("description"))[:1200]
            parsed[link] = {
                "source": source[:160],
                "title": title,
                "summary": summary,
                "link": link,
                "symbols": ",".join(_detect_symbols(f"{title} {summary}")),
                "published_at": _published_dt(entry),
            }

    if not parsed:
        return {"feeds": len(feeds), "feeds_ok": feeds_ok, "fetched": 0, "added": 0}

    existing = set(
        db.scalars(select(NewsItem.link).where(NewsItem.link.in_(list(parsed)))).all()
    )
    added = 0
    for link, item in parsed.items():
        if link in existing:
            continue
        db.add(NewsItem(**item))
        added += 1
    db.commit()
    return {
        "feeds": len(feeds),
        "feeds_ok": feeds_ok,
        "fetched": len(parsed),
        "added": added,
    }


def get_news(
    db: Session,
    symbol: str | None = None,
    before_ms: int | None = None,
    after_ms: int | None = None,
    limit: int = 20,
) -> list[NewsItem]:
    """Query the archive. `before_ms` is the as-of cutoff used by backtests."""
    stmt = select(NewsItem)
    if before_ms is not None:
        stmt = stmt.where(NewsItem.published_at <= _naive_utc(before_ms))
    if after_ms is not None:
        stmt = stmt.where(NewsItem.published_at >= _naive_utc(after_ms))
    if symbol:
        base = base_of(symbol)
        # coin-specific items plus general/market-wide items (empty symbols)
        stmt = stmt.where(or_(NewsItem.symbols.like(f"%{base}%"), NewsItem.symbols == ""))
    stmt = stmt.order_by(NewsItem.published_at.desc()).limit(limit)
    return list(db.scalars(stmt).all())


def to_dict(n: NewsItem) -> dict:
    return {
        "id": n.id,
        "source": n.source,
        "title": n.title,
        "summary": n.summary,
        "link": n.link,
        "symbols": [s for s in (n.symbols or "").split(",") if s],
        "published_at": n.published_at.replace(tzinfo=timezone.utc).isoformat(),
    }


def count(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(NewsItem)) or 0
