"""ORM models. The NewsItem table is the historical archive that makes
news-aware backtesting possible: we ingest RSS over time and later query
items by `published_at <= as_of`."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NewsItem(Base):
    __tablename__ = "news_items"
    __table_args__ = (
        UniqueConstraint("link", name="uq_news_link"),
        Index("ix_news_published_at", "published_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(160), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    link: Mapped[str] = mapped_column(Text)
    # Comma-separated detected tickers (e.g. "BTC,ETH"); empty = general/market-wide.
    symbols: Mapped[str] = mapped_column(String(256), default="")
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
