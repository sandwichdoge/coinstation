"""Application configuration, loaded from environment / .env file."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Market data ---
    binance_base_url: str = "https://api.binance.com"

    # --- Database ---
    database_url: str = "sqlite:///./coinstation.db"

    # --- OpenAI ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None

    # --- News feeds (comma-separated) ---
    rss_feeds: str = (
        "https://www.coindesk.com/arc/outboundfeeds/rss/,"
        "https://cointelegraph.com/rss,"
        "https://decrypt.co/feed,"
        "https://bitcoinmagazine.com/feed"
    )

    # --- CORS (comma-separated) ---
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def rss_feeds_list(self) -> list[str]:
        return [f.strip() for f in self.rss_feeds.split(",") if f.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def ai_enabled(self) -> bool:
        return bool(self.openai_api_key)


settings = Settings()
