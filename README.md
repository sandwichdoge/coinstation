# CoinStation

A Binance-style crypto price-action web app with technical indicators, news
aggregation, rule-based analysis, and a **date-aware backtester**.

- 📈 **Live & historical candlestick charts** (TradingView lightweight-charts) with volume, EMA 20/50/200, Bollinger Bands, **RSI** and **MACD** panes — all time-synced.
- 📰 **News aggregation** from RSS, stored in a timestamped archive.
- 🤖 **Rule-based analysis** — a transparent, auditable scorer turns the technical snapshot into an `action` + 0–100 `confidence` + rationale. No API keys required; every signal that fires is recorded in the rationale.
- ⏪ **Backtesting** with a real *date mechanism* — every data path (prices, indicators, news, analysis) is anchored to an `as_of` timestamp, so you can replay a strategy "as if it were that date."

> ⚠️ Educational tool, **not financial advice**.

---

## Why this stack

| Layer | Choice | Rationale |
|---|---|---|
| Frontend | **TypeScript** + React + Vite + lightweight-charts | Requirement #1; lightweight-charts is the de-facto Binance-style charting lib. |
| Backend | **Python** + FastAPI | Requirement #2 — "ease of integration." Python has the richest ecosystem for *every* other requirement: market data, pandas TA math, `feedparser`, dataframe backtesting. |
| Data | Binance public REST (via `httpx`) | Exact `startTime`/`endTime` control — essential for backtesting. Wrapped behind a provider seam so other exchanges can be added. |
| Storage | SQLite (SQLAlchemy) | Zero-config news archive; swap `DATABASE_URL` for Postgres in prod. |
| Analysis | Transparent rule-based scorer | No external API or key; deterministic and auditable. |

---

## Quick start

### Option A — Docker (one command)

```bash
docker compose up --build
```

- App: **http://localhost:8080**
- API: **http://localhost:8000/api/health** · interactive docs at `/docs`

### Option B — Local dev

**Backend**
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend** (new terminal)
```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173 (proxies /api -> :8000)
```

---

## Configuration (backend `.env`)

| Var | Default | Notes |
|---|---|---|
| `BINANCE_BASE_URL` | `https://api.binance.com` | Market data source. |
| `DATABASE_URL` | `sqlite:///./coinstation.db` | e.g. `postgresql+psycopg://…` in prod. |
| `RSS_FEEDS` | CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine | Comma-separated. |
| `CORS_ORIGINS` | `localhost:5173` | Comma-separated allowed origins. |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status + analysis engine. |
| GET | `/api/coins` | Popular symbols for the picker. |
| GET | `/api/intervals` | Supported candle intervals. |
| GET | `/api/klines` | OHLCV + indicators. `?symbol&interval&start&end&limit&warmup&with_indicators` |
| GET | `/api/news` | Archive query. `?symbol&before&after&limit` (`before` = as-of cutoff). |
| POST | `/api/news/ingest` | Pull RSS feeds into the archive now. |
| GET | `/api/news/stats` | Item count. |
| POST | `/api/analyze` | `{symbol, interval, as_of?}` → action + confidence + rationale. |
| POST | `/api/backtest` | `{symbol, interval, start, end?, strategy, …}` → metrics + equity curve + trades. |

Backtest strategies: `macd`, `rsi`, `ema_cross`, `rules`, `buy_hold`.

---

## How the date mechanism works (and the one honest caveat)

Everything internal is epoch-millisecond UTC, and an `as_of` instant threads through every layer:

- **Prices** — Binance klines fetched with explicit `startTime`/`endTime`. Backtests/indicators fetch ~250 *warm-up* bars **before** the window so EMA200/MACD/RSI are already valid at the first traded bar.
- **News** — queried with `published_at <= as_of`, so a backtest never "sees the future."
- **Analysis** — `analyze` accepts `as_of`; the `rules` backtest strategy calls the *same* function at each evaluation bar. Backtests score technicals only.

**Caveat on historical news:** public RSS feeds only expose their *latest* entries — you cannot retrieve arbitrary old news from them. So the app **accumulates** news into a timestamped archive over time, and historical depth equals what has been ingested. A backtest dated before ingestion began will simply find no news (technicals still run). The schema is ready for a paid historical-news API to backfill older rows. **This is the only part of requirement #5 that can't be fully satisfied from free RSS alone**, and it's handled honestly rather than faked.

---

## Project structure

```
coinstation/
├── docker-compose.yml
├── backend/                 # FastAPI + Python
│   ├── app/
│   │   ├── main.py          # app, CORS, startup news ingest
│   │   ├── config.py        # env settings
│   │   ├── db.py · models.py # SQLAlchemy + NewsItem archive
│   │   ├── schemas.py · timeutil.py
│   │   ├── routers/         # market · news · analysis · backtest
│   │   └── services/
│   │       ├── market_data.py   # Binance, date-aware klines
│   │       ├── indicators.py    # RSI/MACD/EMA/Bollinger/Volume (pandas)
│   │       ├── news.py          # RSS ingest + as-of query
│   │       ├── analysis.py      # rule-based scorer
│   │       └── backtest.py      # date-aware engine
│   └── requirements.txt · Dockerfile
└── frontend/                # React + TypeScript + Vite
    └── src/
        ├── App.tsx · api/client.ts · types.ts
        └── components/
            ├── ChartStack.tsx   # synced price/RSI/MACD panes
            ├── EquityChart.tsx
            ├── Controls.tsx · NewsPanel.tsx
            ├── AnalysisPanel.tsx · BacktestPanel.tsx
```

---

## Extending

- **News aggregation** (designed for growth): add feeds via `RSS_FEEDS`; schedule `POST /api/news/ingest` (cron) to grow the archive; per-item symbol tagging already exists.
- **Historical news**: drop a provider into `services/news.py` to backfill `NewsItem` rows with real publish dates → unlocks deep news-aware backtests.
- **More exchanges**: implement another provider behind the `market_data` seam.
- **Smarter analysis**: extend the scorer in `services/analysis.py` with new signals, or weight bigger/smaller timeframes into a final verdict.
- **More strategies / shorting / position sizing** in `services/backtest.py`.
