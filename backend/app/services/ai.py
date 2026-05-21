"""AI analysis: combine the technical snapshot with recent news into a
recommendation (action + 0-100 confidence + rationale).

Two paths share one output shape:
  * OpenAI when OPENAI_API_KEY is set.
  * A transparent rule-based scorer otherwise (or if the API call fails), so the
    product — including the AI backtest strategy — always works.
"""
from __future__ import annotations

import json

from ..config import settings
from ..schemas import AnalysisResult
from ..timeutil import ms_to_iso

VALID_ACTIONS = ("strong_buy", "buy", "hold", "sell", "strong_sell")


def _horizon(interval: str) -> str:
    return {
        "1m": "intraday (minutes-hours)", "3m": "intraday (minutes-hours)",
        "5m": "intraday (hours)", "15m": "intraday (hours)", "30m": "short-term (1-2 days)",
        "1h": "short-term (1-3 days)", "2h": "short-term (2-4 days)", "4h": "swing (3-7 days)",
        "6h": "swing (1-2 weeks)", "8h": "swing (1-2 weeks)", "12h": "swing (1-3 weeks)",
        "1d": "position (weeks)", "3d": "position (weeks-months)", "1w": "position (months)",
        "1M": "long-term (months+)",
    }.get(interval, "swing (days-weeks)")


# ---- rule-based fallback --------------------------------------------------

def _heuristic(snapshot: dict, headlines: list[dict], interval: str) -> dict:
    score = 0
    factors: list[str] = []
    risks: list[str] = []

    rsi = snapshot.get("rsi")
    if rsi is not None:
        if rsi <= 30:
            score += 2; factors.append(f"RSI {rsi:.0f} is oversold (bullish reversal potential)")
        elif rsi >= 70:
            score -= 2; factors.append(f"RSI {rsi:.0f} is overbought (pullback risk)")
            risks.append("Overbought RSI can persist in strong trends")
        elif rsi < 45:
            score += 1
        elif rsi > 55:
            score -= 1

    hist = snapshot.get("macd_hist")
    if hist is not None:
        if hist > 0:
            score += 1; factors.append("MACD histogram positive (upward momentum)")
        else:
            score -= 1; factors.append("MACD histogram negative (downward momentum)")

    ema50, ema200, price = snapshot.get("ema50"), snapshot.get("ema200"), snapshot.get("price")
    if ema50 is not None and ema200 is not None:
        if ema50 > ema200:
            score += 1; factors.append("EMA50 above EMA200 (uptrend structure)")
        else:
            score -= 1; factors.append("EMA50 below EMA200 (downtrend structure)")
    if price is not None and ema200 is not None:
        score += 1 if price >= ema200 else -1

    if score >= 3:
        action, conf = "strong_buy", 85
    elif score >= 1:
        action, conf = "buy", 65
    elif score <= -3:
        action, conf = "strong_sell", 85
    elif score <= -1:
        action, conf = "sell", 65
    else:
        action, conf = "hold", 55
    confidence = min(90.0, conf + abs(score) * 2.0)

    chg = snapshot.get("window_change_pct")
    chg_txt = f" Price changed {chg:+.1f}% over the window." if chg is not None else ""
    news_txt = (
        f" {len(headlines)} recent headline(s) were available but sentiment is not"
        " modelled in rule-based mode."
        if headlines
        else " No news in the archive for this period."
    )
    summary = f"Rule-based signal: {action.replace('_', ' ').upper()} (score {score:+d})."
    rationale = (
        "Derived from RSI, MACD momentum and EMA trend structure."
        + chg_txt
        + news_txt
        + " Set OPENAI_API_KEY to enable news-aware AI analysis."
    )
    if not risks:
        risks.append("Indicator-only signal; ignores fundamentals and news sentiment.")
    return {
        "action": action,
        "confidence": round(confidence, 1),
        "horizon": _horizon(interval),
        "summary": summary,
        "rationale": rationale,
        "key_factors": factors or ["Insufficient indicator data"],
        "risks": risks,
        "source": "rule-based",
        "model": None,
    }


# ---- OpenAI path ----------------------------------------------------------

_SYSTEM = (
    "You are a disciplined cryptocurrency trading analyst. You are given a "
    "technical-indicator snapshot and recent news headlines for one asset, all "
    "as of a specific timestamp. Reason ONLY from the information provided and "
    "general market principles — do not use any knowledge of events after the "
    "given timestamp. Weigh technicals and news together. Respond with a single "
    "JSON object and nothing else, using exactly these keys: "
    '{"action": one of ["strong_buy","buy","hold","sell","strong_sell"], '
    '"confidence": number 0-100, "horizon": short string, "summary": one sentence, '
    '"rationale": 2-4 sentences, "key_factors": string[], "risks": string[]}. '
    "Confidence reflects conviction, not predicted return."
)


def _normalize(data: dict, interval: str) -> dict:
    action = str(data.get("action", "hold")).lower().strip()
    if action not in VALID_ACTIONS:
        action = "hold"
    try:
        confidence = max(0.0, min(100.0, float(data.get("confidence", 50))))
    except (TypeError, ValueError):
        confidence = 50.0

    def _strlist(v) -> list[str]:
        if isinstance(v, list):
            return [str(x) for x in v][:8]
        return [str(v)] if v else []

    return {
        "action": action,
        "confidence": round(confidence, 1),
        "horizon": str(data.get("horizon") or _horizon(interval)),
        "summary": str(data.get("summary", "")).strip() or "AI analysis.",
        "rationale": str(data.get("rationale", "")).strip(),
        "key_factors": _strlist(data.get("key_factors")),
        "risks": _strlist(data.get("risks")),
        "source": "openai",
        "model": settings.openai_model,
    }


async def _openai(snapshot: dict, headlines: list[dict], symbol: str, interval: str, as_of_iso: str) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
    user_payload = {
        "symbol": symbol,
        "interval": interval,
        "as_of": as_of_iso,
        "technical_snapshot": snapshot,
        "news_headlines": [
            {"title": h["title"], "source": h["source"], "published_at": h["published_at"]}
            for h in headlines
        ],
    }
    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": json.dumps(user_payload, default=str)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = resp.choices[0].message.content or "{}"
    return _normalize(json.loads(content), interval)


# ---- public entrypoint ----------------------------------------------------

async def analyze(
    *,
    symbol: str,
    interval: str,
    as_of_ms: int,
    snapshot: dict,
    headlines: list[dict],
) -> AnalysisResult:
    as_of_iso = ms_to_iso(as_of_ms)
    if settings.ai_enabled:
        try:
            core = await _openai(snapshot, headlines, symbol, interval, as_of_iso)
        except Exception as exc:  # any SDK/network/parse failure → graceful fallback
            core = _heuristic(snapshot, headlines, interval)
            core["summary"] = f"[AI unavailable: {exc}] " + core["summary"]
    else:
        core = _heuristic(snapshot, headlines, interval)

    return AnalysisResult(
        symbol=symbol,
        as_of=as_of_iso,
        interval=interval,
        technical_snapshot=snapshot,
        news_considered=len(headlines),
        headlines=[
            {"title": h["title"], "source": h["source"], "published_at": h["published_at"], "link": h.get("link")}
            for h in headlines
        ],
        **core,
    )
