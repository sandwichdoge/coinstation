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
    """Transparent, internally-consistent scorer.

    Philosophy: trend-following with momentum confirmation. Mean-reversion only
    fires at RSI/Bollinger *extremes* — in the mid-range RSI is read as a mild
    momentum bias *with* the trend, not against it. Each contributor is recorded
    so the rationale is auditable, and confidence is discounted when the signals
    disagree (e.g. a long-term downtrend that price has started to reclaim).
    """
    factors: list[str] = []
    risks: list[str] = []
    # (weight, human-readable factor) for every signal that actually fires.
    contrib: list[tuple[float, str]] = []

    def add(w: float, text: str) -> None:
        contrib.append((w, text))
        if abs(w) >= 0.5:
            factors.append(text)

    def scaled(magnitude_pct: float, full_at: float, weight: float) -> float:
        """Signed fraction of `weight`, growing linearly with |magnitude_pct| and
        saturating at ±weight once it reaches `full_at`. This is the core accuracy
        fix: a reading 1% past a threshold should not score as hard as one 10%
        past it. Marginal setups now produce small scores (and, via the confidence
        formula, modest conviction) instead of full-weight, overconfident calls."""
        return weight * max(-1.0, min(1.0, magnitude_pct / full_at))

    price = snapshot.get("price")
    ema20, ema50, ema200 = snapshot.get("ema20"), snapshot.get("ema50"), snapshot.get("ema200")
    rsi = snapshot.get("rsi")
    macd_line, hist = snapshot.get("macd"), snapshot.get("macd_hist")
    bb_upper, bb_lower = snapshot.get("bb_upper"), snapshot.get("bb_lower")
    chg = snapshot.get("window_change_pct")
    volume, vol_ma = snapshot.get("volume"), snapshot.get("vol_ma")
    cmf = snapshot.get("cmf")
    obv_trend = snapshot.get("obv_trend_pct")
    recent_chg = snapshot.get("recent_change_pct")
    vol_contraction = snapshot.get("vol_contraction")
    price_pos = snapshot.get("price_position")

    # --- Trend regime (medium term): the dominant, but laggy, signal. Scaled by
    #     how far the EMAs have separated — a fresh, barely-crossed regime is far
    #     weaker evidence than a wide, established one. Ties stay neutral. ---
    if ema50 is not None and ema200 is not None and ema200:
        sep_pct = (ema50 - ema200) / ema200 * 100
        w = scaled(sep_pct, full_at=4.0, weight=1.0)
        if abs(w) >= 0.05:
            add(w, "EMA50 above EMA200 (uptrend regime)" if w > 0
                else "EMA50 below EMA200 (downtrend regime)")

    # --- Price vs its mid-term mean: a faster read than EMA200, so we use
    #     EMA50/EMA20 here instead of double-counting the laggy EMA200. Scaled by
    #     % distance: price sitting right on the mean is near-neutral. ---
    ref = ema50 if ema50 is not None else ema20
    if price is not None and ref:
        dist_pct = (price - ref) / ref * 100
        w = scaled(dist_pct, full_at=3.0, weight=1.0)
        if abs(w) >= 0.05:
            add(w, "Price holding above EMA50 (near-term strength)" if w > 0
                else "Price below EMA50 (near-term weakness)")

    # --- Momentum: histogram direction + MACD zero-line context.
    #     A flat (zero) histogram or MACD is momentumless, not bearish. ---
    if hist is not None and price:
        # Normalise the histogram by price so the threshold is comparable across
        # assets/timeframes; a histogram a hair off zero is barely momentum.
        w = scaled(hist / price * 100, full_at=0.6, weight=1.0)
        if abs(w) >= 0.05:
            add(w, "MACD histogram positive (upward momentum)" if w > 0
                else "MACD histogram negative (downward momentum)")
    if macd_line:
        # Above/below zero says whether momentum sits in bull or bear territory.
        add(0.5 if macd_line > 0 else -0.5,
            "MACD above zero line" if macd_line > 0 else "MACD below zero line")

    # --- RSI: mean-reversion only at extremes; mild momentum bias mid-range ---
    if rsi is not None:
        if rsi <= 30:
            add(2.0, f"RSI {rsi:.0f} oversold (bullish reversal potential)")
        elif rsi >= 70:
            add(-1.0, f"RSI {rsi:.0f} overbought (pullback risk)")
            risks.append("Overbought RSI can persist in strong trends")
        elif rsi >= 55:
            add(0.5, f"RSI {rsi:.0f} firm (momentum with trend)")
        elif rsi <= 45:
            add(-0.5, f"RSI {rsi:.0f} soft (waning momentum)")

    # --- Bollinger: stretched-band mean reversion / breakout context ---
    if price is not None and bb_lower is not None and price <= bb_lower:
        add(0.5, "Price at/below lower Bollinger band (stretched low)")
    elif price is not None and bb_upper is not None and price >= bb_upper:
        add(-0.5, "Price at/above upper Bollinger band (stretched high)")
        risks.append("Price extended above the upper Bollinger band")

    # --- Swing support / resistance: the explicit "don't strong-sell the
    #     bottom" guard the trend-follower otherwise lacks. Selling into a
    #     well-tested support (or buying into resistance) is poor risk/reward —
    #     the level caps the move and a bounce/rejection is the base case. The
    #     pull grows the closer price sits to the level and the more times the
    #     level has held, so a fresh STRONG SELL landing on multi-touch support
    #     is tempered toward SELL/HOLD rather than amplified. ---
    support, sup_dist = snapshot.get("support"), snapshot.get("support_dist_pct")
    sup_touches = snapshot.get("support_touches") or 0
    resistance, res_dist = snapshot.get("resistance"), snapshot.get("resistance_dist_pct")
    res_touches = snapshot.get("resistance_touches") or 0

    if support and sup_dist is not None and 0 <= sup_dist <= 4.0 and sup_touches >= 2:
        prox = 1.0 - sup_dist / 4.0                     # 1 at the level → 0 by 4% above
        strength = min(1.0, 0.4 + 0.2 * sup_touches)    # more touches → firmer level
        w = round(2.0 * prox * strength, 2)             # up to +2.0 right on strong support
        if w >= 0.05:
            add(w, f"Testing support ~{support:g} (held {sup_touches}x) — "
                   "limited downside, bounce setup")
            nxt = snapshot.get("support_next")
            if nxt:
                drop = (support - nxt) / support * 100
                risks.append(f"A decisive close below {support:g} opens the next "
                             f"support ~{nxt:g} (~{drop:.1f}% lower).")
            else:
                risks.append(f"A decisive close below {support:g} voids the support "
                             "thesis with no clear level beneath.")
    if resistance and res_dist is not None and 0 <= res_dist <= 4.0 and res_touches >= 2:
        prox = 1.0 - res_dist / 4.0
        strength = min(1.0, 0.4 + 0.2 * res_touches)
        w = round(-2.0 * prox * strength, 2)            # up to -2.0 right under strong resistance
        if abs(w) >= 0.05:
            add(w, f"Capped at resistance ~{resistance:g} (rejected {res_touches}x) — "
                   "limited upside")
            nxt = snapshot.get("resistance_next")
            if nxt:
                rise = (nxt - resistance) / resistance * 100
                risks.append(f"A breakout above {resistance:g} opens the next "
                             f"resistance ~{nxt:g} (~{rise:.1f}% higher).")
            else:
                risks.append(f"A breakout above {resistance:g} clears overhead "
                             "resistance with open air above.")

    # --- Recent realised drift (last ~20 bars): the live momentum, distinct
    #     from the laggy EMA regime. The full-window change is kept only for the
    #     rationale text — scoring it just restates the trend and amplifies lag. ---
    drift = recent_chg if recent_chg is not None else chg
    if drift is not None and abs(drift) >= 3.0:
        w = scaled(drift, full_at=10.0, weight=0.5)
        add(w, f"Up {drift:+.1f}% recently (positive drift)" if w > 0
            else f"Down {drift:+.1f}% recently (negative drift)")

    # --- Money flow (Chaikin): intrabar buying vs selling pressure. A real
    #     read on whether closes are printing near highs (demand) or lows. ---
    if cmf is not None:
        if cmf >= 0.10:
            add(0.5, f"Chaikin money flow positive ({cmf:+.2f}) — net buying pressure")
        elif cmf <= -0.10:
            add(-0.5, f"Chaikin money flow negative ({cmf:+.2f}) — net selling pressure")

    # --- Volume confirmation: a move backed by above-average volume is more
    #     trustworthy than one drifting on thin volume. ---
    above_mean = price is not None and ref is not None and price > ref
    below_mean = price is not None and ref is not None and price < ref
    if volume is not None and vol_ma and volume >= 1.2 * vol_ma and drift is not None:
        if drift > 0 and above_mean:
            add(0.5, "Above-average volume confirms the advance")
        elif drift < 0 and below_mean:
            add(-0.5, "Above-average volume confirms the decline")

    # --- Accumulation / distribution -------------------------------------
    # Wyckoff-style read from OHLCV alone: large players accumulate during a
    # sideways *base* after a decline — price stops falling while money keeps
    # flowing in (rising OBV / positive CMF). Caught before the lagging EMAs and
    # MACD turn, it counters the trend-follower's habit of selling the bottom.
    # Distribution is the mirror: stalling near highs while money flows out.
    downtrend = ema50 is not None and ema200 is not None and ema50 < ema200
    uptrend = ema50 is not None and ema200 is not None and ema50 > ema200
    consolidating = (
        (vol_contraction is not None and vol_contraction < 0.9)  # volatility contracting
        or (recent_chg is not None and abs(recent_chg) < 6.0)    # price gone flat
    )
    inflow = (cmf is not None and cmf > 0.03) or (obv_trend is not None and obv_trend > 0.05)
    outflow = (cmf is not None and cmf < -0.03) or (obv_trend is not None and obv_trend < -0.05)
    not_freefall = recent_chg is None or recent_chg > -12.0  # still falling hard ≠ basing
    not_blowoff = recent_chg is None or recent_chg < 12.0

    if (downtrend or below_mean) and consolidating and inflow and not_freefall:
        add(1.5, "Accumulation: basing after a decline with money flowing in "
                 "(positive CMF/OBV) — potential bottoming")
        risks.append("Accumulation is provisional until price reclaims EMA50; trend is still down.")
    elif (uptrend or above_mean) and consolidating and outflow and not_blowoff and (
        price_pos is None or price_pos > 0.5
    ):
        add(-1.5, "Distribution: stalling near highs with money flowing out "
                  "(negative CMF/OBV) — topping risk")
        risks.append("Distribution can persist; wait for a break below support to confirm.")

    score = sum(w for w, _ in contrib)
    gross = sum(abs(w) for w, _ in contrib)
    agreement = abs(score) / gross if gross else 0.0  # 1 = unanimous, 0 = balanced

    if score >= 3.0:
        action = "strong_buy"
    elif score >= 1.0:
        action = "buy"
    elif score <= -3.0:
        action = "strong_sell"
    elif score <= -1.0:
        action = "sell"
    else:
        action = "hold"

    # Conviction scales with both signal magnitude and how much they agree.
    conf = (50.0 + 8.0 * abs(score)) * (0.6 + 0.4 * agreement)
    confidence = round(min(92.0, max(45.0, conf)), 1)

    if gross and agreement < 0.45 and action != "hold":
        risks.append("Signals conflict (trend vs momentum disagree); low conviction.")

    chg_txt = f" Price changed {chg:+.1f}% over the window." if chg is not None else ""
    news_txt = (
        f" {len(headlines)} recent headline(s) were available but sentiment is not"
        " modelled in rule-based mode."
        if headlines
        else " No news in the archive for this period."
    )
    summary = f"Rule-based signal: {action.replace('_', ' ').upper()} (score {score:+.1f})."
    rationale = (
        "Weighs EMA trend regime, price-vs-mean, MACD momentum, RSI, Bollinger"
        " position, volume, Chaikin money flow, swing support/resistance and an"
        " accumulation/distribution read into a single score; mean-reversion only"
        " at RSI/band extremes or established support/resistance."
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
