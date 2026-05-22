"""Rule-based analysis: turn a technical-indicator snapshot into a
recommendation (action + 0-100 confidence + rationale).

A transparent, auditable scorer — every signal that fires is recorded so the
rationale can be traced back to its inputs. News headlines are accepted for
context/display but are not modelled in the score.
"""
from __future__ import annotations

from ..schemas import AnalysisResult
from ..timeutil import ms_to_iso


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
    rsi_div = snapshot.get("rsi_divergence")
    hist_dir = snapshot.get("macd_hist_dir")
    hist_streak = snapshot.get("macd_hist_streak") or 0
    hist_below_zero = snapshot.get("macd_hist_below_zero")
    climax = snapshot.get("volume_climax")
    wick = snapshot.get("wick_rejection")
    tb_signal = snapshot.get("tb_signal")            # "bottom" | "top" | None
    bottom_score = snapshot.get("bottom_score")
    top_score = snapshot.get("top_score")

    # --- Structural context, computed up front so support/resistance scoring can
    #     reconcile with it. A nearby "resistance" inside an accumulation base is
    #     the range ceiling price is coiling under, not overhead supply far above
    #     — scoring it as a full bearish cap double-counts against the basing read
    #     and (when the symmetric support guard is just out of range) can be the
    #     single contributor that tips a SELL into a STRONG SELL. Mirror for a
    #     support level under active distribution. ---
    ref = ema50 if ema50 is not None else ema20
    above_mean = price is not None and ref is not None and price > ref
    below_mean = price is not None and ref is not None and price < ref
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
    accumulation = (downtrend or below_mean) and consolidating and inflow and not_freefall
    distribution = (
        (uptrend or above_mean) and consolidating and outflow and not_blowoff
        and (price_pos is None or price_pos > 0.5)
    )

    # --- Overextension / blow-off: the mirror of accumulation, and the engine's
    #     explicit "don't chase the top" guard. A trend-follower is structurally
    #     blind here — the price-vs-mean term below saturates a few % above the
    #     mean, so a parabolic move 40% above EMA50 scores the same bullish +1 as
    #     a healthy uptrend. We measure that stretch directly: how far price has
    #     run above its mean (`ext_pct`) and whether the recent move is parabolic.
    #     A blow-off pins price near the top of its range while RSI is overbought.
    ext_pct = (price - ref) / ref * 100 if (price is not None and ref) else None
    overextended = ext_pct is not None and ext_pct >= 15.0
    parabolic = recent_chg is not None and recent_chg >= 25.0
    pinned_high = price_pos is None or price_pos >= 0.6
    # A stretched, parabolic advance is the *setup*; we only call it a blow-off
    # once momentum actually rolls over — otherwise this fires on every overbought
    # bar of a healthy run and bails the whole trend. The rollover tell is a
    # bearish RSI divergence, a sustained MACD-histogram down-turn, or a buying
    # climax: the same exhaustion reads the snapshot already computes.
    rolling_over = (
        rsi_div == "bearish"
        or (hist_dir == "falling" and hist_streak >= 2)
        or climax == "buying"
    )
    blowoff = (uptrend or above_mean) and (overextended or parabolic) and pinned_high and rolling_over

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
    if price is not None and ref:
        dist_pct = (price - ref) / ref * 100
        w = scaled(dist_pct, full_at=3.0, weight=1.0)
        if abs(w) >= 0.05:
            add(w, "Price holding above EMA50 (near-term strength)" if w > 0
                else "Price below EMA50 (near-term weakness)")

    # --- Overextension: once price has stretched well past its mean *and* sits
    #     near the top of its range, each extra percent is mean-reversion risk,
    #     not strength — it counters the saturation of the term above so a spike
    #     no longer reads as plain bullish. Kept deliberately light (caps at -1.0,
    #     grows from +15% to +40% above the mean) and gated on `pinned_high`: it
    #     tempers chasing a vertical move, but won't flip a healthy uptrend short
    #     on its own. The full topping read is the rollover-confirmed blow-off
    #     below; this is just the "don't chase the spike" tap on the brakes. ---
    if ext_pct is not None and ext_pct >= 15.0 and pinned_high:
        w = round(-1.0 * min(1.0, (ext_pct - 15.0) / 25.0), 2)
        if w <= -0.05:
            add(w, f"Price ~{ext_pct:.0f}% above EMA50 (overextended, mean-reversion risk)")
            risks.append("Price is stretched far above its mean; chasing here is poor risk/reward.")

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
            # Scale gently with how stretched it is: -1.0 at 70 ramping to -1.6 by
            # 82. Overbought can persist in strong trends, so this stays modest —
            # but an extreme print is real exhaustion and was previously flat at
            # -1.0, under-weighted versus the +2.0 oversold mirror.
            w = round(-(1.0 + 0.6 * min(1.0, (rsi - 70.0) / 12.0)), 2)
            add(w, f"RSI {rsi:.0f} overbought (pullback risk)")
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

    # A level only binds the move if price is actually near it. When price is
    # stretched to the *opposite* band/extreme, the level on the far side is not
    # the active constraint — capping a stretched-low price for "limited upside"
    # (or a stretched-high one for "limited downside") double-penalises the very
    # mean-reversion the band stretch is flagging, so we damp that term.
    stretched_low = (
        (bb_lower is not None and price is not None and price <= bb_lower)
        or (price_pos is not None and price_pos <= 0.2)
    )
    stretched_high = (
        (bb_upper is not None and price is not None and price >= bb_upper)
        or (price_pos is not None and price_pos >= 0.8)
    )

    if support and sup_dist is not None and 0 <= sup_dist <= 4.0 and sup_touches >= 2:
        prox = 1.0 - sup_dist / 4.0                     # 1 at the level → 0 by 4% above
        strength = min(1.0, 0.4 + 0.2 * sup_touches)    # more touches → firmer level
        w = round(2.0 * prox * strength, 2)             # up to +2.0 right on strong support
        if distribution:
            # price stalling near a level it's distributing from — the floor is
            # giving way, not holding; don't let it manufacture a strong-buy.
            w = round(w * 0.4, 2)
        if stretched_high:
            w = round(w * 0.5, 2)  # price is pinned to the highs, far from this floor
        if w >= 0.05:
            add(w, f"Testing support ~{support:g} (held {sup_touches}x) — "
                   "limited downside, bounce setup")
    if resistance and res_dist is not None and 0 <= res_dist <= 4.0 and res_touches >= 2:
        prox = 1.0 - res_dist / 4.0
        strength = min(1.0, 0.4 + 0.2 * res_touches)
        w = round(-2.0 * prox * strength, 2)            # up to -2.0 right under strong resistance
        if accumulation:
            # the range ceiling price is coiling under during a base, not true
            # overhead supply — don't let it deepen a sell into a strong-sell.
            w = round(w * 0.4, 2)
        if stretched_low:
            w = round(w * 0.5, 2)  # price is pinned to the lows, far from this ceiling
        if abs(w) >= 0.05:
            add(w, f"Capped at resistance ~{resistance:g} (rejected {res_touches}x) — "
                   "limited upside")

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
    # (The `accumulation` / `distribution` flags are computed up top so the
    # support/resistance block above could already reconcile against them.)
    if accumulation:
        add(1.5, "Accumulation: basing after a decline with money flowing in "
                 "(positive CMF/OBV) — potential bottoming")
        risks.append("Accumulation is provisional until price reclaims EMA50; trend is still down.")
    elif distribution:
        add(-1.5, "Distribution: stalling near highs with money flowing out "
                  "(negative CMF/OBV) — topping risk")
        risks.append("Distribution can persist; wait for a break below support to confirm.")
    elif blowoff:
        # Mirror of accumulation: a parabolic, overbought run pinned to the highs.
        # Distribution catches the *quiet* top (sideways stall); this catches the
        # *loud* one (vertical blow-off), which the trend stack otherwise chases.
        add(-2.0, "Blow-off: overbought, overextended run pinned to the highs "
                  "— exhaustion / topping risk")
        risks.append("Blow-off tops reverse sharply; a parabolic advance is not a place to add.")

    # --- Multi-candle reversal & exhaustion ------------------------------
    # Unlike the single-bar reads above, these span a sequence of candles, so
    # they can lean against the laggy trend right at a turn — exactly where a
    # pure trend-follower is blindest and most likely to sell the bottom or
    # chase the top.

    # RSI divergence: a fresh price extreme the oscillator refuses to confirm.
    if rsi_div == "bullish":
        add(1.5, "Bullish RSI divergence (price lower low, RSI higher low) — momentum reversal up")
    elif rsi_div == "bearish":
        add(-1.5, "Bearish RSI divergence (price higher high, RSI lower high) — momentum reversal down")
        risks.append("Bearish divergence flags momentum exhaustion under the high.")

    # MACD-histogram run: a multi-bar turn ahead of the signal-line crossover.
    # Worth most as an *early* tell — a histogram rising while still below zero
    # is the first sign a downtrend's momentum is fading (mirror above zero).
    if hist_streak >= 3 and hist_dir:
        w = min(0.8, 0.2 * hist_streak)
        # A streak this long that has also driven price to the matching range
        # extreme is late-stage, not fresh: downside momentum "building" for 8+
        # bars into the lows is the kind of one-way run that mean-reverts. There
        # we flip the trend-aligned read to a light exhaustion tap rather than
        # pile on. (The counter-trend branches — a histogram turning up while
        # still below zero, or down while above it — are already early-reversal
        # tells, so they keep their weight regardless of maturity.)
        mature = hist_streak >= 8
        at_low = price_pos is not None and price_pos <= 0.25
        at_high = price_pos is not None and price_pos >= 0.75
        if hist_dir == "rising":
            if hist_below_zero:
                add(w, f"MACD histogram rising {hist_streak} bars (downside momentum fading)")
            elif mature and at_high:
                add(-0.5, f"MACD histogram rising {hist_streak} bars but pinned to the highs "
                          "— upside exhaustion, not fresh momentum")
            else:
                add(w, f"MACD histogram rising {hist_streak} bars (upside momentum building)")
        else:
            if not hist_below_zero:
                add(-w, f"MACD histogram falling {hist_streak} bars (upside momentum fading)")
            elif mature and at_low:
                add(0.5, f"MACD histogram falling {hist_streak} bars but pinned to the lows "
                         "— downside exhaustion, not fresh momentum")
            else:
                add(-w, f"MACD histogram falling {hist_streak} bars (downside momentum building)")

    # Volume climax: a spike on a wide bar pinned to the window's extreme.
    if climax == "selling":
        add(1.0, "Selling climax: volume spike at the lows — capitulation / bottoming")
    elif climax == "buying":
        add(-1.0, "Buying climax: volume spike at the highs — blow-off / topping")
        risks.append("Buying climax can mark a local top; chasing strength here is poor risk/reward.")

    # Rejection wick (pin bar): candle anatomy the close-only terms above are
    # blind to. A long lower shadow tagging the lows then closing away is demand
    # defending the level; a long upper shadow at the highs is supply doing the
    # same. Read as a single-bar reversal tell with the trend-stack's blind spot.
    if wick == "bullish":
        add(1.0, "Bullish rejection wick (long lower shadow at the lows) — buyers defending the level")
    elif wick == "bearish":
        add(-1.0, "Bearish rejection wick (long upper shadow at the highs) — sellers defending the level")
        risks.append("Upper-shadow rejection flags supply overhead.")

    # Top/bottom detector: the dedicated, out-of-sample-validated swing-reversal
    # call that fuses the exhaustion reads above into one confirmed bottom/top.
    # It only fires on a *confirmed* turn (price has stopped making new extremes
    # and ticked back), so it is high-conviction and weighted accordingly — and
    # it feeds the strong-call demotion guard below as a structural reversal. The
    # bottom side is the priority and the better-validated of the two.
    if tb_signal == "bottom":
        add(2.0, f"Top/bottom detector: confirmed swing bottom"
                 f"{f' (score {bottom_score:.1f})' if bottom_score is not None else ''} "
                 "— major reversal up")
    elif tb_signal == "top":
        add(-1.5, f"Top/bottom detector: confirmed swing top"
                  f"{f' (score {top_score:.1f})' if top_score is not None else ''} "
                  "— possible reversal down")
        risks.append("Top detection is the weaker side in trending markets; treat the top call as a caution, not a short.")

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

    # --- Structural demotion guard: never issue a *strong* call straight into an
    #     active opposing structural read. A selling climax / bullish divergence /
    #     accumulation base says "this may be the bottom" — so a STRONG SELL there
    #     is the trend-follower's classic mistake (and the mirror at the top). The
    #     score already counts these as positive contributors; this caps the rare
    #     case where the trend/momentum stack still nets past the strong threshold.
    bottoming = (accumulation or climax == "selling" or rsi_div == "bullish"
                 or wick == "bullish" or tb_signal == "bottom")
    topping = (distribution or blowoff or climax == "buying" or rsi_div == "bearish"
               or wick == "bearish" or tb_signal == "top")
    if action == "strong_sell" and bottoming:
        action = "sell"
        risks.append("Demoted from STRONG SELL: an opposing bottoming signal "
                     "(accumulation / capitulation / bullish divergence) is active.")
    elif action == "strong_buy" and topping:
        action = "buy"
        risks.append("Demoted from STRONG BUY: an opposing topping signal "
                     "(distribution / blow-off / bearish divergence) is active.")

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
        " position, volume, Chaikin money flow, swing support/resistance, an"
        " accumulation/distribution read and candle/reversal signals (RSI"
        " divergence, MACD-histogram turns, volume climaxes, rejection wicks) into"
        " a single score;"
        " mean-reversion only at RSI/band extremes or established support/resistance."
        + chg_txt
        + news_txt
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
    }


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
