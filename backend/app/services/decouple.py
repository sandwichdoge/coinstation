"""Decouple meter: how independently a coin trades from BTC — the king, used
here as the global crypto-sentiment proxy.

We align a coin's candles with BTC's over the same window, take per-candle
percentage returns, and regress the coin on BTC. Three reads fall out:

  * correlation (ρ)  — how *coupled* they are: +1 lockstep, 0 independent,
    <0 inverse. The tether to market sentiment.
  * beta (β)         — how *amplified* the coin's response is: β>1 swings harder
    than BTC, 0<β<1 is muted (it moves weakly relative to how far BTC moved in
    the same candle).
  * relative strength — the slice of the coin's move BTC does *not* explain:
    the beta-adjusted residual ("alpha"). Positive = decoupling to the upside
    (independent strength); negative = decoupling to the downside (weakness).

The headline ``decouple_score`` is (1 − R²)·100 with R² = ρ²: the fraction of
the coin's return variance NOT explained by BTC. 0 = every wiggle is BTC's; 100
= it trades entirely on its own. (Inverse correlation also produces a high R²,
so it's flagged separately — moving *against* BTC is its own regime, not
coupling.) Mirrors the auditable, single-source-of-truth style of indicators.py:
given two frames, the math here is all there is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECOUPLE_WINDOW = 30      # candles of return history the regression reads
MIN_SAMPLES = 10          # fewer aligned returns than this and the read isn't meaningful
RS_DEADBAND = 1.0         # |lean %| under this is "tracking", not a real direction

# ``decouple_score`` bands — fraction of variance independent of BTC.
TIGHT_MAX = 33.0          # below this: moves are mostly BTC's
LOOSE_MIN = 66.0          # above this: trades largely on its own
INVERSE_CORR = -0.2       # correlation at/under this is an active inverse regime

_COUPLING_TEXT = {
    "coupled": "tracks BTC closely",
    "loosening": "partly trades on its own",
    "decoupled": "largely trades on its own",
    "inverse": "tends to move opposite BTC",
}


def decouple_meter(coin: pd.DataFrame, btc: pd.DataFrame,
                   window: int = DECOUPLE_WINDOW) -> dict:
    """Compare `coin` against `btc` (both OHLCV frames sharing a `time` column)
    and return the decouple meter. Returns ``{"available": False, "reason": …}``
    when the two can't be aligned on enough candles to judge."""
    if coin.empty or btc.empty:
        return {"available": False, "reason": "No data for the coin or BTC."}

    # Align candle-for-candle on shared open times; only bars both have count.
    merged = pd.merge(
        coin[["time", "close"]].rename(columns={"close": "c_coin"}),
        btc[["time", "close"]].rename(columns={"close": "c_btc"}),
        on="time", how="inner",
    ).sort_values("time")

    # Per-candle percentage returns, trailing `window` of them.
    merged["r_coin"] = merged["c_coin"].pct_change()
    merged["r_btc"] = merged["c_btc"].pct_change()
    rets = merged.dropna(subset=["r_coin", "r_btc"]).tail(window)
    n = len(rets)
    if n < MIN_SAMPLES:
        return {"available": False,
                "reason": f"Only {n} aligned candle(s); need at least {MIN_SAMPLES}."}

    rc = rets["r_coin"].to_numpy(dtype=float)
    rb = rets["r_btc"].to_numpy(dtype=float)

    # Population moments (cov and var consistent with each other so β is exact).
    cov = float(np.mean((rc - rc.mean()) * (rb - rb.mean())))
    var_btc = float(np.var(rb))
    std_c, std_b = float(rc.std()), float(rb.std())
    # Correlation; guard the degenerate case where one side simply didn't move.
    corr = cov / (std_c * std_b) if std_c > 0 and std_b > 0 else 0.0
    corr = max(-1.0, min(1.0, corr))
    beta = cov / var_btc if var_btc > 0 else None

    # R² = ρ²: how much of the coin's variance BTC explains. Its complement is
    # how much the coin is doing on its own — the decouple score.
    decouple_score = round((1.0 - corr ** 2) * 100.0, 1)

    # Window moves, and the two flavours of "are we stronger than BTC":
    coin_change = (float(np.prod(1.0 + rc)) - 1.0) * 100.0
    btc_change = (float(np.prod(1.0 + rb)) - 1.0) * 100.0
    relative_strength = round(coin_change - btc_change, 2)   # raw outperformance

    # Beta-adjusted excess: the part of each move BTC doesn't account for,
    # compounded over the window. The truer "decoupling" direction — it strips
    # out simply being a higher-beta proxy for the same sentiment.
    if beta is not None:
        resid = rc - beta * rb
        alpha = round((float(np.prod(1.0 + resid)) - 1.0) * 100.0, 2)
    else:
        alpha = None

    # Coupling regime. Inverse is checked first: a strong negative correlation
    # has a *low* decouple_score (high R²) yet is the opposite of "tracks BTC".
    if corr <= INVERSE_CORR:
        coupling = "inverse"
    elif decouple_score < TIGHT_MAX:
        coupling = "coupled"
    elif decouple_score < LOOSE_MIN:
        coupling = "loosening"
    else:
        coupling = "decoupled"

    # Direction prefers the beta-adjusted alpha; falls back to raw outperformance.
    lean = alpha if alpha is not None else relative_strength
    if lean is None or abs(lean) < RS_DEADBAND:
        direction = "neutral"
    else:
        direction = "strength" if lean > 0 else "weakness"

    rel_word = ("in line with" if abs(relative_strength) < RS_DEADBAND
                else "stronger than" if relative_strength > 0 else "weaker than")
    alpha_txt = (f", {abs(alpha):.1f}% beta-adjusted {direction}"
                 if alpha is not None and direction != "neutral" else "")
    summary = (
        f"{decouple_score:.0f}/100 decoupled — {_COUPLING_TEXT[coupling]}. "
        f"Over {n} candles it moved {coin_change:+.1f}% vs BTC {btc_change:+.1f}% "
        f"({rel_word} the king{alpha_txt})."
    )

    return {
        "available": True,
        "decouple_score": decouple_score,
        "coupling": coupling,
        "direction": direction,
        "correlation": round(corr, 3),
        "beta": round(beta, 3) if beta is not None else None,
        "relative_strength_pct": relative_strength,
        "alpha_pct": alpha,
        "coin_change_pct": round(coin_change, 2),
        "btc_change_pct": round(btc_change, 2),
        "samples": n,
        "summary": summary,
    }
