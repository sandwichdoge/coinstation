"""Top/bottom (swing-reversal) detector.

A *causal*, per-bar classifier that scores how likely each bar is a local
**bottom** or **top**. It is deliberately distinct from the directional scorer
in ``analysis.py``: that answers "what is the bias now?", this answers "is *this
bar* a turning point?".

Two hard rules make the output trustworthy and back-testable:

  * **Causality.** The score at bar *i* uses only bars ``0..i``. Every feature is
    either a point-in-time read (RSI, Bollinger position, EMA distance) or a
    *trailing* rolling window — never a centered window, never a full-series
    min/max, never a look-ahead pivot. A causal detector structurally cannot
    fire exactly on the low; it confirms a few bars later. That lag is measured
    in the eval harness, not hidden. The leakage tripwire test asserts the score
    at bar *i* is byte-identical whether or not bars ``>i`` exist.

  * **Reuse.** The features are the same exhaustion reads ``indicators.py``
    already trusts (RSI extreme/divergence, volume climax, rejection wick,
    MACD-histogram turn, stretch from the mean, money flow) — here recomputed
    *vectorised across the whole frame* and combined into a single bounded score
    instead of folded into the directional total.

The bottom side carries more confirmation weight than the top side: bottoms are
the priority, so the detector is tuned to catch capitulation lows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Bars required each side of a bar for it to be a confirmed swing pivot. Matches
# indicators.PIVOT_WING — a pivot at index p is only *confirmed* at bar p+WING,
# which is the lag the divergence read pays to stay causal.
WING = 3
RECENT = 20            # trailing window for "near the local extreme" / stretch reads
EVENT_HOLD = 3         # an event (climax/wick) contributes for this many bars
RSI_DIV_MARGIN = 2.0   # min RSI gap (points) for a real divergence
VOL_CLIMAX_MULT = 2.0  # volume >= this * MA to count as a climax bar
EXTREME_TOL = 0.15     # within this fraction of the trailing range counts as "at the extreme"
WICK_DOMINANCE = 2.0   # rejection wick must be >= this * the candle body
WICK_RANGE_FRAC = 0.5  # ... and span >= this fraction of the candle's range

# Per-feature weights. Bottom side is richer (priority); the two sides mirror
# each other where a mirror exists. Scores are raw weighted sums; the eval
# harness sweeps the firing threshold, so absolute scale only sets a default.
BOTTOM_WEIGHTS = {
    "rsi_oversold": 1.0,
    "below_lower_bb": 0.7,
    "stretched_below_mean": 0.8,
    "selling_climax": 1.2,
    "bullish_wick": 1.0,
    "bullish_divergence": 1.3,
    "hist_upturn": 0.6,
    "inflow_while_low": 0.5,
}
TOP_WEIGHTS = {
    "rsi_overbought": 1.0,
    "above_upper_bb": 0.7,
    "stretched_above_mean": 0.8,
    "buying_climax": 1.2,
    "bearish_wick": 1.0,
    "bearish_divergence": 1.3,
    "hist_downturn": 0.6,
    "outflow_while_high": 0.5,
}

# Default firing thresholds, set from the out-of-sample sweep in
# scripts/eval_top_bottom.py: tuned on BTC/ETH 2018-2021 by forward-return
# quality, validated on held-out coins + recent BTC/ETH. At 2.6 the bottom side
# runs ~0.78 precision against 15% ZigZag lows with positive forward returns
# (h50 ≈ +14% OOS); higher conviction, fewer signals — appropriate for the
# "major swing turns" target. Retune via the script if features/horizons change.
BOTTOM_THRESHOLD = 2.6
TOP_THRESHOLD = 2.6


def _clamp01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def _ramp(value: pd.Series, lo: float, hi: float) -> pd.Series:
    """0 at ``lo``, 1 at ``hi``, linear between; clamped. Works either direction
    (``lo`` may be greater than ``hi`` for a falling ramp)."""
    return _clamp01((value - lo) / (hi - lo))


def _divergence_flags(
    low: np.ndarray, high: np.ndarray, rsi: np.ndarray, wing: int = WING
) -> tuple[np.ndarray, np.ndarray]:
    """Causal RSI-divergence state, one forward pass.

    A swing low at index ``p`` is *confirmed* at bar ``p+wing`` (its low is the
    lowest in ``[p-wing, p+wing]``). At each confirmation we compare the new
    swing to the previous confirmed swing low; a bullish divergence (price lower
    low, RSI higher low, in RSI's lower half) latches *on the confirmation bar*
    and stays active until the next swing low confirms. Mirror for highs/bearish.

    Returns ``(bull_active, bear_active)`` boolean arrays the length of the input
    — each True only from the bar a divergence could first be known, never
    earlier. This is the single feature that needs swing structure; everything
    else is point-in-time or a trailing rolling window.
    """
    n = len(low)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    prev_low_p = prev_high_p = -1  # index of the last confirmed swing low / high

    for c in range(wing, n - wing):
        # `c` is the candidate pivot; it becomes *known* at bar c+wing.
        seg = slice(c - wing, c + wing + 1)
        conf = c + wing
        if conf >= n:
            break
        if low[c] == np.min(low[seg]):
            if (
                prev_low_p >= 0
                and not np.isnan(rsi[c]) and not np.isnan(rsi[prev_low_p])
                and low[c] < low[prev_low_p]
                and rsi[c] > rsi[prev_low_p] + RSI_DIV_MARGIN
                and rsi[c] < 50
            ):
                # Latch active from the confirmation bar to the next swing-low confirmation.
                bull[conf:] = True
                # Clear any later swing-high latch overlap is handled independently.
            else:
                bull[conf:] = False
            prev_low_p = c
        if high[c] == np.max(high[seg]):
            if (
                prev_high_p >= 0
                and not np.isnan(rsi[c]) and not np.isnan(rsi[prev_high_p])
                and high[c] > high[prev_high_p]
                and rsi[c] < rsi[prev_high_p] - RSI_DIV_MARGIN
                and rsi[c] > 50
            ):
                bear[conf:] = True
            else:
                bear[conf:] = False
            prev_high_p = c
    return bull, bear


def compute_features(df_ind: pd.DataFrame) -> pd.DataFrame:
    """Per-bar, fully causal reversal features in ``[0, 1]`` each.

    Expects the columns ``indicators.compute_indicator_frame`` produces. Returns
    a new frame (same index) with one column per feature in BOTTOM_WEIGHTS /
    TOP_WEIGHTS, ready to be combined into scores.
    """
    out = pd.DataFrame(index=df_ind.index)
    close = df_ind["close"].astype(float)
    high = df_ind["high"].astype(float)
    low = df_ind["low"].astype(float)
    op = df_ind["open"].astype(float)
    vol = df_ind["volume"].astype(float)
    rsi = df_ind.get("rsi")
    ema50 = df_ind.get("ema50")
    bb_upper, bb_lower, bb_mid = df_ind.get("bb_upper"), df_ind.get("bb_lower"), df_ind.get("bb_mid")
    hist = df_ind.get("macd_hist")
    vol_ma = df_ind.get("vol_ma")
    cmf = df_ind.get("cmf")

    # --- RSI extremes (point-in-time) ---
    if rsi is not None:
        out["rsi_oversold"] = _ramp(rsi, 40.0, 20.0)      # 40→0, 20→1
        out["rsi_overbought"] = _ramp(rsi, 60.0, 80.0)    # 60→0, 80→1
    else:
        out["rsi_oversold"] = out["rsi_overbought"] = 0.0

    # --- Bollinger stretch (point-in-time) ---
    if bb_lower is not None and bb_mid is not None:
        band = (bb_mid - bb_lower).replace(0.0, np.nan)
        out["below_lower_bb"] = _clamp01((bb_lower - close) / band).fillna(0.0)
        out["above_upper_bb"] = _clamp01((close - bb_upper) / band).fillna(0.0)
    else:
        out["below_lower_bb"] = out["above_upper_bb"] = 0.0

    # --- Stretch from the mid-term mean (point-in-time) ---
    if ema50 is not None:
        dist = (close - ema50) / ema50 * 100.0
        out["stretched_below_mean"] = _ramp(-dist, 5.0, 20.0)   # 5% below → 0, 20% below → 1
        out["stretched_above_mean"] = _ramp(dist, 5.0, 20.0)
    else:
        out["stretched_below_mean"] = out["stretched_above_mean"] = 0.0

    # --- Trailing-window extreme position (causal rolling min/max) ---
    roll_low = low.rolling(RECENT, min_periods=RECENT).min()
    roll_high = high.rolling(RECENT, min_periods=RECENT).max()
    rng = (roll_high - roll_low).replace(0.0, np.nan)
    near_low = ((low - roll_low) / rng <= EXTREME_TOL).fillna(False)
    near_high = ((roll_high - high) / rng <= EXTREME_TOL).fillna(False)

    bar_rng = (high - low).replace(0.0, np.nan)
    close_pos = ((close - low) / bar_rng).fillna(0.5)  # 0 = closed at low, 1 = at high

    # --- Volume climax (event → held EVENT_HOLD bars) ---
    if vol_ma is not None:
        spike = (vol >= VOL_CLIMAX_MULT * vol_ma).fillna(False)
        sell_climax = (spike & near_low & (close_pos <= 0.5)).astype(float)
        buy_climax = (spike & near_high & (close_pos >= 0.5)).astype(float)
        out["selling_climax"] = sell_climax.rolling(EVENT_HOLD, min_periods=1).max()
        out["buying_climax"] = buy_climax.rolling(EVENT_HOLD, min_periods=1).max()
    else:
        out["selling_climax"] = out["buying_climax"] = 0.0

    # --- Rejection wick / pin bar (event → held EVENT_HOLD bars) ---
    body = (close - op).abs()
    upper_wick = high - pd.concat([op, close], axis=1).max(axis=1)
    lower_wick = pd.concat([op, close], axis=1).min(axis=1) - low
    long_lower = (lower_wick >= WICK_DOMINANCE * body) & (lower_wick >= WICK_RANGE_FRAC * bar_rng)
    long_upper = (upper_wick >= WICK_DOMINANCE * body) & (upper_wick >= WICK_RANGE_FRAC * bar_rng)
    bull_wick = (near_low & long_lower.fillna(False)).astype(float)
    bear_wick = (near_high & long_upper.fillna(False)).astype(float)
    out["bullish_wick"] = bull_wick.rolling(EVENT_HOLD, min_periods=1).max()
    out["bearish_wick"] = bear_wick.rolling(EVENT_HOLD, min_periods=1).max()

    # --- MACD-histogram turn (point-in-time direction + zero-line context) ---
    if hist is not None:
        dh = hist.diff()
        out["hist_upturn"] = ((dh > 0) & (hist < 0)).astype(float)    # downside momentum fading
        out["hist_downturn"] = ((dh < 0) & (hist > 0)).astype(float)  # upside momentum fading
    else:
        out["hist_upturn"] = out["hist_downturn"] = 0.0

    # --- Money flow while pinned to the extreme (point-in-time) ---
    if cmf is not None:
        out["inflow_while_low"] = ((cmf > 0.03) & near_low).astype(float)
        out["outflow_while_high"] = ((cmf < -0.03) & near_high).astype(float)
    else:
        out["inflow_while_low"] = out["outflow_while_high"] = 0.0

    # --- RSI divergence (causal swing pass) ---
    if rsi is not None:
        bull, bear = _divergence_flags(
            low.to_numpy(), high.to_numpy(), rsi.to_numpy(), WING
        )
        out["bullish_divergence"] = bull.astype(float)
        out["bearish_divergence"] = bear.astype(float)
    else:
        out["bullish_divergence"] = out["bearish_divergence"] = 0.0

    return out.fillna(0.0)


def _turn_confirmation(df_ind: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Causal confirmation that a turn has actually begun — the guard against
    catching a falling knife.

    Exhaustion alone (oversold, stretched, climax) fires *all the way down*: the
    backtest's forward-return lift was negative without this gate because the
    "bottoms" landed mid-plunge, before price turned. So a bottom only confirms
    once price has **stopped making new ``RECENT``-bar lows and ticked up** — we
    trade a couple of bars of lag for signals that actually precede the bounce.
    Mirror for tops.
    """
    close = df_ind["close"].astype(float)
    low = df_ind["low"].astype(float)
    high = df_ind["high"].astype(float)
    roll_low_prev = low.rolling(RECENT, min_periods=RECENT).min().shift(1)
    roll_high_prev = high.rolling(RECENT, min_periods=RECENT).max().shift(1)
    made_new_low = low <= roll_low_prev      # undercut the prior N-bar low → still falling
    made_new_high = high >= roll_high_prev    # printed a fresh N-bar high → still running
    confirm_up = (close > close.shift(1)) & (~made_new_low.fillna(True))
    confirm_down = (close < close.shift(1)) & (~made_new_high.fillna(True))
    return confirm_up.fillna(False), confirm_down.fillna(False)


def top_bottom_scores(
    df_ind: pd.DataFrame,
    bottom_threshold: float = BOTTOM_THRESHOLD,
    top_threshold: float = TOP_THRESHOLD,
    confirm: bool = True,
) -> pd.DataFrame:
    """Add ``bottom_score``, ``top_score`` and discrete ``tb_signal`` columns.

    ``bottom_score`` / ``top_score`` are the raw exhaustion measures. A discrete
    ``tb_signal`` (``+1`` bottom, ``-1`` top, ``0`` none) fires when a side clears
    its threshold *and* (when ``confirm`` is set) the turn-confirmation gate
    agrees. When both sides clear on the same bar (rare), the stronger raw score
    wins.
    """
    feats = compute_features(df_ind)
    bottom = sum(feats[k] * w for k, w in BOTTOM_WEIGHTS.items())
    top = sum(feats[k] * w for k, w in TOP_WEIGHTS.items())

    out = pd.DataFrame(index=df_ind.index)
    out["bottom_score"] = bottom.round(4)
    out["top_score"] = top.round(4)

    fire_b = bottom >= bottom_threshold
    fire_t = top >= top_threshold
    if confirm:
        confirm_up, confirm_down = _turn_confirmation(df_ind)
        fire_b = fire_b & confirm_up
        fire_t = fire_t & confirm_down

    sig = np.where(fire_b & (bottom >= top), 1, np.where(fire_t & (top > bottom), -1, 0))
    out["tb_signal"] = sig.astype(int)
    return out


def latest(df_ind: pd.DataFrame) -> dict:
    """Causal top/bottom read for the most recent bar — for the live snapshot.

    Returns the last bar's scores and a discrete label. Causal by construction:
    ``top_bottom_scores`` uses only trailing data, so the last row is the
    real-time read with no look-ahead.
    """
    if df_ind.empty or len(df_ind) < RECENT:
        return {}
    scored = top_bottom_scores(df_ind)
    last = scored.iloc[-1]
    sig = int(last["tb_signal"])
    return {
        "bottom_score": float(last["bottom_score"]),
        "top_score": float(last["top_score"]),
        "tb_signal": "bottom" if sig > 0 else "top" if sig < 0 else None,
    }
