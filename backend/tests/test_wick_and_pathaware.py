"""Tests for the candle-wick rejection signal, momentum-streak maturity taper,
and the stretched-position resistance/support dampener.

Run from the `backend/` directory: `.venv/bin/python -m pytest`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.analysis import _heuristic
from app.services.indicators import reversal_signals


# ---- helpers --------------------------------------------------------------

def _frame(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLC frame from (open, high, low, close) rows; volume is flat."""
    o, h, l, c = zip(*bars)
    n = len(bars)
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=int),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": np.full(n, 1000.0),
        }
    )


# A snapshot mirroring the BTCUSDT 1h screenshot that scored STRONG SELL (-4.1):
# downtrend, price below mean and pinned to the lower Bollinger band, RSI 38,
# MACD below zero and its histogram falling 11 straight bars, capped under a
# much-tested resistance 0.8% overhead.
def _screenshot_snapshot() -> dict:
    return {
        "price": 76974.18,
        "rsi": 38.4,
        "macd": -150.0,
        "macd_hist": -117.37,
        "ema20": 77600.0,
        "ema50": 77410.04,
        "ema200": 78169.77,
        "bb_upper": 78250.0,
        "bb_lower": 77000.0,          # price <= bb_lower → stretched low
        "volume": 1000.0,
        "vol_ma": 700.0,              # above-average volume
        "cmf": -0.05,
        "window_change_pct": 1.0,
        "recent_change_pct": -2.5,
        "price_position": 0.12,       # pinned near the recent lows
        "resistance": 77595.51,
        "resistance_dist_pct": 0.8,
        "resistance_touches": 36,
        "macd_hist_dir": "falling",
        "macd_hist_streak": 11,
        "macd_hist_below_zero": True,
    }


# ---- rejection-wick detection --------------------------------------------

def test_bullish_rejection_wick_at_the_lows():
    # 40 flat bars, then a hammer: tags a new window low but closes back at the top.
    bars = [(100.0, 101.0, 99.0, 100.0)] * 40
    bars.append((100.0, 100.5, 95.0, 100.0))  # long lower shadow, tiny body
    out = reversal_signals(_frame(bars), price=100.0)
    assert out.get("wick_rejection") == "bullish"
    assert out.get("wick_rejection_bars_ago") == 0


def test_bearish_rejection_wick_at_the_highs():
    bars = [(100.0, 101.0, 99.0, 100.0)] * 40
    bars.append((100.0, 105.0, 99.5, 100.0))  # long upper shadow (shooting star)
    out = reversal_signals(_frame(bars), price=100.0)
    assert out.get("wick_rejection") == "bearish"


def test_no_wick_on_plain_candles():
    # Full-bodied bars with negligible shadows must not register as rejections.
    bars = [(99.0, 101.0, 98.8, 100.8)] * 41
    out = reversal_signals(_frame(bars), price=100.8)
    assert "wick_rejection" not in out


# ---- the screenshot scenario: STRONG SELL should soften -------------------

def test_screenshot_demoted_from_strong_sell():
    res = _heuristic(_screenshot_snapshot(), headlines=[], interval="1h")
    # The path-aware terms (streak exhaustion + stretched-low resistance damp)
    # pull the raw -4.1 back to a defensible SELL rather than STRONG SELL.
    assert res["action"] == "sell"


def test_screenshot_with_rejection_wick_is_not_a_sell_signal():
    snap = _screenshot_snapshot()
    snap["wick_rejection"] = "bullish"  # buyers defended the probe lower
    res = _heuristic(snap, headlines=[], interval="1h")
    assert res["action"] != "strong_sell"
    assert res["action"] in ("sell", "hold")


def test_long_down_streak_into_lows_reads_as_exhaustion():
    res = _heuristic(_screenshot_snapshot(), headlines=[], interval="1h")
    text = " ".join(res["key_factors"] + res["risks"])
    assert "exhaustion" in text.lower()


def test_resistance_cap_damped_when_price_is_stretched_low():
    snap = _screenshot_snapshot()
    stretched = _heuristic(snap, headlines=[], interval="1h")
    # Lift price off the lower band and away from the lows so it is no longer
    # stretched — the same overhead resistance should now bite harder (lower score).
    snap2 = dict(snap, price=77400.0, bb_lower=76000.0, price_position=0.6,
                 resistance_dist_pct=0.3)
    near_res = _heuristic(snap2, headlines=[], interval="1h")
    assert _score(near_res) < _score(stretched)


def _score(res: dict) -> float:
    # The summary embeds "(score +/-N.N)."; parse it back out for comparisons.
    tail = res["summary"].rsplit("score ", 1)[1]
    return float(tail.rstrip(")."))


# ---- regression: a healthy uptrend must still read bullish ----------------

def test_healthy_uptrend_still_bullish():
    snap = {
        "price": 100.0,
        "rsi": 60.0,
        "macd": 1.0,
        "macd_hist": 0.3,
        "ema20": 98.0,
        "ema50": 95.0,
        "ema200": 90.0,
        "bb_upper": 105.0,
        "bb_lower": 92.0,
        "volume": 1000.0,
        "vol_ma": 700.0,
        "cmf": 0.12,
        "window_change_pct": 8.0,
        "recent_change_pct": 5.0,
        "price_position": 0.7,
    }
    res = _heuristic(snap, headlines=[], interval="1h")
    assert res["action"] in ("buy", "strong_buy")
