"""Detector tests, anchored by the leakage tripwire.

The tripwire is the load-bearing test: it rebuilds the indicator frame *and* the
top/bottom scores on truncated histories and asserts the score at bar ``i`` is
identical whether or not bars ``> i`` exist. If any feature ever sneaks a peek at
the future (a centered window, a full-series extreme, a look-ahead pivot) this
fails. Everything else (P&L, precision/recall) is meaningless if this fails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import detectors, indicators


def _synthetic_ohlcv(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Deterministic random-walk OHLCV with a couple of injected swings, so the
    detector has real tops/bottoms to chew on without hitting the network."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.02, n)
    # Inject a sharp drop-and-recover (a bottom) and a parabola-and-fade (a top).
    steps[150:170] -= 0.03
    steps[170:190] += 0.035
    steps[400:415] += 0.04
    steps[415:430] -= 0.045
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(1_000, 5_000, n)
    vol[160:165] *= 4  # volume spike near the injected bottom
    t0 = 1_600_000_000
    return pd.DataFrame({
        "time": [t0 + i * 86_400 for i in range(n)],
        "open": openp, "high": np.maximum.reduce([high, openp, close]),
        "low": np.minimum.reduce([low, openp, close]), "close": close, "volume": vol,
    })


def test_no_lookahead_leakage():
    """Score at bar i must not change when future bars are appended/removed."""
    df = _synthetic_ohlcv()
    ind_full = indicators.compute_indicator_frame(df)
    scored_full = detectors.top_bottom_scores(ind_full)

    # Check a spread of indices, including ones right after the injected swings.
    for i in (300, 350, 405, 420, 500, 560, 599):
        ind_trunc = indicators.compute_indicator_frame(df.iloc[: i + 1])
        scored_trunc = detectors.top_bottom_scores(ind_trunc)
        last = scored_trunc.iloc[-1]
        ref = scored_full.iloc[i]
        assert last["bottom_score"] == pytest.approx(ref["bottom_score"], abs=1e-6), (
            f"bottom_score leak at bar {i}: trunc={last['bottom_score']} full={ref['bottom_score']}"
        )
        assert last["top_score"] == pytest.approx(ref["top_score"], abs=1e-6), (
            f"top_score leak at bar {i}: trunc={last['top_score']} full={ref['top_score']}"
        )
        assert int(last["tb_signal"]) == int(ref["tb_signal"]), f"tb_signal leak at bar {i}"


def test_scores_are_bounded_and_nonneg():
    df = _synthetic_ohlcv()
    scored = detectors.top_bottom_scores(indicators.compute_indicator_frame(df))
    assert (scored["bottom_score"] >= 0).all()
    assert (scored["top_score"] >= 0).all()
    max_b = sum(detectors.BOTTOM_WEIGHTS.values())
    max_t = sum(detectors.TOP_WEIGHTS.values())
    assert scored["bottom_score"].max() <= max_b + 1e-9
    assert scored["top_score"].max() <= max_t + 1e-9


def test_signal_fires_somewhere():
    """On data with injected swings, the detector should produce both a bottom
    and a top signal at least once — otherwise it is inert. Uses a low threshold
    so the test exercises the mechanism independent of the production default."""
    df = _synthetic_ohlcv()
    scored = detectors.top_bottom_scores(
        indicators.compute_indicator_frame(df), bottom_threshold=1.0, top_threshold=1.0
    )
    assert (scored["tb_signal"] == 1).any(), "no bottom signal ever fired"
    assert (scored["tb_signal"] == -1).any(), "no top signal ever fired"


def test_latest_is_causal_dict():
    df = _synthetic_ohlcv()
    out = detectors.latest(indicators.compute_indicator_frame(df))
    assert set(out) >= {"bottom_score", "top_score", "tb_signal"}
    assert out["tb_signal"] in (None, "top", "bottom")


def test_divergence_flags_causal():
    """The divergence flag can only turn on at or after the confirmation bar."""
    df = _synthetic_ohlcv()
    ind = indicators.compute_indicator_frame(df)
    bull, bear = detectors._divergence_flags(
        ind["low"].to_numpy(), ind["high"].to_numpy(), ind["rsi"].to_numpy()
    )
    # First WING bars can never be confirmed pivots → must be False.
    assert not bull[: detectors.WING].any()
    assert not bear[: detectors.WING].any()
