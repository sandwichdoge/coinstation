"""Tests for the evaluation judge: ZigZag labeling, matching, forward-return."""
from __future__ import annotations

import numpy as np

from app.services import evaluation as ev


def test_zigzag_finds_obvious_swings():
    # Triangle wave: 100 -> 130 -> 100 -> 130 (two +30% / -23% legs).
    up1 = np.linspace(100, 130, 30)
    dn1 = np.linspace(130, 100, 30)
    up2 = np.linspace(100, 130, 30)
    close = np.concatenate([up1, dn1, up2])
    pivots = ev.zigzag_pivots(close, close, threshold_pct=15.0)
    kinds = [k for _, k in pivots]
    assert "top" in kinds and "bottom" in kinds
    # Pivots must strictly alternate.
    for a, b in zip(kinds, kinds[1:]):
        assert a != b
    # The top near the first peak (~index 29) should be detected close to it.
    top_idx = [i for i, k in pivots if k == "top"]
    assert any(25 <= i <= 32 for i in top_idx)


def test_zigzag_ignores_subthreshold_noise():
    # Pure 2% wiggles never reverse 15% → no confirmed pivots.
    rng = np.random.default_rng(0)
    close = 100 * (1 + 0.02 * np.sin(np.linspace(0, 20, 200)))
    pivots = ev.zigzag_pivots(close, close, threshold_pct=15.0)
    assert pivots == []


def test_match_side_respects_window_and_lag():
    # Pivot at 100; signal 3 bars late should match, 20 bars late should not.
    m = ev.match_side(signal_idx=[103], pivot_idx=[100], tol=8, early=2)
    assert m.tp == 1 and m.lead_lag == [3]
    m2 = ev.match_side(signal_idx=[120], pivot_idx=[100], tol=8, early=2)
    assert m2.tp == 0


def test_match_side_one_to_one():
    # Two signals near one pivot → only one true positive (precision penalised).
    m = ev.match_side(signal_idx=[101, 104], pivot_idx=[100], tol=8, early=2)
    assert m.tp == 1 and m.n_signals == 2
    assert m.precision == 0.5


def test_debounce_collapses_clusters():
    assert ev.debounce([10, 11, 12, 30, 31], gap=8) == [10, 30]


def test_forward_stats_bottom():
    # Monotonic rise: a bottom signal should win and carry no forward drawdown.
    close = np.linspace(100, 200, 100)
    res = ev.forward_stats(close, signal_idx=[5], horizon=10, side="bottom")
    assert res["cond_mean"] is not None and res["cond_mean"] > 0
    assert res["win_rate"] == 1.0
    assert res["mdd"] == 0.0 or res["mdd"] >= -1e-9  # rising series → no dip after entry
