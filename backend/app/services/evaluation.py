"""Detector evaluation: the *judge* half of the top/bottom system.

Everything here is allowed to use the full series, future included — it defines
ground truth and scores the detector against it. This is the one place look-ahead
is legitimate, and it must stay out of ``detectors.py`` / ``indicators.py`` (the
causal inference path). Keeping the judge in its own module makes that boundary
obvious and lets the labeler be unit-tested in isolation.

Ground truth is a **percent-reversal ZigZag**: the alternating significant highs
and lows a human would circle on the chart, parameterised by the reversal
threshold θ (≈15% for "major swing turns"). Detector signals are then matched to
these pivots within a bar tolerance to give precision / recall / F1 and a
lead-lag distribution, plus a label-independent **forward-return lift** check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---- ground-truth labeling (look-ahead allowed) ---------------------------

def zigzag_pivots(
    high: np.ndarray, low: np.ndarray, threshold_pct: float
) -> list[tuple[int, str]]:
    """Confirmed ZigZag swing pivots as ``(bar_index, 'top'|'bottom')``.

    Walks the series tracking the running extreme of the current leg; a leg
    reverses (and its extreme is confirmed as a pivot) once price retraces
    ``threshold_pct`` from that extreme. The final, unconfirmed leg is dropped —
    so every returned pivot is a *completed* major swing. Uses future bars by
    design; this is ground truth, never fed to the detector.
    """
    n = len(high)
    th = threshold_pct / 100.0
    if n < 2:
        return []
    pivots: list[tuple[int, str]] = []
    trend = 0  # 0 unknown, +1 in an up-leg (seeking a top), -1 down-leg (seeking a bottom)
    hi, hi_i = high[0], 0
    lo, lo_i = low[0], 0

    for i in range(1, n):
        if trend >= 0:
            if high[i] > hi:
                hi, hi_i = high[i], i
        if trend <= 0:
            if low[i] < lo:
                lo, lo_i = low[i], i

        if trend == 0:
            if high[i] >= lo * (1 + th):
                pivots.append((lo_i, "bottom")); trend = 1; hi, hi_i = high[i], i
            elif low[i] <= hi * (1 - th):
                pivots.append((hi_i, "top")); trend = -1; lo, lo_i = low[i], i
        elif trend == 1:
            if low[i] <= hi * (1 - th):
                pivots.append((hi_i, "top")); trend = -1; lo, lo_i = low[i], i
        else:  # trend == -1
            if high[i] >= lo * (1 + th):
                pivots.append((lo_i, "bottom")); trend = 1; hi, hi_i = high[i], i
    return pivots


# ---- signal extraction & debouncing ---------------------------------------

def debounce(signal_idx: list[int], gap: int) -> list[int]:
    """Collapse signals closer than ``gap`` bars into the first of the cluster —
    one reaction, not many. Prevents a detector that latches for several bars
    from inflating its signal count (and deflating precision)."""
    kept: list[int] = []
    for i in sorted(signal_idx):
        if not kept or i - kept[-1] >= gap:
            kept.append(i)
    return kept


# ---- matching & metrics ----------------------------------------------------

@dataclass
class SideMetrics:
    side: str
    n_pivots: int
    n_signals: int
    tp: int
    precision: float
    recall: float
    f1: float
    lead_lag: list[int] = field(default_factory=list)  # signed (signal-pivot); + = late

    @property
    def median_lag(self) -> float | None:
        return float(np.median(self.lead_lag)) if self.lead_lag else None

    def as_row(self) -> dict:
        ll = self.lead_lag
        return {
            "side": self.side, "pivots": self.n_pivots, "signals": self.n_signals,
            "tp": self.tp, "precision": round(self.precision, 3),
            "recall": round(self.recall, 3), "f1": round(self.f1, 3),
            "median_lag": self.median_lag,
            "lag_iqr": (
                [int(np.percentile(ll, 25)), int(np.percentile(ll, 75))] if ll else None
            ),
        }


def match_side(
    signal_idx: list[int], pivot_idx: list[int], tol: int, early: int
) -> SideMetrics:
    """Greedy nearest-match of signals to pivots within ``[-early, +tol]`` bars.

    A causal detector confirms a turn *after* it forms, so the window is
    asymmetric: a signal may sit up to ``tol`` bars late or ``early`` bars before
    a pivot. Each pivot consumes at most one signal and vice-versa.
    """
    pivots = sorted(pivot_idx)
    signals = sorted(signal_idx)
    used = [False] * len(signals)
    tp = 0
    lead_lag: list[int] = []
    for p in pivots:
        best_j, best_d = -1, None
        for j, s in enumerate(signals):
            if used[j]:
                continue
            d = s - p
            if -early <= d <= tol and (best_d is None or abs(d) < abs(best_d)):
                best_j, best_d = j, d
        if best_j >= 0:
            used[best_j] = True
            tp += 1
            lead_lag.append(best_d)
    n_sig, n_piv = len(signals), len(pivots)
    precision = tp / n_sig if n_sig else 0.0
    recall = tp / n_piv if n_piv else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return SideMetrics("", n_piv, n_sig, tp, precision, recall, f1, lead_lag)


def forward_stats(
    close: np.ndarray, signal_idx: list[int], horizon: int, side: str
) -> dict:
    """Forward-outcome stats after a signal — the label-free quality check.

    For trending assets the unconditional baseline is dominated by long bull
    runs, so "beat the average bar" (``lift``) is a harsh, momentum-biased bar
    for a *mean-reversion entry* — a profitable dip-buy can still trail an
    always-long average. So we report what actually matters for the dip-buyer
    alongside it:

      * ``cond_mean`` — absolute mean forward return after the signal.
      * ``win_rate`` — P(forward return > 0); did the call go the right way.
      * ``mdd`` — median *forward* max-drawdown: the worst dip within the horizon
        after entry. A good bottom is followed by a shallow one (you don't get
        stopped out). For tops it is the forward max-*runup* instead (the pain of
        having sold too early), so the sign convention keeps "smaller = better".
      * ``lift`` — kept as a secondary, sign-corrected vs-baseline read.

    The economic verdict still belongs to the P&L backtest (Phase 3); these are
    the threshold-free sanity checks that don't depend on the ZigZag θ.
    """
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = close[horizon:] / close[: n - horizon] - 1.0
    baseline = float(np.nanmean(fwd))
    valid = [s for s in signal_idx if s < n - horizon]
    if not valid:
        return {"horizon": horizon, "n": 0, "cond_mean": None, "baseline": round(baseline, 4),
                "lift": None, "win_rate": None, "mdd": None}
    cond = float(np.nanmean(fwd[valid]))
    raw_lift = cond - baseline
    lift = raw_lift if side == "bottom" else -raw_lift
    if side == "bottom":
        wins = sum(1 for s in valid if fwd[s] > 0)
        # worst close-to-close dip in the `horizon` bars after entry
        mdds = [float(np.min(close[s + 1 : s + horizon + 1] / close[s] - 1.0)) for s in valid]
    else:
        wins = sum(1 for s in valid if fwd[s] < 0)
        mdds = [float(np.max(close[s + 1 : s + horizon + 1] / close[s] - 1.0)) for s in valid]
    return {
        "horizon": horizon, "n": len(valid),
        "cond_mean": round(cond, 4), "baseline": round(baseline, 4), "lift": round(lift, 4),
        "win_rate": round(wins / len(valid), 3),
        "mdd": round(float(np.median(mdds)), 4),
    }


def evaluate(
    df_ind: pd.DataFrame,
    scored: pd.DataFrame,
    zigzag_pct: float = 15.0,
    tol: int = 8,
    early: int = 2,
    debounce_gap: int = 8,
    horizons: tuple[int, ...] = (10, 20, 50),
) -> dict:
    """Full evaluation of one scored frame against ZigZag ground truth.

    Returns per-side detection metrics and forward-return lift at each horizon.
    """
    high = df_ind["high"].to_numpy(dtype=float)
    low = df_ind["low"].to_numpy(dtype=float)
    close = df_ind["close"].to_numpy(dtype=float)

    pivots = zigzag_pivots(high, low, zigzag_pct)
    bottom_pivots = [i for i, k in pivots if k == "bottom"]
    top_pivots = [i for i, k in pivots if k == "top"]

    sig = scored["tb_signal"].to_numpy()
    bottom_sig = debounce([int(i) for i in np.where(sig == 1)[0]], debounce_gap)
    top_sig = debounce([int(i) for i in np.where(sig == -1)[0]], debounce_gap)

    bm = match_side(bottom_sig, bottom_pivots, tol, early); bm.side = "bottom"
    tm = match_side(top_sig, top_pivots, tol, early); tm.side = "top"

    return {
        "n_bars": len(df_ind),
        "zigzag_pct": zigzag_pct,
        "bottom": bm.as_row(),
        "top": tm.as_row(),
        "bottom_fwd": [forward_stats(close, bottom_sig, h, "bottom") for h in horizons],
        "top_fwd": [forward_stats(close, top_sig, h, "top") for h in horizons],
    }
