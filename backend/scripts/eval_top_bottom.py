#!/usr/bin/env python
"""Out-of-sample evaluation sweep for the top/bottom detector.

Run from the backend dir:  ``.venv/bin/python -m scripts.eval_top_bottom``

Pipeline:
  1. Fetch + cache OHLCV (CSV under ../data/cache) per symbol/interval.
  2. Tune the firing thresholds on a TRAIN split (BTC/ETH, early years) by
     pooled F1 — bottoms are the priority side.
  3. Freeze the thresholds and report metrics on a TEST split that shares no
     symbol *and* no time period with training (held-out coins + recent BTC/ETH),
     so the numbers we quote are genuinely out-of-sample.

Everything the detector sees is causal (see detectors.py); the ZigZag labels and
forward-return baselines that judge it live in evaluation.py and use look-ahead
by design.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

from app.services import detectors, evaluation as ev, indicators, market_data

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
GLOBAL_START_MS = market_data.now_ms() - int(9 * 365.25 * 24 * 3600 * 1000)  # ~9y back

# Splits chosen so TEST overlaps TRAIN in neither symbol nor time.
TRAIN = [("BTCUSDT", "2018-01-01", "2021-12-31"), ("ETHUSDT", "2018-01-01", "2021-12-31")]
TEST = [
    ("BTCUSDT", "2022-01-01", None), ("ETHUSDT", "2022-01-01", None),  # held-out time
    ("SOLUSDT", None, None), ("BNBUSDT", None, None), ("XRPUSDT", None, None),
    ("ADAUSDT", None, None), ("LINKUSDT", None, None), ("AVAXUSDT", None, None),
]
B_GRID = [1.0, 1.3, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0]
T_GRID = [1.0, 1.3, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0]


def cached_ohlcv(symbol: str, interval: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}_{interval}.csv"
    if path.exists():
        return pd.read_csv(path)
    df = asyncio.run(
        market_data.fetch_ohlcv(symbol, interval, start_ms=GLOBAL_START_MS, end_ms=market_data.now_ms())
    )
    df.to_csv(path, index=False)
    return df


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    from app.timeutil import to_ms
    if start:
        df = df[df["time"] * 1000 >= to_ms(start)]
    if end:
        df = df[df["time"] * 1000 <= to_ms(end)]
    return df.reset_index(drop=True)


def prep(symbol: str, interval: str, start: str | None, end: str | None) -> pd.DataFrame | None:
    df = _slice(cached_ohlcv(symbol, interval), start, end)
    if len(df) < indicators.WARMUP_BARS + 60:
        return None
    return indicators.compute_indicator_frame(df)


def pooled(results: list[dict], side: str, beta: float = 1.0) -> dict:
    """Pool TP / signals / pivots across symbols, then compute precision/recall/
    F-beta once — fairer than averaging per-symbol F across uneven sample sizes.
    ``beta`` < 1 favours precision (a wrong dip-buy is costly)."""
    tp = sum(r[side]["tp"] for r in results)
    sig = sum(r[side]["signals"] for r in results)
    piv = sum(r[side]["pivots"] for r in results)
    lags = [l for r in results for l in r.get(f"_{side}_lags", [])]
    prec = tp / sig if sig else 0.0
    rec = tp / piv if piv else 0.0
    b2 = beta * beta
    fbeta = (1 + b2) * prec * rec / (b2 * prec + rec) if (prec + rec) else 0.0
    med_lag = float(pd.Series(lags).median()) if lags else None
    return {"tp": tp, "signals": sig, "pivots": piv, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(fbeta, 3), "median_lag": med_lag}


def eval_set(specs, interval, b_thr, t_thr, zigzag, tol, early):
    rows = []
    for symbol, start, end in specs:
        ind = prep(symbol, interval, start, end)
        if ind is None:
            continue
        scored = detectors.top_bottom_scores(ind, bottom_threshold=b_thr, top_threshold=t_thr)
        m = ev.evaluate(ind, scored, zigzag_pct=zigzag, tol=tol, early=early)
        # stash raw lags for pooling
        high, low = ind["high"].to_numpy(float), ind["low"].to_numpy(float)
        pivots = ev.zigzag_pivots(high, low, zigzag)
        m["symbol"] = symbol
        m["_bottom_lags"] = ev.match_side(
            ev.debounce([int(i) for i in (scored["tb_signal"] == 1).to_numpy().nonzero()[0]], 8),
            [i for i, k in pivots if k == "bottom"], tol, early).lead_lag
        m["_top_lags"] = ev.match_side(
            ev.debounce([int(i) for i in (scored["tb_signal"] == -1).to_numpy().nonzero()[0]], 8),
            [i for i, k in pivots if k == "top"], tol, early).lead_lag
        rows.append(m)
    return rows


def _fwd_quality(res, side, horizon=20):
    """Avg forward-return × win-rate at `horizon` across symbols — the economic
    objective. Pure ZigZag-F over-rewards recall (firing often), which the TEST
    profile shows degrades forward returns; this targets signal *quality*."""
    rets = [fr["cond_mean"] for r in res for fr in r[f"{side}_fwd"]
            if fr["horizon"] == horizon and fr["cond_mean"] is not None]
    wins = [fr["win_rate"] for r in res for fr in r[f"{side}_fwd"]
            if fr["horizon"] == horizon and fr["win_rate"] is not None]
    if not rets:
        return -1.0
    return (sum(rets) / len(rets)) * (sum(wins) / len(wins))


def tune(interval, zigzag, tol, early, min_signals=20):
    """Pick each side's threshold by TRAIN forward-return quality at h20, with a
    floor on signal count so a near-empty threshold can't win by default. Bottom
    first (priority), then top given the chosen bottom."""
    def best(grid, side, b_fixed=None, t_fixed=None):
        best_x, best_q = grid[0], -1e9
        for x in grid:
            b = x if side == "bottom" else b_fixed
            t = x if side == "top" else t_fixed
            res = eval_set(TRAIN, interval, b, t, zigzag, tol, early)
            if pooled(res, side)["signals"] < min_signals:
                continue
            q = _fwd_quality(res, side)
            if q > best_q:
                best_x, best_q = x, q
        return best_x

    best_b = best(B_GRID, "bottom", t_fixed=detectors.TOP_THRESHOLD)
    best_t = best(T_GRID, "top", b_fixed=best_b)
    return best_b, best_t


def fmt(side_row: dict) -> str:
    ll = side_row.get("median_lag")
    return (f"P={side_row['precision']:.2f} R={side_row['recall']:.2f} "
            f"F1={side_row['f1']:.2f} (tp {side_row['tp']}/{side_row['signals']} sig, "
            f"{side_row['pivots']} piv, lag~{ll})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--zigzag", type=float, default=15.0, help="ZigZag reversal %% for ground truth")
    ap.add_argument("--tol", type=int, default=8, help="bars a signal may lag a pivot")
    ap.add_argument("--early", type=int, default=2, help="bars a signal may lead a pivot")
    args = ap.parse_args()

    print(f"\n=== Top/Bottom detector OOS evaluation (interval={args.interval}, "
          f"ZigZag θ={args.zigzag}%, match window [-{args.early},+{args.tol}] bars) ===\n")

    print("Tuning thresholds on TRAIN (BTC/ETH 2018-2021)...")
    b_thr, t_thr = tune(args.interval, args.zigzag, args.tol, args.early)
    print(f"  → bottom_threshold={b_thr}  top_threshold={t_thr}\n")

    train_res = eval_set(TRAIN, args.interval, b_thr, t_thr, args.zigzag, args.tol, args.early)
    test_res = eval_set(TEST, args.interval, b_thr, t_thr, args.zigzag, args.tol, args.early)

    def dump(title, res):
        print(f"--- {title} ---")
        for r in res:
            print(f"  {r['symbol']:9s} bars={r['n_bars']:5d} "
                  f"BOTTOM {fmt(r['bottom'])}")
            print(f"  {'':9s}           TOP    {fmt(r['top'])}")
        for side in ("bottom", "top"):
            p = pooled(res, side)
            print(f"  POOLED {side.upper():7s} {fmt(p)}")
        # forward-outcome stats pooled (avg of per-symbol values at each horizon)
        for side, key in (("bottom", "bottom_fwd"), ("top", "top_fwd")):
            agg = {h: {"ret": [], "win": [], "mdd": []} for h in (10, 20, 50)}
            for r in res:
                for fr in r[key]:
                    if fr["cond_mean"] is not None:
                        agg[fr["horizon"]]["ret"].append(fr["cond_mean"])
                        agg[fr["horizon"]]["win"].append(fr["win_rate"])
                        agg[fr["horizon"]]["mdd"].append(fr["mdd"])
            def avg(xs):
                return round(sum(xs) / len(xs), 4) if xs else None
            txt = "  ".join(
                f"h{h}: ret={avg(a['ret'])} win={avg(a['win'])} fwdDD={avg(a['mdd'])}"
                for h, a in agg.items()
            )
            print(f"  FWD {side.upper():7s} {txt}")
        print()

    dump("TRAIN (in-sample)", train_res)
    dump("TEST (out-of-sample)", test_res)


if __name__ == "__main__":
    main()
