"""Technical indicators, implemented directly on pandas so the math is auditable
and identical whether it runs for the live chart or inside the backtest.

`compute_indicator_frame` is the single source of truth: the chart payload, the
AI snapshot and the backtest strategies all derive from the columns it adds."""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- primitive indicators -------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (smoothed with alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # All-gains windows (avg_loss == 0) → RSI 100, not NaN.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: running total of volume signed by the bar's direction.
    Rising OBV while price is flat/falling is the classic accumulation tell."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def chaikin_money_flow(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """Chaikin Money Flow: volume weighted by where each close lands within its
    bar's range, summed over `period`. Positive = buying pressure (closes near
    highs); negative = selling pressure. Bounded roughly to [-1, 1]."""
    rng = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / rng  # money-flow multiplier ∈ [-1, 1]
    mfv = (mfm * volume).fillna(0.0)              # flat-bar (high==low) contributes 0
    vol_sum = volume.rolling(window=period, min_periods=period).sum().replace(0.0, np.nan)
    return mfv.rolling(window=period, min_periods=period).sum() / vol_sum


# ---- combined frame -------------------------------------------------------

EMA_PERIODS = (20, 50, 200)
RSI_PERIOD = 14
MACD_PARAMS = (12, 26, 9)
BB_PARAMS = (20, 2.0)
VOL_MA_PERIOD = 20
CMF_PERIOD = 20
RECENT_WINDOW = 20  # bars of recent context for drift / basing / money-flow reads
PIVOT_WING = 3      # bars required each side of a bar for it to count as a swing pivot
LEVEL_TOL = 0.02    # pivots within 2% of each other are merged into one S/R level
MIN_TOUCH_GAP = 5   # pivots closer than this many bars are one reaction, not distinct touches

# Candles of warm-up to fetch before a window so EMA200 etc. are valid at start.
WARMUP_BARS = 250


def compute_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with indicator columns appended."""
    out = df.copy()
    if out.empty:
        return out
    close = out["close"]
    for p in EMA_PERIODS:
        out[f"ema{p}"] = ema(close, p)
    out["rsi"] = rsi(close, RSI_PERIOD)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close, *MACD_PARAMS)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(close, *BB_PARAMS)
    out["vol_ma"] = sma(out["volume"], VOL_MA_PERIOD)
    out["obv"] = obv(close, out["volume"])
    out["cmf"] = chaikin_money_flow(out["high"], out["low"], close, out["volume"], CMF_PERIOD)
    return out


# ---- support / resistance -------------------------------------------------

def support_resistance(df: pd.DataFrame, price: float, wing: int = PIVOT_WING,
                       tol: float = LEVEL_TOL, min_gap: int = MIN_TOUCH_GAP) -> dict:
    """Swing-pivot support/resistance around `price`, with the next level beyond.

    A *swing pivot* is a bar whose high (low) is the highest (lowest) in a
    window of `wing` bars on each side — a classic fractal. The last `wing` bars
    can't be confirmed pivots, which is correct: it uses no future data. Pivots
    within `tol` of each other are merged into one level.

    A level's strength is its *distinct touch count*: separate reactions to the
    level, not raw pivot bars. Pivots closer than `min_gap` bars (e.g. a flat
    multi-bar base, where every bar shares the window extreme) are one reaction
    and counted once — so the count reflects how often price genuinely returned.

    Returns the nearest support/resistance each side of `price`, plus the next
    level beyond each (``support_next`` / ``resistance_next``) — i.e. where price
    is likely headed if the nearest level is decisively breached.
    """
    n = len(df)
    if n < 2 * wing + 1 or not price:
        return {}
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    # (bar_index, price) for each confirmed swing pivot; index drives touch timing.
    pivots: list[tuple[int, float]] = []
    for i in range(wing, n - wing):
        win = slice(i - wing, i + wing + 1)
        if low[i] == low[win].min():
            pivots.append((i, low[i]))
        if high[i] == high[win].max():
            pivots.append((i, high[i]))
    if not pivots:
        return {}

    # Cluster by price into levels. Anchor each pivot to the running cluster
    # *mean* (not the previous pivot) so a chain of nearby pivots can't drift the
    # level arbitrarily wide past `tol`.
    pivots.sort(key=lambda p: p[1])
    clusters: list[list[tuple[int, float]]] = [[pivots[0]]]
    for idx, p in pivots[1:]:
        mean = sum(q for _, q in clusters[-1]) / len(clusters[-1])
        if p <= mean * (1 + tol):
            clusters[-1].append((idx, p))
        else:
            clusters.append([(idx, p)])

    # Per level: price = mean, touches = pivots separated by >= min_gap bars.
    levels: list[tuple[float, int]] = []
    for c in clusters:
        lv = sum(q for _, q in c) / len(c)
        touches, last = 0, None
        for b in sorted(idx for idx, _ in c):
            if last is None or b - last >= min_gap:
                touches += 1
                last = b
        levels.append((lv, touches))
    levels.sort(key=lambda x: x[0])

    below = [(lv, t) for lv, t in levels if lv <= price]
    above = [(lv, t) for lv, t in levels if lv > price]

    out: dict = {}
    if below:
        lv, t = below[-1]  # nearest level at/under price
        out["support"] = round(lv, 6)
        out["support_touches"] = t
        out["support_dist_pct"] = round((price - lv) / price * 100, 2)
        if len(below) >= 2:
            nlv, nt = below[-2]  # next support down — the target if support breaks
            out["support_next"] = round(nlv, 6)
            out["support_next_touches"] = nt
    if above:
        lv, t = above[0]  # nearest level above price
        out["resistance"] = round(lv, 6)
        out["resistance_touches"] = t
        out["resistance_dist_pct"] = round((lv - price) / price * 100, 2)
        if len(above) >= 2:
            nlv, nt = above[1]  # next resistance up — the target if resistance breaks
            out["resistance_next"] = round(nlv, 6)
            out["resistance_next_touches"] = nt
    return out


# ---- reversal / exhaustion (multi-candle) ---------------------------------

REVERSAL_WINDOW = 2 * RECENT_WINDOW  # bars scanned for divergences / swings
RSI_DIV_MARGIN = 2.0                 # min RSI gap (points) for a real divergence
VOL_CLIMAX_MULT = 2.0                # volume >= this * MA to count as a climax bar
CLIMAX_LOOKBACK = 6                  # only a recent spike is an actionable climax
WICK_DOMINANCE = 2.0                 # rejection wick must be >= this * the candle body
WICK_RANGE_FRAC = 0.5                # ... and span >= this fraction of the candle's range
EXTREME_TOL = 0.15                   # within this fraction of the window range counts as "at the extreme"


def reversal_signals(df_ind: pd.DataFrame, price: float, wing: int = PIVOT_WING,
                     window: int = REVERSAL_WINDOW) -> dict:
    """Reads that span many candles rather than the latest bar — the turns the
    single-bar snapshot is structurally blind to:

      * **RSI divergence** — price prints a lower low (or higher high) that the
        oscillator refuses to confirm: the classic momentum-exhaustion reversal.
      * **MACD-histogram momentum turn** — how many consecutive bars the
        histogram has been rising or falling, catching a turn *before* the
        signal-line crossover the snapshot already reports.
      * **Volume climax** — a volume spike on a wide bar pinned to the window's
        extreme: a selling climax at the low (capitulation / bottom) or a buying
        climax at the high (blow-off / top).

    Uses only confirmed swing pivots (the last `wing` bars stay unconfirmed), so
    like `support_resistance` it never peeks at future data.
    """
    out: dict = {}
    n = len(df_ind)
    if not price or n < 2 * wing + 3:
        return out
    tail = df_ind.tail(window)
    low = tail["low"].to_numpy(dtype=float)
    high = tail["high"].to_numpy(dtype=float)
    close = tail["close"].to_numpy(dtype=float)
    m = len(tail)
    win_low, win_high = float(low.min()), float(high.max())
    rng = win_high - win_low  # window's full range, the yardstick for "at the extreme"

    swing_lows: list[int] = []
    swing_highs: list[int] = []
    for i in range(wing, m - wing):
        seg = slice(i - wing, i + wing + 1)
        if low[i] == low[seg].min():
            swing_lows.append(i)
        if high[i] == high[seg].max():
            swing_highs.append(i)

    def _distinct(idxs: list[int], vals: np.ndarray, pick_min: bool) -> list[int]:
        """Collapse pivots within MIN_TOUCH_GAP bars (one reaction, e.g. a flat
        multi-bar base) into a single swing, keeping the most extreme bar — so
        consecutive bars sharing a window extreme don't masquerade as two swings."""
        kept: list[int] = []
        for i in idxs:
            if kept and i - kept[-1] < MIN_TOUCH_GAP:
                if (vals[i] < vals[kept[-1]]) == pick_min:
                    kept[-1] = i
            else:
                kept.append(i)
        return kept

    swing_lows = _distinct(swing_lows, low, pick_min=True)
    swing_highs = _distinct(swing_highs, high, pick_min=False)

    # --- RSI divergence between the two most recent like swings. Restrict each
    #     side to its own half of the range (lows in <50, highs in >50) so we
    #     only flag exhaustion where it carries weight. If both fire, the more
    #     recent confirming swing wins. ---
    if "rsi" in tail:
        rv = tail["rsi"].to_numpy(dtype=float)
        bull_at = bear_at = -1
        if len(swing_lows) >= 2:
            a, b = swing_lows[-2], swing_lows[-1]
            if (not np.isnan(rv[a]) and not np.isnan(rv[b])
                    and low[b] < low[a] and rv[b] > rv[a] + RSI_DIV_MARGIN and rv[b] < 50):
                bull_at = b
        if len(swing_highs) >= 2:
            a, b = swing_highs[-2], swing_highs[-1]
            if (not np.isnan(rv[a]) and not np.isnan(rv[b])
                    and high[b] > high[a] and rv[b] < rv[a] - RSI_DIV_MARGIN and rv[b] > 50):
                bear_at = b
        if bull_at >= 0 or bear_at >= 0:
            out["rsi_divergence"] = "bullish" if bull_at >= bear_at else "bearish"

    # --- MACD-histogram run: trailing count of same-direction steps. ---
    if "macd_hist" in tail:
        hist = tail["macd_hist"].to_numpy(dtype=float)
        streak, direction = 0, 0
        for d in np.diff(hist)[::-1]:
            if np.isnan(d) or d == 0:
                break
            s = 1 if d > 0 else -1
            if direction == 0:
                direction = s
            if s != direction:
                break
            streak += 1
        if streak >= 2 and direction:
            out["macd_hist_dir"] = "rising" if direction > 0 else "falling"
            out["macd_hist_streak"] = streak
            if not np.isnan(hist[-1]):
                out["macd_hist_below_zero"] = bool(hist[-1] < 0)

    # --- Volume climax: a recent spike on a wide bar pinned to the extreme. ---
    if "vol_ma" in tail and rng > 0:
        vol = tail["volume"].to_numpy(dtype=float)
        vma = tail["vol_ma"].to_numpy(dtype=float)
        for j in range(m - 1, max(m - 1 - CLIMAX_LOOKBACK, -1), -1):
            if np.isnan(vol[j]) or np.isnan(vma[j]) or vma[j] <= 0 or vol[j] < VOL_CLIMAX_MULT * vma[j]:
                continue
            bar_rng = high[j] - low[j]
            if bar_rng <= 0:
                continue
            close_pos = (close[j] - low[j]) / bar_rng          # 0 = closed at low, 1 = at high
            near_low = (low[j] - win_low) / rng <= EXTREME_TOL
            near_high = (win_high - high[j]) / rng <= EXTREME_TOL
            if near_low and close_pos <= 0.5:
                out["volume_climax"] = "selling"
                out["volume_climax_bars_ago"] = m - 1 - j
                break
            if near_high and close_pos >= 0.5:
                out["volume_climax"] = "buying"
                out["volume_climax_bars_ago"] = m - 1 - j
                break

    # --- Rejection wick (pin bar): a recent candle that tags the window extreme
    #     then closes back away from it on a long shadow — buyers (or sellers)
    #     rejecting the probe. The candle-anatomy partner to the volume climax
    #     above, and the read the close-only scoring is otherwise blind to. A
    #     long lower shadow at the lows is demand defending the level (bullish);
    #     a long upper shadow at the highs is supply defending it (bearish). ---
    if "open" in tail and rng > 0:
        op = tail["open"].to_numpy(dtype=float)
        for j in range(m - 1, max(m - 1 - CLIMAX_LOOKBACK, -1), -1):
            o, c, h, l = op[j], close[j], high[j], low[j]
            if any(np.isnan(x) for x in (o, c, h, l)):
                continue
            bar_rng = h - l
            if bar_rng <= 0:
                continue
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            near_low = (l - win_low) / rng <= EXTREME_TOL
            near_high = (win_high - h) / rng <= EXTREME_TOL
            # A long shadow relative to both the body and the bar's own range.
            bull = (near_low and lower_wick >= WICK_DOMINANCE * body
                    and lower_wick >= WICK_RANGE_FRAC * bar_rng)
            bear = (near_high and upper_wick >= WICK_DOMINANCE * body
                    and upper_wick >= WICK_RANGE_FRAC * bar_rng)
            if bull and not bear:
                out["wick_rejection"] = "bullish"
                out["wick_rejection_bars_ago"] = m - 1 - j
                break
            if bear and not bull:
                out["wick_rejection"] = "bearish"
                out["wick_rejection_bars_ago"] = m - 1 - j
                break

    return out


# ---- serialization helpers ------------------------------------------------

def _points(times: pd.Series, series: pd.Series) -> list[dict]:
    return [
        {"time": int(t), "value": round(float(v), 8)}
        for t, v in zip(times, series)
        if pd.notna(v)
    ]


def build_payload(df_ind: pd.DataFrame) -> dict:
    """Lightweight-charts-friendly series, NaN points omitted."""
    if df_ind.empty:
        return {"overlays": {}, "rsi": {}, "macd": {}, "volume": {}}
    t = df_ind["time"]

    overlays = {
        key: _points(t, df_ind[key])
        for key in ("ema20", "ema50", "ema200", "bb_upper", "bb_mid", "bb_lower")
        if key in df_ind
    }

    macd_hist = [
        {
            "time": int(tt),
            "value": round(float(v), 8),
            "color": "#26a69a" if v >= 0 else "#ef5350",
        }
        for tt, v in zip(t, df_ind["macd_hist"])
        if pd.notna(v)
    ]
    volume = [
        {
            "time": int(tt),
            "value": round(float(v), 4),
            "color": "#26a69a80" if c >= o else "#ef535080",
        }
        for tt, v, o, c in zip(t, df_ind["volume"], df_ind["open"], df_ind["close"])
        if pd.notna(v)
    ]

    return {
        "overlays": overlays,
        "rsi": {"period": RSI_PERIOD, "points": _points(t, df_ind["rsi"])},
        "macd": {
            "macd": _points(t, df_ind["macd"]),
            "signal": _points(t, df_ind["macd_signal"]),
            "hist": macd_hist,
        },
        "volume": {"points": volume, "ma": _points(t, df_ind["vol_ma"])},
    }


def latest_snapshot(df_ind: pd.DataFrame) -> dict:
    """Compact technical state of the most recent bar, plus human-readable
    signals. Fed to the AI and shown in the analysis panel."""
    if df_ind.empty:
        return {}
    last = df_ind.iloc[-1]
    prev = df_ind.iloc[-2] if len(df_ind) >= 2 else last

    def f(x) -> float | None:
        return None if pd.isna(x) else round(float(x), 6)

    price = f(last["close"])
    snap: dict = {
        "time": int(last["time"]),
        "price": price,
        "rsi": f(last.get("rsi")),
        "macd": f(last.get("macd")),
        "macd_signal": f(last.get("macd_signal")),
        "macd_hist": f(last.get("macd_hist")),
        "ema20": f(last.get("ema20")),
        "ema50": f(last.get("ema50")),
        "ema200": f(last.get("ema200")),
        "bb_upper": f(last.get("bb_upper")),
        "bb_lower": f(last.get("bb_lower")),
        "volume": f(last.get("volume")),
        "vol_ma": f(last.get("vol_ma")),
        "cmf": f(last.get("cmf")),
    }

    # --- recent-window context: live drift, basing/volatility and net volume
    #     flow over the last RECENT_WINDOW bars. These feed accumulation /
    #     distribution detection in the analysis layer. ---
    n = min(RECENT_WINDOW, len(df_ind) - 1)
    if n >= 5 and price is not None:
        tail = df_ind.tail(n + 1)
        closes = tail["close"].astype(float)
        first_c = float(closes.iloc[0])
        if first_c:
            snap["recent_change_pct"] = round((price - first_c) / first_c * 100, 2)
        hi, lo = float(tail["high"].max()), float(tail["low"].min())
        if hi > lo:
            snap["range_pct"] = round((hi - lo) / price * 100, 2)
            snap["price_position"] = round((price - lo) / (hi - lo), 3)  # 0 = at low, 1 = at high
        if "obv" in df_ind:
            obv_tail = tail["obv"].astype(float)
            vol_sum = float(tail["volume"].iloc[1:].sum())  # volume backing the OBV moves
            if vol_sum:
                # Net OBV change as a fraction of volume traded: +1 = pure inflow, -1 = pure outflow.
                snap["obv_trend_pct"] = round((float(obv_tail.iloc[-1]) - float(obv_tail.iloc[0])) / vol_sum, 3)
        rets = closes.pct_change().dropna()
        if len(rets) >= 6:
            base_vol = float(rets.std())
            recent_vol = float(rets.tail(max(3, len(rets) // 3)).std())
            if base_vol:
                snap["vol_contraction"] = round(recent_vol / base_vol, 3)  # < 1 = volatility contracting

    # --- swing support/resistance: price reaction levels from the whole frame
    #     (warm-up included), so basing near a long-held support is visible. ---
    if price is not None:
        snap.update(support_resistance(df_ind, price))
        snap.update(reversal_signals(df_ind, price))

    signals: list[str] = []
    cmf_v = snap.get("cmf")
    if cmf_v is not None:
        if cmf_v >= 0.05:
            signals.append("Money flow positive (CMF)")
        elif cmf_v <= -0.05:
            signals.append("Money flow negative (CMF)")
    if snap["rsi"] is not None:
        if snap["rsi"] >= 70:
            signals.append("RSI overbought (>=70)")
        elif snap["rsi"] <= 30:
            signals.append("RSI oversold (<=30)")
    m_now, m_prev = last.get("macd"), prev.get("macd")
    s_now, s_prev = last.get("macd_signal"), prev.get("macd_signal")
    if all(pd.notna(x) for x in (m_now, m_prev, s_now, s_prev)):
        if m_prev <= s_prev and m_now > s_now:
            signals.append("MACD bullish crossover")
        elif m_prev >= s_prev and m_now < s_now:
            signals.append("MACD bearish crossover")
        elif snap["macd_hist"] is not None:
            signals.append(
                "MACD momentum positive" if snap["macd_hist"] >= 0 else "MACD momentum negative"
            )
    if snap["ema50"] is not None and snap["ema200"] is not None:
        signals.append(
            "Uptrend (EMA50 > EMA200)" if snap["ema50"] > snap["ema200"] else "Downtrend (EMA50 < EMA200)"
        )
    if price is not None and snap["ema200"] is not None:
        signals.append("Price above EMA200" if price >= snap["ema200"] else "Price below EMA200")
    sup_d, res_d = snap.get("support_dist_pct"), snap.get("resistance_dist_pct")
    if sup_d is not None and sup_d <= 3.0 and (snap.get("support_touches") or 0) >= 2:
        signals.append(f"Testing support (held {snap['support_touches']}x)")
    elif res_d is not None and res_d <= 3.0 and (snap.get("resistance_touches") or 0) >= 2:
        signals.append(f"Capped at resistance (rejected {snap['resistance_touches']}x)")
    if snap.get("rsi_divergence") == "bullish":
        signals.append("Bullish RSI divergence")
    elif snap.get("rsi_divergence") == "bearish":
        signals.append("Bearish RSI divergence")
    if snap.get("volume_climax") == "selling":
        signals.append("Selling climax (volume)")
    elif snap.get("volume_climax") == "buying":
        signals.append("Buying climax (volume)")
    if snap.get("wick_rejection") == "bullish":
        signals.append("Bullish rejection wick")
    elif snap.get("wick_rejection") == "bearish":
        signals.append("Bearish rejection wick")
    if (snap.get("macd_hist_streak") or 0) >= 3:
        signals.append(f"MACD histogram {snap['macd_hist_dir']} {snap['macd_hist_streak']} bars")
    snap["signals"] = signals

    first_close = df_ind.iloc[0]["close"]
    if price is not None and pd.notna(first_close) and float(first_close) != 0:
        snap["window_change_pct"] = round((price - float(first_close)) / float(first_close) * 100, 2)
    return snap
