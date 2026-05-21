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


# ---- combined frame -------------------------------------------------------

EMA_PERIODS = (20, 50, 200)
RSI_PERIOD = 14
MACD_PARAMS = (12, 26, 9)
BB_PARAMS = (20, 2.0)
VOL_MA_PERIOD = 20

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
    }

    signals: list[str] = []
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
    snap["signals"] = signals

    first_close = df_ind.iloc[0]["close"]
    if price is not None and pd.notna(first_close) and float(first_close) != 0:
        snap["window_change_pct"] = round((price - float(first_close)) / float(first_close) * 100, 2)
    return snap
