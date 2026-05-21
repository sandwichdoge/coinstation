"""Date-aware backtest engine.

Pulls warm-up history before `start` so indicators are valid from the first
traded bar, then simulates a long/flat (spot) strategy bar by bar and compares
it to buy & hold. The `ai` strategy calls the same `ai.analyze` used live, with
`as_of` set to each bar's timestamp; backtests run on technicals only (no news).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..schemas import BacktestRequest
from ..timeutil import ms_to_iso, now_ms, to_ms
from . import ai, indicators, market_data

YEAR_MS = 365.25 * 24 * 3600 * 1000


async def _signal(
    req: BacktestRequest,
    position: int,
    row: pd.Series,
    prev: pd.Series,
    ind: pd.DataFrame,
    idx: int,
    j: int,
) -> str:
    """Return 'buy' | 'sell' | 'hold' for the current bar given the position."""
    strat = req.strategy

    if strat == "buy_hold":
        return "buy" if position == 0 else "hold"

    if strat == "rsi":
        r = row.get("rsi")
        if pd.isna(r):
            return "hold"
        if position == 0 and r <= req.rsi_buy:
            return "buy"
        if position == 1 and r >= req.rsi_sell:
            return "sell"
        return "hold"

    if strat == "macd":
        m, s, pm, ps = row.get("macd"), row.get("macd_signal"), prev.get("macd"), prev.get("macd_signal")
        if any(pd.isna(x) for x in (m, s, pm, ps)):
            return "hold"
        if position == 0 and pm <= ps and m > s:
            return "buy"
        if position == 1 and pm >= ps and m < s:
            return "sell"
        return "hold"

    if strat == "ema_cross":
        e50, e200, pe50, pe200 = row.get("ema50"), row.get("ema200"), prev.get("ema50"), prev.get("ema200")
        if any(pd.isna(x) for x in (e50, e200, pe50, pe200)):
            return "hold"
        if position == 0 and pe50 <= pe200 and e50 > e200:  # golden cross
            return "buy"
        if position == 1 and pe50 >= pe200 and e50 < e200:  # death cross
            return "sell"
        return "hold"

    if strat == "ai":
        # Evaluate periodically to bound cost/latency.
        if j % req.ai_rebalance_every != 0:
            return "hold"
        snapshot = indicators.latest_snapshot(ind.iloc[: idx + 1])
        as_of_ms = int(row["time"]) * 1000
        result = await ai.analyze(
            symbol=req.symbol, interval=req.interval, as_of_ms=as_of_ms,
            snapshot=snapshot, headlines=[],
        )
        if (
            position == 0
            and result.action in ("buy", "strong_buy")
            and result.confidence >= req.ai_confidence_threshold
        ):
            return "buy"
        if (
            position == 1
            and result.action in ("sell", "strong_sell")
            and result.confidence >= req.ai_confidence_threshold
        ):
            return "sell"
        return "hold"

    raise ValueError(f"Unknown strategy: {strat}")


def _metrics(equity: list[float], bh: list[float], round_trips: list[float], step_ms: int, initial: float) -> dict:
    final = equity[-1]
    peak, max_dd = -math.inf, 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            max_dd = min(max_dd, v / peak - 1.0)

    rets = np.diff(equity) / np.array(equity[:-1]) if len(equity) > 1 else np.array([])
    sharpe = 0.0
    if rets.size and rets.std() > 0:
        periods_per_year = YEAR_MS / step_ms
        sharpe = float(rets.mean() / rets.std() * math.sqrt(periods_per_year))

    wins = sum(1 for r in round_trips if r > 0)
    return {
        "initial_capital": round(initial, 2),
        "final_equity": round(final, 2),
        "total_return_pct": round((final / initial - 1) * 100, 2),
        "buy_hold_return_pct": round((bh[-1] / initial - 1) * 100, 2),
        "alpha_pct": round((final - bh[-1]) / initial * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "num_trades": len(round_trips),
        "win_rate_pct": round(wins / len(round_trips) * 100, 1) if round_trips else None,
        "exposure_pct": None,  # filled by caller
    }


async def run_backtest(req: BacktestRequest) -> dict:
    start_ms = to_ms(req.start)
    end_ms = to_ms(req.end) or now_ms()
    if start_ms is None:
        raise ValueError("`start` is required.")
    if start_ms >= end_ms:
        raise ValueError("`start` must be before `end`.")

    step = market_data.interval_ms(req.interval)
    warmup_ms = step * indicators.WARMUP_BARS
    df = await market_data.fetch_ohlcv(req.symbol, req.interval, start_ms - warmup_ms, end_ms)
    if df.empty or len(df) < 5:
        raise ValueError("Not enough market data for the requested range/interval.")

    ind = indicators.compute_indicator_frame(df)
    mask = (ind["time"] * 1000) >= start_ms
    window = ind[mask].reset_index(drop=True)
    if window.empty:
        raise ValueError("No candles fall inside the requested window.")
    offset = len(ind) - len(window)  # map window row j -> ind row offset+j

    fee = req.fee_pct / 100.0
    cash = req.initial_capital
    units = 0.0
    position = 0
    entry_price = None
    bars_in_market = 0

    trades: list[dict] = []
    markers: list[dict] = []
    equity_curve: list[dict] = []
    equity_vals: list[float] = []
    bh_vals: list[float] = []
    round_trips: list[float] = []

    bh_units = req.initial_capital * (1 - fee) / float(window.iloc[0]["close"])

    for j in range(len(window)):
        row = window.iloc[j]
        prev = window.iloc[j - 1] if j > 0 else row
        price = float(row["close"])
        t = int(row["time"])

        action = await _signal(req, position, row, prev, ind, offset + j, j)

        if action == "buy" and position == 0:
            units = cash * (1 - fee) / price
            cash = 0.0
            position = 1
            entry_price = price
            trades.append({"time": t, "side": "buy", "price": round(price, 8), "units": round(units, 8)})
            markers.append({"time": t, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "BUY"})
        elif action == "sell" and position == 1:
            cash = units * price * (1 - fee)
            if entry_price:
                round_trips.append(price / entry_price - 1.0)
            trades.append({"time": t, "side": "sell", "price": round(price, 8), "units": round(units, 8), "cash": round(cash, 2)})
            markers.append({"time": t, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "SELL"})
            units = 0.0
            position = 0
            entry_price = None

        if position == 1:
            bars_in_market += 1
        equity = cash + units * price
        bh_equity = bh_units * price
        equity_vals.append(equity)
        bh_vals.append(bh_equity)
        equity_curve.append({"time": t, "equity": round(equity, 2), "buy_hold": round(bh_equity, 2)})

    metrics = _metrics(equity_vals, bh_vals, round_trips, step, req.initial_capital)
    metrics["exposure_pct"] = round(bars_in_market / len(window) * 100, 1)

    payload = indicators.build_payload(window)
    return {
        "symbol": req.symbol.upper(),
        "interval": req.interval,
        "strategy": req.strategy,
        "start": ms_to_iso(start_ms),
        "end": ms_to_iso(end_ms),
        "bars": len(window),
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trades,
        "markers": markers,
        "candles": market_data.to_candles(window),
        "overlays": payload["overlays"],
        "volume": payload["volume"],
        "rsi": payload["rsi"],
        "macd": payload["macd"],
    }
