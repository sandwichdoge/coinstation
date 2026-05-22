"""The top_bottom backtest strategy decision logic (no network)."""
from __future__ import annotations

import asyncio

import pandas as pd

from app.schemas import BacktestRequest
from app.services import backtest as bt


def _decide(position: int, tb_signal: int) -> str:
    req = BacktestRequest(symbol="BTCUSDT", start="2022-01-01", strategy="top_bottom")
    row = pd.Series({"tb_signal": tb_signal})
    return asyncio.run(bt._signal(req, position, row, row, pd.DataFrame(), 0, 0))


def test_buys_flat_on_bottom():
    assert _decide(position=0, tb_signal=1) == "buy"


def test_sells_long_on_top():
    assert _decide(position=1, tb_signal=-1) == "sell"


def test_holds_when_no_signal_or_redundant():
    assert _decide(position=0, tb_signal=0) == "hold"
    assert _decide(position=0, tb_signal=-1) == "hold"  # flat + top → nothing to sell
    assert _decide(position=1, tb_signal=1) == "hold"   # already long + bottom → stay
