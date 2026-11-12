"""Buy at the first fill and hold."""

from __future__ import annotations

import pandas as pd

from backtester.data import PriceData
from backtester.strategies.base import Strategy


class BuyAndHold(Strategy):
    name = "buy_hold"

    def signals(self, prices: PriceData) -> pd.DataFrame:
        return pd.DataFrame(1, index=prices.dates, columns=prices.symbols, dtype=int)
