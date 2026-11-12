"""Long-only moving-average crossover."""

from __future__ import annotations

from backtester.data import PriceData
from backtester.strategies.base import Strategy


class SMACross(Strategy):
    """Be long when the fast average of closes is above the slow average."""

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        fast = int(fast)
        slow = int(slow)
        if fast < 1 or slow < 1:
            raise ValueError("SMA windows must be at least 1.")
        if fast >= slow:
            raise ValueError("Fast SMA window must be shorter than the slow window.")
        self.fast = fast
        self.slow = slow
        self.name = f"sma_cross {fast}/{slow}"

    def signals(self, prices: PriceData):
        close = prices.close
        fast_average = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_average = close.rolling(self.slow, min_periods=self.slow).mean()
        long = (fast_average > slow_average) & slow_average.notna()
        return long.astype(int)
