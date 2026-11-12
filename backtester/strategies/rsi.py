"""Long-only RSI mean reversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.data import PriceData
from backtester.strategies.base import Strategy


class RSIMeanReversion(Strategy):
    """Enter when RSI falls below a low level, exit when it rises above a high one."""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70) -> None:
        period = int(period)
        oversold = float(oversold)
        overbought = float(overbought)
        if period < 2:
            raise ValueError("RSI period must be at least 2.")
        if not (0 < oversold < overbought < 100):
            raise ValueError("Need 0 < oversold < overbought < 100.")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"rsi {period} {oversold:g}/{overbought:g}"

    def signals(self, prices: PriceData) -> pd.DataFrame:
        output = pd.DataFrame(0, index=prices.dates, columns=prices.symbols, dtype=int)
        for symbol in prices.symbols:
            rsi = wilder_rsi(prices.close[symbol], self.period)
            position = 0
            flags = []
            for value in rsi:
                if not np.isnan(value):
                    if position == 0 and value < self.oversold:
                        position = 1
                    elif position == 1 and value > self.overbought:
                        position = 0
                flags.append(position)
            output[symbol] = flags
        return output


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI. The first value uses a simple average of the first `period` changes."""

    delta = close.diff()
    gain = delta.clip(lower=0.0).to_numpy(dtype=float)
    loss = (-delta).clip(lower=0.0).to_numpy(dtype=float)
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    if len(close) > period:
        avg_gain[period] = gain[1 : period + 1].mean()
        avg_loss[period] = loss[1 : period + 1].mean()
        for index in range(period + 1, len(close)):
            avg_gain[index] = (avg_gain[index - 1] * (period - 1) + gain[index]) / period
            avg_loss[index] = (avg_loss[index - 1] * (period - 1) + loss[index]) / period

    rsi = np.full(len(close), np.nan)
    valid = ~np.isnan(avg_loss)
    rsi[valid] = 100.0
    loss_positive = valid & (avg_loss > 0)
    rs = avg_gain[loss_positive] / avg_loss[loss_positive]
    rsi[loss_positive] = 100.0 - (100.0 / (1.0 + rs))
    both_zero = valid & (avg_gain == 0) & (avg_loss == 0)
    rsi[both_zero] = 50.0
    return pd.Series(rsi, index=close.index)
