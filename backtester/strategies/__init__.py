"""Built-in strategies and the registry the CLI uses."""

from backtester.strategies.base import Strategy
from backtester.strategies.buy_hold import BuyAndHold
from backtester.strategies.rsi import RSIMeanReversion
from backtester.strategies.sma import SMACross

STRATEGIES: dict[str, type[Strategy]] = {
    "sma_cross": SMACross,
    "buy_hold": BuyAndHold,
    "rsi": RSIMeanReversion,
}

__all__ = [
    "STRATEGIES",
    "BuyAndHold",
    "RSIMeanReversion",
    "SMACross",
    "Strategy",
]
