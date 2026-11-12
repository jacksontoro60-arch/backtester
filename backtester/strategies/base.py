"""Strategy interface.

A strategy looks at daily bars and decides, at each close, whether it wants
to be long or flat. It must only use prices from that day and earlier.
The engine reads the decision and trades at the next day's open.
"""

from __future__ import annotations

import abc

import pandas as pd

from backtester.data import PriceData


class Strategy(abc.ABC):
    name: str

    @abc.abstractmethod
    def signals(self, prices: PriceData) -> pd.DataFrame:
        """Return 1 for long and 0 for flat.

        Index is the price dates. Columns are the symbols. The value on date
        t is filled at the open of the next bar.
        """
