"""Small OHLC frames for tests. Not market data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtester.data import PriceData

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "synthetic_prices.csv"


def make_prices(dates, symbols: dict[str, dict]) -> PriceData:
    """Build PriceData. Each symbol needs open and close. High and low are filled in."""

    index = pd.DatetimeIndex(pd.to_datetime(dates))
    frames = {}
    for field in ("open", "high", "low", "close", "volume"):
        columns = {}
        for symbol, values in symbols.items():
            if field in values:
                columns[symbol] = np.asarray(values[field], dtype=float)
            elif field == "high":
                columns[symbol] = np.maximum(values["open"], values["close"])
            elif field == "low":
                columns[symbol] = np.minimum(values["open"], values["close"])
            elif field == "volume":
                columns[symbol] = np.ones(len(index))
            else:
                raise KeyError(field)
        frame = pd.DataFrame(columns, index=index)
        frame.index.name = "date"
        frames[field] = frame
    return PriceData(**frames)
