"""Write data/synthetic_prices.csv.

The prices are made up. They are not a sample of a real market.
Regenerate with: python scripts/make_synthetic_prices.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
ROWS = 520
START = "2019-01-02"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_prices.csv"


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START, periods=ROWS)
    frames = [
        _symbol_frame("SYN_A", _trend_close(rng), dates, rng),
        _symbol_frame("SYN_B", _swing_close(rng), dates, rng),
    ]
    table = pd.concat(frames, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# SYNTHETIC DATA\n"
        "# These are not real market prices. Generated for tests and the README example.\n"
        f"# Generator: scripts/make_synthetic_prices.py  seed={SEED}  rows={ROWS} per symbol\n"
    )
    with OUTPUT.open("w", encoding="utf-8") as handle:
        handle.write(header)
        table.to_csv(handle, index=False)
    print(f"Wrote {len(table)} rows to {OUTPUT}")


def _trend_close(rng: np.random.Generator) -> np.ndarray:
    """Drift up, then down, then up, plus small noise. Enough to cross a slow SMA."""

    count = ROWS
    day = np.arange(count)
    drift = np.piecewise(
        day.astype(float),
        [day < 180, (day >= 180) & (day < 300), day >= 300],
        [0.0011, -0.0016, 0.0009],
    )
    noise = rng.normal(0, 0.004, count)
    return 80.0 * np.exp(np.cumsum(drift + noise))


def _swing_close(rng: np.random.Generator) -> np.ndarray:
    """A slower climb with a monthly swing, so RSI has something to do."""

    count = ROWS
    day = np.arange(count, dtype=float)
    noise = rng.normal(0, 0.006, count)
    seasonal = 0.12 * np.sin(2 * np.pi * day / 28.0)
    return 50.0 * np.exp(0.00035 * day + seasonal + np.cumsum(noise) * 0.2)


def _symbol_frame(
    symbol: str,
    close: np.ndarray,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    count = len(close)
    gap = rng.normal(0, 0.002, count)
    opened = np.empty(count)
    opened[0] = close[0]
    opened[1:] = close[:-1] * (1.0 + gap[1:])
    wick = np.abs(rng.normal(0.004, 0.002, count))
    opened = np.round(opened, 4)
    close = np.round(close, 4)
    high = np.round(np.maximum(opened, close) * (1.0 + wick), 4)
    low = np.round(np.minimum(opened, close) * (1.0 - wick), 4)
    high = np.maximum(high, np.maximum(opened, close))
    low = np.minimum(low, np.minimum(opened, close))
    low = np.maximum(low, 0.01)
    volume = rng.integers(200_000, 800_000, count)
    if np.any(high < np.maximum(opened, close)) or np.any(low > np.minimum(opened, close)):
        raise RuntimeError(f"OHLC construction failed for {symbol}.")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "open": np.round(opened, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": volume,
        }
    )


if __name__ == "__main__":
    main()
