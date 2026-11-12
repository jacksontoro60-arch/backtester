"""Load daily OHLC prices from a CSV file or from Yahoo Finance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OHLC_COLUMNS = ("open", "high", "low", "close", "volume")
_PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class PriceData:
    """Daily OHLC aligned across symbols. One column per symbol."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    def __post_init__(self) -> None:
        frames = (self.open, self.high, self.low, self.close, self.volume)
        index = self.close.index
        columns = list(self.close.columns)
        if len(columns) == 0 or len(index) == 0:
            raise ValueError("Price data is empty.")
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("Price rows must be indexed by date.")
        for frame in frames:
            if not frame.index.equals(index) or list(frame.columns) != columns:
                raise ValueError("OHLC frames are not aligned.")
        _validate_ohlc(self)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def copy(self) -> PriceData:
        return PriceData(
            open=self.open.copy(),
            high=self.high.copy(),
            low=self.low.copy(),
            close=self.close.copy(),
            volume=self.volume.copy(),
        )

    def slice(
        self,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> PriceData:
        chosen = list(symbols) if symbols else self.symbols
        missing = [symbol for symbol in chosen if symbol not in self.close.columns]
        if missing:
            raise ValueError(f"Symbols not in the price file: {', '.join(missing)}")
        start_ts = pd.Timestamp(start) if start is not None else self.dates[0]
        end_ts = pd.Timestamp(end) if end is not None else self.dates[-1]
        if start_ts > end_ts:
            raise ValueError("Start date is after end date.")
        mask = (self.dates >= start_ts) & (self.dates <= end_ts)
        if not bool(mask.any()):
            raise ValueError("No price rows in the requested date range.")

        def cut(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.loc[mask, chosen].copy()

        return PriceData(
            open=cut(self.open),
            high=cut(self.high),
            low=cut(self.low),
            close=cut(self.close),
            volume=cut(self.volume),
        )


def read_price_csv(path: str | Path) -> PriceData:
    """Read a long OHLC file: date, symbol, open, high, low, close, volume."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Price file not found: {csv_path}")
    frame = pd.read_csv(csv_path, comment="#")
    return prices_from_long_frame(frame)


def prices_from_long_frame(frame: pd.DataFrame) -> PriceData:
    """Build aligned OHLC frames from a long table."""

    missing = [column for column in ("date", "symbol", *_PRICE_COLUMNS) if column not in frame.columns]
    if missing:
        raise ValueError(f"Price file is missing columns: {', '.join(missing)}")
    table = frame.copy()
    if "volume" not in table.columns:
        table["volume"] = 0
    table["date"] = pd.to_datetime(table["date"], utc=True).dt.tz_convert(None).dt.normalize()
    table["symbol"] = table["symbol"].astype(str).str.strip()
    if (table["symbol"] == "").any():
        raise ValueError("Price file has a blank symbol.")
    for column in OHLC_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[list(_PRICE_COLUMNS)].isna().any().any():
        raise ValueError("Price file has missing OHLC values.")
    table["volume"] = table["volume"].fillna(0)
    if table.duplicated(["date", "symbol"]).any():
        raise ValueError("Price file has duplicate date and symbol rows.")

    symbols = list(dict.fromkeys(table["symbol"].tolist()))
    wide: dict[str, pd.DataFrame] = {}
    for column in OHLC_COLUMNS:
        pivoted = table.pivot(index="date", columns="symbol", values=column)
        wide[column] = pivoted.reindex(columns=symbols).sort_index()
    shared = wide["close"].dropna().index
    if len(shared) == 0:
        raise ValueError("No dates where every symbol has a price.")
    frames = {column: values.loc[shared].astype(float) for column, values in wide.items()}
    for values in frames.values():
        values.index.name = "date"
    return PriceData(**frames)


def load_prices(
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    csv_path: str | Path | None = None,
    cache_dir: str | Path = "cache",
    downloader=None,
    on_status=None,
) -> PriceData:
    """Load prices from ``csv_path`` or download them and cache each symbol."""

    if csv_path:
        prices = read_price_csv(csv_path)
        return prices.slice(symbols=symbols, start=start, end=end)
    if not symbols:
        raise ValueError("Pass symbols or a CSV file.")
    if not start:
        raise ValueError("A start date is required when downloading prices.")
    return fetch_prices(
        symbols,
        start=start,
        end=end,
        cache_dir=cache_dir,
        downloader=downloader,
        on_status=on_status,
    )


def fetch_prices(
    symbols: list[str],
    start: str,
    end: str | None,
    cache_dir: str | Path = "cache",
    downloader=None,
    on_status=None,
) -> PriceData:
    """Download daily bars, using one CSV cache file per symbol and date range."""

    if downloader is None:
        downloader = download_symbol
    frames = []
    folder = Path(cache_dir)
    for symbol in symbols:
        path = cache_file(folder, symbol, start, end)
        if path.is_file():
            if on_status:
                on_status(f"Using cached {symbol}.")
            frame = pd.read_csv(path)
        else:
            if on_status:
                on_status(f"Downloading {symbol}...")
            frame = downloader(symbol, start, end)
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    prices = prices_from_long_frame(combined)
    return prices.slice(symbols=list(symbols), start=start, end=end)


def cache_file(cache_dir: Path, symbol: str, start: str, end: str | None) -> Path:
    safe_symbol = symbol.replace("/", "_").replace("\\", "_").replace(" ", "_")
    end_label = end or "latest"
    return cache_dir / f"{safe_symbol}_{start}_{end_label}.csv"


def download_symbol(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Download adjusted daily OHLC for one symbol with yfinance."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed.") from exc

    kwargs = {
        "start": start,
        "auto_adjust": True,
        "progress": False,
        "threads": False,
    }
    if end is not None:
        # yfinance treats end as exclusive. Add a day so the requested end is kept.
        kwargs["end"] = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        downloaded = yf.download(symbol, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Could not download {symbol}: {exc}") from exc
    return normalize_download(downloaded, symbol)


def normalize_download(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Turn a yfinance frame into date, symbol, open, high, low, close, volume."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError(f"No price data returned for {symbol}.")
    table = frame.copy()
    if isinstance(table.columns, pd.MultiIndex):
        for level in range(table.columns.nlevels):
            values = [str(value) for value in table.columns.get_level_values(level)]
            if symbol in values:
                table = table.xs(symbol, axis=1, level=level)
                break
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.get_level_values(0)
    table.columns = [str(column).strip().lower().replace(" ", "_") for column in table.columns]
    if "close" not in table.columns and "adj_close" in table.columns:
        table = table.rename(columns={"adj_close": "close"})
    missing = [column for column in _PRICE_COLUMNS if column not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol} is missing columns: {', '.join(missing)}")
    if "volume" not in table.columns:
        table["volume"] = 0
    table = table[["open", "high", "low", "close", "volume"]].copy()
    table.index = pd.to_datetime(table.index, utc=True).tz_convert(None).normalize()
    table = table.sort_index()
    table = table.dropna(subset=list(_PRICE_COLUMNS))
    if table.empty:
        raise RuntimeError(f"No price data returned for {symbol}.")
    table["symbol"] = symbol
    table = table.reset_index()
    date_column = table.columns[0]
    table = table.rename(columns={date_column: "date"})
    return table[["date", "symbol", "open", "high", "low", "close", "volume"]]


def file_says_synthetic(path: str | Path) -> bool:
    """True when the CSV header says the prices are synthetic."""

    with Path(path).open(encoding="utf-8") as handle:
        head = handle.read(500).lower()
    return "synthetic" in head


def _validate_ohlc(prices: PriceData) -> None:
    tolerance = 1e-6
    for symbol in prices.symbols:
        opened = prices.open[symbol]
        high = prices.high[symbol]
        low = prices.low[symbol]
        close = prices.close[symbol]
        invalid = (
            (high + tolerance < low)
            | (high + tolerance < opened)
            | (high + tolerance < close)
            | (low - tolerance > opened)
            | (low - tolerance > close)
            | (opened <= 0)
            | (high <= 0)
            | (low <= 0)
            | (close <= 0)
        )
        if bool(invalid.any()):
            when = prices.dates[int(invalid.to_numpy().argmax())]
            raise ValueError(f"Invalid OHLC for {symbol} on {when.date()}.")
