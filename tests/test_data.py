from pathlib import Path

import pandas as pd
import pytest

from backtester.data import (
    download_symbol,
    load_prices,
    normalize_download,
    prices_from_long_frame,
    read_price_csv,
)
from tests.helpers import SAMPLE_CSV


def test_sample_csv_is_labeled_synthetic_and_loads():
    text = SAMPLE_CSV.read_text(encoding="utf-8")
    assert "SYNTHETIC" in text
    assert "not real market prices" in text
    prices = read_price_csv(SAMPLE_CSV)
    assert prices.symbols == ["SYN_A", "SYN_B"]
    assert len(prices.dates) == 520
    assert prices.close.min().min() > 0
    assert (prices.high + 1e-6 >= prices.close).all().all()
    assert (prices.low - 1e-6 <= prices.close).all().all()


def test_csv_filters_symbols_and_dates_inclusive():
    prices = load_prices(
        symbols=["SYN_B"],
        start="2019-01-02",
        end="2019-01-31",
        csv_path=SAMPLE_CSV,
    )
    assert prices.symbols == ["SYN_B"]
    assert prices.dates[0] == pd.Timestamp("2019-01-02")
    assert prices.dates[-1] <= pd.Timestamp("2019-01-31")
    assert prices.dates[-1] >= pd.Timestamp("2019-01-30")


def test_unknown_symbol_and_bad_rows_are_rejected():
    with pytest.raises(ValueError, match="not in the price file"):
        load_prices(symbols=["AAPL"], csv_path=SAMPLE_CSV)

    with pytest.raises(ValueError, match="missing columns"):
        prices_from_long_frame(pd.DataFrame({"date": ["2020-01-02"], "symbol": ["X"], "open": [1]}))

    rows = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-02"],
            "symbol": ["X", "X"],
            "open": [10, 10],
            "high": [11, 11],
            "low": [9, 9],
            "close": [10, 10],
            "volume": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        prices_from_long_frame(rows)

    bad = pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "symbol": ["X"],
            "open": [10],
            "high": [9],
            "low": [8],
            "close": [10],
            "volume": [1],
        }
    )
    with pytest.raises(ValueError, match="Invalid OHLC"):
        prices_from_long_frame(bad)


def test_csv_load_does_not_download(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("download was called")

    monkeypatch.setattr("backtester.data.download_symbol", fail)
    prices = load_prices(csv_path=SAMPLE_CSV)
    assert len(prices.dates) == 520


def _yahoo_rows(columns):
    index = pd.to_datetime(["2020-01-02", "2020-01-03"])
    values = [[10.0, 11.0, 9.0, 10.5, 1000], [10.5, 12.0, 10.0, 11.0, 1100]]
    return pd.DataFrame(values, index=index, columns=columns)


def test_normalize_download_accepts_flat_and_multiindex_columns():
    flat = _yahoo_rows(["Open", "High", "Low", "Close", "Volume"])
    ticker_first = _yahoo_rows(
        pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
    )
    field_first = _yahoo_rows(
        pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["AAA"]])
    )
    for frame in (flat, ticker_first, field_first):
        table = normalize_download(frame, "AAA")
        assert list(table.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
        assert table["symbol"].tolist() == ["AAA", "AAA"]
        assert table["close"].tolist() == [10.5, 11.0]


def test_normalize_download_rejects_empty_or_incomplete_frames():
    with pytest.raises(RuntimeError, match="No price data"):
        normalize_download(pd.DataFrame(), "AAA")
    incomplete = pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2020-01-02"]))
    with pytest.raises(RuntimeError, match="missing columns"):
        normalize_download(incomplete, "AAA")


def test_download_uses_yfinance_and_keeps_the_end_date(monkeypatch):
    captured = {}

    def fake_download(symbol, **kwargs):
        captured["symbol"] = symbol
        captured.update(kwargs)
        index = pd.to_datetime(["2020-01-02", "2020-01-03"])
        return pd.DataFrame(
            {
                "Open": [10.0, 10.0],
                "High": [11.0, 11.0],
                "Low": [9.0, 9.0],
                "Close": [10.5, 10.5],
                "Volume": [100, 120],
            },
            index=index,
        )

    import yfinance

    monkeypatch.setattr(yfinance, "download", fake_download)
    table = download_symbol("AAPL", "2020-01-02", "2020-01-03")
    assert captured["symbol"] == "AAPL"
    assert captured["start"] == "2020-01-02"
    assert captured["end"] == "2020-01-04"
    assert captured["auto_adjust"] is True
    assert table["symbol"].tolist() == ["AAPL", "AAPL"]


def test_download_cache_is_reused(tmp_path: Path):
    calls = []

    def fake(symbol, start, end):
        calls.append((symbol, start, end))
        return pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-03"],
                "symbol": [symbol, symbol],
                "open": [10.0, 11.0],
                "high": [12.0, 12.0],
                "low": [9.0, 10.0],
                "close": [11.0, 11.5],
                "volume": [100, 100],
            }
        )

    kwargs = {
        "symbols": ["AAA"],
        "start": "2020-01-02",
        "end": "2020-01-03",
        "cache_dir": tmp_path,
        "downloader": fake,
    }
    from backtester.data import fetch_prices

    first = fetch_prices(**kwargs)
    second = fetch_prices(**kwargs)
    fetch_prices(**{**kwargs, "end": "2020-01-02"})
    assert calls == [("AAA", "2020-01-02", "2020-01-03"), ("AAA", "2020-01-02", "2020-01-02")]
    assert first.close["AAA"].tolist() == second.close["AAA"].tolist()
    assert list(tmp_path.glob("*.csv"))
