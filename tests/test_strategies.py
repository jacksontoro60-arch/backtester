import pandas as pd
import pytest

from backtester.data import read_price_csv
from backtester.strategies import BuyAndHold, RSIMeanReversion, SMACross
from backtester.strategies.rsi import wilder_rsi
from tests.helpers import SAMPLE_CSV, make_prices


def test_sma_cross_matches_a_hand_worked_example():
    close = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]
    prices = make_prices(
        pd.bdate_range("2020-01-01", periods=len(close)),
        {"X": {"open": close, "close": close}},
    )
    signals = SMACross(fast=2, slow=4).signals(prices)
    assert signals["X"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]


def test_sma_rejects_a_fast_window_that_is_not_shorter():
    with pytest.raises(ValueError):
        SMACross(fast=20, slow=20)
    with pytest.raises(ValueError):
        SMACross(fast=50, slow=20)


def test_rsi_matches_wilder_values_and_the_band_rule():
    close = pd.Series([10, 11, 12, 11, 10, 9, 10, 12, 14], dtype=float)
    rsi = wilder_rsi(close, period=3)
    assert pd.isna(rsi.iloc[2])
    assert rsi.iloc[3] == pytest.approx(66.66666666666667)
    assert rsi.iloc[4] == pytest.approx(44.44444444444444)
    assert rsi.iloc[5] == pytest.approx(29.62962962962963)
    assert rsi.iloc[6] == pytest.approx(53.08641975308642)
    assert rsi.iloc[7] == pytest.approx(76.54320987654322)

    prices = make_prices(
        pd.bdate_range("2020-01-01", periods=len(close)),
        {"X": {"open": close.tolist(), "close": close.tolist()}},
    )
    signals = RSIMeanReversion(period=3, oversold=30, overbought=70).signals(prices)
    assert signals["X"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0]


def test_rsi_rejects_bad_levels():
    with pytest.raises(ValueError):
        RSIMeanReversion(period=1)
    with pytest.raises(ValueError):
        RSIMeanReversion(oversold=70, overbought=30)


def test_buy_and_hold_is_always_long():
    prices = make_prices(
        ["2020-01-01", "2020-01-02"],
        {"A": {"open": [1, 1], "close": [1, 1]}, "B": {"open": [2, 2], "close": [2, 2]}},
    )
    signals = BuyAndHold().signals(prices)
    assert signals.shape == (2, 2)
    assert signals.to_numpy().tolist() == [[1, 1], [1, 1]]


@pytest.mark.parametrize("strategy", [SMACross(20, 50), RSIMeanReversion(), BuyAndHold()])
def test_builtin_signals_on_a_prefix_match_the_full_series(strategy):
    prices = read_price_csv(SAMPLE_CSV)
    full = strategy.signals(prices)
    cut = prices.dates[120]
    prefix = prices.slice(end=cut)
    got = strategy.signals(prefix)
    pd.testing.assert_frame_equal(got, full.loc[got.index])
