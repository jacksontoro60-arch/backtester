"""Guards for the next-open fill rule.

A signal is decided at the close. It is filled at the next bar's open.
Changing prices after that close must not change earlier equity.
"""

import pandas as pd
import pytest

from backtester.data import read_price_csv
from backtester.engine import run_backtest
from backtester.strategies import BuyAndHold, SMACross
from backtester.strategies.base import Strategy
from tests.helpers import SAMPLE_CSV, make_prices


class Above(Strategy):
    """Long when today's close is at or above a level. Uses no future bars."""

    name = "above"

    def __init__(self, level: float):
        self.level = level

    def signals(self, prices):
        return (prices.close >= self.level).astype(int)


def test_same_bar_jump_is_not_tradable():
    prices = make_prices(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
        {"A": {"open": [100, 100, 100, 200], "close": [100, 100, 200, 200]}},
    )
    result = run_backtest(prices, Above(150), starting_cash=10_000)
    # The close jumps on day 3. The buy fills at the next open, which is already 200.
    assert result.signals["A"].tolist() == [0, 0, 1, 1]
    assert result.equity.iloc[2] == pytest.approx(10_000)
    assert result.invested.iloc[2] == pytest.approx(0)
    assert result.equity.iloc[3] == pytest.approx(10_000)
    assert result.trades.iloc[0].entry_price == pytest.approx(200)


def test_signal_fills_at_the_next_open_and_only_then():
    up_day = {"open": [100, 110], "close": [100, 150]}
    dates = ["2020-01-01", "2020-01-02"]
    prices = make_prices(dates, {"A": up_day})

    early = run_backtest(prices, Above(100), starting_cash=1000)
    late = run_backtest(prices, Above(150), starting_cash=1000)

    # Signal on day 0 buys the day-1 open at 110, then marks the close at 150.
    assert early.equity.iloc[0] == pytest.approx(1000)
    assert early.equity.iloc[1] == pytest.approx(1000 / 110 * 150)
    assert early.trades.iloc[0].entry_price == pytest.approx(110)

    # Signal on the jump day has no next open, so the jump is not captured.
    assert late.signals["A"].tolist() == [0, 1]
    assert late.equity.tolist() == pytest.approx([1000, 1000])
    assert late.metrics.n_trades == 0


@pytest.mark.parametrize("strategy", [SMACross(20, 50), BuyAndHold()])
def test_future_prices_do_not_change_earlier_equity(strategy):
    prices = read_price_csv(SAMPLE_CSV)
    original = run_backtest(prices, strategy, starting_cash=10_000)

    mutated = prices.copy()
    cut = len(prices.dates) - 40
    for frame in (mutated.open, mutated.high, mutated.low, mutated.close):
        frame.iloc[cut + 1 :] = frame.iloc[cut + 1 :] * 1.8
    changed = run_backtest(mutated, strategy, starting_cash=10_000)

    pd.testing.assert_series_equal(original.equity.iloc[: cut + 1], changed.equity.iloc[: cut + 1])
    pd.testing.assert_frame_equal(original.signals.iloc[: cut + 1], changed.signals.iloc[: cut + 1])
    # Buy-and-hold is in the market, so the mutated marks have to show up later.
    if isinstance(strategy, BuyAndHold):
        assert not original.equity.iloc[cut + 1 :].equals(changed.equity.iloc[cut + 1 :])
