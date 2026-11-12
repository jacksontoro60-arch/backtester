import pandas as pd
import pytest

from backtester.data import read_price_csv
from backtester.engine import run_backtest
from backtester.strategies import BuyAndHold, SMACross
from backtester.strategies.base import Strategy
from tests.helpers import SAMPLE_CSV, make_prices


class Flat(Strategy):
    name = "flat"

    def signals(self, prices):
        return pd.DataFrame(0, index=prices.dates, columns=prices.symbols)


class LongSymbols(Strategy):
    def __init__(self, names: set[str]):
        self.names = names
        self.name = "long_some"

    def signals(self, prices):
        frame = pd.DataFrame(0, index=prices.dates, columns=prices.symbols)
        for symbol in self.names:
            frame[symbol] = 1
        return frame


class LongThenFlat(Strategy):
    """Long on the first close only, so the next open buys and the one after sells."""

    name = "long_then_flat"

    def signals(self, prices):
        frame = pd.DataFrame(0, index=prices.dates, columns=prices.symbols)
        frame.iloc[0] = 1
        return frame


def test_flat_strategy_never_trades():
    prices = make_prices(
        pd.bdate_range("2020-01-01", periods=4),
        {"A": {"open": [10, 11, 12, 13], "close": [10, 11, 12, 13]}},
    )
    result = run_backtest(prices, Flat(), starting_cash=1000)
    assert result.equity.tolist() == [1000, 1000, 1000, 1000]
    assert result.invested.sum() == 0
    assert result.metrics.n_trades == 0
    assert result.metrics.total_return == 0
    assert result.metrics.exposure == 0


def test_buy_and_hold_equity_drawdown_and_exposure_match_a_hand_example():
    prices = make_prices(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
        {"A": {"open": [100, 100, 120, 80], "close": [100, 120, 80, 100]}},
    )
    result = run_backtest(prices, BuyAndHold(), starting_cash=1000)
    assert result.equity.tolist() == pytest.approx([1000, 1200, 800, 1000])
    assert result.metrics.total_return == pytest.approx(0)
    assert result.metrics.cagr == pytest.approx(0)
    assert result.metrics.max_drawdown == pytest.approx(-1 / 3)
    assert result.metrics.exposure == pytest.approx(0.75)
    assert result.metrics.n_trades == 1
    assert result.metrics.win_rate == pytest.approx(0)
    trade = result.trades.iloc[0]
    assert trade.entry_price == pytest.approx(100)
    assert trade.exit_price == pytest.approx(100)
    assert trade.exit_reason == "mark_to_market"
    assert trade.pnl == pytest.approx(0)


def test_costs_are_charged_on_the_buy_and_the_sell():
    prices = make_prices(
        ["2020-01-01", "2020-01-02", "2020-01-03"],
        {"A": {"open": [100, 100, 100], "close": [100, 100, 100]}},
    )
    result = run_backtest(
        prices,
        LongThenFlat(),
        starting_cash=1000,
        commission_bps=100,
        slippage_bps=100,
    )
    # Signal on day 0 buys on day 1. Signal on day 1 is flat, so day 2 sells.
    shares = 1000 / (101 * 1.01)
    expected_cash = shares * 99 * 0.99
    assert result.trades.iloc[0].entry_price == pytest.approx(101)
    assert result.trades.iloc[0].exit_price == pytest.approx(99)
    assert result.equity.iloc[-1] == pytest.approx(expected_cash)
    assert result.equity.iloc[-1] < 1000


def test_equal_cash_sleeves_do_not_borrow_from_each_other():
    dates = ["2020-01-01", "2020-01-02"]
    both = {
        "A": {"open": [100, 100], "close": [100, 150]},
        "B": {"open": [100, 100], "close": [100, 150]},
    }
    prices = make_prices(dates, both)
    both_long = run_backtest(prices, BuyAndHold(), starting_cash=1000)
    one_long = run_backtest(prices, LongSymbols({"A"}), starting_cash=1000)
    single = run_backtest(
        make_prices(dates, {"A": both["A"]}),
        BuyAndHold(),
        starting_cash=1000,
    )
    assert both_long.metrics.total_return == pytest.approx(single.metrics.total_return)
    assert one_long.metrics.total_return == pytest.approx(single.metrics.total_return / 2)
    assert both_long.equity.iloc[0] == pytest.approx(1000)
    assert one_long.invested.iloc[0] == 0
    assert one_long.invested.iloc[-1] == pytest.approx(one_long.equity.iloc[-1] - 500)


def test_sample_sma_trades_account_and_start_in_cash():
    prices = read_price_csv(SAMPLE_CSV)
    result = run_backtest(prices, SMACross(20, 50), starting_cash=10_000)
    assert result.equity.iloc[0] == pytest.approx(10_000)
    assert result.invested.iloc[0] == 0
    assert result.metrics.n_trades > 0
    assert result.metrics.max_drawdown <= 0
    assert 0 <= result.metrics.exposure <= 1
    assert result.metrics.sharpe == result.metrics.sharpe
    for trade in result.trades.itertuples():
        assert trade.entry_date > prices.dates[0]
        assert trade.entry_price == pytest.approx(prices.open.loc[trade.entry_date, trade.symbol])
        assert trade.pnl == pytest.approx(trade.shares * (trade.exit_price - trade.entry_price))
        entry_loc = prices.dates.get_loc(pd.Timestamp(trade.entry_date))
        assert result.signals.iloc[entry_loc - 1][trade.symbol] == 1
        if entry_loc >= 2:
            assert result.signals.iloc[entry_loc - 2][trade.symbol] == 0
        if trade.exit_reason == "signal":
            exit_loc = prices.dates.get_loc(pd.Timestamp(trade.exit_date))
            assert trade.exit_price == pytest.approx(prices.open.loc[trade.exit_date, trade.symbol])
            assert result.signals.iloc[exit_loc - 1][trade.symbol] == 0
        else:
            assert trade.exit_reason == "mark_to_market"
            assert trade.exit_date == prices.dates[-1]
            assert trade.exit_price == pytest.approx(prices.close.loc[trade.exit_date, trade.symbol])


def test_bad_inputs_are_rejected():
    prices = make_prices(["2020-01-01"], {"A": {"open": [1], "close": [1]}})
    with pytest.raises(ValueError):
        run_backtest(prices, Flat(), starting_cash=0)
    with pytest.raises(ValueError):
        run_backtest(prices, Flat(), commission_bps=-1)

    class NotAFrame(Strategy):
        name = "bad"

        def signals(self, prices):
            return prices.close.iloc[:, 0]

    with pytest.raises(TypeError):
        run_backtest(prices, NotAFrame())
