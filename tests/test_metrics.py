import numpy as np
import pandas as pd
import pytest

from backtester.metrics import TRADING_DAYS, compute_metrics, drawdown, render_table


def _curve():
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    equity = pd.Series([100.0, 110.0, 90.0, 95.0, 120.0], index=index)
    trades = pd.DataFrame({"pnl": [10.0, -5.0, 1.0]})
    invested = pd.Series([0.0, 110.0, 90.0, 95.0, 120.0], index=index)
    return equity, trades, invested


def test_metrics_match_hand_calculations():
    equity, trades, invested = _curve()
    metrics = compute_metrics(equity, trades, invested)

    assert metrics.total_return == pytest.approx(0.2)
    assert metrics.max_drawdown == pytest.approx(90 / 110 - 1)
    assert metrics.max_drawdown == pytest.approx(float(drawdown(equity).min()))
    assert metrics.max_drawdown <= 0
    assert metrics.n_trades == 3
    assert metrics.win_rate == pytest.approx(2 / 3)
    assert metrics.exposure == pytest.approx(0.8)

    returns = equity.iloc[1:].to_numpy() / equity.iloc[:-1].to_numpy() - 1
    daily_std = float(np.std(returns, ddof=1))
    assert metrics.volatility == pytest.approx(daily_std * np.sqrt(TRADING_DAYS))
    assert metrics.sharpe == pytest.approx(float(np.mean(returns)) / daily_std * np.sqrt(TRADING_DAYS))

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    assert metrics.cagr == pytest.approx(1.2 ** (1 / years) - 1)


def test_flat_equity_has_zero_risk_stats_and_no_win_rate():
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    equity = pd.Series([100.0, 100.0, 100.0], index=index)
    invested = pd.Series([0.0, 0.0, 0.0], index=index)
    metrics = compute_metrics(equity, pd.DataFrame(), invested)
    assert metrics.total_return == 0
    assert metrics.cagr == 0
    assert metrics.max_drawdown == 0
    assert metrics.sharpe == 0
    assert metrics.volatility == 0
    assert metrics.n_trades == 0
    assert metrics.win_rate != metrics.win_rate
    assert metrics.exposure == 0


def test_table_labels_the_benchmark_and_skips_a_missing_win_rate():
    equity, trades, invested = _curve()
    metrics = compute_metrics(equity, trades, invested)
    flat = compute_metrics(
        pd.Series([100.0, 100.0], index=equity.index[:2]),
        pd.DataFrame(),
        pd.Series([0.0, 0.0], index=equity.index[:2]),
    )
    table = render_table("sma_cross 20/50", metrics, flat)
    assert "Buy & hold" in table
    assert "Total return" in table
    assert "Sharpe" in table
    assert "n/a" in table
