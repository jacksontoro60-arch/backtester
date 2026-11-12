"""Equal-cash, long-only backtest. Signals fill at the next bar's open."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.data import PriceData
from backtester.metrics import Metrics, compute_metrics
from backtester.strategies.base import Strategy

TRADE_COLUMNS = [
    "symbol",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "shares",
    "pnl",
    "return",
    "exit_reason",
]


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    invested: pd.Series
    trades: pd.DataFrame
    signals: pd.DataFrame
    metrics: Metrics


def run_backtest(
    prices: PriceData,
    strategy: Strategy,
    starting_cash: float = 10_000.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    """Run one strategy.

    Starting cash is split evenly across symbols. Each sleeve trades only its
    own symbol: in when the signal is long, cash when it is flat. A signal at
    today's close is filled at the next session's open. Nothing trades on the
    first bar, and the last bar's signal has no later open to fill.
    """

    starting_cash = float(starting_cash)
    commission_bps = float(commission_bps)
    slippage_bps = float(slippage_bps)
    if starting_cash <= 0:
        raise ValueError("Starting cash must be positive.")
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("Commission and slippage must be zero or positive.")

    signals = _align_signals(strategy.signals(prices), prices)
    allocations = _split_cash(starting_cash, len(prices.symbols))
    equities = []
    invested = []
    trades = []
    for symbol, sleeve_cash in zip(prices.symbols, allocations):
        sleeve = _run_sleeve(
            dates=prices.dates,
            open_prices=prices.open[symbol],
            close_prices=prices.close[symbol],
            signals=signals[symbol],
            cash=sleeve_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            symbol=symbol,
        )
        equities.append(sleeve["equity"])
        invested.append(sleeve["invested"])
        trades.extend(sleeve["trades"])

    equity = pd.concat(equities, axis=1).sum(axis=1)
    equity.name = "equity"
    invested_value = pd.concat(invested, axis=1).sum(axis=1)
    invested_value.name = "invested"
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    if not trade_frame.empty:
        trade_frame = trade_frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    metrics = compute_metrics(equity, trade_frame, invested_value)
    return BacktestResult(
        equity=equity,
        invested=invested_value,
        trades=trade_frame,
        signals=signals,
        metrics=metrics,
    )


def _run_sleeve(
    dates: pd.DatetimeIndex,
    open_prices: pd.Series,
    close_prices: pd.Series,
    signals: pd.Series,
    cash: float,
    commission_bps: float,
    slippage_bps: float,
    symbol: str,
) -> dict:
    commission_rate = commission_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    shares = 0.0
    in_position = False
    entry: dict | None = None
    trades: list[dict] = []
    equity_values = []
    invested_values = []

    for index in range(len(dates)):
        if index > 0:
            want_long = float(signals.iloc[index - 1]) > 0
            if want_long and not in_position:
                fill = float(open_prices.iloc[index]) * (1.0 + slippage_rate)
                entry_cash = cash
                shares = cash / (fill * (1.0 + commission_rate))
                cash = 0.0
                in_position = True
                entry = {
                    "entry_date": dates[index],
                    "entry_price": fill,
                    "shares": shares,
                    "entry_cash": entry_cash,
                }
            elif (not want_long) and in_position:
                fill = float(open_prices.iloc[index]) * (1.0 - slippage_rate)
                proceeds = shares * fill
                fee = proceeds * commission_rate
                exit_cash = proceeds - fee
                trades.append(_trade_row(symbol, entry, dates[index], fill, exit_cash, "signal"))
                cash = exit_cash
                shares = 0.0
                in_position = False
                entry = None

        mark = float(close_prices.iloc[index])
        equity_values.append(cash + shares * mark)
        invested_values.append(shares * mark)

    if in_position and entry is not None:
        last_close = float(close_prices.iloc[-1])
        trades.append(
            _trade_row(symbol, entry, dates[-1], last_close, shares * last_close, "mark_to_market")
        )

    equity = pd.Series(equity_values, index=dates, name=symbol)
    invested = pd.Series(invested_values, index=dates, name=symbol)
    return {"equity": equity, "invested": invested, "trades": trades}


def _trade_row(
    symbol: str,
    entry: dict,
    exit_date,
    exit_price: float,
    exit_cash: float,
    reason: str,
) -> dict:
    pnl = exit_cash - entry["entry_cash"]
    ret = pnl / entry["entry_cash"] if entry["entry_cash"] else float("nan")
    return {
        "symbol": symbol,
        "entry_date": entry["entry_date"],
        "entry_price": entry["entry_price"],
        "exit_date": exit_date,
        "exit_price": exit_price,
        "shares": entry["shares"],
        "pnl": pnl,
        "return": ret,
        "exit_reason": reason,
    }


def _split_cash(starting_cash: float, count: int) -> list[float]:
    """Split cash evenly. The last sleeve absorbs rounding so the sum matches."""

    allocations = [starting_cash / count] * count
    allocations[-1] = starting_cash - sum(allocations[:-1])
    return allocations


def _align_signals(signals: pd.DataFrame, prices: PriceData) -> pd.DataFrame:
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals() must return a DataFrame of 0 and 1.")
    extra = sorted(set(signals.columns) - set(prices.symbols))
    if extra:
        raise ValueError(f"Strategy returned unknown symbols: {', '.join(extra)}")
    aligned = signals.reindex(index=prices.dates, columns=prices.symbols).fillna(0.0)
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Strategy signals must be finite numbers.")
    return aligned.astype(float)
