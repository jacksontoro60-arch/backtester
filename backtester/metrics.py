"""Performance stats for one equity curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    volatility: float
    win_rate: float
    n_trades: int
    exposure: float


def drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown as a fraction of the running peak. Zero or negative."""

    return equity / equity.cummax() - 1.0


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    invested: pd.Series,
) -> Metrics:
    """Summarize a backtest.

    Sharpe uses daily simple returns, a zero risk-free rate, and 252 days.
    It is 0 when there are fewer than two returns or when volatility is 0.
    A position still open at the end should already be in ``trades`` if it
    is meant to count toward win rate.
    """

    if equity.empty:
        raise ValueError("Equity curve is empty.")
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    if initial <= 0:
        raise ValueError("Starting equity must be positive.")
    total_return = final / initial - 1.0
    cagr = _cagr(equity.index[0], equity.index[-1], initial, final)
    max_drawdown = float(drawdown(equity).min())
    sharpe, volatility = _sharpe_and_volatility(equity)
    n_trades = int(len(trades))
    if n_trades == 0:
        win_rate = float("nan")
    else:
        win_rate = float((trades["pnl"] > 0).sum()) / n_trades
    exposure = float((invested / equity).mean())
    return Metrics(
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        volatility=volatility,
        win_rate=win_rate,
        n_trades=n_trades,
        exposure=exposure,
    )


def render_table(strategy_name: str, strategy: Metrics, benchmark: Metrics) -> str:
    """Plain-text comparison of the strategy and buy-and-hold."""

    rows = [
        ("Total return", _pct(strategy.total_return), _pct(benchmark.total_return)),
        ("CAGR", _pct(strategy.cagr), _pct(benchmark.cagr)),
        ("Max drawdown", _pct(strategy.max_drawdown), _pct(benchmark.max_drawdown)),
        ("Sharpe", _num(strategy.sharpe), _num(benchmark.sharpe)),
        ("Volatility", _pct(strategy.volatility), _pct(benchmark.volatility)),
        ("Win rate", _pct(strategy.win_rate), _pct(benchmark.win_rate)),
        ("Trades", str(strategy.n_trades), str(benchmark.n_trades)),
        ("Exposure", _pct(strategy.exposure), _pct(benchmark.exposure)),
    ]
    label_width = max(len(row[0]) for row in rows)
    strategy_width = max(len(strategy_name), *(len(row[1]) for row in rows))
    benchmark_width = max(len("Buy & hold"), *(len(row[2]) for row in rows))
    header = (
        f"{'':<{label_width}}  {strategy_name:>{strategy_width}}  "
        f"{'Buy & hold':>{benchmark_width}}"
    )
    rule = "-" * len(header)
    lines = [header, rule]
    for label, left, right in rows:
        lines.append(
            f"{label:<{label_width}}  {left:>{strategy_width}}  {right:>{benchmark_width}}"
        )
    return "\n".join(lines)


def _cagr(start, end, initial: float, final: float) -> float:
    elapsed_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    if elapsed_days <= 0 or final <= 0:
        return 0.0 if final == initial else float("nan")
    years = elapsed_days / 365.25
    return (final / initial) ** (1.0 / years) - 1.0


def _sharpe_and_volatility(equity: pd.Series) -> tuple[float, float]:
    returns = equity.iloc[1:] / equity.iloc[:-1].to_numpy() - 1.0
    if len(returns) < 2:
        return 0.0, 0.0
    daily_std = float(returns.std(ddof=1))
    if daily_std == 0.0 or np.isnan(daily_std):
        return 0.0, 0.0
    scale = np.sqrt(TRADING_DAYS)
    return float(returns.mean()) / daily_std * scale, daily_std * scale


def _pct(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.2f}"
