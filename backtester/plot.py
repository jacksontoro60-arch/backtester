"""Equity and drawdown chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtester.metrics import drawdown

_STRATEGY_COLOR = "#0b3a82"
_BENCHMARK_COLOR = "#b35c00"


def save_chart(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    path: str | Path,
    title: str,
    note: str,
) -> Path:
    """Write a two-panel PNG: equity, then drawdown, against buy-and-hold."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (equity_axis, drawdown_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1]},
    )
    fig.patch.set_facecolor("white")

    equity_axis.plot(
        strategy_equity.index,
        strategy_equity.values,
        color=_STRATEGY_COLOR,
        linewidth=1.6,
        label="Strategy",
    )
    equity_axis.plot(
        benchmark_equity.index,
        benchmark_equity.values,
        color=_BENCHMARK_COLOR,
        linewidth=1.4,
        linestyle="--",
        label="Buy & hold",
    )
    equity_axis.set_ylabel("Equity")
    equity_axis.ticklabel_format(useOffset=False, style="plain", axis="y")
    equity_axis.legend(frameon=False, loc="best")
    equity_axis.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
    equity_axis.spines["top"].set_visible(False)
    equity_axis.spines["right"].set_visible(False)

    strategy_dd = drawdown(strategy_equity) * 100
    benchmark_dd = drawdown(benchmark_equity) * 100
    drawdown_axis.plot(strategy_dd.index, strategy_dd.values, color=_STRATEGY_COLOR, linewidth=1.2)
    drawdown_axis.plot(
        benchmark_dd.index,
        benchmark_dd.values,
        color=_BENCHMARK_COLOR,
        linewidth=1.1,
        linestyle="--",
    )
    drawdown_axis.axhline(0, color="#888888", linewidth=0.6)
    drawdown_axis.set_ylabel("Drawdown %")
    drawdown_axis.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
    drawdown_axis.spines["top"].set_visible(False)
    drawdown_axis.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.10)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output
