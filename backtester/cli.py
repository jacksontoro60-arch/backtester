"""Command line: python -m backtester run ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtester.data import file_says_synthetic, load_prices
from backtester.engine import run_backtest
from backtester.metrics import render_table
from backtester.plot import save_chart
from backtester.strategies import STRATEGIES, BuyAndHold, RSIMeanReversion, SMACross
from backtester.strategies.base import Strategy


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    symbols = _split_symbols(args.symbols) if args.symbols else None
    on_status = lambda message: print(message, file=sys.stderr)
    prices = load_prices(
        symbols=symbols,
        start=args.start,
        end=args.end,
        csv_path=args.csv,
        cache_dir=args.cache_dir,
        on_status=on_status,
    )
    strategy = _build_strategy(args)
    result = run_backtest(
        prices,
        strategy,
        starting_cash=args.cash,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )
    benchmark = run_backtest(
        prices,
        BuyAndHold(),
        starting_cash=args.cash,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )

    synthetic = bool(args.csv) and file_says_synthetic(args.csv)
    start = prices.dates[0].date().isoformat()
    end = prices.dates[-1].date().isoformat()
    symbol_list = ", ".join(prices.symbols)
    print(f"Backtest: {strategy.name} on {symbol_list}")
    print(
        f"{start} to {end}   cash {_cash(args.cash)}   "
        f"commission {args.commission_bps:g} bps   slippage {args.slippage_bps:g} bps"
    )
    print("Signals fill at the next open. Long only, equal cash per symbol.")
    if synthetic:
        print("Synthetic prices. Not real market data.")
    print()
    print(render_table(strategy.name, result.metrics, benchmark.metrics))
    print()

    note = "Synthetic prices. Not real market data." if synthetic else "Daily bars. Signals fill at the next open."
    chart = save_chart(
        result.equity,
        benchmark.equity,
        args.chart,
        title=f"{strategy.name} vs buy and hold",
        note=note,
    )
    print(f"Saved chart to {chart}")
    if args.trades_csv:
        trades_path = Path(args.trades_csv)
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        result.trades.to_csv(trades_path, index=False)
        print(f"Saved trades to {trades_path}")
    return 0


def _build_strategy(args: argparse.Namespace) -> Strategy:
    if args.strategy == "sma_cross":
        return SMACross(fast=args.fast, slow=args.slow)
    if args.strategy == "rsi":
        return RSIMeanReversion(
            period=args.rsi_period,
            oversold=args.oversold,
            overbought=args.overbought,
        )
    if args.strategy == "buy_hold":
        return BuyAndHold()
    raise ValueError(f"Unknown strategy: {args.strategy}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtester",
        description="Backtest a few long-only daily stock strategies. Not financial advice.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run one backtest and write a chart.")
    run.add_argument("--symbols", help="Comma-separated symbols, for example AAPL,MSFT.")
    run.add_argument("--start", help="First date, YYYY-MM-DD. Required when downloading.")
    run.add_argument("--end", help="Last date, YYYY-MM-DD. Inclusive. Optional.")
    run.add_argument(
        "--strategy",
        required=True,
        choices=sorted(STRATEGIES),
        help="sma_cross, buy_hold, or rsi.",
    )
    run.add_argument("--fast", type=int, default=20, help="Fast SMA window. Default 20.")
    run.add_argument("--slow", type=int, default=50, help="Slow SMA window. Default 50.")
    run.add_argument("--rsi-period", type=int, default=14, help="RSI lookback. Default 14.")
    run.add_argument("--oversold", type=float, default=30, help="RSI entry level. Default 30.")
    run.add_argument("--overbought", type=float, default=70, help="RSI exit level. Default 70.")
    run.add_argument("--cash", type=float, default=10_000, help="Starting cash. Default 10000.")
    run.add_argument("--commission-bps", type=float, default=0, help="Commission in basis points.")
    run.add_argument("--slippage-bps", type=float, default=0, help="Slippage in basis points.")
    run.add_argument("--csv", help="OHLC CSV to use instead of downloading. Works offline.")
    run.add_argument("--cache-dir", default="cache", help="Where downloaded CSV files are stored.")
    run.add_argument("--chart", default="output/equity.png", help="Where to write the PNG.")
    run.add_argument("--trades-csv", help="Optional path for the trade list.")
    return parser


def _split_symbols(text: str) -> list[str]:
    symbols = [part.strip() for part in text.split(",") if part.strip()]
    if not symbols:
        raise ValueError("No symbols given.")
    return symbols


def _cash(amount: float) -> str:
    if float(amount).is_integer():
        return f"{amount:,.0f}"
    return f"{amount:,.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
