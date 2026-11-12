import pandas as pd

from backtester.cli import main
from backtester.plot import save_chart
from tests.helpers import SAMPLE_CSV, make_prices


def test_chart_file_is_a_png(tmp_path):
    prices = make_prices(
        ["2020-01-01", "2020-01-02", "2020-01-03"],
        {"A": {"open": [10, 10, 11], "close": [10, 12, 11]}},
    )
    equity = pd.Series([100.0, 120.0, 110.0], index=prices.dates)
    path = tmp_path / "out" / "equity.png"
    save_chart(equity, equity, path, title="test", note="Synthetic prices.")
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_runs_offline_on_the_sample(tmp_path, capsys):
    chart = tmp_path / "chart.png"
    trades = tmp_path / "trades.csv"
    code = main(
        [
            "run",
            "--csv",
            str(SAMPLE_CSV),
            "--strategy",
            "sma_cross",
            "--fast",
            "20",
            "--slow",
            "50",
            "--chart",
            str(chart),
            "--trades-csv",
            str(trades),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    header = trades.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("symbol,entry_date,entry_price")
    assert "Total return" in captured.out
    assert "Buy & hold" in captured.out
    assert "Synthetic" in captured.out
    assert "Downloading" not in captured.err


def test_cli_runs_rsi_offline(tmp_path, capsys):
    chart = tmp_path / "rsi.png"
    code = main(
        [
            "run",
            "--csv",
            str(SAMPLE_CSV),
            "--strategy",
            "rsi",
            "--rsi-period",
            "14",
            "--chart",
            str(chart),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "rsi 14" in captured.out
    assert chart.exists()


def test_cli_rejects_a_bad_window(tmp_path, capsys):
    code = main(
        [
            "run",
            "--csv",
            str(SAMPLE_CSV),
            "--strategy",
            "sma_cross",
            "--fast",
            "50",
            "--slow",
            "20",
            "--chart",
            str(tmp_path / "nope.png"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "error:" in captured.err
