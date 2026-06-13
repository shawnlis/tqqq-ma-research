import pytest

from research.data import load_prices


def test_cash_symbol_creates_zero_return_price_series() -> None:
    prices = load_prices(
        start="2020-01-01",
        end="2020-01-10",
        symbols=["CASH"],
        use_csv_if_exists=False,
    )

    assert list(prices.columns) == ["CASH"]
    assert not prices.empty
    assert prices["CASH"].tolist() == pytest.approx([1.0] * len(prices))


def test_cash_symbol_can_create_synthetic_ohlc_columns() -> None:
    prices = load_prices(
        start="2020-01-01",
        end="2020-01-10",
        symbols=["CASH"],
        use_csv_if_exists=False,
        include_ohlc=True,
    )

    assert ["CASH", "CASH_OPEN", "CASH_HIGH", "CASH_LOW", "CASH_CLOSE"] == list(prices.columns)
    assert prices["CASH_OPEN"].tolist() == pytest.approx([1.0] * len(prices))
