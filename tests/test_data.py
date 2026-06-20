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


def test_cached_load_prices_honors_start_and_end(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", "2020-01-10"),
            "AAA": [100.0 + i for i in range(8)],
        }
    ).to_csv(cache_dir / "aaa_prices.csv", index=False)

    prices = load_prices(
        start="2020-01-03",
        end="2020-01-07",
        symbols=["AAA"],
        cache_dir=str(cache_dir),
        use_csv_if_exists=True,
    )

    assert prices.index.min() == pd.Timestamp("2020-01-03")
    assert prices.index.max() == pd.Timestamp("2020-01-07")
