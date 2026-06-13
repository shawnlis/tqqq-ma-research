import pandas as pd
import pytest

from research.data import make_synthetic_leveraged_price, make_synthetic_leveraged_returns


def test_leverage_one_equals_base_returns_minus_fees() -> None:
    base_returns = pd.Series([0.01, -0.02], index=pd.date_range("2020-01-01", periods=2))

    synthetic = make_synthetic_leveraged_returns(
        base_returns,
        leverage=1.0,
        annual_expense_ratio=0.252,
        annual_financing_spread=0.0,
    )

    assert synthetic.tolist() == pytest.approx([0.009, -0.021])


def test_leverage_three_produces_three_times_daily_returns_before_fees() -> None:
    base_returns = pd.Series([0.01, -0.02], index=pd.date_range("2020-01-01", periods=2))

    synthetic = make_synthetic_leveraged_returns(
        base_returns,
        leverage=3.0,
        annual_expense_ratio=0.0,
        annual_financing_spread=0.0,
    )

    assert synthetic.tolist() == pytest.approx([0.03, -0.06])


def test_synthetic_equity_prefix_does_not_use_future_data() -> None:
    prices = pd.Series(
        [100.0, 101.0, 99.0, 150.0],
        index=pd.date_range("2020-01-01", periods=4),
    )

    prefix_equity = make_synthetic_leveraged_price(
        prices.iloc[:3],
        leverage=3.0,
        annual_expense_ratio=0.0,
        annual_financing_spread=0.0,
    )
    full_equity = make_synthetic_leveraged_price(
        prices,
        leverage=3.0,
        annual_expense_ratio=0.0,
        annual_financing_spread=0.0,
    )

    assert full_equity.iloc[:3].tolist() == pytest.approx(prefix_equity.tolist())
