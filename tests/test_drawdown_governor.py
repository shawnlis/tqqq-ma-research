import pandas as pd

from research.strategies.drawdown_governor import (
    DrawdownGovernorParams,
    apply_drawdown_governor,
)


def test_drawdown_governor_caps_exposure_on_portfolio_drawdown() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    desired = pd.Series(1.25, index=index)
    equity = pd.Series([1.00, 0.98, 0.76, 0.75, 0.74], index=index)
    qqq = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0], index=index)

    governed = apply_drawdown_governor(
        strategy_equity=equity,
        desired_position=desired,
        qqq_close=qqq,
        tqqq_close=qqq,
        params=DrawdownGovernorParams(
            use_drawdown_governor=True,
            portfolio_dd_trigger=-0.20,
            qqq_dd_trigger=-0.50,
            reduced_exposure=0.50,
            recovery_ma_window=2,
            recovery_momentum_window=1,
            recovery_momentum_threshold=0.05,
            max_days_reduced=20,
        ),
    )

    assert governed.loc[index[2], "governed_position"] == 0.50
    assert governed.loc[index[2], "governor_active"] == 1


def test_drawdown_governor_restores_on_recovery_signal() -> None:
    index = pd.bdate_range("2020-01-01", periods=6)
    desired = pd.Series(1.00, index=index)
    equity = pd.Series([1.00, 0.78, 0.76, 0.76, 0.76, 0.76], index=index)
    qqq = pd.Series([100.0, 92.0, 90.0, 94.0, 98.0, 101.0], index=index)

    governed = apply_drawdown_governor(
        strategy_equity=equity,
        desired_position=desired,
        qqq_close=qqq,
        tqqq_close=qqq,
        params=DrawdownGovernorParams(
            use_drawdown_governor=True,
            portfolio_dd_trigger=-0.20,
            qqq_dd_trigger=-0.50,
            reduced_exposure=0.25,
            recovery_ma_window=2,
            recovery_momentum_window=1,
            recovery_momentum_threshold=0.03,
            max_days_reduced=20,
        ),
    )

    assert governed.loc[index[1], "governed_position"] == 0.25
    assert governed.loc[index[3], "recovery_signal"] == 1
    assert governed.loc[index[3], "governed_position"] == 1.00
    assert governed.loc[index[4], "governed_position"] == 1.00


def test_drawdown_governor_gradually_restores_after_max_days_without_new_low() -> None:
    index = pd.bdate_range("2020-01-01", periods=7)
    desired = pd.Series(1.00, index=index)
    equity = pd.Series([1.00, 0.75, 0.74, 0.74, 0.74, 0.74, 0.74], index=index)
    qqq = pd.Series([100.0, 90.0, 91.0, 91.5, 92.0, 92.5, 93.0], index=index)

    governed = apply_drawdown_governor(
        strategy_equity=equity,
        desired_position=desired,
        qqq_close=qqq,
        tqqq_close=qqq,
        params=DrawdownGovernorParams(
            use_drawdown_governor=True,
            portfolio_dd_trigger=-0.20,
            qqq_dd_trigger=-0.50,
            reduced_exposure=0.25,
            recovery_ma_window=50,
            recovery_momentum_window=1,
            recovery_momentum_threshold=0.50,
            max_days_reduced=2,
        ),
    )

    assert governed.loc[index[1], "governed_position"] == 0.25
    assert governed.loc[index[3], "governed_position"] > 0.25
    assert governed.loc[index[4], "governed_position"] > governed.loc[index[3], "governed_position"]
