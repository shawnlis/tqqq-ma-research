import pandas as pd

from research.strategies.rebound import apply_rebound_reentry


def test_rebound_reentry_forces_position_after_drawdown_reclaim() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    qqq = pd.Series([100.0, 100.0, 100.0, 80.0, 82.0, 85.0, 88.0], index=index)
    current_position = pd.Series(0.25, index=index)

    rebound = apply_rebound_reentry(
        qqq_close=qqq,
        tqqq_close=qqq,
        current_position=current_position,
        rolling_high_window=3,
        drawdown_trigger=-0.10,
        rebound_momentum_window=1,
        rebound_momentum_threshold=0.02,
        reclaim_ma_window=2,
        rebound_position=1.25,
        rebound_hold_days=2,
        extreme_crash_vol_threshold=1.0,
    )

    assert rebound.loc[index[4]] == 1.25
    assert rebound.loc[index[5]] == 1.25
    assert rebound.min() >= current_position.min()


def test_rebound_reentry_blocked_by_extreme_volatility() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    qqq = pd.Series([100.0, 100.0, 100.0, 80.0, 82.0, 85.0, 88.0], index=index)
    current_position = pd.Series(0.25, index=index)

    rebound = apply_rebound_reentry(
        qqq_close=qqq,
        tqqq_close=qqq,
        current_position=current_position,
        rolling_high_window=3,
        drawdown_trigger=-0.10,
        rebound_momentum_window=1,
        rebound_momentum_threshold=0.02,
        reclaim_ma_window=2,
        rebound_position=1.25,
        rebound_hold_days=2,
        extreme_crash_vol_window=2,
        extreme_crash_vol_threshold=0.000001,
    )

    assert rebound.max() == 0.25
