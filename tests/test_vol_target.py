import pandas as pd

from research.backtest import backtest_vol_target_with_positions
from research.strategies.vol_target_strategy import (
    VolTargetParams,
    build_vol_target_signal,
    make_vol_target_param_grid,
)


def _params() -> VolTargetParams:
    return VolTargetParams(
        target_ann_vol=0.60,
        realized_vol_window=2,
        min_exposure=0.25,
        max_exposure=2.00,
        trend_window=3,
        momentum_window=2,
        trend_multiplier_below_ma=0.50,
        momentum_boost=1.15,
        crash_vol_cutoff=0.50,
        crash_exposure=0.0,
    )


def test_vol_target_position_is_shifted_one_day() -> None:
    index = pd.bdate_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {
            "TQQQ": [100.0, 101.0, 102.0, 103.0, 104.0, 106.0, 108.0, 110.0],
            "QQQ": [100.0, 100.5, 101.0, 101.5, 102.0, 103.0, 104.0, 105.0],
        },
        index=index,
    )

    bt = backtest_vol_target_with_positions(
        data=data,
        params=_params(),
        transaction_cost_bps=0.0,
    )

    expected_position = bt["target_exposure"].shift(1).fillna(0.0)
    pd.testing.assert_series_equal(
        bt["position"],
        expected_position.rename("position"),
        check_names=True,
    )


def test_vol_target_crash_cut_uses_realized_daily_vol_and_trend() -> None:
    index = pd.bdate_range("2020-01-01", periods=8)
    tqqq = pd.Series([100.0, 110.0, 90.0, 112.0, 88.0, 114.0, 86.0, 116.0], index=index)
    qqq = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0], index=index)
    params = VolTargetParams(
        target_ann_vol=0.60,
        realized_vol_window=2,
        min_exposure=0.25,
        max_exposure=2.00,
        trend_window=3,
        momentum_window=2,
        trend_multiplier_below_ma=0.50,
        momentum_boost=1.00,
        crash_vol_cutoff=0.001,
        crash_exposure=0.0,
    )

    signal = build_vol_target_signal(tqqq, qqq, params)

    assert signal["crash_regime"].iloc[-1] == 1
    assert signal["target_exposure"].iloc[-1] == 0.0


def test_vol_target_grid_contains_expected_cartesian_product() -> None:
    grid = make_vol_target_param_grid(
        target_ann_vols=[0.45, 0.60],
        realized_vol_windows=[10],
        min_exposures=[0.25],
        max_exposures=[1.0, 1.5],
        trend_windows=[100],
        momentum_windows=[20],
        trend_multiplier_below_ma_options=[0.25],
        momentum_boosts=[1.0],
        crash_vol_cutoffs=[0.04],
        crash_exposures=[0.0, 0.25],
    )

    assert len(grid) == 8
