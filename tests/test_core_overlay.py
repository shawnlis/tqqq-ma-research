import pandas as pd
import pytest

import research.backtest as backtest
from research.strategies.core_overlay_strategy import (
    CoreOverlayParams,
    build_core_overlay_signal,
)


def _prices(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"TQQQ": values, "QQQ": values}, index=index)


def test_core_overlay_signal_respects_bounds_and_crash_floor() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    tqqq = pd.Series([100, 101, 103, 98, 91, 89, 88, 87], index=index, dtype=float)
    qqq = pd.Series([100, 101, 103, 98, 91, 89, 88, 87], index=index, dtype=float)
    params = CoreOverlayParams(
        core_exposure=0.50,
        overlay_max=0.65,
        max_total_exposure=1.00,
        trend_window=3,
        fast_trend_window=2,
        momentum_window=1,
        vol_window=2,
        vol_cap=0.000001,
        crash_cut_exposure=0.25,
        rebound_boost=False,
    )

    signal = build_core_overlay_signal(tqqq, qqq, params)

    assert signal.min() >= 0.25
    assert signal.max() <= 1.00
    assert signal.iloc[-1] == pytest.approx(0.25)


def test_core_overlay_backtest_shifts_position_and_charges_cost(monkeypatch) -> None:
    data = _prices([100.0, 110.0, 121.0, 121.0])
    signal = pd.Series([1.0, 1.0, 0.5, 0.5], index=data.index)
    monkeypatch.setattr(
        backtest,
        "build_core_overlay_signal",
        lambda tqqq, qqq, params: signal,
    )
    params = CoreOverlayParams(
        core_exposure=0.50,
        overlay_max=0.50,
        max_total_exposure=1.00,
        trend_window=3,
        fast_trend_window=2,
        momentum_window=1,
        vol_window=2,
        vol_cap=0.03,
        crash_cut_exposure=0.25,
        rebound_boost=False,
    )

    result = backtest.backtest_core_overlay_with_positions(
        data=data,
        params=params,
        transaction_cost_bps=100.0,
    )

    assert result["position"].tolist() == [0.0, 1.0, 1.0, 0.5]
    assert result["turnover"].tolist() == [0.0, 1.0, 0.0, 0.5]
    assert result.loc[data.index[1], "ret"] == pytest.approx(0.10 - 0.01)
    assert result.loc[data.index[3], "ret"] == pytest.approx(-0.005)
