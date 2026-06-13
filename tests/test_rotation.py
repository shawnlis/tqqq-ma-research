import pandas as pd
import pytest
import numpy as np

import research.backtest as backtest
from research.strategies.rotation_strategy import RotationParams, build_rotation_signal


def _prices(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "TQQQ": values,
            "SOXL": values,
            "UPRO": values,
            "QQQ": values,
            "CASH": 1.0,
        },
        index=index,
    )


def test_rotation_backtest_shifts_weights_and_charges_all_asset_turnover(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _prices([100.0, 110.0, 121.0, 121.0])
    signal = pd.DataFrame(
        {
            "target_weight_TQQQ": [1.0, 1.0, 0.0, 0.0],
            "target_weight_SOXL": [0.0, 0.0, 1.0, 1.0],
            "target_weight_CASH": [0.0, 0.0, 0.0, 0.0],
            "selected_assets": ["TQQQ", "TQQQ", "SOXL", "SOXL"],
            "selected_score": [1.0, 1.0, 1.0, 1.0],
            "is_rebalance_date": [1, 0, 1, 0],
            "trend_filter_allows_risk": [1, 1, 1, 1],
            "available_risk_assets": ["TQQQ;SOXL"] * 4,
        },
        index=data.index,
    )
    monkeypatch.setattr(backtest, "build_rotation_signal", lambda data, params: signal)

    bt = backtest.backtest_rotation_with_positions(
        data=data,
        params=RotationParams(risk_assets=("TQQQ", "SOXL"), defensive_asset="CASH"),
        transaction_cost_bps=100.0,
    )

    assert bt["weight_TQQQ"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert bt["weight_SOXL"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert bt["turnover"].tolist() == [0.0, 1.0, 0.0, 2.0]
    assert bt.loc[data.index[1], "ret"] == pytest.approx(0.10 - 0.01)
    assert bt.loc[data.index[3], "ret"] == pytest.approx(-0.02)


def test_rotation_signal_uses_defensive_asset_when_trend_filter_blocks_risk() -> None:
    index = pd.bdate_range("2020-01-01", periods=230)
    steps = np.arange(230, dtype=float)
    downtrend = pd.Series(100.0 - steps * 0.1, index=index)
    data = pd.DataFrame(
        {
            "TQQQ": 100.0 + steps,
            "SOXL": 100.0 + steps * 1.1,
            "UPRO": 100.0 + steps * 0.8,
            "QQQ": downtrend,
            "CASH": 1.0,
        },
        index=index,
    )

    signal = build_rotation_signal(
        data,
        RotationParams(
            risk_assets=("TQQQ", "SOXL", "UPRO"),
            rebalance_frequency="monthly",
            defensive_asset="CASH",
            trend_filter_symbol="QQQ",
            trend_filter_window=100,
        ),
    )

    last = signal.iloc[-1]
    assert last["target_weight_CASH"] == pytest.approx(1.0)
    assert last["selected_assets"] == "CASH"


def test_rotation_eval_piece_ignores_missing_asset_before_eval_window(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    data = pd.DataFrame(
        {
            "TQQQ": [100.0, 101.0, 102.0, 103.0],
            "SGOV": [np.nan, np.nan, 1.0, 1.0],
            "QQQ": [100.0, 101.0, 102.0, 103.0],
        },
        index=index,
    )
    signal = pd.DataFrame(
        {
            "target_weight_SGOV": [1.0, 1.0, 1.0, 1.0],
            "selected_assets": ["SGOV"] * 4,
            "selected_score": [np.nan] * 4,
            "is_rebalance_date": [1, 0, 0, 0],
            "trend_filter_allows_risk": [0, 0, 0, 0],
            "available_risk_assets": ["TQQQ"] * 4,
        },
        index=index,
    )
    monkeypatch.setattr(backtest, "build_rotation_signal", lambda data, params: signal)

    piece = backtest.make_rotation_eval_piece(
        full_data=data,
        params=RotationParams(risk_assets=("TQQQ",), defensive_asset="SGOV"),
        eval_start=index[2],
        eval_end=index[3],
        transaction_cost_bps=0.0,
    )

    assert piece["weight_SGOV"].tolist() == [1.0, 1.0]
